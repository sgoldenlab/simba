import os
from typing import Union, Optional
import pandas as pd
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_file_exist_and_readable, check_valid_boolean
from simba.utils.errors import InvalidInputError
from simba.utils.enums import Keys
from simba.roi_tools.roi_utils import get_rectangle_df_headers, get_circle_df_headers, get_polygon_df_headers
from simba.utils.warnings import NoFileFoundWarning, MissingFileWarning
from simba.utils.printing import stdout_success
from simba.utils.read_write import find_all_videos_in_directory



VIDEO = 'Video'
EXPECTED_RECT_COLS = ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'Center_X', 'Center_Y', 'topLeftX', 'topLeftY', 'Bottom_right_X', 'Bottom_right_Y', 'width', 'height', 'width_cm', 'height_cm', 'area_cm','Tags', 'Ear_tag_size']
EXPECTED_CIRC_COLS = ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'centerX', 'centerY', 'radius', 'radius_cm', 'area_cm', 'Tags', 'Ear_tag_size', 'Center_X', 'Center_Y']
EXPECTED_POLY_COLS = ['Video', 'Shape_type', 'Name', 'Color name', 'Color BGR', 'Thickness', 'Center_X', 'Center_Y', 'vertices', 'center', 'area', 'max_vertice_distance', 'area_cm', 'Tags', 'Ear_tag_size']

#EXPECTED_RECT_COLS, EXPECTED_CIRC_COLS, EXPECTED_POLY_COLS = get_rectangle_df_headers(), get_circle_df_headers(), get_polygon_df_headers()

class ROIDefinitionsCSVImporter(ConfigReader):
    """
    Import ROI definitions from CSV files into SimBA H5 format.

    Converts human-readable CSV files containing ROI definitions (rectangles, circles, and/or polygons)
    into SimBA's native H5 format for use in ROI analysis workflows. At least one CSV file path
    (rectangles, circles, or polygons) must be provided.

    .. note::
       ROI CSV files can be created from existing SimBA H5 ROI definitions using :func:`simba.utils.data.convert_roi_definitions`.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param Optional[Union[str, os.PathLike]] rectangles_path: Path to CSV file containing rectangle ROI definitions. If None, no rectangles will be imported.
    :param Optional[Union[str, os.PathLike]] circles_path: Path to CSV file containing circle ROI definitions. If None, no circles will be imported.
    :param Optional[Union[str, os.PathLike]] polygon_path: Path to CSV file containing polygon ROI definitions. If None, no polygons will be imported.
    :param bool append: If True, append ROI definitions to existing ROI_definitions.h5 file. If False, overwrite existing file. Default: False.

    :example:
    >>> importer = ROIDefinitionsCSVImporter(
    ...     config_path=r"project_folder/project_config.ini",
    ...     rectangles_path=r"logs/measures/rectangles_20251122110043.csv",
    ...     circles_path=r"logs/measures/circles_20251122110043.csv",
    ...     polygon_path=r"logs/measures/polygons_20251122110043.csv",
    ...     append=False
    ... )
    >>> importer.run()

    :example:
    >>> importer = ROIDefinitionsCSVImporter(
    ...     config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini",
    ...     circles_path=r"C:\troubleshooting\mouse_open_field\project_folder\logs\measures\circles_20251122110043.csv",
    ...     rectangles_path=r"C:\troubleshooting\mouse_open_field\project_folder\logs\measures\rectangles_20251122110043.csv",
    ...     polygon_path=r"C:\troubleshooting\mouse_open_field\project_folder\logs\measures\polygons_20251122110043.csv"
    ... )
    >>> importer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 rectangles_path: Optional[Union[str, os.PathLike]] = None,
                 circles_path: Optional[Union[str, os.PathLike]] = None,
                 polygon_path: Optional[Union[str, os.PathLike]] = None,
                 append: bool = False):

        if rectangles_path is None and circles_path is None and polygon_path is None:
            raise InvalidInputError(msg='Please pass at path to rectangles, circles, and/or polygon CSVs. They are all None', source=self.__class__.__name__)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        if rectangles_path is not None: check_file_exist_and_readable(file_path=rectangles_path, raise_error=True)
        if circles_path is not None:  check_file_exist_and_readable(file_path=circles_path, raise_error=True)
        if polygon_path is not None: check_file_exist_and_readable(file_path=polygon_path, raise_error=True)
        check_valid_boolean(value=append, source=f'{self.__class__.__name__} append', raise_error=True)
        if append and not os.path.isfile(self.roi_coordinates_path):
            NoFileFoundWarning(msg='Cannot APPEND ROI CSV data, the expected file {} does not exist to append data to. A new file will be created', source=self.__class__.__name__)
            append = False
        self.video_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=False, sort_alphabetically=True)
        self.project_video_names = list(self.video_dict.keys())
        self.write_mode = 'w' if not append else 'a'
        self.circles_df, self.rectangles_df, self.polygon_df, self.append = None, None, None, append
        self.new_rect_cnt, self.new_circ_cnt, self.new_poly_cnt, self.new_video_names = 0, 0, 0, set()
        if rectangles_path is not None:
            self.rectangles_df = pd.read_csv(filepath_or_buffer=rectangles_path, index_col=0)
            self.new_rect_cnt += len(self.rectangles_df)
            self.new_video_names.update(list(self.rectangles_df[VIDEO]))
            missing_cols = [x for x in list(self.rectangles_df.columns) if x not in EXPECTED_RECT_COLS]
            if len(missing_cols) > 0:
                raise InvalidInputError(msg=f'The RECTANGLE ROI CSV file {rectangles_path} is missing the expected columns {missing_cols}', source=self.__class__.__name__)
        else:
            self.rectangles_df = pd.DataFrame(columns=get_rectangle_df_headers())
        if circles_path is not None:
            self.circles_df = pd.read_csv(filepath_or_buffer=circles_path, index_col=0)
            self.new_circ_cnt += len(self.circles_df)
            self.new_video_names.update(list(self.circles_df[VIDEO]))
            missing_cols = [x for x in list(self.circles_df.columns) if x not in EXPECTED_CIRC_COLS]
            if len(missing_cols) > 0:
                raise InvalidInputError(msg=f'The CIRCLE ROI CSV file {circles_path} is missing the expected columns {missing_cols}', source=self.__class__.__name__)
        else:
            self.circles_df = pd.DataFrame(columns=get_circle_df_headers())
        if polygon_path is not None:
            self.polygon_df = pd.read_csv(filepath_or_buffer=polygon_path, index_col=0)
            self.new_poly_cnt += len(self.polygon_df)
            self.new_video_names.update(list(self.polygon_df[VIDEO]))
            missing_cols = [x for x in list(self.polygon_df.columns) if x not in EXPECTED_POLY_COLS]
            if len(missing_cols) > 0:
                raise InvalidInputError(msg=f'The POLYGON ROI CSV file {polygon_path} is missing the expected columns {missing_cols}', source=self.__class__.__name__)
        else:
            self.polygon_df = pd.DataFrame(columns=get_polygon_df_headers())

    def run(self):
        store = pd.HDFStore(self.roi_coordinates_path, mode=self.write_mode)
        store[Keys.ROI_RECTANGLES.value] = self.rectangles_df
        store[Keys.ROI_CIRCLES.value] = self.circles_df
        store[Keys.ROI_POLYGONS.value] = self.polygon_df
        store[Keys.ROI_RECTANGLES.value]  = store[Keys.ROI_RECTANGLES.value].drop_duplicates()
        store[Keys.ROI_CIRCLES.value] = store[Keys.ROI_CIRCLES.value].drop_duplicates()
        store[Keys.ROI_POLYGONS.value] = store[Keys.ROI_POLYGONS.value].drop_duplicates()
        store.close()

        new_roi_cnt = self.new_poly_cnt + self.new_rect_cnt + self.new_circ_cnt
        new_video_names = list(self.new_video_names)
        missing_videos = [x for x in new_video_names if x not in self.project_video_names]
        if len(missing_videos) > 0:
            MissingFileWarning(msg=f'Imported ROIs for {len(missing_videos)} videos which do not exist in the {self.video_dir} directory: {missing_videos}.', source=self.__class__.__name__)
        self.timer.stop_timer()
        stdout_success(msg=f'{new_roi_cnt} new ROIs for {len(new_video_names)} videos imported to SimBA project.', source=self.__class__.__name__, elapsed_time=self.timer.elapsed_time_str)





# u = ROIDefinitionsCSVImporter(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini",  circles_path=r"C:\troubleshooting\mouse_open_field\project_folder\logs\measures\circles_20251122110043.csv", rectangles_path=r"C:\troubleshooting\mouse_open_field\project_folder\logs\measures\rectangles_20251122110043.csv", polygon_path=r"C:\troubleshooting\mouse_open_field\project_folder\logs\measures\polygons_20251122110043.csv")
# u.run()
