__author__ = "Simon Nilsson"

import glob
import itertools
import json
import logging
import logging.config
import os
import shutil
from ast import literal_eval
from configparser import ConfigParser
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.enums import ConfigKey, Defaults, Dtypes, Keys, Paths
from simba.utils.errors import (BodypartColumnNotFoundError, DataHeaderError,
                                DuplicationError, InvalidInputError,
                                MissingProjectConfigEntryError,
                                NoFilesFoundError, NoROIDataError,
                                NotDirectoryError, ParametersFileError,
                                PermissionError)
from simba.utils.lookups import (create_color_palettes, get_color_dict,
                                 get_emojis, get_log_config)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt, get_all_clf_names,
                                    get_fn_ext, read_config_file, read_df,
                                    read_project_path_and_file_type, write_df)
from simba.utils.warnings import (BodypartColumnNotFoundWarning,
                                  InvalidValueWarning, NoDataFoundWarning,
                                  NoFileFoundWarning)


class ConfigReader(object):
    """
    Methods for reading SimBA configparser.Configparser project config and associated project data.

    :param configparser.Configparser config_path: path to SimBA project_config.ini
    :param bool read_video_info: if true, read the project_folder/logs/video_info.csv file.
    """

    def __init__(
        self, config_path: str, read_video_info: bool = True, create_logger: bool = True
    ):

        self.timer = SimbaTimer(start=True)
        self.config_path = config_path
        self.config = read_config_file(config_path=config_path)
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        self.project_path, self.file_type = read_project_path_and_file_type(
            config=self.config
        )
        self.input_csv_dir = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        self.features_dir = os.path.join(
            self.project_path, Paths.FEATURES_EXTRACTED_DIR.value
        )
        self.targets_folder = os.path.join(
            self.project_path, Paths.TARGETS_INSERTED_DIR.value
        )
        self.input_frames_dir = os.path.join(
            self.project_path, Paths.INPUT_FRAMES_DIR.value
        )
        self.machine_results_dir = os.path.join(
            self.project_path, Paths.MACHINE_RESULTS_DIR.value
        )
        self.directionality_df_dir = os.path.join(
            self.project_path, Paths.DIRECTIONALITY_DF_DIR.value
        )
        self.body_part_directionality_df_dir = os.path.join(
            self.project_path, Paths.BODY_PART_DIRECTIONALITY_DF_DIR.value
        )
        self.outlier_corrected_dir = os.path.join(
            self.project_path, Paths.OUTLIER_CORRECTED.value
        )
        self.outlier_corrected_movement_dir = os.path.join(
            self.project_path, Paths.OUTLIER_CORRECTED_MOVEMENT.value
        )
        self.heatmap_clf_location_dir = os.path.join(
            self.project_path, Paths.HEATMAP_CLF_LOCATION_DIR.value
        )
        self.heatmap_location_dir = os.path.join(
            self.project_path, Paths.HEATMAP_LOCATION_DIR.value
        )
        self.line_plot_dir = os.path.join(self.project_path, Paths.LINE_PLOT_DIR.value)
        self.probability_plot_dir = os.path.join(
            self.project_path, Paths.PROBABILITY_PLOTS_DIR.value
        )
        self.gantt_plot_dir = os.path.join(
            self.project_path, Paths.GANTT_PLOT_DIR.value
        )
        self.path_plot_dir = os.path.join(self.project_path, Paths.PATH_PLOT_DIR.value)
        self.shap_logs_path = os.path.join(self.project_path, Paths.SHAP_LOGS.value)
        self.video_info_path = os.path.join(self.project_path, Paths.VIDEO_INFO.value)
        self.frames_output_dir = os.path.join(
            self.project_path, Paths.FRAMES_OUTPUT_DIR.value
        )
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.roi_features_save_dir = os.path.join(
            self.project_path, Paths.ROI_FEATURES.value
        )
        self.configs_meta_dir = os.path.join(self.project_path, "configs")
        self.sklearn_plot_dir = os.path.join(
            self.project_path, Paths.SKLEARN_RESULTS.value
        )
        self.detailed_roi_data_dir = os.path.join(
            self.project_path, Paths.DETAILED_ROI_DATA_DIR.value
        )
        self.directing_animals_video_output_path = os.path.join(
            self.project_path, Paths.DIRECTING_BETWEEN_ANIMALS_OUTPUT_PATH.value
        )
        self.directing_body_part_animal_video_output_path = os.path.join(
            self.project_path,
            Paths.DIRECTING_BETWEEN_ANIMAL_BODY_PART_OUTPUT_PATH.value,
        )
        self.animal_cnt = self.read_config_entry(
            config=self.config,
            section=ConfigKey.GENERAL_SETTINGS.value,
            option=ConfigKey.ANIMAL_CNT.value,
            data_type=Dtypes.INT.value,
        )
        self.bodypart_direction = self.read_config_entry(
            self.config,
            section=ConfigKey.DIRECTIONALITY_SETTINGS.value,
            option=ConfigKey.BODYPART_DIRECTION_VALUE.value,
            data_type=Dtypes.STR.value,
            default_value=Dtypes.NONE.value,
        )
        self.clf_cnt = self.read_config_entry(self.config, ConfigKey.SML_SETTINGS.value, ConfigKey.TARGET_CNT.value, Dtypes.INT.value)
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.clf_cnt)
        self.feature_file_paths = glob.glob(self.features_dir + f"/*.{self.file_type}")
        self.target_file_paths = glob.glob(self.targets_folder + f"/*.{self.file_type}")
        self.input_csv_paths = glob.glob(self.input_csv_dir + f"/*.{self.file_type}")
        self.body_part_directionality_paths = glob.glob(
            self.body_part_directionality_df_dir + f"/*.{self.file_type}"
        )
        self.outlier_corrected_paths = glob.glob(
            self.outlier_corrected_dir + f"/*.{self.file_type}"
        )
        self.outlier_corrected_movement_paths = glob.glob(
            self.outlier_corrected_movement_dir + f"/*.{self.file_type}"
        )
        self.cpu_cnt, self.cpu_to_use = find_core_cnt()
        self.machine_results_paths = glob.glob(
            self.machine_results_dir + f"/*.{self.file_type}"
        )
        self.logs_path = os.path.join(self.project_path, "logs")
        self.body_parts_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
        check_file_exist_and_readable(file_path=self.body_parts_path)
        self.body_parts_lst = (
            pd.read_csv(self.body_parts_path, header=None).iloc[:, 0].to_list()
        )
        self.body_parts_lst = [x for x in self.body_parts_lst if str(x) != "nan"]
        self.get_body_part_names()
        self.get_bp_headers()
        self.bp_col_names = self.x_cols + self.y_cols + self.p_cols
        self.pose_setting = self.read_config_entry(
            config=self.config,
            section=ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            option=ConfigKey.POSE_SETTING.value,
            data_type=Dtypes.STR.value,
        )
        self.roi_coordinates_path = os.path.join(
            self.logs_path, Paths.ROI_DEFINITIONS.value
        )
        self.video_dir = os.path.join(self.project_path, "videos")
        self.clf_validation_dir = os.path.join(
            self.project_path, Paths.CLF_VALIDATION_DIR.value
        )
        self.clf_data_validation_dir = os.path.join(
            self.project_path, "csv", "validation"
        )
        self.annotated_frm_dir = os.path.join(
            self.project_path, Paths.ANNOTATED_FRAMES_DIR.value
        )
        self.single_validation_video_save_dir = os.path.join(
            self.project_path, Paths.SINGLE_CLF_VALIDATION.value
        )
        self.data_table_path = os.path.join(self.project_path, Paths.DATA_TABLE.value)
        self.check_multi_animal_status()
        self.multiprocess_chunksize = Defaults.CHUNK_SIZE.value
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.clr_lst = create_color_palettes(self.animal_cnt, int(len(self.x_cols)) + 1)
        self.animal_bp_dict = self.create_body_part_dictionary(
            self.multi_animal_status,
            self.multi_animal_id_list,
            self.animal_cnt,
            self.x_cols,
            self.y_cols,
            self.p_cols,
            self.clr_lst,
        )
        self.project_bps = list(set([x[:-2] for x in self.bp_headers]))
        self.color_dict = get_color_dict()
        if create_logger:
            self.create_logger()
        self.emojis = get_emojis()
        if read_video_info:
            self.video_info_df = self.read_video_info_csv(
                file_path=self.video_info_path
            )

    def read_roi_data(self) -> None:
        """
        Method to read in ROI definitions from SimBA project
        """

        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(msg="SIMBA ERROR: No ROI definitions were found in your SimBA project. Please draw some ROIs before analyzing your ROI data", source=self.__class__.__name__,
            )
        else:
            self.rectangles_df = pd.read_hdf(
                self.roi_coordinates_path, key=Keys.ROI_RECTANGLES.value
            )
            if ("Center_X" in self.rectangles_df.columns) and (
                self.rectangles_df["Center_X"].isnull().values.any()
            ):
                for idx, row in self.rectangles_df.iterrows():
                    self.rectangles_df.loc[idx]["Center_X"] = row["Tags"]["Center tag"][
                        0
                    ]
                    self.rectangles_df.loc[idx]["Center_Y"] = row["Tags"]["Center tag"][
                        1
                    ]
            self.circles_df = pd.read_hdf(
                self.roi_coordinates_path, key=Keys.ROI_CIRCLES.value
            ).dropna(how="any")
            self.polygon_df = pd.read_hdf(
                self.roi_coordinates_path, key=Keys.ROI_POLYGONS.value
            )
            if "Center_XCenter_Y" in self.polygon_df.columns:
                self.polygon_df = self.polygon_df.drop(["Center_XCenter_Y"], axis=1)
            self.polygon_df = self.polygon_df.dropna(how="any")
            self.shape_names = list(itertools.chain(self.rectangles_df["Name"].unique(), self.polygon_df["Name"].unique(), self.circles_df["Name"].unique()))
            self.roi_dict = {
                Keys.ROI_RECTANGLES.value: self.rectangles_df,
                Keys.ROI_CIRCLES.value: self.circles_df,
                Keys.ROI_POLYGONS.value: self.polygon_df,
            }
            self.roi_types_names_lst = set()
            for idx, r in self.roi_dict[Keys.ROI_RECTANGLES.value].iterrows():
                self.roi_types_names_lst.add(f'Rectangle: {r["Name"]}')
            for idx, r in self.roi_dict[Keys.ROI_CIRCLES.value].iterrows():
                self.roi_types_names_lst.add(f'Circle: {r["Name"]}')
            for idx, r in self.roi_dict[Keys.ROI_POLYGONS.value].iterrows():
                self.roi_types_names_lst.add(f'Polygon: {r["Name"]}')
            self.roi_types_names_lst = list(self.roi_types_names_lst)
            for shape_type, shape_data in self.roi_dict.items():
                if shape_type == Keys.ROI_CIRCLES.value:
                    self.roi_dict[Keys.ROI_CIRCLES.value]["Center_X"] = self.roi_dict[
                        Keys.ROI_CIRCLES.value
                    ]["centerX"]
                    self.roi_dict[Keys.ROI_CIRCLES.value]["Center_Y"] = self.roi_dict[
                        Keys.ROI_CIRCLES.value
                    ]["centerY"]
                elif shape_type == Keys.ROI_RECTANGLES.value:
                    self.roi_dict[Keys.ROI_RECTANGLES.value][
                        "Center_X"
                    ] = self.roi_dict[Keys.ROI_RECTANGLES.value]["Bottom_right_X"] - (
                        self.roi_dict[Keys.ROI_RECTANGLES.value]["width"] / 2
                    )
                    self.roi_dict[Keys.ROI_RECTANGLES.value][
                        "Center_Y"
                    ] = self.roi_dict[Keys.ROI_RECTANGLES.value]["Bottom_right_Y"] - (
                        self.roi_dict[Keys.ROI_RECTANGLES.value]["height"] / 2
                    )
                elif shape_type == Keys.ROI_POLYGONS.value:
                    try:
                        self.roi_dict[Keys.ROI_POLYGONS.value]["Center_X"] = (
                            self.roi_dict[Keys.ROI_POLYGONS.value]["Center_X"]
                        )
                        self.roi_dict[Keys.ROI_POLYGONS.value]["Center_Y"] = (
                            self.roi_dict[Keys.ROI_POLYGONS.value]["Center_Y"]
                        )
                    except KeyError:
                        pass
            self.video_names_w_rois = set(
                list(self.rectangles_df["Video"])
                + list(self.circles_df["Video"])
                + list(self.polygon_df["Video"])
            )

    def get_all_clf_names(self) -> List[str]:
        """
        Helper to return all classifier names in SimBA project

        :return List[str]:
        """

        model_names = []
        for i in range(self.clf_cnt):
            entry_name = "target_name_{}".format(str(i + 1))
            model_names.append(
                self.read_config_entry(
                    self.config,
                    ConfigKey.SML_SETTINGS.value,
                    entry_name,
                    data_type=Dtypes.STR.value,
                )
            )
        return model_names

    def insert_column_headers_for_outlier_correction(
        self, data_df: pd.DataFrame, new_headers: List[str], filepath: str
    ) -> pd.DataFrame:
        """
        Helper to insert new column headers onto a dataframe.

        :param pd.DataFrame data_df:  Dataframe where headers to to-bo replaced.
        :param List[str] new_headers:  Names of new headers.
        :param str filepath:  Path to where ``data_df`` is stored on disk
        :returns pd.DataFrame: Dataframe with new headers
        :raises DataHeaderWarning: If new headers are fewer/more than columns in dataframe

        :example:
        >>> df = pd.DataFrame(data=[[1, 2, 3], [1, 2, 3]], columns=['Feature_1', 'Feature_2', 'Feature_3'])
        >>> ConfigReader.insert_column_headers_for_outlier_correction(data_df=df, new_headers=['Feature_4', 'Feature_5', 'Feature_6'], filepath='test/my_test_file.csv')
        """

        if len(new_headers) != len(data_df.columns):
            difference = int(len(data_df.columns) - len(new_headers))
            bp_missing = int(abs(difference) / 3)
            if difference < 0:
                raise DataHeaderError(
                    msg=f"SIMBA ERROR: SimBA expects {len(new_headers)} columns of data inside the files within project_folder/csv/input_csv directory. However, within file {filepath} file, SimBA found {len(data_df.columns)} columns. Thus, there is {abs(difference)} missing data columns in the imported data, which may represent {int(bp_missing)} bodyparts if each body-part has an x, y and p value. Either revise the SimBA project pose-configuration with {bp_missing} less body-part, or include {bp_missing} more body-part in the imported data",
                    source=self.__class__.__name__,
                )
            else:
                raise DataHeaderError(
                    msg=f"SIMBA ERROR: SimBA expects {len(new_headers)} columns of data inside the files within project_folder/csv/input_csv directory. However, within file {filepath} file, SimBA found {len(data_df.columns)} columns. Thus, there is {abs(difference)} more data columns in the imported data than anticipated, which may represent {int(bp_missing)} bodyparts if each body-part has an x, y and p value. Either revise the SimBA project pose-configuration with {bp_missing} more body-part, or include {bp_missing} less body-part in the imported data",
                    source=self.__class__.__name__,
                )
        else:
            data_df.columns = new_headers
            return data_df

    def get_number_of_header_columns_in_df(self, df: pd.DataFrame) -> int:
        """
        Helper to find the count of non-numerical rows at the top of a dataframe.

        :param pd.DataFrame data_df: Dataframe to find the count non-numerical header rows in.
        :returns int
        :raises DataHeaderError: All rows are non-numerical.

        :example:
        >>> ConfigReader.get_number_of_header_columns_in_df(df=pd.DataFrame(data=[[1, 2, 3], [1, 2, 3]]))
        >>> 0
        >>> ConfigReader.get_number_of_header_columns_in_df(df=pd.DataFrame(data=[['Head_1', 'Body_2', 'Tail_3'], ['Some_nonsense', 'A_mistake', 'Maybe_multi_headers?'], [11, 99, 109], [122, 43, 2091]]))
        >>> 2
        """

        for i in range(len(df)):
            try:
                _ = df.iloc[i:].apply(pd.to_numeric).reset_index(drop=True)
                return i
            except ValueError:
                pass
        raise DataHeaderError(
            msg="Could find the count of header columns in dataframe. All values appear to be non-numeric",
            source=self.__class__.__name__,
        )

    def find_video_of_file(
        self,
        video_dir: Union[str, os.PathLike],
        filename: str,
        raise_error: bool = False,
    ) -> Union[str, os.PathLike]:
        """
        Helper to find the video file representing a known data file basename.

        :param Union[str, os.PathLike] video_dir: Directory holding putative video file.
        :param str filename: Data file name, e.g., ``Video_1``.
        :param bool raise_error: If True, raise error if no video can be found.
        :return Union[str, os.PathLike]: Path to video file.


        :example:
        >>> config_reader = ConfigReader(config_path='My_SimBA_Config')
        >>> config_reader.find_video_of_file(video_dir=config_reader.video_dir, filename='Video1')
        >>> '/project_folder/videos/Video1.mp4'
        """
        try:
            all_files_in_video_folder = [
                f for f in next(os.walk(video_dir))[2] if not f[0] == "."
            ]
        except StopIteration:
            raise NoFilesFoundError(
                msg=f"No files found in the {video_dir} directory",
                source=self.__class__.__name__,
            )
        all_files_in_video_folder = [
            os.path.join(video_dir, x) for x in all_files_in_video_folder
        ]
        return_path = None
        for file_path in all_files_in_video_folder:
            _, video_filename, ext = get_fn_ext(file_path)
            if (video_filename == filename) and (
                (ext.lower() == ".mp4") or (ext.lower() == ".avi")
            ):
                return_path = file_path

        if return_path is None and not raise_error:
            NoFileFoundWarning(
                msg=f"SimBA could not find a video file representing {filename} in the project video directory",
                source=self.__class__.__name__,
            )
        if return_path is None and raise_error:
            raise NoFilesFoundError(
                msg=f"SimBA could not find a video file representing {filename} in the project video directory",
                source=self.__class__.__name__,
            )
        return return_path

    def add_missing_ROI_cols(self, shape_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to add missing ROI definitions (``Color BGR``, ``Thickness``, ``Color name``) in ROI info
        dataframes created by the first version of the SimBA ROI user-interface but analyzed using newer versions of SimBA.


        :param pd.DataFrame shape_df: Dataframe holding ROI definitions.
        :return pd.DataFrame with ``Color BGR``, ``Thickness``, ``Color name`` fields
        """

        if not "Color BGR" in shape_df.columns:
            shape_df["Color BGR"] = [(255, 255, 255)] * len(shape_df)
        if not "Thickness" in shape_df.columns:
            shape_df["Thickness"] = [5] * len(shape_df)
        if not "Color name" in shape_df.columns:
            shape_df["Color name"] = "White"

        return shape_df

    def remove_a_folder(
        self, folder_dir: str, raise_error: Optional[bool] = False
    ) -> None:
        """
        Helper to remove single directory.

        :param folder_dir: Directory to remove.
        :param bool raise_error: If True, raise ``NotDirectoryError`` error of folder does not exist.
        :raises NotDirectoryError: If ``raise_error`` and directory does not exist.

        :example:
        >>> self.remove_a_folder(folder_dir'gerbil/gerbil_data/featurized_data/temp')
        """
        shutil.rmtree(folder_dir, ignore_errors=True)

    def remove_multiple_folders(
        self, folders: List[os.PathLike], raise_error: Optional[bool] = False
    ) -> None:
        """
        Helper to remove multiple directories.

        :param folders List[os.PathLike]: List of directory paths.
        :param bool raise_error: If True, raise ``NotDirectoryError`` error of folder does not exist. if False, then pass. Default False.
        :raises NotDirectoryError: If ``raise_error`` and directory does not exist.

        :example:
        >>> self.remove_multiple_folders(folders= ['gerbil/gerbil_data/featurized_data/temp'])
        """
        folders = [x for x in folders if x is not None]
        for folder_path in folders:
            if raise_error and not os.path.isdir(folder_path):
                raise NotDirectoryError(
                    msg=f"Cannot delete directory {folder_path}. The directory does not exist.",
                    source=self.__class__.__name__,
                )
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path, ignore_errors=True)
            else:
                pass

    def find_animal_name_from_body_part_name(self, bp_name: str, bp_dict: dict) -> str:
        """
        Given body-part name and animal body-part dict, returns the animal name

        :param str bp_name: Name of the body-part. E.g., ``Ear_1``.
        :param dict bp_dict: Nested dict holding animal names as keys and body-part names as and coordinates as values. Created by :meth:`simba.mixins.config_reader.ConfigReader.create_body_part_dictionary`
        :returns str

        :example:
        >>> config_reader = ConfigReader(config_path='tests/data/test_projects/two_c57/project_folder/project_config.ini')
        >>> ConfigReader.find_animal_name_from_body_part_name(bp_name='Ear_1', bp_dict=config_reader.animal_bp_dict)
        >>> 'simon'
        """

        for animal_name, animal_bps in bp_dict.items():
            if bp_name in [x[:-2] for x in animal_bps["X_bps"]]:
                return animal_name

    def create_body_part_dictionary(
        self,
        multi_animal_status: bool,
        animal_id_lst: list,
        animal_cnt: int,
        x_cols: List[str],
        y_cols: List[str],
        p_cols: Optional[List[str]] = None,
        colors: Optional[List[List[Tuple[int, int, int]]]] = None,
    ) -> Dict[str, Union[List[str], List[Tuple]]]:
        """
        Helper to create dict of dict lookup of body-parts where the keys are animal names, and
        values are the body-part names.

        :param bool multi_animal_status: If True, it is a multi-animal SimBA project.
        :param List[str] multi_animal_id_lst: Animal names. Eg., ['Simon, 'JJ']. Note: If a single animal project, this will be overridden and set to `Animal_1`.
        :param int animal_cnt: Number of animals in the SimBA project.
        :param List[str] x_cols: column names for body-part coordinates on x-axis. Returned by  :meth:`simba.mixins.config_reader.ConfigReader.get_body_part_names`
        :param List[str] y_cols: column names for body-part coordinates on y-axis. Returned by  :meth:`simba.mixins.config_reader.ConfigReader.get_body_part_names`
        :param List[str] p_cols: column names for body-part pose-estimation probability values. Returned by  :meth:`simba.mixins.config_reader.ConfigReader.get_body_part_names`
        :param Optional[List[List[Tuple[int, int, int]]]] colors: Optional bgr colors to associate with the body-parts. Returned by :meth:`simba.utils.data.create_color_palettes`.
        :returns dict

        :example:
        >>> ConfigReader.create_body_part_dictionary(multi_animal_status=True, animal_id_lst=['simon',])
        >>> {'simon': {'X_bps': ['Nose_1_x', 'Ear_left_1_x', 'Ear_right_1_x', 'Center_1_x', 'Lat_left_1_x', 'Lat_right_1_x', 'Tail_base_1_x', 'Tail_end_1_x'], 'Y_bps': ['Nose_1_y', 'Ear_left_1_y', 'Ear_right_1_y', 'Center_1_y', 'Lat_left_1_y', 'Lat_right_1_y', 'Tail_base_1_y', 'Tail_end_1_y'], 'colors': [[255.0, 0.0, 255.0], [223.125, 31.875, 255.0], [191.25, 63.75, 255.0], [159.375, 95.625, 255.0], [127.5, 127.5, 255.0], [95.625, 159.375, 255.0], [63.75, 191.25, 255.0], [31.875, 223.125, 255.0], [0.0, 255.0, 255.0]], 'P_bps': ['Nose_1_p', 'Ear_left_1_p', 'Ear_right_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p', 'Tail_base_1_p', 'Tail_end_1_p']}, 'jj': {'X_bps': ['Nose_2_x', 'Ear_left_2_x', 'Ear_right_2_x', 'Center_2_x', 'Lat_left_2_x', 'Lat_right_2_x', 'Tail_base_2_x', 'Tail_end_2_x'], 'Y_bps': ['Nose_2_y', 'Ear_left_2_y', 'Ear_right_2_y', 'Center_2_y', 'Lat_left_2_y', 'Lat_right_2_y', 'Tail_base_2_y', 'Tail_end_2_y'], 'colors': [[102.0, 127.5, 0.0], [102.0, 143.4375, 31.875], [102.0, 159.375, 63.75], [102.0, 175.3125, 95.625], [102.0, 191.25, 127.5], [102.0, 207.1875, 159.375], [102.0, 223.125, 191.25], [102.0, 239.0625, 223.125], [102.0, 255.0, 255.0]], 'P_bps': ['Nose_2_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Center_2_p', 'Lat_left_2_p', 'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p']}}
        """

        animal_bp_dict = {}

        if multi_animal_status:
            for animal in range(animal_cnt):
                animal_bp_dict[animal_id_lst[animal]] = {}
                animal_bp_dict[animal_id_lst[animal]]["X_bps"] = [
                    i for i in x_cols if animal_id_lst[animal] in i
                ]
                animal_bp_dict[animal_id_lst[animal]]["Y_bps"] = [
                    i for i in y_cols if animal_id_lst[animal] in i
                ]
                if colors:
                    animal_bp_dict[animal_id_lst[animal]]["colors"] = colors[animal]
                if p_cols:
                    animal_bp_dict[animal_id_lst[animal]]["P_bps"] = [
                        i for i in p_cols if animal_id_lst[animal] in i
                    ]
                if not animal_bp_dict[animal_id_lst[animal]]["X_bps"]:
                    multi_animal_status = False
                    break

        if not multi_animal_status:
            if animal_cnt > 1:
                for animal in range(animal_cnt):
                    currAnimalName = f"Animal_{str(animal + 1)}"
                    search_string_x = f"_{str(animal + 1)}_x"
                    search_string_y = f"_{str(animal + 1)}_y"
                    search_string_p = f"_{str(animal + 1)}_p"
                    animal_bp_dict[currAnimalName] = {}
                    animal_bp_dict[currAnimalName]["X_bps"] = [
                        i for i in x_cols if i.endswith(search_string_x)
                    ]
                    animal_bp_dict[currAnimalName]["Y_bps"] = [
                        i for i in y_cols if i.endswith(search_string_y)
                    ]
                    if colors:
                        animal_bp_dict[currAnimalName]["colors"] = colors[animal]
                    if p_cols:
                        animal_bp_dict[currAnimalName]["P_bps"] = [
                            i for i in p_cols if i.endswith(search_string_p)
                        ]
                if animal_id_lst[0] != "":
                    for animal in range(len(animal_id_lst)):
                        currAnimalName = f"Animal_{str(animal + 1)}"
                        animal_bp_dict[animal_id_lst[animal]] = animal_bp_dict.pop(
                            currAnimalName
                        )

            else:
                animal_bp_dict["Animal_1"] = {}
                animal_bp_dict["Animal_1"]["X_bps"] = [i for i in x_cols]
                animal_bp_dict["Animal_1"]["Y_bps"] = [i for i in y_cols]
                if colors:
                    animal_bp_dict["Animal_1"]["colors"] = colors[0]
                if p_cols:
                    animal_bp_dict["Animal_1"]["P_bps"] = [i for i in p_cols]
        return animal_bp_dict

    def get_body_part_names(self):
        """
        Helper to extract pose-estimation data field names (x, y, p)

        :example:
        >>> config_reader = ConfigReader(config_path='test/project_config.csv')
        >>> config_reader.get_body_part_names()
        """

        self.x_cols, self.y_cols, self.p_cols = [], [], []
        for bp in self.body_parts_lst:
            self.x_cols.append(f"{bp}_x")
            self.y_cols.append(f"{bp}_y")
            self.p_cols.append(f"{bp}_p")

    def drop_bp_cords(
        self, df: pd.DataFrame, raise_error: bool = False
    ) -> pd.DataFrame:
        """
        Helper to remove pose-estimation fields from dataframe.

        :param pd.DataFrame df: pandas dataframe containing pose-estimation fields (body-part x, y, p fields)
        :param bool raise_error: If True, raise error if body-parts cant be found. Else, print warning
        :return pd.DataFrame: ``df`` without pose-estimation fields

        :example:
        >>> config_reader = ConfigReader(config_path='test/project_folder/project_config.csv')
        >>> df = read_df(config_reader.machine_results_paths[0], file_type='csv')
        >>> df = config_reader.drop_bp_cords(df=df)
        """
        missing_body_part_fields = list(set(self.bp_col_names) - set(list(df.columns)))
        if len(missing_body_part_fields) > 0 and not raise_error:
            BodypartColumnNotFoundWarning(
                msg=f"SimBA could not drop body-part coordinates, some body-part names are missing in dataframe. SimBA expected the following body-parts, that could not be found inside the file: {missing_body_part_fields}",
                source=self.__class__.__name__,
            )
            return df.drop(self.bp_col_names, axis=1, errors="ignore")
        elif len(missing_body_part_fields) > 0 and raise_error:
            raise BodypartColumnNotFoundError(
                msg=f"SimBA could not drop body-part coordinates, some body-part names are missing in dataframe. SimBA expected the following body-parts, that could not be found inside the file: {missing_body_part_fields}",
                source=self.__class__.__name__,
            )
        else:
            return df.drop(self.bp_col_names, axis=1)

    def get_bp_headers(self) -> None:
        """
        Helper to create ordered list of all column header fields for SimBA project dataframes.

        >>> config_reader = ConfigReader(config_path='test/project_folder/project_config.ini')
        >>> config_reader.get_bp_headers()
        """

        self.bp_headers = []
        for bp in self.body_parts_lst:
            c1, c2, c3 = (f"{bp}_x", f"{bp}_y", f"{bp}_p")
            self.bp_headers.extend((c1, c2, c3))

    def read_config_entry(
        self,
        config: ConfigParser,
        section: str,
        option: str,
        data_type: Literal["str", "int", "float", "folder_path"],
        default_value: Optional[Any] = None,
        options: Optional[List[Any]] = None,
    ) -> Union[str, int, float]:
        """
        Helper to read entry from a configparser.ConfigParser object

        :param ConfigParser config: Parsed SimBA project config
        :param str section: Project config section name
        :param str option: Project config option name
        :param str data_type: Type of data. E.g., ``str``, ``int``, ``float``, ``folder_path``.
        :param Optional[Any] default_value: If entry cannot be found, then default to this value.
        :param Optional[List[Any]] options: If not ``None``, then list of viable entries.
        :raise InvalidInputError: If returned value is not in ``options``.
        :raise MissingProjectConfigEntryError: If no entry is found and no ``default_value`` is provided.

        :return Union[str, float, int, os.Pathlike]

        :example:
        >>> config = ConfigReader(config_path='tests/data/test_projects/two_c57/project_folder/project_config.ini')
        >>> config.read_config_entry(config=self.config, section='Multi animal IDs', option='id_list', data_type='str')
        >>> 'simon,jj'
        """
        try:
            if config.has_option(section, option):
                if data_type == Dtypes.FLOAT.value:
                    value = config.getfloat(section, option)
                elif data_type == Dtypes.INT.value:
                    value = config.getint(section, option)
                elif data_type == Dtypes.STR.value:
                    value = config.get(section, option).strip()
                elif data_type == Dtypes.FOLDER.value:
                    value = config.get(section, option).strip()
                    if not os.path.isdir(value):
                        raise NotDirectoryError(
                            msg=f"The SimBA config file includes paths to a folder ({value}) that does not exist.",
                            source=self.__class__.__name__,
                        )
                if options != None:
                    if value not in options:
                        raise InvalidInputError(
                            msg=f"{option} is set to {str(value)} in SimBA, but this is not among the valid options: ({options})",
                            source=self.__class__.__name__,
                        )
                    else:
                        return value
                return value
            elif default_value != None:
                return default_value
            else:
                raise MissingProjectConfigEntryError(
                    msg=f"SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.",
                    source=self.__class__.__name__,
                )
        except ValueError:
            if default_value != None:
                return default_value
            else:
                raise MissingProjectConfigEntryError(
                    msg=f"SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.",
                    source=self.__class__.__name__,
                )

    def read_video_info_csv(self, file_path: str) -> pd.DataFrame:
        """
        Helper to read the project_folder/logs/video_info.csv of the SimBA project in as a pd.DataFrame
        Parameters
        ----------
        file_path: str

        Returns
        -------
        pd.DataFrame
        """

        if not os.path.isfile(file_path):
            raise NoFilesFoundError(
                msg=f"Could not find the video_info.csv table in your SimBA project. Create it using the [Video parameters] tab. SimBA expects the file at location {file_path}",
                source=self.__class__.__name__,
            )
        info_df = pd.read_csv(file_path)
        for c in [
            "Video",
            "fps",
            "Resolution_width",
            "Resolution_height",
            "Distance_in_mm",
            "pixels/mm",
        ]:
            if c not in info_df.columns:
                raise ParametersFileError(
                    msg=f'The project "project_folder/logs/video_info.csv" does not not have an anticipated header ({c}). Please re-create the file and make sure each video has a {c} value',
                    source=self.__class__.__name__,
                )
        info_df["Video"] = info_df["Video"].astype(str)
        for c in [
            "fps",
            "Resolution_width",
            "Resolution_height",
            "Distance_in_mm",
            "pixels/mm",
        ]:
            try:
                info_df[c] = info_df[c].astype(float)
            except:
                raise ParametersFileError(
                    msg=f'One or more values in the {c} column of the "project_folder/logs/video_info.csv" file could not be interpreted as a numeric value. Please re-create the file and make sure the entries in the {c} column are all numeric.',
                    source=self.__class__.__name__,
                )
        if info_df["fps"].min() <= 1:
            InvalidValueWarning(
                msg="Videos in your SimBA project have an FPS of 1 or less. Please use videos with more than one frame per second, or correct the inaccurate fps inside the `project_folder/logs/videos_info.csv` file",
                source=self.__class__.__name__,
            )
        return info_df

    def read_video_info(
        self, video_name: str, raise_error: Optional[bool] = True
    ) -> (pd.DataFrame, float, float):
        """
        Helper to read the meta-data (pixels per mm, resolution, fps) from the video_info.csv for a single input file.

        :param str video_name: The name of the video without extension to get the metadata for
        :param Optional[bool] raise_error: If True, raise error if video info for the video name cannot be found. Default: True.
        :raise ParametersFileError: If ``raise_error`` and video metadata info is not found
        :raise DuplicationError: If file contains multiple entries for the same video.
        :return (pd.DataFrame, float, float) representing all video info, pixels per mm, and fps
        """

        video_settings = self.video_info_df.loc[
            self.video_info_df["Video"] == video_name
        ]
        if len(video_settings) > 1:
            raise DuplicationError(
                msg=f"SimBA found multiple rows in the project_folder/logs/video_info.csv named {str(video_name)}. Please make sure that each video name is represented ONCE in the video_info.csv",
                source=self.__class__.__name__,
            )
        elif len(video_settings) < 1:
            if raise_error:
                raise ParametersFileError(
                    msg=f"SimBA could not find {str(video_name)} in the video_info.csv file. Make sure all videos analyzed are represented in the project_folder/logs/video_info.csv file.",
                    source=self.__class__.__name__,
                )
            else:
                return (None, None, None)
        else:
            try:
                px_per_mm = float(video_settings["pixels/mm"])
                fps = float(video_settings["fps"])
                return video_settings, px_per_mm, fps
            except TypeError:
                raise ParametersFileError(
                    msg=f"Make sure the videos that are going to be analyzed are represented with APPROPRIATE VALUES inside the project_folder/logs/video_info.csv file in your SimBA project. Could not interpret the fps, pixels per millimeter and/or fps as numerical values for video {video_name}",
                    source=self.__class__.__name__,
                )

    def check_multi_animal_status(self) -> None:
        """
        Helper to check if the project is a multi-animal SimBA project.

        """
        multi_animal_id_lst = []
        if not self.config.has_section("Multi animal IDs"):
            for animal in range(self.animal_cnt):
                multi_animal_id_lst.append("Animal_" + str(animal + 1))
            multi_animal_status = False

        else:
            multi_animal_id_str = self.read_config_entry(
                config=self.config,
                section="Multi animal IDs",
                option="id_list",
                data_type="str",
            )
            multi_animal_id_lst = [x.lstrip() for x in multi_animal_id_str.split(",")]
            multi_animal_id_lst = [x for x in multi_animal_id_lst if x != "None"]
            if (self.animal_cnt > 1) and (len(multi_animal_id_lst) > 1):
                multi_animal_status = True
            else:
                for animal in range(self.animal_cnt):
                    multi_animal_id_lst.append("Animal_{}".format(str(animal + 1)))
                multi_animal_status = False

        self.multi_animal_status = multi_animal_status
        self.multi_animal_id_list = multi_animal_id_lst[: self.animal_cnt]

    def remove_roi_features(self, data_dir: Union[str, os.PathLike]) -> None:
        """
        Helper to remove ROI-based features from datasets within a directory. The identified ROI-based fields are move to the
        ``project_folder/logs/ROI_data_{datetime}`` directory.

        .. note::
           ROI-based features are identified based on the combined criteria of (i) The prefix of the field is a named ROI in
           the ``project_folder/logs/ROI_definitions.h5`` file, and (ii) the suffix of the field is contained in the ['in zone',
           'n zone_cumulative_time', 'in zone_cumulative_percent', 'distance', 'facing']

        :param Union[str, os.PathLike] data_dir: directory with data to remove ROi features from.

        :example:
        >>> self.remove_roi_features('/project_folder/csv/features_extracted')

        """
        ROI_COL_SUFFIXES = [
            "in zone",
            "n zone_cumulative_time",
            "in zone_cumulative_percent",
            "distance",
            "facing",
        ]

        check_if_dir_exists(in_dir=data_dir)
        timer = SimbaTimer(start=True)
        filepaths = glob.glob(data_dir + f"/*.{self.file_type}")
        roi_dir = os.path.join(
            self.project_path,
            "logs",
            f'ROI_data_{datetime.now().strftime("%Y%m%d%H%M%S")}',
        )
        check_if_filepath_list_is_empty(
            filepaths=filepaths,
            error_msg=f"No .{self.file_type} files found in {data_dir}.",
        )
        for file_cnt, file_path in enumerate(filepaths):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            roi_cols = set()
            df = read_df(file_path=file_path, file_type=self.file_type)
            for shape_name in self.shape_names:
                roi_cols.update([x for x in df.columns if shape_name in x])
            for suffix in ROI_COL_SUFFIXES:
                roi_cols.update([x for x in roi_cols if x.endswith(suffix)])
            if len(roi_cols) == 0:
                NoDataFoundWarning(
                    msg=f"NO ROI data found in {file_path}",
                    source=self.__class__.__name__,
                )
            else:
                roi_df = df[list(roi_cols)]
                df = df.drop(list(roi_cols), axis=1)
                roi_save_path = os.path.join(roi_dir, video_name + "." + self.file_type)
                if not os.path.exists(roi_dir):
                    os.makedirs(roi_dir)
                write_df(df=roi_df, file_type=self.file_type, save_path=roi_save_path)
                write_df(df=df, file_type=self.file_type, save_path=file_path)
                print(f"ROI features (N={len(roi_cols)}) removed from {video_name}...")
        timer.stop_timer()
        stdout_success(
            msg=f"ROI features removed from {len(filepaths)} files in the {data_dir} directory. The ROI features has been moved to the {roi_dir} directory",
            elapsed_time=timer.elapsed_time_str,
            source=self.__class__.__name__,
        )

    def create_log_msg_from_init_args(self, locals: dict):

        def has_methods_or_classes(obj):
            return any(callable(attr) for attr in dir(obj))

        def remove_keys_with_methods_or_classes(dictionary):
            return {
                key: value
                for key, value in dictionary.items()
                if not has_methods_or_classes(value)
            }

        msg = ""
        locals.pop("self", None)
        locals.pop("__class__", None)
        locals = remove_keys_with_methods_or_classes(locals)

        for cnt, (k, v) in enumerate(locals.items()):
            msg += f"{k}: {v}"
            if cnt != len(list(locals.keys())) - 1:
                msg += ", "
        return msg

    def create_logger(self) -> None:
        if self.__class__ not in logging.Logger.manager.loggerDict.keys():
            log_config = get_log_config()
            log_config["handlers"]["file_handler"]["filename"] = os.path.join(
                self.logs_path, "project_log.log"
            )
            logging.config.dictConfig(log_config)
            self.logger = logging.getLogger(str(self.__class__))
        if not os.path.isfile(os.path.join(self.logs_path, "project_log.log")):
            self.logger.info(f"Logging initiated in project {self.project_path}...")

    def create_third_part_append_logger(self, path: str) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(filename=path, encoding="utf-8")  # or whatever
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    def add_video_to_video_info(
        self,
        video_name: str,
        fps: float,
        width: int,
        height: int,
        pixels_per_mm: int,
        distance_mm: float = 0.0,
    ) -> None:
        self.video_info_df.loc[len(self.video_info_df)] = [
            video_name,
            fps,
            width,
            height,
            distance_mm,
            pixels_per_mm,
        ]
        self.video_info_df.drop_duplicates(subset=["Video"], inplace=True, keep="last")
        self.video_info_df = self.video_info_df.set_index("Video")
        try:
            self.video_info_df.to_csv(
                os.path.join(self.project_path, Paths.VIDEO_INFO.value)
            )
        except:
            raise PermissionError(
                msg="SimBA tried to write to project_folder/logs/video_info.csv, but was not allowed. If this file is open in another program, try closing it.",
                source=self.__class__.__name__,
            )
        stdout_success(
            msg="Video info has been UPDATED at project_folder/logs/video_info.csv",
            source=self.__class__.__name__,
        )


# config = ConfigReader(config_path='/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/two_c57/project_folder/project_config.ini', read_video_info=False)

# config.read_roi_data()


# config = ConfigReader(config_path='/Users/simon/Desktop/envs/troubleshooting/Nastacia_unsupervised/project_folder/project_config.ini', read_video_info=False)
# config.read_roi_data()
# config.remove_roi_features('/Users/simon/Desktop/envs/troubleshooting/Nastacia_unsupervised/project_folder/csv/features_extracted')

# remove_roi_features(
#     config_path='/Users/simon/Desktop/envs/troubleshooting/Nastacia_unsupervised/project_folder/project_config.ini',
#     data_dir='/Users/simon/Desktop/envs/troubleshooting/Nastacia_unsupervised/project_folder/csv/features_extracted')

# test = ConfigReader(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
