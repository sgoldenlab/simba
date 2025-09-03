__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from itertools import combinations

from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.roi_tools.roi_utils import get_roi_dict_from_dfs
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_dir_exists,
    check_same_files_exist_in_all_directories, check_valid_boolean,
    check_valid_dataframe, check_valid_lst, check_video_has_rois)
from simba.utils.enums import ROI_SETTINGS, Formats
from simba.utils.errors import (DuplicationError, InvalidInputError,
                                NoFilesFoundError, NoROIDataError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (copy_files_in_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, remove_a_folder,
                                    remove_multiple_folders, write_df)

SHAPE_TYPE = "Shape_type"
TWO_POINT_BP_DISTANCES = 'TWO-POINT BODY-PART DISTANCES (MM)'
WITHIN_ANIMAL_THREE_POINT_ANGLES = 'WITHIN-ANIMAL THREE-POINT BODY-PART ANGLES (DEGREES)'
WITHIN_ANIMAL_THREE_POINT_HULL = "WITHIN-ANIMAL THREE-POINT CONVEX HULL PERIMETERS (MM)"
WITHIN_ANIMAL_FOUR_POINT_HULL = "WITHIN-ANIMAL FOUR-POINT CONVEX HULL PERIMETERS (MM)"
ANIMAL_CONVEX_HULL_PERIMETER = 'ENTIRE ANIMAL CONVEX HULL PERIMETERS (MM)'
ANIMAL_CONVEX_HULL_AREA = "ENTIRE ANIMAL CONVEX HULL AREA (MM2)"
FRAME_BP_MOVEMENT = "FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)"
FRAME_BP_TO_ROI_CENTER = "FRAME-BY-FRAME BODY-PART DISTANCES TO ROI CENTERS (MM)"
FRAME_BP_INSIDE_ROI = "FRAME-BY-FRAME BODY-PARTS INSIDE ROIS (BOOLEAN)"
ARENA_EDGE = "BODY-PART DISTANCES TO VIDEO FRAME EDGE (MM)"




FEATURE_FAMILIES = [TWO_POINT_BP_DISTANCES,
                    WITHIN_ANIMAL_THREE_POINT_ANGLES,
                    WITHIN_ANIMAL_THREE_POINT_HULL,
                    WITHIN_ANIMAL_FOUR_POINT_HULL,
                    ANIMAL_CONVEX_HULL_PERIMETER,
                    ANIMAL_CONVEX_HULL_AREA,
                    FRAME_BP_MOVEMENT,
                    FRAME_BP_TO_ROI_CENTER,
                    FRAME_BP_INSIDE_ROI,
                    ARENA_EDGE]


class FeatureSubsetsCalculator(ConfigReader, TrainModelMixin):
    """
    Computes a subset of features from pose for non-ML downstream purposes.
    E.g., returns the size of animal convex hull in each frame.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str save_dir: directory where to store results.
    :param List[str] feature_family: List of feature subtype to calculate. E.g., ['TWO-POINT BODY-PART DISTANCES (MM)"].
    :param bool file_checks: If true, checks that the files which the data is appended too contains the anticipated number of rows and no duplicate columns after appending. Default False.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the data. If None, then the data is only appended.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory of pose-estimation data to compute feature subsets for. If None, then the `/project_folder/csv/outlier_corrected_movement_locations` directory.
    :param bool append_to_features_extracted: If True, appends the data to the file sin the `features_extracted` directory. Default: False.
    :param bool append_to_targets_inserted: If True, appends the data to the file sin the `targets_inserted` directory. Default: False.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/feature_subsets.md>_`

    .. image:: _static/img/feature_subsets.png
       :width: 400
       :align: center

    :example:
    >>> test = FeatureSubsetsCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
    >>>                               feature_families=[FRAME_BP_MOVEMENT, WITHIN_ANIMAL_THREE_POINT_ANGLES],
    >>>                               append_to_features_extracted=False,
    >>>                               file_checks=False,
    >>>                               append_to_targets_inserted=False,
    >>>                               save_dir=r"C:\troubleshooting\mitra\project_folder\csv\new_features")
    >>> test.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 feature_families: List[str],
                 file_checks: bool = False,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 append_to_features_extracted: bool = False,
                 append_to_targets_inserted: bool = False):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        check_valid_boolean(value=file_checks, source=f'{self.__class__.__name__} file_checks', raise_error=True)
        check_valid_boolean(value=append_to_features_extracted, source=f'{self.__class__.__name__} append_to_features_extracted', raise_error=True)
        check_valid_boolean(value=append_to_targets_inserted, source=f'{self.__class__.__name__} append_to_targets_inserted', raise_error=True)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir)
        check_valid_lst(data=feature_families, source=f'{self.__class__.__name__} feature_families', valid_dtypes=(str,), valid_values=FEATURE_FAMILIES, min_len=1, raise_error=True)
        self.file_checks, self.feature_families, self.save_dir = file_checks, feature_families, save_dir
        self.append_to_features_extracted = append_to_features_extracted
        self.append_to_targets_inserted = append_to_targets_inserted
        if data_dir is None:
            self.data_dir = self.outlier_corrected_dir
            self.data_paths = self.outlier_corrected_paths
        else:
            self.data_dir = data_dir
            check_if_dir_exists(in_dir=data_dir)
            self.data_paths  = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['csv'], raise_error=True)
        if self.append_to_features_extracted:
            if not check_same_files_exist_in_all_directories(dirs=[self.data_dir, self.features_dir], file_type=self.file_type, raise_error=False):
                raise NoFilesFoundError(msg=f'Cannot append feature subset to files in {self.features_dir} directory: To proceed, the files in the {self.features_dir} and the {self.data_dir} directories has to contain the same number of files with the same filenames.', source=self.__class__.__name__)
        if self.append_to_targets_inserted:
            if not check_same_files_exist_in_all_directories(dirs=[self.data_dir, self.targets_folder], file_type=self.file_type, raise_error=False):
                raise NoFilesFoundError(msg=f'Cannot append feature subset to files in {self.targets_folder} directory: To proceed, the files in the {self.targets_folder} and the {self.data_dir} directories has to contain the same number of files with the same filenames.', source=self.__class__.__name__)
        self.video_names = [get_fn_ext(filepath=x)[1] for x in self.data_paths]
        for file_path in self.data_paths: check_file_exist_and_readable(file_path=file_path)
        self.temp_dir = os.path.join(self.data_dir, f"temp_data_{self.datetime}")
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)
        if (FRAME_BP_TO_ROI_CENTER in feature_families) or (FRAME_BP_INSIDE_ROI in feature_families):
            if not os.path.isfile(self.roi_coordinates_path):
                raise NoROIDataError(msg=f'Cannot compute ROI features: The SimBA project has no ROI data defined.')
            self.read_roi_data(); check_video_has_rois(roi_dict=self.roi_dict)
            self.roi_dict = get_roi_dict_from_dfs(rectangle_df=self.rectangles_df, circle_df=self.circles_df, polygon_df=self.polygon_df, video_name_nesting=True)
            missing_roi_videos = [x for x in self.video_names if x not in list(self.roi_dict.keys())]
            if len(missing_roi_videos) > 0:
                raise NoROIDataError(msg=f'Cannot compute ROI features: The following videos have no ROIs: {missing_roi_videos}')
        self.__get_bp_combinations()

    def __get_bp_combinations(self):
        self.two_point_combs = np.array(list(combinations(self.project_bps, 2)))
        self.within_animal_three_point_combs = {}
        self.within_animal_four_point_combs = {}
        self.animal_bps = {}
        for animal, animal_data in self.animal_bp_dict.items():
            animal_bps = [x[:-2] for x in animal_data["X_bps"]]
            self.animal_bps[animal] = animal_bps
            self.within_animal_three_point_combs[animal] = np.array(list(combinations(animal_bps, 3)))
            self.within_animal_four_point_combs[animal] = np.array(list(combinations(animal_bps, 4)))

    def _get_two_point_bp_distances(self):
        for c in self.two_point_combs:
            x1, y1, x2, y2 = list(sum([(f"{x}_x", f"{y}_y") for (x, y) in zip(c, c)], ()))
            bp1 = self.data_df[[x1, y1]].values
            bp2 = self.data_df[[x2, y2]].values
            self.results[f"Distance (mm) {c[0]}-{c[1]}"] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=bp1.astype(np.float64), location_2=bp2.astype(np.float64), px_per_mm=np.float64(self.px_per_mm), centimeter=False)

    def __get_three_point_angles(self):
        for animal, points in self.within_animal_three_point_combs.items():
            for point in points:
                col_names = list(sum([(f"{x}_x", f"{y}_y") for (x, y) in zip(point, point)], ()))
                self.results[f"Angle (degrees) {point[0]}-{point[1]}-{point[2]}"] = (FeatureExtractionMixin.angle3pt_vectorized(data=self.data_df[col_names].values))

    def __get_three_point_hulls(self):
        for animal, points in self.within_animal_three_point_combs.items():
            for point in points:
                col_names = list(sum([(f"{x}_x", f"{y}_y") for (x, y) in zip(point, point)], ()))
                three_point_arr = np.reshape(self.data_df[col_names].values, (len(self.data_df / 2), -1, 2)).astype(np.float32)
                self.results[f"{animal} three-point convex hull perimeter (mm) {point[0]}-{point[1]}-{point[2]}"] = (jitted_hull(points=three_point_arr, target=Formats.PERIMETER.value) / self.px_per_mm)

    def __get_four_point_hulls(self):
        for animal, points in self.within_animal_four_point_combs.items():
            for point in points:
                col_names = list(sum([(f"{x}_x", f"{y}_y") for (x, y) in zip(point, point)], ()))
                four_point_arr = np.reshape(self.data_df[col_names].values, (len(self.data_df / 2), -1, 2) ).astype(np.float32)
                self.results[f"{animal} four-point convex perimeter (mm) {point[0]}-{point[1]}-{point[2]}-{point[3]}"] = (jitted_hull(points=four_point_arr, target=Formats.PERIMETER.value) / self.px_per_mm)


    def __get_convex_hulls(self, method: str):
        for animal, point in self.animal_bps.items():
            col_names = list(sum([(f"{x}_x", f"{y}_y") for (x, y) in zip(point, point)], ()))
            animal_point_arr = np.reshape(self.data_df[col_names].values, (len(self.data_df / 2), -1, 2)).astype(np.float32)
            if method == 'perimeter':
                self.results[f"{animal} convex hull perimeter (mm)"] = (jitted_hull(points=animal_point_arr, target=Formats.PERIMETER.value)/ self.px_per_mm)
            else:
                self.results[f"{animal} convex hull area (mm2)"] = (jitted_hull(points=animal_point_arr, target=Formats.AREA.value) / self.px_per_mm)


    def __get_framewise_movement(self):
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                check_valid_dataframe(df=self.data_df, source=self.file_path, required_fields=[f"{bp}_x", f"{bp}_y"])
                bp_arr = FeatureExtractionMixin.create_shifted_df(df=self.data_df[[f"{bp}_x", f"{bp}_y"]]).values
                x, y = bp_arr[:, 0:2], bp_arr[:, 2:4]
                self.results[f"{animal} movement {bp} (mm)"] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=x.astype(np.float64), location_2=y.astype(np.float64), px_per_mm=np.float64(self.px_per_mm), centimeter=False)

    def __get_roi_center_distances(self):
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                check_valid_dataframe(df=self.data_df, source=self.file_path, required_fields=[f"{bp}_x", f"{bp}_y"])
                bp_arr = self.data_df[[f"{bp}_x", f"{bp}_y"]].values.astype(np.float32)
                for roi_name, roi_data in self.roi_dict[self.video_name].items():
                    center_point = np.array([roi_data['Center_X'], roi_data['Center_Y']]).astype(np.int32)
                    distance = FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=bp_arr, location_2=center_point, px_per_mm=self.px_per_mm)
                    self.results[f"{animal} {bp} to {roi_name} center distance (mm)"] = distance

    def __get_distances_to_frm_edge(self):
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                check_valid_dataframe(df=self.data_df, source=self.file_path, required_fields=[f"{bp}_x", f"{bp}_y"])
                bp_arr = self.data_df[[f"{bp}_x", f"{bp}_y"]].values.astype(np.float32)
                distance = FeatureExtractionSupplemental().border_distances(data=bp_arr, pixels_per_mm=self.px_per_mm, img_resolution=np.array([self.video_width, self.video_height], dtype=np.int32), time_window=1, fps=1)
                self.results[f"{animal} {bp} to left video edge distance (mm)"] = distance[:, 0]
                self.results[f"{animal} {bp} to right video edge distance (mm)"] = distance[:, 1]
                self.results[f"{animal} {bp} to top video edge distance (mm)"] = distance[:, 2]
                self.results[f"{animal} {bp} to bottom video edge distance (mm)"] = distance[:, 3]

    def __get_inside_roi(self):
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                check_valid_dataframe(df=self.data_df, source=self.file_path, required_fields=[f"{bp}_x", f"{bp}_y"])
                bp_arr = self.data_df[[f"{bp}_x", f"{bp}_y"]].values.astype(np.float32)
                for roi_name, roi_data in self.roi_dict[self.video_name].items():
                    if roi_data[SHAPE_TYPE] == ROI_SETTINGS.RECTANGLE.value:
                        roi_coords = np.array([[roi_data['topLeftX'], roi_data['topLeftY']], [roi_data['Bottom_right_X'], roi_data['Bottom_right_Y']]])
                        r = FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_arr, roi_coords=roi_coords)
                        self.results[f"{animal} {bp} inside rectangle {roi_name} (Boolean)"] = r
                    elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.CIRCLE.value:
                        circle_center = np.array([roi_data['Center_X'], roi_data['Center_Y']]).astype(np.int32)
                        r = FeatureExtractionMixin.is_inside_circle(bp=bp_arr, roi_center=circle_center, roi_radius=roi_data['radius'])
                        self.results[f"{animal} {bp} inside circle {roi_name} (Boolean)"] = r
                    elif roi_data[SHAPE_TYPE] == ROI_SETTINGS.POLYGON.value:
                        vertices = roi_data['vertices'].astype(np.int32)
                        r = FeatureExtractionMixin.framewise_inside_polygon_roi(bp_location=bp_arr, roi_coords=vertices)
                        self.results[f"{animal} {bp} inside polygon {roi_name} (Boolean)"] = r

    def __check_files(self, x: pd.DataFrame, y: pd.DataFrame, path_x: str, path_y: str):
        if len(x) != len(y):
            remove_multiple_folders(folders=[self.temp_append_dir, self.temp_dir], raise_error=False)
            raise InvalidInputError(msg=f'The files at {path_x} and {path_y} do not contain the same number of rows: {len(x)} vs {len(y)}', source=self.__class__.__name__)
        duplicated_x_cols = [i for i in x.columns if i in y.columns]
        if len(duplicated_x_cols) > 0:
            remove_multiple_folders(folders=[self.temp_append_dir, self.temp_dir], raise_error=False)
            raise DuplicationError(msg=f'Cannot append the new features to {path_y}. This file already has the following columns: {duplicated_x_cols}', source=self.__class__.__name__)

    def __append_to_data_in_dir(self, dir: Union[str, os.PathLike]):
        temp_files = find_files_of_filetypes_in_directory(directory=self.temp_dir, extensions=[f'.{self.file_type}'], as_dict=True)
        self.temp_append_dir = os.path.join(dir, f'temp_{self.datetime}')
        os.makedirs(self.temp_append_dir)
        for file_cnt, (file_name, file_path) in enumerate(temp_files.items()):
            print(f'Appending features to {file_name} ({file_cnt+1}/{len(list(temp_files.keys()))})')
            old_df = read_df(file_path=os.path.join(dir, f'{file_name}.{self.file_type}'), file_type=self.file_type).reset_index(drop=True)
            new_features_df = read_df(file_path=file_path, file_type=self.file_type).reset_index(drop=True)
            if self.file_checks:
                self.__check_files(x=new_features_df, y=old_df, path_x=file_path, path_y=os.path.join(dir, f'{file_name}.{self.file_type}'))
            save_path = os.path.join(self.temp_append_dir, f'{file_name}.{self.file_type}')
            out_df = pd.concat([old_df, new_features_df], axis=1)
            write_df(df=out_df, file_type=self.file_type, save_path=save_path)
        prior_dir = os.path.join(dir, f"Prior_to_feature_subset_append_{self.datetime}")
        os.makedirs(prior_dir)
        copy_files_in_directory(in_dir=dir, out_dir=prior_dir, filetype=self.file_type, raise_error=True)
        copy_files_in_directory(in_dir=self.temp_append_dir, out_dir=dir, filetype=self.file_type, raise_error=True)
        remove_a_folder(folder_dir=self.temp_append_dir, ignore_errors=False)

    def __append_to_targets_inserted(self, dir: Union[str, os.PathLike]):
        temp_files = find_files_of_filetypes_in_directory(directory=self.temp_dir, extensions=[f'.{self.file_type}'], as_dict=True)
        self.temp_append_dir = os.path.join(dir, f'temp_{self.datetime}')
        os.makedirs(self.temp_append_dir)
        for file_cnt, (file_name, file_path) in enumerate(temp_files.items()):
            old_df = read_df(file_path=os.path.join(dir, f'{file_name}.{self.file_type}'), file_type=self.file_type).reset_index(drop=True)
            new_features_df = read_df(file_path=file_path, file_type=self.file_type).reset_index(drop=True)
            if self.file_checks:
                self.__check_files(x=new_features_df, y=old_df, path_x=file_path, path_y=os.path.join(dir, f'{file_name}.{self.file_type}'))
            save_path = os.path.join(self.temp_append_dir, f'{file_name}.{self.file_type}')
            clf_cols = [x for x in self.clf_names if x in list(old_df.columns)]
            clf_df, old_df = old_df[clf_cols], old_df.drop(clf_cols, axis=1)
            out_df = pd.concat([old_df, new_features_df, clf_df], axis=1)
            write_df(df=out_df, file_type=self.file_type, save_path=save_path)
        prior_dir = os.path.join(dir, f"Prior_to_feature_subset_append_{self.datetime}")
        os.makedirs(prior_dir)
        copy_files_in_directory(in_dir=dir, out_dir=prior_dir, filetype=self.file_type, raise_error=True)
        copy_files_in_directory(in_dir=self.temp_append_dir, out_dir=dir, filetype=self.file_type, raise_error=True)
        remove_a_folder(folder_dir=self.temp_append_dir, ignore_errors=False)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            self.file_path = file_path
            self.video_name = get_fn_ext(filepath=file_path)[1]
            video_timer = SimbaTimer(start=True)
            save_path = os.path.join(self.temp_dir, f'{self.video_name}.{self.file_type}')
            self.results = pd.DataFrame()
            print(f'Analyzing video {self.video_name}... ({file_cnt+1}/{len(self.data_paths)})')
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
            self.video_width, self.video_height = self.video_info['Resolution_width'].values[0], self.video_info['Resolution_height'].values[0]
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            for family_cnt, feature_family in enumerate(self.feature_families):
                print(f"Analyzing {self.video_name} and {feature_family} (Video {file_cnt + 1}/{len(self.outlier_corrected_paths)}, Family {family_cnt + 1}/{len(self.feature_families)})...")
                if feature_family == TWO_POINT_BP_DISTANCES:
                    self._get_two_point_bp_distances()
                elif feature_family == WITHIN_ANIMAL_THREE_POINT_ANGLES:
                    self.__get_three_point_angles()
                elif feature_family == WITHIN_ANIMAL_THREE_POINT_HULL:
                    self.__get_three_point_hulls()
                elif feature_family == WITHIN_ANIMAL_FOUR_POINT_HULL:
                    self.__get_four_point_hulls()
                elif feature_family == ANIMAL_CONVEX_HULL_PERIMETER:
                    self.__get_convex_hulls(method='perimeter')
                elif feature_family == ANIMAL_CONVEX_HULL_AREA:
                    self.__get_convex_hulls(method='area')
                elif feature_family == FRAME_BP_MOVEMENT:
                    self.__get_framewise_movement()
                elif feature_family == FRAME_BP_TO_ROI_CENTER:
                    self.__get_roi_center_distances()
                elif feature_family == FRAME_BP_INSIDE_ROI:
                    self.__get_inside_roi()
                elif feature_family == ARENA_EDGE:
                    self.__get_distances_to_frm_edge()

            self.results = self.results.add_suffix('_FEATURE_SUBSET')
            self.results = self.results[sorted(self.results.columns)]
            write_df(df=self.results.fillna(-1), file_type=self.file_type, save_path=save_path)
            video_timer.stop_timer()
            print(f"Feature subsets computed for {self.video_name} complete (elapsed time {video_timer.elapsed_time_str}s)...")
        if self.append_to_features_extracted:
            print(f'Appending new feature to files in {self.features_dir}...')
            self.__append_to_data_in_dir(dir=self.features_dir)
        if self.append_to_targets_inserted:
            print(f'Appending new feature to files in {self.targets_folder}...')
            self.__append_to_targets_inserted(dir=self.targets_folder)
        if self.save_dir is not None:
            print(f"Storing new features in {self.save_dir}...")
            copy_files_in_directory(in_dir=self.temp_dir, out_dir=self.save_dir, filetype=self.file_type, raise_error=True)
        remove_a_folder(folder_dir=self.temp_dir, ignore_errors=False)
        self.timer.stop_timer()
        stdout_success(msg="Feature sub-sets calculations complete!", elapsed_time=self.timer.elapsed_time_str)




# test = FeatureSubsetsCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                 feature_families=[ARENA_EDGE],
#                                 append_to_features_extracted=False,
#                                 file_checks=True,
#                                 append_to_targets_inserted=False,
#                                 save_dir=r"C:\troubleshooting\mitra\project_folder\csv\feature_subset")
# test.run()

#
# test = FeatureSubsetsCalculator(config_path=r"D:\Stretch\Stretch\project_folder\project_config.ini",
#                                 feature_families=[TWO_POINT_BP_DISTANCES],
#                                 append_to_features_extracted=True,
#                                 file_checks=True,
#                                 append_to_targets_inserted=True,
#                                 save_dir=r"D:\Stretch\Stretch\project_folder\new_features")
# test.run()



# test = FeatureSubsetsCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                 feature_families=[TWO_POINT_BP_DISTANCES],
#                                 append_to_features_extracted=False,
#                                 file_checks=False,
#                                 append_to_targets_inserted=False,
#                                 save_dir=r"C:\troubleshooting\mitra\project_folder\csv\new_features")
# test.run()



# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-parts inside ROIs (Boolean)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()
#
#
# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-part movements (mm)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()
#
#
# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_families=['Frame-by-frame body-part distances to ROI centers (mm)', 'Frame-by-frame body-parts inside ROIs (Boolean)'],
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data',
#                                 include_file_checks=True,
#                                 append_to_features_extracted=False,
#                                 append_to_targets_inserted=True)
# test.run()
