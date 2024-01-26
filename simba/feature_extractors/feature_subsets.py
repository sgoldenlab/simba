__author__ = "Simon Nilsson"

import glob
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import shutil
from itertools import combinations

from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_filepath_list_is_empty,
                                check_if_headers_in_dfs_are_unique,
                                check_same_number_of_rows_in_dfs,
                                check_that_column_exist)
from simba.utils.enums import Formats, Options
from simba.utils.errors import CountError, DataHeaderError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class FeatureSubsetsCalculator(ConfigReader, FeatureExtractionMixin, TrainModelMixin):
    """
    Computes a subset of features from pose for non-ML downstream purposes.
    E.g., returns the size of animal convex hull in each frame.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str save_dir: directory where to store results.
    :parameter str feature_family: Feature subtype to calculate. E.g., "Two-point body-part distances (mm)".

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/feature_subsets.md>_`

    .. image:: _static/img/feature_subsets.png
       :width: 400
       :align: center

    Examples
    ----------
    >>> _ = FeatureSubsetsCalculator(config_path='project_folder/project_config.ini', feature_family='Frame-by-frame body-parts inside ROIs (Boolean)', save_dir='data').run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        save_dir: Union[str, os.PathLike, None],
        feature_families: List[str],
        include_file_checks: Optional[bool] = False,
        append_to_features_extracted: Optional[bool] = False,
        append_to_targets_inserted: Optional[bool] = False,
    ):
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        check_if_filepath_list_is_empty(
            filepaths=self.outlier_corrected_paths,
            error_msg=f"SIMBA ERROR: Zero data files found in {self.outlier_corrected_paths} directory",
        )
        self.feature_families, self.save_dir = feature_families, save_dir
        self.append_to_features, self.append_to_targets, self.checks_files = (
            append_to_features_extracted,
            append_to_targets_inserted,
            include_file_checks,
        )
        self.create_temp_folders()

    def create_temp_folders(self):
        self.general_temp_folder = os.path.join(
            self.project_path, "csv", f"temp_data_{self.datetime}"
        )
        if not os.path.isdir(self.general_temp_folder):
            os.makedirs(self.general_temp_folder)
        self.features_extracted_temp_path, self.targets_inserted_temp_path = None, None
        if self.append_to_features or self.append_to_targets:
            if self.append_to_features:
                self.features_extracted_temp_path = os.path.join(
                    self.features_dir, f"temp_data_{self.datetime}"
                )
                if not os.path.isdir(self.features_extracted_temp_path):
                    os.makedirs(self.features_extracted_temp_path)
            if self.append_to_targets:
                self.targets_inserted_temp_path = os.path.join(
                    self.targets_folder, f"temp_data_{self.datetime}"
                )
                if not os.path.isdir(self.targets_inserted_temp_path):
                    os.makedirs(self.targets_inserted_temp_path)

    def perform_clean_up(self, directories: List[str]):
        print("Performing clean up and deleting temporary directories...")
        self.remove_multiple_folders(folders=directories)

    def __get_bp_combinations(self):
        self.two_point_combs = np.array(list(combinations(self.project_bps, 2)))
        self.within_animal_three_point_combs = {}
        self.within_animal_four_point_combs = {}
        self.animal_bps = {}
        for animal, animal_data in self.animal_bp_dict.items():
            animal_bps = [x[:-2] for x in animal_data["X_bps"]]
            self.animal_bps[animal] = animal_bps
            self.within_animal_three_point_combs[animal] = np.array(
                list(combinations(animal_bps, 3))
            )
            self.within_animal_four_point_combs[animal] = np.array(
                list(combinations(animal_bps, 4))
            )

    def run(self):
        self.__get_bp_combinations()
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            self.current_file_path = file_path
            self.video_timer = SimbaTimer(start=True)
            self.results = pd.DataFrame()
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            _, self.pixel_per_mm, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            self.df = read_df(file_path=file_path, file_type=self.file_type)
            for family_cnt, feature_family in enumerate(self.feature_families):
                print(
                    f"Analyzing {self.video_name} and {feature_family} (Video {file_cnt+1}/{len(self.outlier_corrected_paths)}, Family {family_cnt+1}/{len(self.feature_families)})..."
                )
                if feature_family == "Two-point body-part distances (mm)":
                    self.calc_distances()
                elif (
                    feature_family
                    == "Within-animal three-point body-part angles (degrees)"
                ):
                    self.calc_angles()
                elif (
                    feature_family
                    == "Within-animal three-point convex hull perimeters (mm)"
                ):
                    self.calc_three_point_hulls()
                elif (
                    feature_family
                    == "Within-animal four-point convex hull perimeters (mm)"
                ):
                    self.calc_four_point_hulls()
                elif feature_family == "Entire animal convex hull perimeters (mm)":
                    self.animal_convex_hulls()
                elif feature_family == "Entire animal convex hull area (mm2)":
                    self.calc_animal_convex_hulls_area()
                elif feature_family == "Frame-by-frame body-part movements (mm)":
                    self.calc_movements()
                elif (
                    feature_family
                    == "Frame-by-frame body-part distances to ROI centers (mm)"
                ):
                    self.calc_roi_center_distances()
                elif (
                    feature_family == "Frame-by-frame body-parts inside ROIs (Boolean)"
                ):
                    self.calc_inside_roi()
            self.__save_to_temp()
            self.video_timer.stop_timer()
            print(
                f"Additional features computed for {self.video_name} complete (elapsed time {self.video_timer.elapsed_time_str}s)..."
            )
        if self.append_to_features or self.append_to_targets:
            print("Appending data to existing files...")
            self.append_to_data()
        self.replace_files_in_folders()
        self.perform_clean_up(
            directories=[
                self.targets_inserted_temp_path,
                self.features_extracted_temp_path,
                self.general_temp_folder,
            ]
        )
        self.timer.stop_timer()
        stdout_success(
            msg="Feature sub-sets calculations complete!",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def calc_distances(self):
        for c in self.two_point_combs:
            col_names = list(sum([(x + "_x", y + "_y") for (x, y) in zip(c, c)], ()))

            self.results[f"Distance (mm) {c[0]}-{c[1]}"] = self.euclidean_distance(
                self.df[col_names[0]].values,
                self.df[col_names[2]].values,
                self.df[col_names[1]].values,
                self.df[col_names[3]].values,
                self.pixel_per_mm,
            )

    def calc_angles(self):
        for animal, points in self.within_animal_three_point_combs.items():
            for point in points:
                col_names = list(
                    sum([(x + "_x", y + "_y") for (x, y) in zip(point, point)], ())
                )
                self.results[f"Angle (degrees) {point[0]}-{point[1]}-{point[2]}"] = (
                    self.angle3pt_serialized(data=self.df[col_names].values)
                )

    def calc_three_point_hulls(self):
        for animal, points in self.within_animal_three_point_combs.items():
            for point in points:
                col_names = list(
                    sum([(x + "_x", y + "_y") for (x, y) in zip(point, point)], ())
                )
                three_point_arr = np.reshape(
                    self.df[col_names].values, (len(self.df / 2), -1, 2)
                ).astype(np.float32)
                self.results[
                    f"{animal} three-point convex hull perimeter (mm) {point[0]}-{point[1]}-{point[2]}"
                ] = (
                    jitted_hull(points=three_point_arr, target=Formats.PERIMETER.value)
                    / self.pixel_per_mm
                )

    def calc_four_point_hulls(self):
        for animal, points in self.within_animal_four_point_combs.items():
            for point in points:
                col_names = list(
                    sum([(x + "_x", y + "_y") for (x, y) in zip(point, point)], ())
                )
                four_point_arr = np.reshape(
                    self.df[col_names].values, (len(self.df / 2), -1, 2)
                ).astype(np.float32)
                self.results[
                    f"{animal} four-point convex perimeter (mm) {point[0]}-{point[1]}-{point[2]}-{point[3]}"
                ] = (
                    jitted_hull(points=four_point_arr, target=Formats.PERIMETER.value)
                    / self.pixel_per_mm
                )

    def animal_convex_hulls(self):
        for animal, point in self.animal_bps.items():
            col_names = list(
                sum([(x + "_x", y + "_y") for (x, y) in zip(point, point)], ())
            )
            animal_point_arr = np.reshape(
                self.df[col_names].values, (len(self.df / 2), -1, 2)
            ).astype(np.float32)
            self.results[f"{animal} convex hull perimeter (mm)"] = (
                jitted_hull(points=animal_point_arr, target=Formats.PERIMETER.value)
                / self.pixel_per_mm
            )

    def calc_movements(self):
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                check_that_column_exist(
                    df=self.df, column_name=f"{bp}_x", file_name=self.current_file_path
                )
                check_that_column_exist(
                    df=self.df, column_name=f"{bp}_y", file_name=self.current_file_path
                )
                bp_df = self.df[[f"{bp}_x", f"{bp}_y"]]
                shift = bp_df.shift(periods=1).add_suffix("_shift")
                bp_df = pd.concat([bp_df, shift], axis=1, join="inner").reset_index(
                    drop=True
                )
                for c in shift.columns:
                    bp_df[c] = bp_df[c].fillna(bp_df[c[:-6]])
                self.results[f"{animal} movement {bp} (mm)"] = self.euclidean_distance(
                    bp_df[f"{bp}_x_shift"].values,
                    bp_df[f"{bp}_x"].values,
                    bp_df[f"{bp}_y_shift"].values,
                    bp_df[f"{bp}_y"].values,
                    self.pixel_per_mm,
                )

    def calc_animal_convex_hulls_area(self):
        for animal, point in self.animal_bps.items():
            col_names = list(
                sum([(x + "_x", y + "_y") for (x, y) in zip(point, point)], ())
            )
            animal_point_arr = np.reshape(
                self.df[col_names].values, (len(self.df / 2), -1, 2)
            ).astype(np.float32)
            self.results[f"{animal} convex hull area (mm2)"] = (
                jitted_hull(points=animal_point_arr, target=Formats.AREA.value)
                / self.pixel_per_mm
            )

    def calc_roi_center_distances(self):
        self.read_roi_data()
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                check_that_column_exist(
                    df=self.df, column_name=f"{bp}_x", file_name=self.current_file_path
                )
                check_that_column_exist(
                    df=self.df, column_name=f"{bp}_y", file_name=self.current_file_path
                )
                bp_arr = self.df[[f"{bp}_x", f"{bp}_y"]].values.astype(float)
                for shape_type, shape_data in self.roi_dict.items():
                    for shape_name in shape_data["Name"].unique():
                        center_point = (
                            shape_data.loc[
                                (shape_data["Video"] == self.video_name)
                                & (shape_data["Name"] == shape_name)
                            ][["Center_X", "Center_Y"]]
                            .astype(float)
                            .values[0]
                        )
                        self.results[
                            f"{animal} {bp} to {shape_name} center distance (mm)"
                        ] = self.framewise_euclidean_distance_roi(
                            location_1=bp_arr,
                            location_2=center_point,
                            px_per_mm=self.pixel_per_mm,
                        ).astype(
                            "float32"
                        )

    def calc_inside_roi(self):
        self.read_roi_data()
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                check_that_column_exist(
                    df=self.df, column_name=f"{bp}_x", file_name=self.current_file_path
                )
                check_that_column_exist(
                    df=self.df, column_name=f"{bp}_y", file_name=self.current_file_path
                )
                bp_arr = self.df[[f"{bp}_x", f"{bp}_y"]].values.astype(float)
                for shape_type, shape_info in self.roi_dict.items():
                    for shape_name in shape_info["Name"].unique():
                        if shape_type == "rectangles":
                            shape_data = shape_info.loc[
                                (shape_info["Video"] == self.video_name)
                                & (shape_info["Name"] == shape_name)
                            ][
                                [
                                    "topLeftX",
                                    "topLeftY",
                                    "Bottom_right_X",
                                    "Bottom_right_Y",
                                ]
                            ].values.reshape(
                                2, 2
                            )
                            self.results[
                                f"{animal} {bp} inside rectangle {shape_name} (Boolean)"
                            ] = self.framewise_inside_rectangle_roi(
                                bp_location=bp_arr, roi_coords=shape_data
                            )
                        if shape_type == "polygons":
                            shape_data = shape_info.loc[
                                (shape_info["Video"] == self.video_name)
                                & (shape_info["Name"] == shape_name)
                            ][["vertices"]].values[0][0]
                            self.results[
                                f"{animal} {bp} inside polygon {shape_name} (Boolean)"
                            ] = self.framewise_inside_polygon_roi(
                                bp_location=bp_arr, roi_coords=shape_data
                            )

    def __save_to_temp(self):
        save_path = os.path.join(
            self.general_temp_folder, self.video_name + f".{self.file_type}"
        )
        write_df(
            df=self.results.fillna(-1), file_type=self.file_type, save_path=save_path
        )

    def append_to_data(self):
        for file_path in glob.glob(self.general_temp_folder + f"/*.{self.file_type}"):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            new_df = read_df(file_path=file_path, file_type=self.file_type)
            if self.append_to_features:
                print(f"Appending new data to features extracted from {video_name}...")
                data_save_path = os.path.join(
                    self.features_extracted_temp_path, video_name + f".{self.file_type}"
                )
                features_path = os.path.join(
                    self.features_dir, f"{video_name}.{self.file_type}"
                )
                check_file_exist_and_readable(file_path=file_path)
                data_df = read_df(file_path=features_path, file_type=self.file_type)
                if not check_same_number_of_rows_in_dfs(dfs=[data_df, new_df]):
                    self.remove_multiple_folders(
                        folders=[
                            self.targets_inserted_temp_path,
                            self.features_extracted_temp_path,
                            self.general_temp_folder,
                        ]
                    )
                    raise CountError(
                        f"Cannot append feature subsets to features file of {video_name}. The files contain an unequal number of rows ({len(new_df)} vs {len(data_df)})"
                    )
                if len(check_if_headers_in_dfs_are_unique(dfs=[data_df, new_df])) > 0:
                    self.remove_multiple_folders(
                        folders=[
                            self.targets_inserted_temp_path,
                            self.features_extracted_temp_path,
                            self.general_temp_folder,
                        ]
                    )
                    raise DataHeaderError(
                        msg=f"Attempted to append new feature data to features extracted file {features_path}. At least one of the new feature names aleady exist in the file."
                    )
                save_df = pd.concat([data_df, new_df], axis=1)
                write_df(df=save_df, file_type=self.file_type, save_path=data_save_path)

            if self.append_to_targets:
                print(f"Appending new data to targets inserted from {video_name}...")
                data_save_path = os.path.join(
                    self.targets_inserted_temp_path, video_name + f".{self.file_type}"
                )
                target_path = os.path.join(
                    self.targets_folder, f"{video_name}.{self.file_type}"
                )
                check_file_exist_and_readable(file_path=file_path)
                data_df = read_df(file_path=target_path, file_type=self.file_type)
                targets_clf_df = pd.DataFrame()
                if not check_same_number_of_rows_in_dfs(dfs=[data_df, new_df]):
                    self.perform_clean_up(
                        directories=[
                            self.targets_inserted_temp_path,
                            self.features_extracted_temp_path,
                            self.general_temp_folder,
                        ]
                    )
                    raise CountError(
                        f"Cannot append feature subsets to targets file of {video_name}. The files contain an unequal number of rows ({len(new_df)} vs {len(data_df)})"
                    )
                if len(check_if_headers_in_dfs_are_unique(dfs=[data_df, new_df])) > 0:
                    self.perform_clean_up(
                        directories=[
                            self.targets_inserted_temp_path,
                            self.features_extracted_temp_path,
                            self.general_temp_folder,
                        ]
                    )
                    raise DataHeaderError(
                        msg=f"Attempted to append new feature data to targets file of {target_path}. At least one of the nes feature names aleady exist in the file."
                    )
                for clf in self.clf_names:
                    if clf in data_df.columns:
                        targets_clf_df[clf] = data_df[clf]
                        data_df.drop(clf, axis=1, inplace=True)
                save_df = pd.concat([data_df, new_df, targets_clf_df], axis=1)
                write_df(df=save_df, file_type=self.file_type, save_path=data_save_path)

        if self.checks_files:
            if self.append_to_features:
                print('Confirming integrity of new "features_extracted" data...')
                file_paths = glob.glob(
                    self.features_extracted_temp_path + f"/*.{self.file_type}"
                )
                self.data_df = self.read_all_files_in_folder_mp_futures(
                    file_paths=file_paths, file_type=self.file_type
                )
                self.check_raw_dataset_integrity(
                    df=self.data_df, logs_path=self.logs_path
                )
                del self.data_df

            if self.append_to_targets:
                print('Confirming integrity of new "targets_inserted" data...')
                file_paths = glob.glob(
                    self.targets_inserted_temp_path + f"/*.{self.file_type}"
                )
                self.data_df = self.read_all_files_in_folder_mp_futures(
                    file_paths=file_paths, file_type=self.file_type
                )
                self.check_raw_dataset_integrity(
                    df=self.data_df, logs_path=self.logs_path
                )
                del self.data_df

    def replace_files_in_folders(self):
        if self.append_to_features:
            prior_to_features_append_dir = os.path.join(
                self.features_dir, f"Prior_to_feature_subset_append_{self.datetime}"
            )
            print(
                f'Replacing data in "features_extracted" directory with additional data (For safety, previous files are copied to {prior_to_features_append_dir})...'
            )
            if not os.path.isdir(prior_to_features_append_dir):
                os.makedirs(prior_to_features_append_dir)
            for file_path in glob.glob(self.features_dir + f"/*.{self.file_type}"):
                shutil.move(
                    file_path,
                    os.path.join(
                        prior_to_features_append_dir, os.path.basename(file_path)
                    ),
                )
            for file_path in glob.glob(
                self.features_extracted_temp_path + f"/*.{self.file_type}"
            ):
                shutil.move(
                    file_path,
                    os.path.join(self.features_dir, os.path.basename(file_path)),
                )

        if self.append_to_targets:
            prior_to_targets_append_dir = os.path.join(
                self.targets_folder, f"Prior_to_feature_subset_append_{self.datetime}"
            )
            print(
                f'Replacing data in "targets_inserted" directory with additional data (For safety, previous files are copied to {prior_to_targets_append_dir})...'
            )
            if not os.path.isdir(prior_to_targets_append_dir):
                os.makedirs(prior_to_targets_append_dir)
            for file_path in glob.glob(self.targets_folder + f"/*.{self.file_type}"):
                shutil.move(
                    file_path,
                    os.path.join(
                        prior_to_targets_append_dir, os.path.basename(file_path)
                    ),
                )
            for file_path in glob.glob(
                self.targets_inserted_temp_path + f"/*.{self.file_type}"
            ):
                shutil.move(
                    file_path,
                    os.path.join(self.targets_folder, os.path.basename(file_path)),
                )

        if self.save_dir != None:
            print(f"Storing new features in {self.save_dir}...")
            for file_path in glob.glob(
                self.general_temp_folder + f"/*.{self.file_type}"
            ):
                shutil.move(
                    file_path, os.path.join(self.save_dir, os.path.basename(file_path))
                )


# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-parts inside ROIs (Boolean)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()


# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-part movements (mm)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()

#
# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_families=['Frame-by-frame body-part distances to ROI centers (mm)', 'Frame-by-frame body-parts inside ROIs (Boolean)'],
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data',
#                                 include_file_checks=True,
#                                 append_to_features_extracted=False,
#                                 append_to_targets_inserted=True)
# test.run()
