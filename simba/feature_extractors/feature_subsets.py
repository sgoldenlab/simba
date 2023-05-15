__author__ = "Simon Nilsson"

import pandas as pd
import os
import numpy as np
from typing import Union
try:
    from typing import Literal
except:
    from typing_extensions import Literal

from itertools import combinations
from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.utils.enums import Formats, Options
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.read_write import get_fn_ext, read_df, write_df


class FeatureSubsetsCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Computes a subset of features from pose for non-ML downstream purposes.
    E.g., returns the size of animal convex hull in each frame.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str save_dir: directory where to store results.
    :parameter str feature_family: Feature subtype to calculate. E.g., "Two-point body-part distances (mm)".

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/feature_subsets.md>_`

    Examples
    ----------
    >>> _ = FeatureSubsetsCalculator(config_path='project_folder/project_config.ini', feature_family='Frame-by-frame body-parts inside ROIs (Boolean)', save_dir='data').run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 feature_family: Literal[Options.FEATURE_SUBSET_OPTIONS]):

        FeatureExtractionMixin.__init__(self, config_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        check_if_filepath_list_is_empty(
            filepaths=self.outlier_corrected_paths,
            error_msg=f"SIMBA ERROR: Zero data files found in {self.outlier_corrected_paths} directory",
        )
        self.feature_family, self.save_dir = feature_family, save_dir

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
        for file_path in self.outlier_corrected_paths:
            self.video_timer = SimbaTimer(start=True)
            self.results = pd.DataFrame()
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            print(f"Analyzing {self.video_name} ({self.feature_family})...")
            _, self.pixel_per_mm, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            self.df = read_df(file_path=file_path, file_type=self.file_type)
            if self.feature_family == "Two-point body-part distances (mm)":
                self.calc_distances()
            elif (
                self.feature_family
                == "Within-animal three-point body-part angles (degrees)"
            ):
                self.calc_angles()
            elif (
                self.feature_family
                == "Within-animal three-point convex hull perimeters (mm)"
            ):
                self.calc_three_point_hulls()
            elif (
                self.feature_family
                == "Within-animal four-point convex hull perimeters (mm)"
            ):
                self.calc_four_point_hulls()
            elif self.feature_family == "Entire animal convex hull perimeters (mm)":
                self.animal_convex_hulls()
            elif self.feature_family == "Entire animal convex hull area (mm2)":
                self.calc_animal_convex_hulls_area()
            elif self.feature_family == "Frame-by-frame body-part movements (mm)":
                self.calc_movements()
            elif (
                self.feature_family
                == "Frame-by-frame body-part distances to ROI centers (mm)"
            ):
                self.calc_roi_center_distances()
            elif (
                self.feature_family == "Frame-by-frame body-parts inside ROIs (Boolean)"
            ):
                self.calc_inside_roi()
            self.__save()
            self.video_timer.stop_timer()
            print(
                f"Video {self.video_name} complete (elapsed time {self.video_timer.elapsed_time_str}s)..."
            )
        self.timer.stop_timer()
        stdout_success(
            msg=f"{self.feature_family} for {str(len(self.outlier_corrected_paths))} videos saved in {self.save_dir}",
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
                self.results[
                    f"Angle (degrees) {point[0]}-{point[1]}-{point[2]}"
                ] = self.angle3pt_serialized(data=self.df[col_names].values)

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
                        )

    def calc_inside_roi(self):
        self.read_roi_data()
        for animal, animal_bps in self.animal_bps.items():
            for bp in animal_bps:
                bp_arr = self.df[[f"{bp}_x", f"{bp}_y"]].values.astype(float)
                for shape_type, shape_data in self.roi_dict.items():
                    for shape_name in shape_data["Name"].unique():
                        if shape_type == "rectangles":
                            shape_data = shape_data.loc[
                                (shape_data["Video"] == self.video_name)
                                & (shape_data["Name"] == shape_name)
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
                            shape_data = shape_data.loc[
                                (shape_data["Video"] == self.video_name)
                                & (shape_data["Name"] == shape_name)
                            ][["vertices"]].values[0][0]
                            self.results[
                                f"{animal} {bp} inside polygon {shape_name} (Boolean)"
                            ] = self.framewise_inside_polygon_roi(
                                bp_location=bp_arr, roi_coords=shape_data
                            )

    def __save(self):
        save_path = os.path.join(self.save_dir, f"{self.video_name}.csv")
        write_df(
            df=self.results.round(2), file_type=Formats.CSV.value, save_path=save_path
        )


# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-parts inside ROIs (Boolean)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()


# test = FeatureSubsetsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 feature_family='Frame-by-frame body-part movements (mm)',
#                                 save_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/data')
# test.run()
