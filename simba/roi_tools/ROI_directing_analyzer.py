__author__ = "Simon Nilsson"

import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from numba import jit, prange

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.data import slice_roi_dict_for_video
from simba.utils.errors import (InvalidInputError, NoDataError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_data_paths, read_df


class DirectingROIAnalyzer(ConfigReader, FeatureExtractionMixin):
    """
    Compute aggregate statistics for animals directing towards ROIs.

    :param str config_path: Path to SimBA project config file in Configparser format
    :param Optional[Union[str, os.PathLike]] data_path: Path to folder or file holding the data used to calculate ROI aggregate statistics. If None, then defaults to the `project_folder/csv/outlier_corrected_movement_location` directory of the SimBA project. Default: None.

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

       `Example expected output file <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    :example:
    >>> test = DirectingROIAnalyzer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    >>> test.run()
    >>> test.save()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_path: Optional[Union[str, os.PathLike]] = None,
    ):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        self.data_paths = read_data_paths(
            path=data_path,
            default=self.outlier_corrected_paths,
            default_name=self.outlier_corrected_dir,
            file_type=self.file_type,
        )
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(
                expected_file_path=self.roi_coordinates_path
            )
        if not self.check_directionality_viable()[0]:
            raise InvalidInputError(
                msg="Cannot compute directionality towards ROIs. The ear and nose data is tracked in the project",
                source=self.__class__.__name__,
            )
        self.read_roi_data()
        self.direct_bp_dict = self.check_directionality_cords()

    def __format_direction_data(
        self,
        direction_data: np.ndarray,
        nose_arr: np.ndarray,
        roi_center: np.ndarray,
        animal_name: str,
        shape_name: str,
    ) -> pd.DataFrame:

        x_min = np.minimum(direction_data[:, 1], nose_arr[:, 0])
        y_min = np.minimum(direction_data[:, 2], nose_arr[:, 1])
        delta_x = abs((direction_data[:, 1] - nose_arr[:, 0]) / 2)
        delta_y = abs((direction_data[:, 2] - nose_arr[:, 1]) / 2)
        x_middle, y_middle = np.add(x_min, delta_x), np.add(y_min, delta_y)
        direction_data = np.concatenate(
            (y_middle.reshape(-1, 1), direction_data), axis=1
        )
        direction_data = np.concatenate(
            (x_middle.reshape(-1, 1), direction_data), axis=1
        )
        direction_data = np.delete(direction_data, [2, 3, 4], 1)
        bp_data = pd.DataFrame(
            direction_data, columns=["Eye_x", "Eye_y", "Directing_BOOL"]
        )
        bp_data["ROI_x"] = roi_center[0]
        bp_data["ROI_y"] = roi_center[1]
        bp_data = bp_data[["Eye_x", "Eye_y", "ROI_x", "ROI_y", "Directing_BOOL"]]
        bp_data.insert(loc=0, column="ROI", value=shape_name)
        bp_data.insert(loc=0, column="Animal", value=animal_name)
        bp_data.insert(loc=0, column="Video", value=self.video_name)
        bp_data = bp_data.reset_index().rename(columns={"index": "Frame"})
        bp_data = bp_data[bp_data["Directing_BOOL"] == 1].reset_index(drop=True)
        return bp_data

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def ccw(roi_lines: np.array, eye_lines: np.array, shape_type: str):
        def calc(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        results = np.full((eye_lines.shape[0], 4), -1)
        for i in prange(eye_lines.shape[0]):
            eye, roi = eye_lines[i][0:2], eye_lines[i][2:4]
            min_distance = np.inf

            if shape_type == "Circle":
                reversed_roi_lines = roi_lines[::-1]
                for j in prange(roi_lines.shape[0]):
                    dist_1 = np.sqrt(
                        (eye[0] - roi_lines[j][0]) ** 2
                        + (eye[1] - roi_lines[j][1]) ** 2
                    )
                    dist_2 = np.sqrt(
                        (eye[0] - roi_lines[j][2]) ** 2
                        + (eye[1] - roi_lines[j][3]) ** 2
                    )
                    if (dist_1 < min_distance) or (dist_2 < min_distance):
                        min_distance = min(dist_1, dist_2)
                        results[i] = reversed_roi_lines[j]

            else:
                for j in prange(roi_lines.shape[0]):
                    line_a, line_b = roi_lines[j][0:2], roi_lines[j][2:4]
                    center_x, center_y = (
                        line_a[0] + line_b[0] // 2,
                        line_a[1] + line_b[1] // 2,
                    )
                    if calc(eye, line_a, line_b) != calc(roi, line_a, line_b) or calc(
                        eye, roi, line_a
                    ) != calc(eye, roi, line_b):
                        distance = np.sqrt(
                            (eye[0] - center_x) ** 2 + (eye[1] - center_y) ** 2
                        )
                        if distance < min_distance:
                            results[i] = roi_lines[j]
                            min_distance = distance

        return results

    def __find_roi_intersections(self, bp_data: pd.DataFrame, shape_info: dict):

        eye_lines = bp_data[["Eye_x", "Eye_y", "ROI_x", "ROI_y"]].values.astype(int)
        roi_lines = None
        if shape_info["Shape_type"] == "Rectangle":
            top_left_x, top_left_y = (shape_info["topLeftX"], shape_info["topLeftY"])
            bottom_right_x, bottom_right_y = (
                shape_info["Bottom_right_X"],
                shape_info["Bottom_right_Y"],
            )
            top_right_x, top_right_y = top_left_x + shape_info["width"], top_left_y
            bottom_left_x, bottom_left_y = (
                bottom_right_x - shape_info["width"],
                bottom_right_y,
            )
            roi_lines = np.array(
                [
                    [top_left_x, top_left_y, bottom_left_x, bottom_left_y],
                    [bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y],
                    [bottom_right_x, bottom_right_y, top_right_x, top_right_y],
                    [top_right_x, top_right_y, top_left_x, top_left_y],
                ]
            )

        elif shape_info["Shape_type"] == "Polygon":
            roi_lines = np.full((shape_info["vertices"].shape[0], 4), np.nan)
            roi_lines[-1] = np.hstack(
                (shape_info["vertices"][0], shape_info["vertices"][-1])
            )
            for i in range(shape_info["vertices"].shape[0] - 1):
                roi_lines[i] = np.hstack(
                    (shape_info["vertices"][i], shape_info["vertices"][i + 1])
                )

        elif shape_info["Shape_type"] == "Circle":
            center = shape_info[["centerX", "centerY"]].values.astype(int)
            roi_lines = np.full((2, 4), np.nan)
            roi_lines[0] = np.array(
                [
                    center[0],
                    center[1] - shape_info["radius"],
                    center[0],
                    center[1] + shape_info["radius"],
                ]
            )
            roi_lines[1] = np.array(
                [
                    center[0] - shape_info["radius"],
                    center[1],
                    center[0] + shape_info["radius"],
                    center[1],
                ]
            )

        return self.ccw(
            roi_lines=roi_lines,
            eye_lines=eye_lines,
            shape_type=shape_info["Shape_type"],
        )

    def run(self):
        self.results = []
        for file_cnt, file_path in enumerate(self.data_paths):
            _, self.video_name, _ = get_fn_ext(file_path)
            video_timer = SimbaTimer(start=True)
            print(f"Analyzing ROI directionality in video {self.video_name}...")
            data_df = read_df(file_path=file_path, file_type=self.file_type)
            video_roi_dict, shape_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=self.video_name)
            for animal_name, bps in self.direct_bp_dict.items():
                ear_left_arr = data_df[[bps["Ear_left"]["X_bps"], bps["Ear_left"]["Y_bps"]]].values
                ear_right_arr = data_df[[bps["Ear_right"]["X_bps"], bps["Ear_right"]["Y_bps"]]].values
                nose_arr = data_df[[bps["Nose"]["X_bps"], bps["Nose"]["Y_bps"]]].values
                for roi_type, roi_type_data in video_roi_dict.items():
                    for _, row in roi_type_data.iterrows():
                        roi_center = np.array([row["Center_X"], row["Center_Y"]])
                        roi_name = row["Name"]
                        direction_data = FeatureExtractionMixin.jitted_line_crosses_to_static_targets(
                            left_ear_array=ear_left_arr,
                            right_ear_array=ear_right_arr,
                            nose_array=nose_arr,
                            target_array=roi_center,
                        )
                        bp_data = self.__format_direction_data(
                            direction_data=direction_data,
                            nose_arr=nose_arr,
                            roi_center=roi_center,
                            animal_name=animal_name,
                            shape_name=roi_name,
                        )

                        eye_roi_intersections = pd.DataFrame(
                            self.__find_roi_intersections(
                                bp_data=bp_data, shape_info=row
                            ),
                            columns=[
                                "ROI_edge_1_x",
                                "ROI_edge_1_y",
                                "ROI_edge_2_x",
                                "ROI_edge_2_y",
                            ],
                        )
                        self.results.append(
                            pd.concat([bp_data, eye_roi_intersections], axis=1)
                        )
            video_timer.stop_timer()
            print(f"ROI directionality analyzed in video {self.video_name}... (elapsed time: {video_timer.elapsed_time_str}s)")
        if len(self.results) == 0:
            raise NoDataError(msg=f'No ROI DATA exists for data files {self.data_paths}', source=self.__class__.__name__)
        self.results_df = pd.concat(self.results, axis=0)

    def save(self, path: Optional[Union[str, os.PathLike]] = None):
        if not hasattr(self, "results_df"):
            raise InvalidInputError(msg="Run the ROI direction analyzer before saving")
        if path is None:
            path = os.path.join(
                self.logs_path, f"ROI_directionality_summary_{self.datetime}.csv"
            )
        self.results_df.to_csv(path)
        stdout_success(
            msg=f"Detailed ROI directionality data saved in {path}",
            source=self.__class__.__name__,
        )


#
# test = DirectingROIAnalyzer(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini')
# test.run()
# test.save()
