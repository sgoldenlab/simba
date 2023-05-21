__author__ = "Simon Nilsson"

import os
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from joblib import Parallel, delayed
from shapely.geometry import Point

from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import Formats
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import stdout_success
from simba.utils.read_write import read_df, write_df


class BoundaryStatisticsCalculator(ConfigReader):
    """
    Compute boundary intersection statistics.

    :parameter str config_path: Path to SimBA project config file in Configparser format.
    :parameter bool roi_intersections: If True, calculates intersection of animal-anchored ROIs
    :parameter bool roi_keypoint_intersections: If True, calculates intersection of animal-anchored ROIs and pose-estimated animal key-points.
    :parameter str save_format: Output data format. OPTIONS: CSV, PARQUET, PICKLE.

    Notes
    ----------
    `Bounding boxes tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/anchored_rois.md>`_.

    Examples
    ----------
    >>> boundary_stats_calculator = BoundaryStatisticsCalculator(config_path='MyConfigFile',roi_intersections=True, roi_keypoint_intersections=True, save_format='CSV')
    >>> boundary_stats_calculator.save_results()
    """

    def __init__(
        self,
        config_path: str,
        roi_intersections: bool,
        roi_keypoint_intersections: bool,
        save_format: str,
    ):
        ConfigReader.__init__(self, config_path=config_path)
        self.save_format, self.roi_intersections, self.roi_keypoint_intersections = (
            save_format,
            roi_intersections,
            roi_keypoint_intersections,
        )
        self.anchored_roi_path = os.path.join(
            self.project_path, "logs", "anchored_rois.pickle"
        )
        self.save_folder = os.path.join(self.project_path, "csv", "anchored_roi_data")
        if not os.path.isfile(self.anchored_roi_path):
            raise NoFilesFoundError(
                msg=f"No anchored ROI data detected. Extract anchored ROIs before computing statistics. File expected at path {self.anchored_roi_path}"
            )
        self.polygons = read_df(
            file_path=self.anchored_roi_path, file_type=Formats.PICKLE.value
        )
        self.calculate_statistics()

    def _find_intersections(self, animal_roi: list, other_animals: dict):
        results = []
        for first_animal, second_animal in zip(animal_roi, other_animals):
            results.append(first_animal.intersects(second_animal))
        return results

    def _find_points_in_roi(self, animal_roi: list, second_animal_bps: list):
        results = []
        for polygon_cnt, polygon in enumerate(animal_roi):
            frm_results = []
            for k, v in second_animal_bps[polygon_cnt].items():
                frm_results.append(Point(v).within(polygon))
            results.append(frm_results)
        return results

    def _sort_keypoint_results(self, first_animal_name: str, second_animal_name: str):
        results = defaultdict(list)
        for frm in self.results:
            for body_part_cnt, body_part in enumerate(frm):
                body_part_name = self.animal_bp_dict[second_animal_name]["X_bps"][
                    body_part_cnt
                ][:-4]
                results[
                    "{}:{}:{}".format(
                        first_animal_name, second_animal_name, body_part_name
                    )
                ].append(int(body_part))
        return pd.DataFrame(results)

    def _sort_intersection_results(self):
        results = defaultdict(list)
        for animal_one_name, animal_one_data in self.intersecting_rois.items():
            for animal_two_name, animal_two_data in animal_one_data.items():
                results["{}:{}:ROI_ONLY".format(animal_one_name, animal_two_name)] = [
                    int(x) for x in animal_two_data
                ]
        return pd.DataFrame(results)

    def calculate_statistics(self):
        self.intersection_dfs = {}
        self.keypoint_dfs = {}
        for video_cnt, (video_name, video_data) in enumerate(self.polygons.items()):
            print("Calculating statistics for video {}...".format(video_name))
            if self.roi_intersections:
                self.intersecting_rois = {}
                print(
                    "Calculating intersecting anchored ROIs for video {}...".format(
                        video_name
                    )
                )
                for first_animal in self.animal_bp_dict.keys():
                    first_animal_anchored_rois = [
                        video_data[first_animal][i : i + 100]
                        for i in range(0, len(video_data[first_animal]), 100)
                    ]
                    self.intersecting_rois[first_animal] = {}
                    for second_animal in {
                        k: v for k, v in video_data.items() if k != first_animal
                    }.keys():
                        second_animal_anchored_rois = [
                            video_data[second_animal][i : i + 100]
                            for i in range(0, len(video_data[second_animal]), 100)
                        ]
                        results = Parallel(n_jobs=5, verbose=2, backend="threading")(
                            delayed(self._find_intersections)(i, j)
                            for i, j in zip(
                                first_animal_anchored_rois, second_animal_anchored_rois
                            )
                        )
                        self.intersecting_rois[first_animal][second_animal] = [
                            i for s in results for i in s
                        ]
                self.intersection_dfs[video_name] = self._sort_intersection_results()

            if self.roi_keypoint_intersections:
                self.data_df = read_df(
                    os.path.join(
                        self.outlier_corrected_dir, video_name + "." + self.file_type
                    ),
                    self.file_type,
                ).astype(int)
                keypoints_df_lst = []
                print(
                    "Calculate intersecting anchored ROIs and keypoints for video {}...".format(
                        video_name
                    )
                )
                for first_animal in self.animal_bp_dict.keys():
                    first_animal_anchored_rois = [
                        video_data[first_animal][i : i + 100]
                        for i in range(0, len(video_data[first_animal]), 100)
                    ]
                    for second_animal in {
                        k: v
                        for k, v in self.animal_bp_dict.items()
                        if k != first_animal
                    }.keys():
                        second_animal_name = deepcopy(second_animal)
                        second_animal_df_tuples = pd.DataFrame()
                        for x_col, y_col in zip(
                            self.animal_bp_dict[second_animal]["X_bps"],
                            self.animal_bp_dict[second_animal]["Y_bps"],
                        ):
                            second_animal_df_tuples[x_col[:-4]] = list(
                                zip(self.data_df[x_col], self.data_df[y_col])
                            )
                        second_animal = second_animal_df_tuples.to_dict(
                            orient="records"
                        )
                        second_animal = [
                            second_animal[i : i + 100]
                            for i in range(0, len(second_animal), 100)
                        ]
                        results = Parallel(n_jobs=5, verbose=1, backend="threading")(
                            delayed(self._find_points_in_roi)(i, j)
                            for i, j in zip(first_animal_anchored_rois, second_animal)
                        )
                        self.results = [i for s in results for i in s]
                        keypoints_df_lst.append(
                            self._sort_keypoint_results(
                                first_animal_name=first_animal,
                                second_animal_name=second_animal_name,
                            )
                        )
                self.keypoint_dfs[video_name] = pd.concat(keypoints_df_lst, axis=1)

    def save_results(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        out_df = None
        for video_cnt, (video_name, video_data) in enumerate(self.polygons.items()):
            save_path = os.path.join(
                self.save_folder, f"{video_name}.{self.save_format.lower()}"
            )
            if (self.roi_intersections) and (self.roi_keypoint_intersections):
                out_df = pd.concat(
                    [self.intersection_dfs[video_name], self.keypoint_dfs[video_name]],
                    axis=1,
                )
            elif self.roi_intersections:
                out_df = self.intersection_dfs[video_name]
            elif self.roi_keypoint_intersections:
                out_df = self.keypoint_dfs[video_name]
            write_df(df=out_df, file_type=self.save_format.lower(), save_path=save_path)
            print(f"Data for video {video_name} saved...")

        stdout_success(
            msg=f"Data for {str(len(self.polygons.keys()))} videos saved in {self.save_folder}"
        )


# boundary_stats_calculator = BoundaryStatisticsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini',
#                                                          roi_intersections=True,
#                                                          roi_keypoint_intersections=True,
#                                                          save_format='PICKLE')
# boundary_stats_calculator.save_results()
