import os
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from numba import float32, float64, njit, prange

from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.mixins.circular_statistics import CircularStatisticsMixin
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


@njit(
    [
        (float32[:, :], float32[:, :], float64),
        (float64[:, :], float64[:, :], float64),
        (float64[:, :], float32[:, :], float64),
    ]
)
def calculate_weighted_avg(
    bp: np.ndarray, p: Union[np.ndarray, None], threshold: float
):
    results = np.full((bp.shape[0]), np.nan)
    n = bp.shape[0]
    for i in prange(n):
        if p is not None:
            p_thresh_idx = np.argwhere(p[i] > threshold).flatten()
            if p_thresh_idx.shape[0] > 0:
                p_vals = p[i][p_thresh_idx]
                bp_vals = bp[i][p_thresh_idx]
                weighted_sum = 0
                for x in range(p_vals.shape[0]):
                    weighted_sum += bp_vals[x] * p_vals[x]
                frm_result = weighted_sum / np.sum(p_vals)
            else:
                frm_result = np.mean(bp[i])
        else:
            frm_result = np.mean(bp[i])
        results[i] = frm_result
    return results


@njit([(float32[:, :], float32[:, :], float32[:, :], float64)])
def polygon_fill(
    x_bps: np.ndarray, y_bps: np.ndarray, p_bps: np.ndarray, threshold: float
):
    results = np.full((x_bps.shape[0], x_bps[0].shape[0], 2), np.nan)
    for frm_cnt in prange(x_bps.shape[0]):
        frm_bp_cnt = 0
        for bp_cnt in range(p_bps[frm_cnt].shape[0]):
            if p_bps[frm_cnt][bp_cnt] > threshold:
                results[frm_cnt][bp_cnt] = np.array(
                    [x_bps[frm_cnt, bp_cnt], y_bps[frm_cnt, bp_cnt]]
                )
                frm_bp_cnt += 1
        if frm_bp_cnt < 3:
            results[frm_cnt] = np.zeros((x_bps[0].shape[0], 2))
        elif np.sum(np.isnan(results[frm_cnt])) > 0:
            fill_idx = np.argwhere(~np.isnan(results[frm_cnt]))[0]
            fill_val = results[frm_cnt][fill_idx[0]]
            results[frm_cnt][np.isnan(results[frm_cnt][:, 0])] = fill_val[0]
            results[frm_cnt][np.isnan(results[frm_cnt][:, 1])] = fill_val[1]

    return results


def get_circle_fit_angle(x: np.ndarray, y: np.ndarray, p: np.ndarray, threshold: float):
    combined_arr = np.stack((x, y), axis=2).astype(np.float64)
    circles = CircularStatisticsMixin.fit_circle(data=combined_arr)
    diff_x_last = x[:, -1] - circles[:, 0]
    diff_y_last = y[:, -1] - circles[:, 1]
    diff_x_first = x[:, 0] - circles[:, 0]
    diff_y_first = y[:, 0] - circles[:, 1]
    angles = np.degrees(
        np.arctan2(diff_y_last, diff_x_last) - np.arctan2(diff_y_first, diff_x_first)
    )
    angles = angles + 360 * (angles < 0)

    below_thresh_idx = np.argwhere(np.average(p, axis=1) < threshold)
    angles[below_thresh_idx] = 0.0

    return angles


class AmberFeatureExtractor(ConfigReader, FeatureExtractionMixin):
    """
    Class for extracting features for the AMBER pipeline.

    The AMBER pipeline was developed for quantifying home-cage maternal and mother-pup interactions from side-view recordings.
    Downstream behavior classifiers can be trained to assess maternal nest attendance, nursing, pup-directed licking and grooming, self-directed grooming, eating, and drinking.

    .. note::
       `AMBER publication <https://www.nature.com/articles/s41598-023-45495-4>`_.
       `AMBER GitHub Repository <https://github.com/lapphe/AMBER-pipeline>`_.
       `AMBER OSF Repository <https://osf.io/e3dyc/>`_.

       For more info, contact Hannah E. Lapp `Hannah.Lapp@austin.utexas.edu <Hannah.Lapp@austin.utexas.edu/>`_.

    :example:
    >>> AmberFeatureExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

    References
    ----------
    .. [1] Lapp H et al., Automated maternal behavior during early life in rodents (AMBER) pipeline, Sci. Rep.,
           2023.
    """

    def __init__(self, config_path: str):
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        self.timer = SimbaTimer(start=True)
        print(f"Extracting features from {len(self.files_found)} file(s)...")
        self.pup_threshold = 0.4
        self.dam_threshold = 0.4
        self.roll_windows_values = [
            1,
            2,
            5,
            8,
            0.5,
        ]  # values used to calculate rolling average across frames

    def run(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            _, file_name, _ = get_fn_ext(file_path)
            print(
                f"Extracting features for video {file_name} ({file_cnt + 1}/{len(self.files_found)})..."
            )
            video_settings, self.px_per_mm, fps = self.read_video_info(
                video_name=file_name
            )
            data_df = read_df(file_path=file_path, file_type=self.file_type)

            body_part_names = [
                x.replace("_x", "") for x in list(data_df.columns) if "_x" in x
            ]
            pup_bp_names = [x for x in body_part_names if "pup" in x]
            dam_bp_names = [x for x in body_part_names if x not in pup_bp_names]
            p_col_names = [p + "_p" for p in body_part_names]

            print("Calculating dam points, centroids, and convex hulls...")

            data_df["arm_x"] = np.where(
                data_df["left_armpit_p"] > self.dam_threshold,
                data_df["left_armpit_x"],
                data_df["right_armpit_x"],
            )
            data_df["arm_y"] = np.where(
                data_df["left_armpit_p"] > self.dam_threshold,
                data_df["left_armpit_y"],
                data_df["right_armpit_y"],
            )
            data_df["arm_p"] = np.where(
                data_df["left_armpit_p"] > self.dam_threshold,
                data_df["left_armpit_p"],
                data_df["right_armpit_p"],
            )

            data_df["side_x"] = np.where(
                data_df["left_ventrum_side_p"] > self.dam_threshold,
                data_df["left_ventrum_side_x"],
                data_df["right_ventrum_side_x"],
            )
            data_df["side_y"] = np.where(
                data_df["left_ventrum_side_p"] > self.dam_threshold,
                data_df["left_ventrum_side_y"],
                data_df["right_ventrum_side_y"],
            )
            data_df["side_p"] = np.where(
                data_df["left_ventrum_side_p"] > self.dam_threshold,
                data_df["left_ventrum_side_p"],
                data_df["right_ventrum_side_p"],
            )

            dam_x_bps, dam_y_bps, dam_p_bps = (
                [x + "_x" for x in dam_bp_names],
                [x + "_y" for x in dam_bp_names],
                [x + "_p" for x in dam_bp_names],
            )
            data_df["dam_centroid_x"] = calculate_weighted_avg(
                bp=data_df[dam_x_bps].values,
                p=data_df[dam_p_bps].values,
                threshold=self.dam_threshold,
            )
            data_df["dam_centroid_y"] = calculate_weighted_avg(
                bp=data_df[dam_y_bps].values,
                p=data_df[dam_p_bps].values,
                threshold=self.dam_threshold,
            )

            dam_polygon_points = polygon_fill(
                x_bps=data_df[dam_x_bps].values,
                y_bps=data_df[dam_y_bps].values,
                p_bps=data_df[dam_p_bps].values,
                threshold=0.2,
            )
            data_df["dam_convex_hull"] = jitted_hull(
                points=dam_polygon_points.astype(np.float32), target="area"
            ) / (self.px_per_mm**2)

            dam_head_parts = [
                "dam_nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "top_head_dam",
            ]
            dam_head_x, dam_head_y, dam_head_p = (
                [x + "_x" for x in dam_head_parts],
                [x + "_y" for x in dam_head_parts],
                [x + "_p" for x in dam_head_parts],
            )

            data_df["head_centroid_x"] = calculate_weighted_avg(
                bp=data_df[dam_head_x].values,
                p=data_df[dam_head_p].values,
                threshold=self.dam_threshold,
            )
            data_df["head_centroid_y"] = calculate_weighted_avg(
                bp=data_df[dam_head_y].values,
                p=data_df[dam_head_p].values,
                threshold=self.dam_threshold,
            )

            head_polygon = polygon_fill(
                x_bps=data_df[dam_head_x].values,
                y_bps=data_df[dam_head_y].values,
                p_bps=data_df[dam_head_p].values,
                threshold=0.2,
            )

            data_df["head_convex_hull"] = jitted_hull(
                points=head_polygon.astype(np.float32), target="area"
            ) / (self.px_per_mm**2)

            print("Calculating pup centroids...")
            pup_x_bps, pup_y_bps, pup_p_bps = (
                [x + "_x" for x in pup_bp_names],
                [x + "_y" for x in pup_bp_names],
                [x + "_p" for x in pup_bp_names],
            )

            data_df["pups_centroid_x"] = calculate_weighted_avg(
                bp=data_df[pup_x_bps].values,
                p=data_df[pup_p_bps].values,
                threshold=self.pup_threshold,
            )
            data_df["pups_centroid_y"] = calculate_weighted_avg(
                bp=data_df[pup_y_bps].values,
                p=data_df[pup_p_bps].values,
                threshold=self.pup_threshold,
            )

            print("Calculating pup convex hull...")
            pups_polygon = polygon_fill(
                x_bps=data_df[pup_x_bps].values,
                y_bps=data_df[pup_y_bps].values,
                p_bps=data_df[pup_p_bps].values,
                threshold=0.2,
            )
            data_df["pups_convex_hull"] = jitted_hull(
                points=pups_polygon.astype(np.float32), target="area"
            ) / (self.px_per_mm**2)

            print("Calculating high probably body part counts...")
            data_df["pup_avg_p"] = data_df[pup_p_bps].mean(axis=1)

            data_df["high_p_pup_bp"] = FeatureExtractionMixin().count_values_in_range(
                data=data_df[pup_p_bps].fillna(0).values, ranges=np.array([[0.2, 1.0]])
            )
            data_df["high_p_dam_bp"] = FeatureExtractionMixin().count_values_in_range(
                data=data_df[dam_p_bps].fillna(0).values, ranges=np.array([[0.2, 1.0]])
            )

            data_df["high_p_pup_bp_no_zero"] = data_df["high_p_pup_bp"].replace(
                to_replace=0, value=1
            )

            data_df["pups_centroid_mult_x"] = (
                data_df["pups_centroid_x"] * data_df["high_p_pup_bp_no_zero"]
            )
            data_df["pups_centroid_mult_y"] = (
                data_df["pups_centroid_y"] * data_df["high_p_pup_bp_no_zero"]
            )

            roll_rows_30 = int(30 * 60 * fps)
            data_df["pups_centroid_x_roll_mean_30m"] = (
                data_df["pups_centroid_mult_x"]
                .rolling(roll_rows_30, min_periods=0, center=True)
                .sum()
                / data_df["high_p_pup_bp_no_zero"]
                .rolling(roll_rows_30, min_periods=0, center=True)
                .sum()
            )
            data_df["pups_centroid_y_roll_mean_30m"] = (
                data_df["pups_centroid_mult_y"]
                .rolling(roll_rows_30, min_periods=0, center=True)
                .sum()
                / data_df["high_p_pup_bp_no_zero"]
                .rolling(roll_rows_30, min_periods=0, center=True)
                .sum()
            )

            roll_rows_60 = int(60 * 60 * fps)
            data_df["pups_centroid_x_roll_mean_60m"] = (
                data_df["pups_centroid_mult_x"]
                .rolling(roll_rows_60, min_periods=0, center=True)
                .sum()
                / data_df["high_p_pup_bp_no_zero"]
                .rolling(roll_rows_60, min_periods=0, center=True)
                .sum()
            )
            data_df["pups_centroid_y_roll_mean_60m"] = (
                data_df["pups_centroid_mult_y"]
                .rolling(roll_rows_60, min_periods=0, center=True)
                .sum()
                / data_df["high_p_pup_bp_no_zero"]
                .rolling(roll_rows_60, min_periods=0, center=True)
                .sum()
            )

            print("Calculating movements...")

            movement_columns = deepcopy(dam_bp_names)
            data_df_shifted = FeatureExtractionMixin.create_shifted_df(
                df=data_df, periods=1
            )

            for col in movement_columns:
                bp_cols, s_bp_cols = [f"{col}_x", f"{col}_y"], [
                    f"{col}_x_shifted",
                    f"{col}_y_shifted",
                ]
                bp_arr, s_bp_arr = (
                    data_df_shifted[bp_cols].values,
                    data_df_shifted[s_bp_cols].values,
                )
                data_df[f"{col}_movement"] = (
                    FeatureExtractionMixin.framewise_euclidean_distance(
                        location_1=bp_arr, location_2=s_bp_arr, px_per_mm=self.px_per_mm
                    )
                )

            back_point_movements = [
                "back_2_movement",
                "back_3_movement",
                "back_4_movement",
                "back_5_movement",
                "back_6_movement",
                "back_7_movement",
                "back_8_movement",
                "back_9_movement",
                "back_10_movement",
            ]
            back_point_movements_p = [
                "back_2_p",
                "back_3_p",
                "back_4_p",
                "back_5_p",
                "back_6_p",
                "back_7_p",
                "back_8_p",
                "back_9_p",
                "back_10_p",
            ]

            data_df["back_avg_movement"] = calculate_weighted_avg(
                bp=data_df[back_point_movements].values,
                p=data_df[back_point_movements_p].values,
                threshold=self.dam_threshold,
            )

            head_point_movements = [
                "dam_nose_movement",
                "right_eye_movement",
                "left_eye_movement",
                "left_ear_movement",
                "right_ear_movement",
            ]
            head_point_movements_p = [
                "dam_nose_p",
                "right_eye_p",
                "left_eye_p",
                "left_ear_p",
                "right_ear_p",
            ]

            data_df["head_avg_movement"] = calculate_weighted_avg(
                bp=data_df[head_point_movements].values,
                p=data_df[head_point_movements_p].values,
                threshold=self.dam_threshold,
            )
            #
            data_df["head_max_movement"] = np.ma.max(
                [
                    data_df["dam_nose_movement"],
                    data_df["right_eye_movement"],
                    data_df["left_eye_movement"],
                    data_df["left_ear_movement"],
                    data_df["right_ear_movement"],
                ],
                axis=0,
            )

            data_df["ventrum_side_movement"] = calculate_weighted_avg(
                bp=data_df[
                    ["left_ventrum_side_movement", "right_ventrum_side_movement"]
                ].values,
                p=data_df[["left_ventrum_side_p", "right_ventrum_side_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["leg_front_movement"] = calculate_weighted_avg(
                bp=data_df[
                    ["left_leg_front_movement", "right_leg_front_movement"]
                ].values,
                p=data_df[["left_leg_front_p", "right_leg_front_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["leg_behind_movement"] = calculate_weighted_avg(
                bp=data_df[
                    ["left_leg_behind_movement", "right_leg_behind_movement"]
                ].values,
                p=data_df[["left_leg_behind_p", "right_leg_behind_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["wrist_movement"] = calculate_weighted_avg(
                bp=data_df[["left_wrist_movement", "right_wrist_movement"]].values,
                p=data_df[["left_wrist_p", "right_wrist_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["armpit_movement"] = calculate_weighted_avg(
                bp=data_df[["left_armpit_movement", "right_armpit_movement"]].values,
                p=data_df[["left_armpit_p", "right_armpit_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["shoulder_movement"] = calculate_weighted_avg(
                bp=data_df[
                    ["left_shoulder_movement", "right_shoulder_movement"]
                ].values,
                p=data_df[["left_shoulder_p", "right_shoulder_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["eye_movement"] = calculate_weighted_avg(
                bp=data_df[["left_eye_movement", "right_eye_movement"]].values,
                p=data_df[["left_eye_p", "right_eye_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["ear_movement"] = calculate_weighted_avg(
                bp=data_df[["left_ear_movement", "right_ear_movement"]].values,
                p=data_df[["left_ear_p", "right_ear_p"]].values,
                threshold=self.dam_threshold,
            )

            data_df["ankle_movement"] = calculate_weighted_avg(
                bp=data_df[["left_ankle_movement", "right_ankle_movement"]].values,
                p=data_df[["left_ankle_p", "right_ankle_p"]].values,
                threshold=self.dam_threshold,
            )

            print("Calculating distances...")

            data_df["dam_pup30m_distance"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["dam_centroid_x", "dam_centroid_y"]].values,
                    location_2=data_df[
                        [
                            "pups_centroid_x_roll_mean_30m",
                            "pups_centroid_y_roll_mean_30m",
                        ]
                    ].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["head_pup30m_distance"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["head_centroid_x", "head_centroid_y"]].values,
                    location_2=data_df[
                        [
                            "pups_centroid_x_roll_mean_30m",
                            "pups_centroid_y_roll_mean_30m",
                        ]
                    ].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["dam_pup60m_distance"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["dam_centroid_x", "dam_centroid_y"]].values,
                    location_2=data_df[
                        [
                            "pups_centroid_x_roll_mean_60m",
                            "pups_centroid_y_roll_mean_60m",
                        ]
                    ].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["head_pup60m_distance"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["head_centroid_x", "head_centroid_y"]].values,
                    location_2=data_df[
                        [
                            "pups_centroid_x_roll_mean_60m",
                            "pups_centroid_y_roll_mean_60m",
                        ]
                    ].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["dam_pup_distance"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["dam_centroid_x", "dam_centroid_y"]].values,
                    location_2=data_df[["pups_centroid_x", "pups_centroid_y"]].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["head_pup_distance"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["head_centroid_x", "head_centroid_y"]].values,
                    location_2=data_df[["pups_centroid_x", "pups_centroid_y"]].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["back_length"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["back_2_x", "back_2_y"]].values,
                    location_2=data_df[["back_10_x", "back_10_y"]].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["nose_back10_length"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["dam_nose_x", "dam_nose_y"]].values,
                    location_2=data_df[["back_10_x", "back_10_y"]].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["left_wrist_nose_length"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["left_wrist_x", "left_wrist_y"]].values,
                    location_2=data_df[["dam_nose_x", "dam_nose_y"]].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["right_wrist_nose_length"] = (
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=data_df[["right_wrist_x", "right_wrist_y"]].values,
                    location_2=data_df[["dam_nose_x", "dam_nose_y"]].values,
                    px_per_mm=self.px_per_mm,
                )
            )

            data_df["wrist_nose_length"] = calculate_weighted_avg(
                bp=data_df[
                    ["left_wrist_nose_length", "right_wrist_nose_length"]
                ].values,
                p=data_df[["left_wrist_p", "right_wrist_p"]].values,
                threshold=self.dam_threshold,
            )
            data_df.drop(
                inplace=True,
                columns=["left_wrist_nose_length", "right_wrist_nose_length"],
            )
            dam_p = [
                "dam_nose_p",
                "left_eye_p",
                "right_eye_p",
                "left_ear_p",
                "right_ear_p",
                "left_shoulder_p",
                "right_shoulder_p",
                "arm_p",
                "side_p",
            ]
            data_df["avg_dam_bp_p"] = np.mean(data_df[dam_p].values, axis=1)

            data_df["sum_probabilities"] = data_df[p_col_names].sum(axis=1)

            print("Calculating fields for dam back curve...")
            back_points_x = [
                "back_1_center_x",
                "back_2_x",
                "back_3_x",
                "back_4_x",
                "back_5_x",
                "back_6_x",
                "back_7_x",
                "back_8_x",
                "back_9_x",
                "back_10_x",
            ]
            back_points_y = [
                "back_1_center_y",
                "back_2_y",
                "back_3_y",
                "back_4_y",
                "back_5_y",
                "back_6_y",
                "back_7_y",
                "back_8_y",
                "back_9_y",
                "back_10_y",
            ]
            back_points_p = [
                "back_1_center_p",
                "back_2_p",
                "back_3_p",
                "back_4_p",
                "back_5_p",
                "back_6_p",
                "back_7_p",
                "back_8_p",
                "back_9_p",
                "back_10_p",
            ]

            print("Calculating dam angles...")

            data_df["dam_back_angle"] = get_circle_fit_angle(
                x=data_df[back_points_x].values,
                y=data_df[back_points_y].values,
                p=data_df[back_points_p].values,
                threshold=0.5,
            )

            head_angle_cords = [
                "back_10_x",
                "back_10_y",
                "back_1_center_x",
                "back_1_center_y",
                "top_head_dam_x",
                "top_head_dam_y",
            ]
            data_df["dam_head_angle"] = FeatureExtractionMixin.angle3pt_serialized(
                data=data_df[head_angle_cords].values
            )

            top_angle_cords = [
                "back_10_x",
                "back_10_y",
                "back_4_x",
                "back_4_y",
                "back_1_center_x",
                "back_1_center_y",
            ]
            data_df["dam_back_top_angle"] = FeatureExtractionMixin.angle3pt_serialized(
                data=data_df[top_angle_cords].values
            )

            head_angle_cords = [
                "back_2_x",
                "back_2_y",
                "back_1_center_x",
                "back_1_center_y",
                "top_head_dam_x",
                "top_head_dam_y",
            ]
            data_df["dam_head_angle2"] = FeatureExtractionMixin.angle3pt_serialized(
                data=data_df[head_angle_cords].values
            )

            print("Calculating rolling averages...")

            roll_windows = []
            for j in range(len(self.roll_windows_values)):
                roll_windows.append(int(fps / self.roll_windows_values[j]))

            data_df["head_avg_movement_roll_mean_1s"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )

            data_df["head_avg_movement_roll_mean_1ds"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["head_avg_movement_roll_mean_2s"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["back_avg_movement_roll_mean_1s"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["back_avg_movement_roll_mean_1ds"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["back_avg_movement_roll_mean_2s"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["head_back_rel_roll_mean_1s"] = data_df[
                "head_avg_movement_roll_mean_1s"
            ] / (
                data_df["head_avg_movement_roll_mean_1s"]
                + data_df["back_avg_movement_roll_mean_1s"]
            )
            data_df["head_back_rel_roll_mean_1ds"] = data_df[
                "head_avg_movement_roll_mean_1ds"
            ] / (
                data_df["head_avg_movement_roll_mean_1ds"]
                + data_df["back_avg_movement_roll_mean_1ds"]
            )
            data_df["head_back_rel_roll_mean_2s"] = data_df[
                "head_avg_movement_roll_mean_2s"
            ] / (
                data_df["head_avg_movement_roll_mean_2s"]
                + data_df["back_avg_movement_roll_mean_2s"]
            )

            data_df["pups_convex_hull_roll_mean_1s"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["pups_convex_hull_roll_mean_1ds"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["pups_convex_hull_roll_mean_2s"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_pup_distance_roll_mean_1s"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_pup_distance_roll_mean_1ds"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_pup_distance_roll_mean_2s"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_pup30m_distance_roll_sum_1s"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_pup30m_distance_roll_mean_1ds"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_pup30m_distance_roll_mean_2s"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_pup60m_distance_roll_mean_1s"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_pup60m_distance_roll_mean_1ds"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_pup60m_distance_roll_mean_2s"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_back_angle_roll_mean_1s"] = (
                data_df["dam_back_angle"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            # csv_df['dam_back_angle_roll_mean_1ds'] = csv_df['dam_back_angle'].rolling(roll_windows[2], min_periods=0, center=True).mean()
            data_df["dam_back_angle_roll_mean_2s"] = (
                data_df["dam_back_angle"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_head_angle_roll_mean_1s"] = (
                data_df["dam_head_angle"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_head_angle_roll_mean_1ds"] = (
                data_df["dam_head_angle"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_head_angle_roll_mean_2s"] = (
                data_df["dam_head_angle"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_head_angle2_roll_mean_1s"] = (
                data_df["dam_head_angle2"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_head_angle2_roll_mean_1ds"] = (
                data_df["dam_head_angle2"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_head_angle2_roll_mean_2s"] = (
                data_df["dam_head_angle2"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_back_top_angle_roll_mean_1s"] = (
                data_df["dam_back_top_angle"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            # csv_df['dam_back_top_angle_roll_mean_1ds'] = csv_df['dam_back_top_angle'].rolling(roll_windows[2], min_periods=0, center=True).mean()
            data_df["dam_back_top_angle_roll_mean_2s"] = (
                data_df["dam_back_top_angle"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_nose_movement_roll_mean_1s"] = (
                data_df["dam_nose_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_nose_movement_roll_mean_1ds"] = (
                data_df["dam_nose_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_nose_movement_roll_mean_2s"] = (
                data_df["dam_nose_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["eye_movement_roll_mean_1s"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["eye_movement_roll_mean_1ds"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["eye_movement_roll_mean_2s"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["ear_movement_roll_mean_1s"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["ear_movement_roll_mean_1ds"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["ear_movement_roll_mean_2s"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["top_head_dam_movement_roll_mean_1s"] = (
                data_df["top_head_dam_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["top_head_dam_movement_roll_mean_1ds"] = (
                data_df["top_head_dam_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["top_head_dam_movement_roll_mean_2s"] = (
                data_df["top_head_dam_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            # data_df['ventrum_side_movement_roll_mean_1s'] = data_df['ventrum_side_movement'].rolling(roll_windows[0], min_periods=0, center=True).mean()
            # data_df['ventrum_side_movement_roll_mean_1ds'] = data_df['ventrum_side_movement'].rolling(roll_windows[2], min_periods=0, center=True).mean()
            # data_df['ventrum_side_movement_roll_mean_2s'] = data_df['ventrum_side_movement'].rolling(roll_windows[4], min_periods=0, center=True).mean()

            # data_df['leg_front_movement_roll_mean_1s'] = data_df['leg_front_movement'].rolling(roll_windows[0], min_periods=0, center=True).mean()
            # data_df['leg_front_movement_roll_mean_1ds'] = data_df['leg_front_movement'].rolling(roll_windows[2], min_periods=0, center=True).mean()
            # data_df['leg_front_movement_roll_mean_2s'] = data_df['leg_front_movement'].rolling(roll_windows[4], min_periods=0, center=True).mean()

            # data_df['leg_behind_movement_roll_mean_1s'] = data_df['leg_behind_movement'].rolling(roll_windows[0], min_periods=0, center=True).mean()
            # data_df['leg_behind_movement_roll_mean_1ds'] = data_df['leg_behind_movement'].rolling(roll_windows[2], min_periods=0, center=True).mean()
            # data_df['leg_behind_movement_roll_mean_2s'] = data_df['leg_behind_movement'].rolling(roll_windows[4], min_periods=0, center=True).mean()

            data_df["wrist_movement_roll_mean_1s"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["wrist_movement_roll_mean_1ds"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["wrist_movement_roll_mean_2s"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["armpit_movement_roll_mean_1s"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["armpit_movement_roll_mean_1ds"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["armpit_movement_roll_mean_2s"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["dam_convex_hull_roll_mean_1s"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_convex_hull_roll_mean_1ds"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["dam_convex_hull_roll_mean_2s"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["head_convex_hull_roll_mean_1s"] = (
                data_df["head_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["head_convex_hull_roll_mean_1ds"] = (
                data_df["head_convex_hull"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["head_convex_hull_roll_mean_2s"] = (
                data_df["head_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["high_p_pup_bp_roll_mean_1s"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["high_p_pup_bp_roll_mean_1ds"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["high_p_pup_bp_roll_mean_2s"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["high_p_dam_bp_roll_mean_1s"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["high_p_dam_bp_roll_mean_1ds"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["high_p_dam_bp_roll_mean_2s"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            data_df["nose_back10_length_roll_mean_1s"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .mean()
            )
            data_df["nose_back10_length_roll_mean_1ds"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .mean()
            )
            data_df["nose_back10_length_roll_mean_2s"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .mean()
            )

            print("Calculating rolling window sums ...")

            data_df["head_avg_movement_roll_sum_1s"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["head_avg_movement_roll_sum_1ds"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["head_avg_movement_roll_sum_2s"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["head_max_movement_roll_sum_1s"] = (
                data_df["head_max_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["head_max_movement_roll_sum_1ds"] = (
                data_df["head_max_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["head_max_movement_roll_sum_2s"] = (
                data_df["head_max_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["back_avg_movement_roll_sum_1s"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["back_avg_movement_roll_sum_1ds"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["back_avg_movement_roll_sum_2s"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["head_back_rel_roll_sum_1s"] = data_df[
                "head_avg_movement_roll_sum_1s"
            ] / (
                data_df["head_avg_movement_roll_sum_1s"]
                + data_df["back_avg_movement_roll_sum_1s"]
            )
            data_df["head_back_rel_roll_sum_1ds"] = data_df[
                "head_avg_movement_roll_sum_1ds"
            ] / (
                data_df["head_avg_movement_roll_sum_1ds"]
                + data_df["back_avg_movement_roll_sum_1ds"]
            )
            data_df["head_back_rel_roll_sum_2s"] = data_df[
                "head_avg_movement_roll_sum_2s"
            ] / (
                data_df["head_avg_movement_roll_sum_2s"]
                + data_df["back_avg_movement_roll_sum_2s"]
            )

            data_df["pups_convex_hull_roll_sum_1s"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["pups_convex_hull_roll_sum_1ds"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["pups_convex_hull_roll_sum_2s"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["dam_pup_distance_roll_sum_1s"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_pup_distance_roll_sum_1ds"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_pup_distance_roll_sum_2s"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["dam_pup30m_distance_roll_sum_1s"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_pup30m_distance_roll_sum_1ds"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_pup30m_distance_roll_sum_2s"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["dam_pup60m_distance_roll_sum_1s"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_pup60m_distance_roll_sum_1ds"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_pup60m_distance_roll_sum_2s"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["top_head_dam_movement_roll_sum_1s"] = (
                data_df["top_head_dam_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["top_head_dam_movement_roll_sum_1ds"] = (
                data_df["top_head_dam_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["top_head_dam_movement_roll_sum_2s"] = (
                data_df["top_head_dam_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["dam_back_angle_roll_sum_1s"] = (
                data_df["dam_back_angle"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_back_angle_roll_sum_1ds"] = (
                data_df["dam_back_angle"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_back_angle_roll_sum_2s"] = (
                data_df["dam_back_angle"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["dam_nose_movement_roll_sum_1s"] = (
                data_df["dam_nose_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_nose_movement_roll_sum_1ds"] = (
                data_df["dam_nose_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_nose_movement_roll_sum_2s"] = (
                data_df["dam_nose_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["eye_movement_roll_sum_1s"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["eye_movement_roll_sum_1ds"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["eye_movement_roll_sum_2s"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["ear_movement_roll_sum_1s"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["ear_movement_roll_sum_1ds"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["ear_movement_roll_sum_2s"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            # csv_df['ventrum_side_movement_roll_sum_1s'] = csv_df['ventrum_side_movement'].rolling(roll_windows[0], min_periods=0, center=True).sum()
            # csv_df['ventrum_side_movement_roll_sum_1ds'] = csv_df['ventrum_side_movement'].rolling(roll_windows[2], min_periods=0, center=True).sum()
            # csv_df['ventrum_side_movement_roll_sum_2s'] = csv_df['ventrum_side_movement'].rolling(roll_windows[4], min_periods=0, center=True).sum()

            # csv_df['leg_front_movement_roll_sum_1s'] = csv_df['leg_front_movement'].rolling(roll_windows[0], min_periods=0, center=True).sum()
            # csv_df['leg_front_movement_roll_sum_1ds'] = csv_df['leg_front_movement'].rolling(roll_windows[2], min_periods=0, center=True).sum()
            # csv_df['leg_front_movement_roll_sum_2s'] = csv_df['leg_front_movement'].rolling(roll_windows[4], min_periods=0, center=True).sum()

            # csv_df['leg_behind_movement_roll_sum_1s'] = csv_df['leg_behind_movement'].rolling(roll_windows[0], min_periods=0, center=True).sum()
            # csv_df['leg_behind_movement_roll_sum_1ds'] = csv_df['leg_behind_movement'].rolling(roll_windows[2], min_periods=0, center=True).sum()
            # csv_df['leg_behind_movement_roll_sum_2s'] = csv_df['leg_behind_movement'].rolling(roll_windows[4], min_periods=0, center=True).sum()

            data_df["wrist_movement_roll_sum_1s"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["wrist_movement_roll_sum_1ds"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["wrist_movement_roll_sum_2s"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["armpit_movement_roll_sum_1s"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["armpit_movement_roll_sum_1ds"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["armpit_movement_roll_sum_2s"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["dam_convex_hull_roll_sum_1s"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_convex_hull_roll_sum_1ds"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["dam_convex_hull_roll_sum_2s"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            # csv_df['head_convex_hull_roll_sum_1s'] = csv_df['head_convex_hull'].rolling(roll_windows[0], min_periods=0, center=True).sum()
            # csv_df['head_convex_hull_roll_sum_1ds'] = csv_df['head_convex_hull'].rolling(roll_windows[2], min_periods=0, center=True).sum()
            # csv_df['head_convex_hull_roll_sum_2s'] = csv_df['head_convex_hull'].rolling(roll_windows[4], min_periods=0, center=True).sum()

            data_df["high_p_pup_bp_roll_sum_1s"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["high_p_pup_bp_roll_sum_1ds"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["high_p_pup_bp_roll_sum_2s"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["high_p_dam_bp_roll_sum_1s"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["high_p_dam_bp_roll_sum_1ds"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["high_p_dam_bp_roll_sum_2s"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            data_df["nose_back10_length_roll_sum_1s"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .sum()
            )
            data_df["nose_back10_length_roll_sum_1ds"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[2], min_periods=0, center=True)
                .sum()
            )
            data_df["nose_back10_length_roll_sum_2s"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .sum()
            )

            print("Calculating rolling window standard deviations...")
            data_df["head_avg_movement_roll_std_1s"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[0], min_periods=0)
                .std()
            )
            data_df["head_avg_movement_roll_std_2s"] = (
                data_df["head_avg_movement"]
                .rolling(roll_windows[4], min_periods=0)
                .std()
            )

            data_df["back_avg_movement_roll_std_1s"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[0], min_periods=0)
                .std()
            )
            data_df["back_avg_movement_roll_std_2s"] = (
                data_df["back_avg_movement"]
                .rolling(roll_windows[4], min_periods=0)
                .std()
            )

            data_df["pups_convex_hull_roll_std_1s"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["pups_convex_hull_roll_std_2s"] = (
                data_df["pups_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["dam_pup_distance_roll_std_1s"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["dam_pup_distance_roll_std_2s"] = (
                data_df["dam_pup_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["dam_pup30m_distance_roll_std_1s"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["dam_pup30m_distance_roll_std_2s"] = (
                data_df["dam_pup30m_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["dam_pup60m_distance_roll_std_1s"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["dam_pup60m_distance_roll_std_2s"] = (
                data_df["dam_pup60m_distance"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["eye_movement_roll_std_1s"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["eye_movement_roll_std_2s"] = (
                data_df["eye_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["ear_movement_roll_std_1s"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["ear_movement_roll_std_2s"] = (
                data_df["ear_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["wrist_movement_roll_std_1s"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["wrist_movement_roll_std_2s"] = (
                data_df["wrist_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["armpit_movement_roll_std_1s"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["armpit_movement_roll_std_2s"] = (
                data_df["armpit_movement"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["dam_convex_hull_roll_std_1s"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["dam_convex_hull_roll_std_2s"] = (
                data_df["dam_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["head_convex_hull_roll_std_1s"] = (
                data_df["head_convex_hull"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["head_convex_hull_roll_std_2s"] = (
                data_df["head_convex_hull"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["high_p_pup_bp_roll_std_1s"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["high_p_pup_bp_roll_std_2s"] = (
                data_df["high_p_pup_bp"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["high_p_dam_bp_roll_std_1s"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["high_p_dam_bp_roll_std_2s"] = (
                data_df["high_p_dam_bp"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["nose_back10_length_roll_std_1s"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["nose_back10_length_roll_std_2s"] = (
                data_df["nose_back10_length"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["dam_head_angle_roll_std_1s"] = (
                data_df["dam_head_angle"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["dam_head_angle_roll_std_2s"] = (
                data_df["dam_head_angle"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["dam_head_angle2_roll_std_1s"] = (
                data_df["dam_head_angle2"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["dam_head_angle2_roll_std_2s"] = (
                data_df["dam_head_angle2"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            data_df["dam_back_top_angle_roll_std_1s"] = (
                data_df["dam_back_top_angle"]
                .rolling(roll_windows[0], min_periods=0, center=True)
                .std()
            )
            data_df["dam_back_top_angle_roll_std_2s"] = (
                data_df["dam_back_top_angle"]
                .rolling(roll_windows[4], min_periods=0, center=True)
                .std()
            )

            drop_cols = [
                "side_x",
                "arm_x",
                "head_centroid_x",
                "dam_centroid_x",
                "left_wrist_movement",
                "right_wrist_movement",
                "left_eye_movement",
                "right_eye_movement",
                "left_armpit_movement",
                "right_armpit_movement",
                "left_palm_movement",
                "right_palm_movement",
                "left_ear_movement",
                "right_ear_movement",
                "left_ankle_movement",
                "right_ankle_movement",
                "left_leg_behind_movement",
                "right_leg_behind_movement",
                "left_leg_front_movement",
                "right_leg_front_movement",
                "pups_centroid_x_roll_mean_30m",
                "pups_centroid_x_roll_mean_60m",
                "pups_centroid_y_roll_mean_30m",
                "pups_centroid_y_roll_mean_60m",
                "pups_centroid_mult_x",
                "pups_centroid_mult_y",
                "pups_centroid_x",
                "pups_centroid_y",
                "high_p_pup_bp_no_zero",
            ]

            data_df = data_df.drop(drop_cols, axis=1)

            print(f"Correcting coordinate units with {self.px_per_mm} pixels per mm...")
            correction_bps = ["dam_centroid", "head_centroid"]
            correction_columns = [bp + "_y" for bp in correction_bps]

            for col in correction_columns:
                data_df[col] = data_df[col] / self.px_per_mm

            data_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            data_df = data_df.fillna(0).apply(pd.to_numeric)

            print(f"Saving features for video {file_name}...")
            save_path = os.path.join(
                self.features_dir, file_name + "." + self.file_type
            )
            write_df(data_df, self.file_type, save_path)
            video_timer.stop_timer()
            print(
                f"Feature extraction complete for video {file_name} (elapsed time: {video_timer.elapsed_time_str}s)"
            )

        self.timer.stop_timer()
        stdout_success(
            f"Feature extraction complete for {str(len(self.files_found))} video(s). Results are saved inside the project_folder/csv/features_extracted directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


#
# extractor = AmberFeatureExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# extractor.run()
