__author__ = "Simon Nilsson"

import itertools
import os
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd

import simba
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_df_field_is_boolean,
                                check_if_dir_exists, check_instance, check_int,
                                check_str, check_that_column_exist)
from simba.utils.enums import Paths
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.warnings import ShapWarning

SIMBA_DIR = os.path.dirname(simba.__file__)


class ShapAggregateStatisticsVisualizer(ConfigReader):
    """
    Calculate aggregate (binned) SHAP value statistics where individual bins represent reaulated features.
    and create line chart visualizations reprsenting aggregations of behavior-present SHAP values.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/SHAP.md#step-3-interpreting-the-shap-value-ouput-generated-by-simba>`__.
       `Example output <https://github.com/sgoldenlab/simba/blob/master/images/example_shap_graph.png>`__.

    .. image:: _static/img/shap.png
       :width: 600
       :align: center


    :parameter str config_path: Path to SimBA project config file in Configparser format
    :param str classifier_name: Name of classifier (e.g., Attack).
    :param pd.DataFrame shap_df: Dataframe with non-aggregated SHAP values where rows represent frames and columns represent features.
    :param float shap_baseline_value: SHAP expected value (computed by ``simba.train_model_functions.create_shap_log``).
    :param str save_path: Directory where to store the results

    :example:
    >>> _ = ShapAggregateStatisticsVisualizer(config_path='SimBAConfigFilePath', classifier_name='Attack', shap_df='tests/test_data/test_shap/data/test_shap.csv', shap_baseline_value=4, save_path='SaveDirectory')
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 shap_df: pd.DataFrame,
                 classifier_name: str,
                 shap_baseline_value: int,
                 visualization: Optional[bool] = True,
                 save_path: Optional[Union[str, os.PathLike]] = None):

        check_file_exist_and_readable(file_path=config_path)
        check_instance(
            source=f"{self.__class__.__name__} shap_df",
            instance=shap_df,
            accepted_types=(pd.DataFrame,),
        )
        check_str(
            name=f"{self.__class__.__name__} classifier_name", value=classifier_name
        )
        check_that_column_exist(
            df=shap_df, column_name=classifier_name, file_name="shap dataframe"
        )
        check_if_df_field_is_boolean(df=shap_df, field=classifier_name)
        check_int(
            name=f"{self.__class__.__name__} shap_baseline_value",
            value=shap_baseline_value,
            max_value=100,
            min_value=0,
        )

        ConfigReader.__init__(self, config_path=config_path)
        if (self.pose_setting != "14") and (self.pose_setting != "16"):
            ShapWarning(msg="SHAP visualizations/aggregate stats skipped (only viable for projects with two animals and default 7 or 8 body-parts per animal) ...", source=self.__class__.__name__)
        else:
            self.classifier_name, self.shap_df, self.shap_baseline_value = (
                classifier_name,
                shap_df,
                shap_baseline_value,
            )
            if not os.path.exists(self.shap_logs_path):
                os.makedirs(self.shap_logs_path)
            self.img_save_path = os.path.join(
                self.shap_logs_path,
                f"SHAP_summary_line_graph_{self.classifier_name}_{self.datetime}.png",
            )
            feature_categories_csv_path = os.path.join(
                SIMBA_DIR, Paths.SIMBA_SHAP_CATEGORIES_PATH.value
            )
            check_file_exist_and_readable(file_path=feature_categories_csv_path)
            self.feature_categories_df = pd.read_csv(
                feature_categories_csv_path, header=[0, 1]
            )
            self.unique_feature_category_names, self.unique_time_bin_names = set(
                list(self.feature_categories_df.columns.levels[0])
            ), set(list(self.feature_categories_df.columns.levels[1]))
            self.__run()
            if visualization:
                self.__create_base_shap_img()
                self.__insert_data_in_img()

    def __save_aggregate_scores(self):
        """
        Private helper to convert results in dict format to CSV file.
        """
        results = pd.DataFrame(
            columns=self.results[list(self.results.keys())[0]],
            index=self.results.keys(),
        )
        for row_name, time_bins in self.results.items():
            for column_name, value in time_bins.items():
                results.loc[row_name, column_name] = value
        results.reindex(sorted(results.columns, reverse=True), axis=1).to_csv(
            self.df_save_path
        )

    def __run(self):
        self.agg_stats_timer = SimbaTimer(start=True)
        for clf_state, clf_state_name in zip(range(2), ["ABSENT", "PRESENT"]):
            self.results = {}
            self.df_save_path = os.path.join(
                self.shap_logs_path,
                f"SHAP_summary_{self.classifier_name}_{clf_state_name}_{self.datetime}.csv",
            )
            shap_clf_sliced = self.shap_df[
                self.shap_df[self.classifier_name] == clf_state
            ]

            for feature_category, feature_time_bin in itertools.product(
                self.unique_feature_category_names, self.unique_time_bin_names
            ):
                if feature_category not in self.results.keys():
                    self.results[feature_category] = {}
                feature_names_sliced = list(
                    self.feature_categories_df.loc[
                        :, (feature_category, feature_time_bin)
                    ]
                )
                feature_names_sliced = [
                    x
                    for x in feature_names_sliced
                    if str(x) != "nan" and x in shap_clf_sliced
                ]
                self.results[feature_category][feature_time_bin] = round(
                    shap_clf_sliced[feature_names_sliced].sum(axis=1).mean() * 100, 6
                )
            self.__save_aggregate_scores()
        self.agg_stats_timer.stop_timer()
        self.visualization_timer = SimbaTimer(start=True)
        stdout_success(
            msg=f"Aggregate SHAP statistics saved in {self.shap_logs_path} directory",
            elapsed_time=self.agg_stats_timer.elapsed_time_str,
        )

    def __create_base_shap_img(self):
        """
        Private helper to create the base (axes, icons ticks etc.) of the aggregate shap value visualization.
        """
        shap_img_path = os.path.join(SIMBA_DIR, Paths.SIMBA_SHAP_IMG_PATH.value)
        check_if_dir_exists(in_dir=shap_img_path)
        self.scale_img_dict = {
            "baseline_scale": os.path.join(shap_img_path, "baseline_scale.jpg"),
            "small_arrow": os.path.join(shap_img_path, "down_arrow.jpg"),
            "side_scale": os.path.join(shap_img_path, "side_scale.jpg"),
            "color_bar": os.path.join(shap_img_path, "color_bar.jpg"),
        }
        self.category_img_dict = {
            "Animal distances": {
                "icon": os.path.join(shap_img_path, "animal_distances.jpg")
            },
            "Intruder movement": {
                "icon": os.path.join(shap_img_path, "intruder_movement.jpg")
            },
            "Resident+intruder movement": {
                "icon": os.path.join(shap_img_path, "resident_intruder_movement.jpg")
            },
            "Resident movement": {
                "icon": os.path.join(shap_img_path, "resident_movement.jpg")
            },
            "Intruder shape": {
                "icon": os.path.join(shap_img_path, "intruder_shape.jpg")
            },
            "Resident+intruder shape": {
                "icon": os.path.join(shap_img_path, "resident_intruder_shape.jpg")
            },
            "Resident shape": {
                "icon": os.path.join(shap_img_path, "resident_shape.jpg")
            },
        }
        for k, v in self.scale_img_dict.items():
            check_file_exist_and_readable(file_path=v)
        for k, v in self.category_img_dict.items():
            check_file_exist_and_readable(file_path=v["icon"])
        self.positive_arrow_colors = [
            (253, 141, 60),
            (252, 78, 42),
            (227, 26, 28),
            (189, 0, 38),
            (128, 0, 38),
        ]
        self.negative_arrow_colors = [
            (65, 182, 196),
            (29, 145, 192),
            (34, 94, 168),
            (37, 52, 148),
            (8, 29, 88),
        ]
        self.ranges_lst = [
            list(range(0, 20)),
            list(range(20, 40)),
            list(range(40, 60)),
            list(range(60, 80)),
            list(range(80, 101)),
        ]
        self.img = 255 * np.ones([1680, 1680, 3], dtype=np.uint8)
        self.baseline_scale_img = cv2.imread(self.scale_img_dict["baseline_scale"])
        self.baseline_scale_top_left = (100, 800)
        baseline_scale_bottom_right = (
            self.baseline_scale_top_left[0] + self.baseline_scale_img.shape[0],
            self.baseline_scale_top_left[1] + self.baseline_scale_img.shape[1],
        )
        baseline_scale_middle = ((int(700 + self.baseline_scale_img.shape[1] / 2)), 85)
        self.img[
            self.baseline_scale_top_left[0] : baseline_scale_bottom_right[0],
            self.baseline_scale_top_left[1] : baseline_scale_bottom_right[1],
        ] = self.baseline_scale_img
        cv2.putText(
            self.img,
            "BASELINE SHAP",
            baseline_scale_middle,
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        self.small_arrow_img = cv2.imread(self.scale_img_dict["small_arrow"])
        small_arrow_top_left = (
            baseline_scale_bottom_right[0],
            int(
                self.baseline_scale_top_left[1]
                + (self.baseline_scale_img.shape[1] / 100) * (self.shap_baseline_value)
            ),
        )
        small_arrow_bottom_right = (
            small_arrow_top_left[0] + self.small_arrow_img.shape[0],
            small_arrow_top_left[1] + self.small_arrow_img.shape[1],
        )
        self.img[
            small_arrow_top_left[0] : small_arrow_bottom_right[0],
            small_arrow_top_left[1] : small_arrow_bottom_right[1],
        ] = self.small_arrow_img

        self.color_bar_img = cv2.imread(self.scale_img_dict["color_bar"])

        side_scale_img = cv2.imread(self.scale_img_dict["side_scale"])
        side_scale_top_left = (
            small_arrow_bottom_right[0] + 50,
            self.baseline_scale_top_left[1] - 50,
        )
        self.side_scale_y_tick_cords = [
            (side_scale_top_left[0], side_scale_top_left[1] - 75)
        ]
        for i in range(1, 7):
            self.side_scale_y_tick_cords.append(
                (
                    int(side_scale_top_left[0] + (side_scale_img.shape[0] / 4) * i),
                    int(side_scale_top_left[1] - 75),
                )
            )
        self.arrow_start = (
            int(small_arrow_top_left[1] + (self.small_arrow_img.shape[1] / 2)),
            side_scale_top_left[0],
        )

        for img_cnt, (img_name, img_data) in enumerate(self.category_img_dict.items()):
            icon_img = cv2.resize(
                cv2.imread(img_data["icon"]),
                None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_CUBIC,
            )
            icon_top_left = (
                self.side_scale_y_tick_cords[img_cnt][0] - int(icon_img.shape[0] / 2),
                self.side_scale_y_tick_cords[img_cnt][1] - 100,
            )
            icon_bottom_right = (
                icon_top_left[0] + icon_img.shape[0],
                self.side_scale_y_tick_cords[img_cnt][1] + icon_img.shape[1] - 100,
            )
            text_location = (
                int(
                    icon_bottom_right[0]
                    - (icon_bottom_right[0] - icon_top_left[0])
                    + 100
                ),
                int(icon_bottom_right[1] - (icon_bottom_right[1] - icon_top_left[1]))
                - 380,
            )
            cv2.putText(
                self.img,
                str(img_name),
                (text_location[1], text_location[0]),
                cv2.FONT_HERSHEY_COMPLEX,
                0.75,
                (0, 0, 0),
                1,
            )
            self.img[
                icon_top_left[0] : icon_bottom_right[0],
                icon_top_left[1] : icon_bottom_right[1],
            ] = icon_img

    def __insert_data_in_img(self):
        """
        Private helper to insert the data (i.e., colored arrows, text etc.) into the aggregate shap value visualization
        and save the results.
        """
        data_df = pd.read_csv(
            os.path.join(
                self.shap_logs_path,
                f"SHAP_summary_{self.classifier_name}_PRESENT_{self.datetime}.csv",
            ),
            index_col=0,
        )
        for feature_category in self.unique_feature_category_names:
            self.category_img_dict[feature_category]["value"] = int(
                data_df.loc[feature_category, :].sum()
            )

        for row_cnt, (feature_category_name, feature_data) in enumerate(
            self.category_img_dict.items()
        ):
            arrow_width = int(
                (self.baseline_scale_img.shape[1] / 100) * abs(feature_data["value"])
            )
            if feature_data["value"] > 0:
                arrow_end = (self.arrow_start[0] + arrow_width, self.arrow_start[1])
                arrow_middle = int(
                    ((arrow_end[1] - self.arrow_start[1]) / 2) + self.arrow_start[1] - 7
                )
                for bracket_no, bracket in enumerate(self.ranges_lst):
                    if abs(feature_data["value"]) in bracket:
                        color = (
                            self.positive_arrow_colors[bracket_no][2],
                            self.positive_arrow_colors[bracket_no][1],
                            self.positive_arrow_colors[bracket_no][0],
                        )
                cv2.arrowedLine(
                    self.img, self.arrow_start, arrow_end, color, 5, tipLength=0.1
                )
                cv2.putText(
                    self.img,
                    "+" + str(abs(feature_data["value"])) + "%",
                    (arrow_end[0] - 7, arrow_middle - 15),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    color,
                    2,
                )

            else:
                arrow_end = (self.arrow_start[0] - arrow_width, self.arrow_start[1])
                arrow_middle = int(
                    ((self.arrow_start[1] - arrow_end[1]) / 2) + arrow_end[1] - 7
                )
                for bracket_no, bracket in enumerate(self.ranges_lst):
                    if abs(feature_data["value"]) in bracket:
                        color = (
                            self.negative_arrow_colors[bracket_no][2],
                            self.negative_arrow_colors[bracket_no][1],
                            self.negative_arrow_colors[bracket_no][0],
                        )
                cv2.arrowedLine(
                    self.img, self.arrow_start, arrow_end, color, 5, tipLength=0.1
                )
                cv2.putText(
                    self.img,
                    "-" + str(abs(feature_data["value"])) + "%",
                    (arrow_end[0] - 7, arrow_middle - 15),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    color,
                    2,
                )

            if row_cnt != (len(list(self.category_img_dict.keys())) - 1):
                self.arrow_start = (
                    arrow_end[0],
                    self.side_scale_y_tick_cords[row_cnt + 1][0],
                )

        small_arrow_top_left = (
            int(arrow_end[1]) + 20,
            int(arrow_end[0] - self.small_arrow_img.shape[1] / 2),
        )
        small_arrow_bottom_right = (
            small_arrow_top_left[0] + self.small_arrow_img.shape[0],
            small_arrow_top_left[1] + self.small_arrow_img.shape[1],
        )
        self.img[
            small_arrow_top_left[0] : small_arrow_bottom_right[0],
            small_arrow_top_left[1] : small_arrow_bottom_right[1],
        ] = self.small_arrow_img
        color_bar_top_left = (
            arrow_end[1] + self.small_arrow_img.shape[0] + 25,
            self.baseline_scale_top_left[1],
        )
        color_bar_bottom_right = (
            color_bar_top_left[0] + self.color_bar_img.shape[0],
            color_bar_top_left[1] + self.color_bar_img.shape[1],
        )
        self.img[
            color_bar_top_left[0] : color_bar_bottom_right[0],
            color_bar_top_left[1] : color_bar_bottom_right[1],
        ] = self.color_bar_img

        color_bar_middle = (
            (int(580 + self.baseline_scale_img.shape[1] / 2)),
            color_bar_bottom_right[0] + 50,
        )
        cv2.putText(
            self.img,
            "CLASSIFICATION PROBABILITY",
            color_bar_middle,
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.imwrite(self.img_save_path, self.img)
        self.visualization_timer.stop_timer()
        stdout_success(
            msg=f"SHAP summary graph saved at {self.img_save_path}",
            elapsed_time=self.visualization_timer.elapsed_time_str,
        )


# shap_df = pd.read_csv('/Users/simon/Desktop/envs/simba_dev/tests/test_data/test_shap/data/test_shap.csv', index_col=0)
# test = ShapAggregateStatisticsVisualizer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                          classifier_name='Attack',
#                                          shap_df=shap_df,
#                                          shap_baseline_value=40,
#                                          save_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/shap')

# shap_df = pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0)
# test = ShapAggregateStatisticsVisualizer(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini',
#                                          classifier_name='target',
#                                          shap_df=shap_df,
#                                          shap_baseline_value=40,
#                                          save_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/shap')
#
