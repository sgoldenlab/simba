__author__ = "Simon Nilsson"

import itertools
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

import simba
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_df_field_is_boolean,
                                check_if_dir_exists, check_if_valid_img,
                                check_instance, check_int, check_str,
                                check_that_column_exist, check_valid_boolean,
                                check_valid_dataframe, check_valid_tuple)
from simba.utils.enums import Paths
from simba.utils.errors import FeatureNumberMismatchError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (bgr_to_rgb_tuple,
                                    read_shap_feature_categories_csv,
                                    read_shap_img_paths)

SIMBA_DIR = os.path.dirname(simba.__file__)

def _create_shap_base_plot(baseline_value: int) -> Tuple[np.ndarray, Tuple[int, int], List[Tuple[int, int]]]:

    shap_img_path = os.path.join(SIMBA_DIR, Paths.SIMBA_SHAP_IMG_PATH.value)
    check_if_dir_exists(in_dir=shap_img_path)
    scale_img_paths, category_img_paths = read_shap_img_paths()
    for k, v in category_img_paths.items(): check_file_exist_and_readable(file_path=v)
    img = 255 * np.ones([1680, 1680, 3], dtype=np.uint8)
    baseline_scale_img = cv2.imread(scale_img_paths["baseline_scale"])
    baseline_scale_top_left = (100, 800)
    baseline_scale_bottom_right = (baseline_scale_top_left[0] + baseline_scale_img.shape[0], baseline_scale_top_left[1] + baseline_scale_img.shape[1])
    baseline_scale_middle = ((int(700 + baseline_scale_img.shape[1] / 2)), 85)
    img[baseline_scale_top_left[0] : baseline_scale_bottom_right[0], baseline_scale_top_left[1] : baseline_scale_bottom_right[1]] = baseline_scale_img
    cv2.putText(img, "BASELINE SHAP", baseline_scale_middle, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    small_arrow_img = cv2.imread(scale_img_paths["small_arrow"])
    small_arrow_top_left = (baseline_scale_bottom_right[0], int(baseline_scale_top_left[1] + (baseline_scale_img.shape[1] / 100) * (baseline_value)))
    small_arrow_bottom_right = (small_arrow_top_left[0] + small_arrow_img.shape[0], small_arrow_top_left[1] + small_arrow_img.shape[1])
    img[small_arrow_top_left[0] : small_arrow_bottom_right[0], small_arrow_top_left[1] : small_arrow_bottom_right[1]] = small_arrow_img

    side_scale_img = cv2.imread(scale_img_paths["side_scale"])
    side_scale_top_left = (small_arrow_bottom_right[0] + 50, baseline_scale_top_left[1] - 50)
    side_scale_y_tick_cords = [(side_scale_top_left[0], side_scale_top_left[1] - 75)]
    for i in range(1, 7):
        side_scale_y_tick_cords.append((int(side_scale_top_left[0] + (side_scale_img.shape[0] / 4) * i), int(side_scale_top_left[1] - 75)))
    arrow_start = (int(small_arrow_top_left[1] + (small_arrow_img.shape[1] / 2)), side_scale_top_left[0])
    for img_cnt, (img_name, img_data) in enumerate(category_img_paths.items()):
        icon_img = cv2.resize(cv2.imread(img_data), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        icon_top_left = (side_scale_y_tick_cords[img_cnt][0] - int(icon_img.shape[0] / 2), side_scale_y_tick_cords[img_cnt][1] - 100)
        icon_bottom_right = (icon_top_left[0] + icon_img.shape[0], side_scale_y_tick_cords[img_cnt][1] + icon_img.shape[1] - 100)
        text_location = (int(icon_bottom_right[0] - (icon_bottom_right[0] - icon_top_left[0]) + 100), int(icon_bottom_right[1] - (icon_bottom_right[1] - icon_top_left[1])) - 380)
        cv2.putText(img, str(img_name), (text_location[1], text_location[0]), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 1)
        img[icon_top_left[0] : icon_bottom_right[0], icon_top_left[1] : icon_bottom_right[1]] = icon_img

    return img, arrow_start, side_scale_y_tick_cords

def _insert_data_in_base_shap_plot(img: np.ndarray,
                                   arrow_start: Tuple[int, int],
                                   present_df: pd.DataFrame,
                                   side_scale_y_tick_cords: List[Tuple[int, int]]) -> np.ndarray:

    check_if_valid_img(data=img, source=f'{_insert_data_in_base_shap_plot.__name__} img')
    check_valid_tuple(x=arrow_start, min_integer=0)
    check_valid_dataframe(df=present_df, source=f'{_insert_data_in_base_shap_plot.__name__} present_df')
    scale_img_paths, category_img_paths = read_shap_img_paths()
    positive_arrow_colors = [(253, 141, 60), (252, 78, 42), (227, 26, 28), (189, 0, 38), (128, 0, 38)]
    negative_arrow_colors = [(65, 182, 196), (29, 145, 192), (34, 94, 168), (37, 52, 148), (8, 29, 88)]
    positive_arrow_colors = [bgr_to_rgb_tuple(x) for x in positive_arrow_colors]
    negative_arrow_colors = [bgr_to_rgb_tuple(x) for x in negative_arrow_colors]
    ranges_lst = [list(range(0, 20)), list(range(20, 40)), list(range(40, 60)), list(range(60, 80)), list(range(80, 101))]

    baseline_scale_img = cv2.imread(scale_img_paths["baseline_scale"])
    small_arrow_img = cv2.imread(scale_img_paths["small_arrow"])
    color_bar_img = cv2.imread(scale_img_paths["color_bar"])
    baseline_scale_top_left, arrow_end = (100, 800), None

    for row_cnt, (x_name, x_data) in enumerate(present_df.iterrows()):
        x_cat_sum = int(x_data.sum())
        arrow_width = int((baseline_scale_img.shape[1] / 100) * abs(x_cat_sum))
        if x_cat_sum > 0:
            arrow_end = (arrow_start[0] + arrow_width, arrow_start[1])
            arrow_middle = int(((arrow_end[1] - arrow_start[1]) / 2) + arrow_start[1] - 7)
            for bracket_num, bracket in enumerate(ranges_lst):
                if abs(x_cat_sum) in bracket:
                    color = positive_arrow_colors[bracket_num]
                    cv2.arrowedLine(img, arrow_start, arrow_end, color, 5, tipLength=0.1)
                    cv2.putText(img, f"+{x_cat_sum}%", (arrow_end[0] - 7, arrow_middle - 15), cv2.FONT_HERSHEY_COMPLEX, 1,  color, 2,)
        else:
            arrow_end = (arrow_start[0] - arrow_width, arrow_start[1])
            arrow_middle = int(((arrow_start[1] - arrow_end[1]) / 2) + arrow_end[1] - 7)
            for bracket_num, bracket in enumerate(ranges_lst):
                if abs(x_cat_sum) in bracket:
                    color = negative_arrow_colors[bracket_num]
                    cv2.arrowedLine(img, arrow_start, arrow_end, color, 5, tipLength=0.1)
                    cv2.putText(img, f"-{abs(x_cat_sum)}%", (arrow_end[0] - 7, arrow_middle - 15), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        if row_cnt != (len(present_df) - 1):
            arrow_start = (arrow_end[0], side_scale_y_tick_cords[row_cnt + 1][0])

    small_arrow_top_left = (int(arrow_end[1]) + 20, int(arrow_end[0] - small_arrow_img.shape[1] / 2))
    small_arrow_bottom_right = (small_arrow_top_left[0] + small_arrow_img.shape[0], small_arrow_top_left[1] + small_arrow_img.shape[1])
    img[small_arrow_top_left[0] : small_arrow_bottom_right[0], small_arrow_top_left[1] : small_arrow_bottom_right[1]] = small_arrow_img
    color_bar_top_left = (arrow_end[1] + small_arrow_img.shape[0] + 25, baseline_scale_top_left[1])
    color_bar_bottom_right = (color_bar_top_left[0] + color_bar_img.shape[0], color_bar_top_left[1] + color_bar_img.shape[1])
    img[color_bar_top_left[0] : color_bar_bottom_right[0], color_bar_top_left[1] : color_bar_bottom_right[1]] = color_bar_img
    color_bar_middle = ((int(580 + baseline_scale_img.shape[1] / 2)), color_bar_bottom_right[0] + 50)
    cv2.putText(img, "CLASSIFICATION PROBABILITY", color_bar_middle, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    return img

class ShapAggregateStatisticsCalculator():
    """
    Calculate aggregate (binned) SHAP value statistics where individual bins represent reaulated features.
    and create line chart visualizations reprsenting aggregations of behavior-present SHAP values.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/SHAP.md#step-3-interpreting-the-shap-value-ouput-generated-by-simba>`__.
       `Example output <https://github.com/sgoldenlab/simba/blob/master/images/example_shap_graph.png>`__.

    .. image:: _static/img/shap.png
       :width: 600
       :align: center

    :param pd.DataFrame shap_df: Data with framewise SHAP values.
    :param str classifier_name: Name of classifier (e.g., Attack).
    :param pd.DataFrame shap_df: Dataframe with non-aggregated SHAP values where rows represent frames and columns represent features.
    :param float shap_baseline_value: SHAP expected value (computed by ``simba.train_model_functions.create_shap_log``).
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to store the results. If None, then return the results instead of saving it.
    :param Optional[Any] filename_suffix: Optional suffix to add to the shap output filenames. Useful for gridsearches and multiple shap data output files are to-be stored in the same `save_dir`.
    :param bool plot: If True, creates a visualization of the aggregate SHAP values. Default True.

    :example:
    >>> shap_df = pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0)
    >>> test = ShapAggregateStatisticsCalculator(classifier_name='target',
    >>>                                          shap_df=shap_df,
    >>>                                          shap_baseline_value=40,
    >>>                                          save_dir=None) #'/Users/simon/Desktop/feltz'
    >>> dfs, img = test.run()
    """

    def __init__(self,
                 shap_df: pd.DataFrame,
                 classifier_name: str,
                 shap_baseline_value: int,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 filename_suffix: Optional[int] = None,
                 plot: bool = True):

        check_instance(source=f"{self.__class__.__name__} shap_df", instance=shap_df, accepted_types=(pd.DataFrame,))
        check_str(name=f"{self.__class__.__name__} classifier_name", value=classifier_name)
        check_that_column_exist(df=shap_df, column_name=classifier_name, file_name="shap dataframe")
        check_if_df_field_is_boolean(df=shap_df, field=classifier_name)
        check_int(name=f"{self.__class__.__name__} shap_baseline_value", value=shap_baseline_value, max_value=100, min_value=0)
        check_valid_boolean(value=[plot], source=f'{self.__class__.__name__} plot')
        self.clf_name, self.shap_df, self.shap_baseline_value, self.plot = classifier_name, shap_df, shap_baseline_value, plot
        self.save_path, self.datetime = None, datetime.now().strftime("%Y%m%d%H%M%S")
        self.present_save_path, self.absent_save_path, img_save_path = None, None, None
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir)
            if filename_suffix is None:
                self.img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{self.clf_name}.png")
                self.df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{self.clf_name}_PRESENT.csv"),
                                      'ABSENT': os.path.join(save_dir, f"SHAP_summary_{self.clf_name}_ABSENT.csv")}
            else:
                self.img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{self.clf_name}_{filename_suffix}.png")
                self.df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{self.clf_name}_PRESENT_{filename_suffix}.csv"),
                                      'ABSENT': os.path.join(save_dir, f"SHAP_summary_{self.clf_name}_ABSENT_{filename_suffix}.csv")}
        self.feature_categories_df, self.x_names, self.unique_feature_category_names, self.unique_time_bin_names = read_shap_feature_categories_csv()
        self.save_dir = save_dir
        v = [x for x in list(shap_df.columns) if x in self.x_names]
        if len(v) == 0:
            raise FeatureNumberMismatchError('No feature names in the shap dataframe is defined in the shap aggregation classes', source=self.__class__.__name__)

    def run(self):
        timer = SimbaTimer(start=True)
        self.results, self.img = {}, None
        for state, clf_state_name in zip(range(2), ["ABSENT", "PRESENT"]):
            state_result = {}
            shap_df = self.shap_df[self.shap_df[self.clf_name] == state]
            for x_category, x_time_bin in itertools.product(self.unique_feature_category_names, self.unique_time_bin_names):
                if x_category not in state_result.keys():
                    state_result[x_category] = {}
                feature_names_sliced = list(self.feature_categories_df.loc[:, (x_category, x_time_bin)])
                feature_names_sliced = [x for x in feature_names_sliced if str(x) != "nan" and x in feature_names_sliced]
                feature_names_sliced = [x for x in feature_names_sliced if x in shap_df.columns]
                state_result[x_category][x_time_bin] = round(shap_df[feature_names_sliced].sum(axis=1).mean() * 100, 6)
            results = pd.DataFrame(columns=state_result[list(state_result.keys())[0]], index=state_result.keys())
            for row_name, time_bins in state_result.items():
                for column_name, value in time_bins.items():
                    results.loc[row_name, column_name] = value
            results.reindex(sorted(results.columns, reverse=True), axis=1)
            self.results[clf_state_name] = results
            if self.save_dir is not None:
                self.results[clf_state_name].to_csv(self.df_save_paths[clf_state_name])

        if self.plot:
            base_plot, arrow_start, side_scale_y_tick_cords = _create_shap_base_plot(baseline_value=self.shap_baseline_value)
            self.img = _insert_data_in_base_shap_plot(img=base_plot, arrow_start=arrow_start, present_df=self.results['PRESENT'], side_scale_y_tick_cords=side_scale_y_tick_cords)
            if self.save_dir is not None:
                cv2.imwrite(self.img_save_path, self.img)

        timer.stop_timer()
        if self.save_dir is not None:
            stdout_success(msg=f"SHAP summary graph saved at {self.img_save_path}", elapsed_time=timer.elapsed_time_str)
        else:
            return (self.results, self.img)


# shap_df = pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0)
# test = ShapAggregateStatisticsCalculator(classifier_name='target',
#                                          shap_df=shap_df,
#                                          shap_baseline_value=40,
#                                          save_dir=None) #'/Users/simon/Desktop/feltz'
# dfs, img = test.run()

# shap_df = pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0)
# test = ShapAggregateStatisticsVisualizer(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini',
#                                          classifier_name='target',
#                                          shap_df=shap_df,
#                                          shap_baseline_value=40,
#                                          save_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/shap')
#
