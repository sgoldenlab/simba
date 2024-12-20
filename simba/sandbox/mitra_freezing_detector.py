import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from numba import typed

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists,
    check_int, check_str, check_valid_dataframe)
from simba.utils.data import detect_bouts, plug_holes_shortest_bout
from simba.utils.enums import Formats
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, read_video_info)

NAPE_X, NAPE_Y = 'nape_x', 'nape_y'
FREEZING = 'FREEZING'

class MitraFreezingDetector(ConfigReader):

    """
    Detects freezing behavior in rodent movement data as outlined in `Sabnis et al.` (2024).


    This implementation follows the methodology for freezing detection using supervised machine learning,
    specifically applied to mouse behavioral analysis. The freezing detection algorithm is based on movement
    thresholds calculated from tracked body parts (nose, ears, and tail base) over time.

    The detector identifies freezing episodes by analyzing reduced movement across the defined body parts,
    applying bout detection, and classifying freezing bouts based on predefined criteria, such as a
    minimum movement threshold over a specified bout duration.

    :example:
    >>> MitraFreezingDetector(data_dir=r'D:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location', config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")

    References
    ----------
    .. [1] Sabnis et al., Visual detection of seizures in mice using supervised machine learning, `biorxiv`, doi: https://doi.org/10.1101/2024.05.29.596520.

    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 config_path: Union[str, os.PathLike],
                 nose_name: Optional[str] = 'nose',
                 left_ear_name: Optional[str] = 'Left_ear',
                 right_ear_name: Optional[str] = 'right_ear',
                 tail_base_name: Optional[str] = 'tail_base',
                 time_window: Optional[int] = 3,
                 movement_threshold: Optional[int] = 5,
                 save_dir: Optional[Union[str, os.PathLike]] = None):

        check_if_dir_exists(in_dir=data_dir)
        for bp_name in [nose_name, left_ear_name, right_ear_name, tail_base_name]: check_str(name='body part name', value=bp_name, allow_blank=False)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        self.nose_heads = [f'{nose_name}_x'.lower(), f'{nose_name}_y'.lower()]
        self.left_ear_heads = [f'{left_ear_name}_x'.lower(), f'{left_ear_name}_y'.lower()]
        self.right_ear_heads = [f'{right_ear_name}_x'.lower(), f'{right_ear_name}_y'.lower()]
        self.tail_base_heads = [f'{tail_base_name}_x'.lower(), f'{tail_base_name}_y'.lower()]
        self.required_field = self.nose_heads + self.left_ear_heads + self.right_ear_heads + self.tail_base_heads
        check_int(name='time_window', value=time_window, min_value=1)
        check_int(name='movement_threshold', value=movement_threshold, min_value=1)
        self.save_dir = save_dir
        if self.save_dir is None:
            self.save_dir = os.path.join(self.logs_path, f'freezing_data_time_{time_window}s_{self.datetime}')
            os.makedirs(self.save_dir)
        else:
            check_if_dir_exists(in_dir=self.save_dir)
        self.time_window, self.movement_threshold = time_window, movement_threshold
        self.movement_threshold = movement_threshold
        self.run()

    def run(self):
        agg_results = pd.DataFrame(columns=['VIDEO', 'FREEZING FRAMES', 'FREEZING TIME (S)', 'FREEZING BOUT COUNTS', 'FREEZING PCT OF SESSION', 'VIDEO TOTAL FRAMES', 'VIDEO TOTAL TIME (S)'])
        agg_results_path = os.path.join(self.save_dir, 'aggregate_freezing_results.csv')
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_name = get_fn_ext(filepath=file_path)[1]
            print(f'Analyzing {video_name}...')
            save_file_path = os.path.join(self.save_dir, f'{video_name}.csv')
            df = read_df(file_path=file_path, file_type='csv').reset_index(drop=True)
            _, px_per_mm, fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
            df.columns = [str(x).lower() for x in df.columns]
            check_valid_dataframe(df=df, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.required_field)
            nose_shifted = FeatureExtractionMixin.create_shifted_df(df[self.nose_heads])
            nose_1, nose_2 = nose_shifted.iloc[:, 0:2].values, nose_shifted.iloc[:, 2:4].values
            nose_movement = FeatureExtractionMixin.euclidean_distance(bp_1_x=nose_1[:, 0].flatten(), bp_2_x=nose_2[:, 0].flatten(), bp_1_y=nose_1[:, 1].flatten(), bp_2_y=nose_2[:, 1].flatten(), px_per_mm=px_per_mm)
            tail_base_shifted = FeatureExtractionMixin.create_shifted_df(df[self.tail_base_heads])
            tail_base_shifted_1, tail_base_shifted_2 = tail_base_shifted.iloc[:, 0:2].values, tail_base_shifted.iloc[:, 2:4].values
            tail_base_movement = FeatureExtractionMixin.euclidean_distance(bp_1_x=tail_base_shifted_1[:, 0].flatten(), bp_2_x=tail_base_shifted_2[:, 0].flatten(), bp_1_y=tail_base_shifted_1[:, 1].flatten(), bp_2_y=tail_base_shifted_2[:, 1].flatten(), px_per_mm=px_per_mm)
            left_ear_arr = df[self.left_ear_heads].values.astype(np.int64)
            right_ear_arr = df[self.right_ear_heads].values.astype(np.int64)
            nape_arr = pd.DataFrame(FeatureExtractionMixin.find_midpoints(bp_1=left_ear_arr, bp_2=right_ear_arr, percentile=np.float64(0.5)), columns=[NAPE_X, NAPE_Y])
            nape_shifted = FeatureExtractionMixin.create_shifted_df(nape_arr[[NAPE_X, NAPE_Y]])
            nape_shifted_1, nape_shifted_2 = nape_shifted.iloc[:, 0:2].values, nape_shifted.iloc[:, 2:4].values
            nape_movement = FeatureExtractionMixin.euclidean_distance(bp_1_x=nape_shifted_1[:, 0].flatten(), bp_2_x=nape_shifted_2[:, 0].flatten(), bp_1_y=nape_shifted_1[:, 1].flatten(),  bp_2_y=nape_shifted_2[:, 1].flatten(), px_per_mm=px_per_mm)
            movement = np.hstack([nose_movement.reshape(-1, 1), nape_movement.reshape(-1, 1), tail_base_movement.reshape(-1, 1)])
            mean_movement = np.mean(movement, axis=1)
            mm_s = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=mean_movement.astype(np.float32), window_sizes=np.array([1], dtype=np.float64), sample_rate=int(fps), statistics=typed.List(["sum"]))[0].flatten()
            freezing_idx = np.argwhere(mm_s <= self.movement_threshold).astype(np.int32).flatten()
            df[FREEZING] = 0
            df.loc[freezing_idx, FREEZING] = 1
            df = plug_holes_shortest_bout(data_df=df, clf_name=FREEZING, fps=fps, shortest_bout=100)
            bouts = detect_bouts(data_df=df, target_lst=[FREEZING], fps=fps)
            bouts = bouts[bouts['Bout_time'] >= self.time_window]
            if len(bouts) > 0:
                freezing_idx = list(bouts.apply(lambda x: list(range(int(x["Start_frame"]), int(x["End_frame"]) + 1)), 1))
                freezing_idx = [x for xs in freezing_idx for x in xs]
                df.loc[freezing_idx, FREEZING] = 1
            else:
                freezing_idx = []

            df.to_csv(save_file_path)
            agg_results.loc[len(agg_results)] = [video_name, len(freezing_idx), round(len(freezing_idx) / fps, 4), len(bouts), round((len(freezing_idx) / len(df)) * 100, 4), len(df), round(len(df)/fps, 2) ]
        agg_results.to_csv(agg_results_path)
        stdout_success(msg=f'Results saved in {self.save_dir} directory.')







            #
            # freezing_idx = np.argwhere(movement_mean <= movement_threshold)
            #

            # freezing_df = video_df[video_df['Movement_2s'] <= 66]


#
#
#
#
#     print(file_cnt)
#     video_name = get_fn_ext(filepath=file_path)[1].lower()
#     if '_cno_' in video_name:
#         drug = 'cno'
#     elif '_saline_' in video_name:
#         drug = 'saline'
#     else:
#         drug = 'dzo'
#     if '_gi_' in video_name:
#         group = 'gi'
#     elif '_gq_' in video_name:
#         group = 'gq'
#     else:
#         group = 'dzo'
#     df = read_df(file_path=file_path, file_type='csv')
#     nose_shifted = FeatureExtractionMixin.create_shifted_df(df[['Nose_x', 'Nose_y']])
#     nose_1 = nose_shifted.iloc[:, 0:2].values
#     nose_2 = nose_shifted.iloc[:, 2:4].values
#     nose_movement = FeatureExtractionMixin.euclidean_distance(bp_1_x=nose_1[:, 0].flatten(),
#                                                               bp_2_x=nose_2[:, 0].flatten(),
#                                                               bp_1_y=nose_1[:, 1].flatten(),
#                                                               bp_2_y=nose_2[:, 1].flatten(), px_per_mm=2.15)
#
#
#     tail_base_shifted = FeatureExtractionMixin.create_shifted_df(df[['Tail_base_x', 'Tail_base_y']])
#     tail_base_shifted_1 = tail_base_shifted.iloc[:, 0:2].values
#     tail_base_shifted_2 = tail_base_shifted.iloc[:, 2:4].values
#     tail_base_movement = FeatureExtractionMixin.euclidean_distance(bp_1_x=tail_base_shifted_1[:, 0].flatten(),
#                                                                           bp_2_x=tail_base_shifted_2[:, 0].flatten(),
#                                                                           bp_1_y=tail_base_shifted_1[:, 1].flatten(),
#                                                                           bp_2_y=tail_base_shifted_2[:, 1].flatten(), px_per_mm=2.15)
#
#
#     left_ear_arr = df[['Left_ear_x', 'Left_ear_y']].values.astype(np.int64)
#     right_ear_arr = df[['Right_ear_x', 'Right_ear_y']].values.astype(np.int64)
#     nape_arr = pd.DataFrame(FeatureExtractionMixin.find_midpoints(bp_1=left_ear_arr, bp_2=right_ear_arr, percentile=np.float64(0.5)), columns=['Nape_x', 'Nape_y'])
#     nape_shifted = FeatureExtractionMixin.create_shifted_df(nape_arr[['Nape_x', 'Nape_y']])
#
#     nape_shifted_1 = nape_shifted.iloc[:, 0:2].values
#     nape_shifted_2 = nape_shifted.iloc[:, 2:4].values
#     nape_movement = FeatureExtractionMixin.euclidean_distance(bp_1_x=nape_shifted_1[:, 0].flatten(),
#                                                               bp_2_x=nape_shifted_2[:, 0].flatten(),
#                                                               bp_1_y=nape_shifted_1[:, 1].flatten(),
#                                                               bp_2_y=nape_shifted_2[:, 1].flatten(), px_per_mm=2.15)
#
#     movement = np.hstack([nose_movement.reshape(-1, 1), nape_movement.reshape(-1, 1), tail_base_movement.reshape(-1, 1)])
#     mean_movement = np.mean(movement, axis=1)
#
#     movement_mean = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=nose_movement.astype(np.float32), window_sizes=np.array([2, 3, 4, 6], dtype=np.float64), sample_rate=30, statistics=typed.List(["sum"]))[0]
#     movement_df = pd.DataFrame(movement_mean, columns=['Movement_2s', 'Movement_3s', 'Movement_4s', 'Movement_6s'])
#     out = pd.concat([movement_df], axis=1)
#     out['video'] = get_fn_ext(filepath=file_path)[1]
#     out['drug'] = drug
#     out['group'] = group
#     out['condition'] = f'{group}_{drug}'.upper()
#     results.append(out)
#
# out = pd.concat(results, axis=0)
# final_results = pd.DataFrame(columns=['DRUG', 'GROUP', 'CONDITION', '% SESSION'])
# for video_cnt, video in enumerate(out['video'].unique()):
#     print(video_cnt)
#     video_df = out[out['video'] == video].reset_index(drop=True)
#     drug, group, condition = video_df['drug'].iloc[0], video_df['group'].iloc[0], video_df['condition'].iloc[0]
#     if condition == 'GQ_CNO' or condition == 'GQ_SALINE':
#         freezing_df = video_df[video_df['Movement_2s'] <= 66]
#         if len(freezing_df) == 0:
#             time = 0
#         else:
#             time = (len(freezing_df) / len(video_df))
#         final_results.loc[len(final_results)] = [drug, group, condition, time]
#
#
#
# cno = final_results[final_results['CONDITION'] == 'GQ_CNO']['% SESSION'].values
# saline = final_results[final_results['CONDITION'] == 'GQ_SALINE']['% SESSION'].values
#
# ttest_ind(cno, saline, equal_var=True)
#
#
# plot = sns.stripplot(data=final_results, x='CONDITION', y='% SESSION', linewidth=2)
# plot = sns.barplot(data=final_results, x='CONDITION', y='% SESSION')

MitraFreezingDetector(data_dir=r'D:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location',
                      config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")