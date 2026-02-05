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
                                    get_current_time, get_fn_ext, read_df,
                                    read_video_info)

NAPE_X, NAPE_Y = 'nape_x', 'nape_y'
FREEZING = 'FREEZING'

class FreezingDetector(ConfigReader):
    """
    Detect freezing behavior using heuristic rules based on movement velocity thresholds.
    Analyzes pose-estimation data to detect freezing episodes by computing the mean velocity
    of key body parts (nape, nose, and tail-base) and identifying periods where movement falls below
    a specified threshold for a minimum duration.

    .. important::
        Freezing is detected as `present` when **the velocity (computed from the mean movement of the nape, nose, and tail-base body-parts) falls below
        the movement threshold for the duration (and longer) of the specified time-window**.
        Freezing is detected as `absent` when not present.

    .. note::
       The method uses the left and right ear body-parts to compute the `nape` location of the animal
       as the midpoint between the ears. The nape, nose, and tail-base movements are averaged to compute
       overall animal movement velocity.

    :param Union[str, os.PathLike] data_dir: Path to directory containing pose-estimated body-part data in CSV format. Each CSV file should contain pose estimation data for one video.
    :param Union[str, os.PathLike] config_path: Path to SimBA project config file (`.ini` format) containing project settings and video information.
    :param Optional[str] nose_name: The name of the pose-estimated nose body-part column (without _x/_y suffix). Defaults to 'nose'.
    :param Optional[str] left_ear_name: The name of the pose-estimated left ear body-part column (without _x/_y suffix). Defaults to 'Left_ear'.
    :param Optional[str] right_ear_name: The name of the pose-estimated right ear body-part column (without _x/_y suffix). Defaults to 'right_ear'.
    :param Optional[str] tail_base_name: The name of the pose-estimated tail base body-part column (without _x/_y suffix). Defaults to 'tail_base'.
    :param Optional[int] time_window: The minimum time window in seconds that movement must be below the threshold to be considered freezing. Only freezing bouts lasting at least this duration are retained. Defaults to 3.
    :param Optional[int] movement_threshold: Movement threshold in millimeters per second. Frames with mean velocity below this threshold are considered potential freezing. Defaults to 5.
    :param Optional[int] shortest_bout: Minimum duration in milliseconds for a freezing bout to be considered valid. Shorter bouts are filtered out. Defaults to 100.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to store the results. If None, then results are stored in a timestamped subdirectory within the ``logs`` directory of the SimBA project.
    :returns: None. Results are saved to CSV files in the specified save directory:
        - Individual video results: One CSV file per video with freezing annotations added as a 'FREEZING' column (1 = freezing, 0 = not freezing)
        - Aggregate results: `aggregate_freezing_results.csv` containing summary statistics for all videos

    :example:
    >>> x = FreezingDetector(data_dir=r'D:\\troubleshooting\\mitra\\project_folder\\csv\\outlier_corrected_movement_location', config_path=r"D:\\troubleshooting\\mitra\\project_folder\\project_config.ini", time_window=3, movement_threshold=5, shortest_bout=100
    >>> x.run()

    references
    ----------
    ..
    .. [1] Sabnis et al., Visual detection of seizures in mice using supervised machine learning, `biorxiv`, doi: https://doi.org/10.1101/2024.05.29.596520.
    .. [2] Lopez et al., Region-specific Nucleus Accumbens Dopamine Signals Encode Distinct Aspects of Avoidance Learning, `biorxiv`, doi: https://doi.org/10.1101/2024.08.28.610149
    .. [3] Lopez, Gabriela C., Louis D. Van Camp, Ryan F. Kovaleski, et al. "Region-Specific Nucleus Accumbens Dopamine Signals Encode Distinct Aspects of Avoidance Learning." `Cell Biology`, Volume 35, Issue 10p2433-2443.e5May 19, 2025. DOI: 10.1016/j.cub.2025.04.006
    .. [4] Lazaro et al., Brainwide Genetic Capture for Conscious State Transitions, `biorxiv`, doi: https://doi.org/10.1101/2025.03.28.646066
    .. [5] Sabnis et al., Visual detection of seizures in mice using supervised machine learning, 2025, Cell Reports Methods 5, 101242 December 15, 2025.
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 nose_name: str = 'nose',
                 left_ear_name: str = 'Left_ear',
                 right_ear_name: str = 'right_ear',
                 tail_base_name: str = 'tail_base',
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 time_window: int = 4,
                 movement_threshold: int = 5,
                 shortest_bout: int = 100,
                 save_dir: Optional[Union[str, os.PathLike]] = None):
        for bp_name in [nose_name, left_ear_name, right_ear_name, tail_base_name]: check_str(name='body part name', value=bp_name, allow_blank=False)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        if data_dir is not None:
            check_if_dir_exists(in_dir=data_dir)
        else:
            data_dir = self.outlier_corrected_dir
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        self.nose_heads = [f'{nose_name}_x'.lower(), f'{nose_name}_y'.lower()]
        self.left_ear_heads = [f'{left_ear_name}_x'.lower(), f'{left_ear_name}_y'.lower()]
        self.right_ear_heads = [f'{right_ear_name}_x'.lower(), f'{right_ear_name}_y'.lower()]
        self.tail_base_heads = [f'{tail_base_name}_x'.lower(), f'{tail_base_name}_y'.lower()]
        self.required_field = self.nose_heads + self.left_ear_heads + self.right_ear_heads + self.tail_base_heads
        check_int(name='time_window', value=time_window, min_value=1)
        check_int(name='movement_threshold', value=movement_threshold, min_value=1)
        self.save_dir = save_dir
        if self.save_dir is None:
            self.save_dir = os.path.join(self.logs_path, f'freezing_data_time_{time_window}s_{movement_threshold}mm_{self.datetime}')
            os.makedirs(self.save_dir)
        else:
            check_if_dir_exists(in_dir=self.save_dir)
        self.time_window, self.movement_threshold = time_window, movement_threshold
        self.movement_threshold, self.shortest_bout = movement_threshold, shortest_bout
    def run(self):
        agg_results = pd.DataFrame(columns=['VIDEO', 'FREEZING FRAMES', 'FREEZING TIME (S)', 'FREEZING BOUT COUNTS', 'FREEZING PCT OF SESSION', 'VIDEO TOTAL FRAMES', 'VIDEO TOTAL TIME (S)'])
        agg_results_path = os.path.join(self.save_dir, f'aggregate_freezing_results_{self.datetime}.csv')
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_name = get_fn_ext(filepath=file_path)[1]
            print(f'[{get_current_time()}] Analyzing freezing {video_name}...(video {file_cnt+1}/{len(self.data_paths)})')
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
            df[f'Probability_{FREEZING}'] = 0
            df[FREEZING] = 0
            df.loc[freezing_idx, FREEZING] = 1
            df = plug_holes_shortest_bout(data_df=df, clf_name=FREEZING, fps=fps, shortest_bout=self.shortest_bout)
            bouts = detect_bouts(data_df=df, target_lst=[FREEZING], fps=fps)
            bouts = bouts[bouts['Bout_time'] >= self.time_window]
            if len(bouts) > 0:
                df[FREEZING] = 0
                freezing_idx = list(bouts.apply(lambda x: list(range(int(x["Start_frame"]), int(x["End_frame"]) + 1)), 1))
                freezing_idx = [x for xs in freezing_idx for x in xs]
                df.loc[freezing_idx, FREEZING] = 1
                df.loc[freezing_idx, f'Probability_{FREEZING}'] = 1
            else:
                df[FREEZING] = 0
                freezing_idx = []
            df.to_csv(save_file_path)
            #print(video_name, len(freezing_idx), round(len(freezing_idx) / fps, 4), df[FREEZING].sum())
            agg_results.loc[len(agg_results)] = [video_name, len(freezing_idx), round(len(freezing_idx) / fps, 4), len(bouts), round((len(freezing_idx) / len(df)) * 100, 4), len(df), round(len(df)/fps, 2) ]
        agg_results.to_csv(agg_results_path)
        self.timer.stop_timer(); stdout_success(msg=f'Results saved in {self.save_dir} directory.', elapsed_time=self.timer.elapsed_time_str)



# FreezingDetector(
#     data_dir=r'E:\troubleshooting\mitra_emergence_hour\project_folder\csv\outlier_corrected_movement_location',
#     config_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini",
#     time_window=3,
#     movement_threshold=5,
#     shortest_bout=100
# ).run()