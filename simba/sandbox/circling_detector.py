import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from numba import typed

from simba.mixins.circular_statistics import CircularStatisticsMixin
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

CIRCLING = 'CIRCLING'

class CirclingDetector(ConfigReader):

    """
    Detect circling using heuristic rules.

    .. important::
        Circling is detected as :underline:`present` when **the circular range of the animal is above the ``circular_range_threshold`` within the preceding ``time_threshold``** AND
        **the movement of the animal (defined as the sum of the center movement) is above the  ``movement_threshold`` within the preceding ``time_threshold``.**

    :param Union[str, os.PathLike] data_dir: Path to directory containing pose-estimated body-part data in CSV format.
    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param Optional[str] nose_name: The name of the pose-estimated nose body-part. Defaults to 'nose'.
    :param Optional[str] left_ear: The name of the pose-estimated left ear body-part. Defaults to 'left_ear'.
    :param Optional[str] right_ear: The name of the pose-estimated right ear body-part. Defaults to 'right_ear'.
    :param Optional[str] tail_base_name: The name of the pose-estimated tail base body-part. Defaults to 'tail_base'.
    :param Optional[str] center_name: The name of the pose-estimated center body-part. Defaults to 'center'.
    :param Optional[int] time_threshold: The time window in preceding seconds in which to evaluate the animals circular range. Default: 10.
    :param Optional[int] circular_range_threshold: A value in degrees, between 0-360.
    :param Optional[int] movement_threshold: A movement threshold in millimeters.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to store the results. If None, then results are stored in the ``logs`` directory of the SimBA project.

    References
    ----------
    .. [1] Sabnis et al., Visual detection of seizures in mice using supervised machine learning, `biorxiv`, doi: https://doi.org/10.1101/2024.05.29.596520.

    :example:
    >>> CirclingDetector(data_dir=r'D:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location', config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 config_path: Union[str, os.PathLike],
                 nose_name: Optional[str] = 'nose',
                 left_ear_name: Optional[str] = 'left_ear',
                 right_ear_name: Optional[str] = 'right_ear',
                 tail_base_name: Optional[str] = 'tail_base',
                 center_name: Optional[str] = 'center',
                 time_threshold: Optional[int] = 10,
                 circular_range_threshold: Optional[int] = 320,
                 movement_threshold: Optional[int] = 60,
                 save_dir: Optional[Union[str, os.PathLike]] = None):

        check_if_dir_exists(in_dir=data_dir)
        for bp_name in [nose_name, left_ear_name, right_ear_name, tail_base_name]: check_str(name='body part name', value=bp_name, allow_blank=False)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        self.nose_heads = [f'{nose_name}_x'.lower(), f'{nose_name}_y'.lower()]
        self.left_ear_heads = [f'{left_ear_name}_x'.lower(), f'{left_ear_name}_y'.lower()]
        self.right_ear_heads = [f'{right_ear_name}_x'.lower(), f'{right_ear_name}_y'.lower()]
        self.center_heads = [f'{center_name}_x'.lower(), f'{center_name}_y'.lower()]
        self.required_field = self.nose_heads + self.left_ear_heads + self.right_ear_heads
        self.save_dir = save_dir
        if self.save_dir is None:
            self.save_dir = os.path.join(self.logs_path, f'circling_data_{self.datetime}')
            os.makedirs(self.save_dir)
        else:
            check_if_dir_exists(in_dir=self.save_dir)
        self.time_threshold, self.circular_range_threshold, self.movement_threshold = time_threshold, circular_range_threshold, movement_threshold

    def run(self):
        agg_results = pd.DataFrame(columns=['VIDEO', 'CIRCLING FRAMES', 'CIRCLING TIME (S)', 'CIRCLING BOUT COUNTS', 'CIRCLING PCT OF SESSION', 'VIDEO TOTAL FRAMES', 'VIDEO TOTAL TIME (S)'])
        agg_results_path = os.path.join(self.save_dir, 'aggregate_circling_results.csv')
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_name = get_fn_ext(filepath=file_path)[1]
            print(f'Analyzing {video_name} ({file_cnt+1}/{len(self.data_paths)})...')
            save_file_path = os.path.join(self.save_dir, f'{video_name}.csv')
            df = read_df(file_path=file_path, file_type='csv').reset_index(drop=True)
            _, px_per_mm, fps = read_video_info(video_info_df=self.video_info_df, video_name=video_name)
            df.columns = [str(x).lower() for x in df.columns]
            check_valid_dataframe(df=df, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.required_field)

            nose_arr = df[self.nose_heads].values.astype(np.float32)
            left_ear_arr = df[self.left_ear_heads].values.astype(np.float32)
            right_ear_arr = df[self.right_ear_heads].values.astype(np.float32)

            center_shifted = FeatureExtractionMixin.create_shifted_df(df[self.center_heads])
            center_1, center_2 = center_shifted.iloc[:, 0:2].values, center_shifted.iloc[:, 2:4].values

            angle_degrees = CircularStatisticsMixin().direction_three_bps(nose_loc=nose_arr, left_ear_loc=left_ear_arr, right_ear_loc=right_ear_arr).astype(np.float32)
            sliding_circular_range = CircularStatisticsMixin().sliding_circular_range(data=angle_degrees, time_windows=np.array([self.time_threshold], dtype=np.float64), fps=int(fps)).flatten()
            movement = FeatureExtractionMixin.euclidean_distance(bp_1_x=center_1[:, 0].flatten(), bp_2_x=center_2[:, 0].flatten(), bp_1_y=center_1[:, 1].flatten(), bp_2_y=center_2[:, 1].flatten(), px_per_mm=2.15)
            movement_sum = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=movement.astype(np.float32), window_sizes=np.array([self.time_threshold], dtype=np.float64), sample_rate=fps, statistics=typed.List(["sum"])).astype(np.int32)[0].flatten()

            circling_idx = np.argwhere(sliding_circular_range >= self.circular_range_threshold).astype(np.int32).flatten()
            movement_idx = np.argwhere(movement_sum >= self.movement_threshold).astype(np.int32).flatten()
            circling_idx = [x for x in movement_idx if x in circling_idx]
            df[CIRCLING] = 0
            df.loc[circling_idx, CIRCLING] = 1
            bouts = detect_bouts(data_df=df, target_lst=[CIRCLING], fps=fps)
            df = plug_holes_shortest_bout(data_df=df, clf_name=CIRCLING, fps=fps, shortest_bout=100)
            df.to_csv(save_file_path)
            agg_results.loc[len(agg_results)] = [video_name, len(circling_idx), round(len(circling_idx) / fps, 4), len(bouts), round((len(circling_idx) / len(df)) * 100, 4), len(df), round(len(df)/fps, 2) ]

        agg_results.to_csv(agg_results_path)
        stdout_success(msg=f'Results saved in {self.save_dir} directory.')



# detector = CirclingDetector(data_dir=r'D:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location', config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")
# detector.run()

