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
from simba.utils.errors import InvalidInputError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_current_time, get_fn_ext, read_df,
                                    read_video_info)

NAPE_X, NAPE_Y = 'nape_x', 'nape_y'
FREEZING = 'FREEZING'
IMMOBILITY = 'IMMOBILITY'

class FreezingDetector(ConfigReader):
    """
    Detect freezing behavior using heuristic rules based on movement velocity thresholds.
    Analyzes pose-estimation data to detect freezing episodes by computing the mean velocity
    of key body parts (nape, nose, and tail-base) and identifying periods where movement falls below
    a specified threshold for a minimum duration.

    .. seealso::
       :class:`~simba.data_processors.freezing_detector.FreezingDetector` - the original version, with
       ``time_window`` / ``shortest_bout`` parameters and freezing-only output.


    .. video:: _static/img/FreezingDetector.webm
       :width: 1000
       :autoplay:
       :loop:
       :muted:
       :align: center

    .. video:: _static/img/FreezingDetector_2.webm
       :width: 1000
       :autoplay:
       :loop:
       :muted:
       :align: center

    .. image:: _static/img/FreezingDetector_pipeline.png
       :alt: Freezing Detector pipeline
       :width: 1000
       :align: center

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
    :param Optional[int] min_freezing_ms: Minimum duration in milliseconds of sustained low movement to be scored as freezing. Shorter bouts are ignored. This is the lower edge of the freezing band. Defaults to 2000.
    :param Optional[int] movement_threshold: Movement threshold in millimeters per second. Frames with mean velocity below this threshold are considered potential freezing. Higher and more freezing will be detected. Lower and less freezing will be detected. Defaults to 5.
    :param Optional[int] clean_ms: De-noising window in milliseconds. Brief sub-threshold movement interruptions shorter than this are bridged (so jitter does not split one freeze into two), and stray bouts shorter than this are dropped, before the duration tests. Keep small relative to ``min_freezing_ms``. Defaults to 100.
    :param Optional[int] min_immobility_ms: Minimum duration in milliseconds at which sustained freezing is instead scored as immobility. Bouts at least this long are flagged in an `IMMOBILITY` column (1 = immobility, 0 = not) and removed from `FREEZING` (the two are mutually exclusive), with extra aggregate statistics; bouts between ``min_freezing_ms`` and this remain freezing. Must be greater than or equal to ``min_freezing_ms`` (when equal, there is no freezing band and every sustained bout is immobility). If None, no immobility is computed and every bout at least ``min_freezing_ms`` long is freezing. Defaults to None.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to store the results. If None, then results are stored in a timestamped subdirectory within the ``logs`` directory of the SimBA project.
    :return: None. Results are saved to CSV files in the specified save directory:
        - Individual video results: One CSV file per video with freezing annotations added as a 'FREEZING' column (1 = freezing, 0 = not freezing)
        - Aggregate results: `aggregate_freezing_results.csv` containing summary statistics for all videos

    :example:
    >>> x = FreezingDetector(data_dir=r'D:\\troubleshooting\\mitra\\project_folder\\csv\\outlier_corrected_movement_location', config_path=r"D:\\troubleshooting\\mitra\\project_folder\\project_config.ini", min_freezing_ms=1500, movement_threshold=5, min_immobility_ms=2000)
    >>> x.run()

    References
    ----------
    .. [1] Sabnis, G. S., et al. (2024). Visual detection of seizures in mice using supervised machine learning.
           `bioRxiv <https://doi.org/10.1101/2024.05.29.596520>`_.
    .. [2] Lopez, G. C., et al. (2024). Region-specific nucleus accumbens dopamine signals encode distinct aspects of avoidance learning.
           `bioRxiv <https://doi.org/10.1101/2024.08.28.610149>`_.
    .. [3] Lopez, G. C., Van Camp, L. D., Kovaleski, R. F., et al. (2025). Region-specific nucleus accumbens dopamine signals encode distinct aspects of avoidance learning.
           `Current Biology, 35(10), 2433–2443.e5 <https://doi.org/10.1016/j.cub.2025.04.006>`_.
    .. [4] Lazaro et al. (2025). Brainwide genetic capture for conscious state transitions.
           `bioRxiv <https://doi.org/10.1101/2025.03.28.646066>`_.
    .. [5] Sabnis, G. S., et al. (2025). Visual detection of seizures in mice using supervised machine learning.
           `Cell Reports Methods, 5(12), 101242 <https://doi.org/10.1016/j.crmeth.2025.101242>`_.
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 nose_name: str = 'nose',
                 left_ear_name: str = 'left_ear',
                 right_ear_name: str = 'right_ear',
                 tail_base_name: str = 'tail_base',
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 min_freezing_ms: int = 2000,
                 movement_threshold: int = 5,
                 clean_ms: int = 100,
                 min_immobility_ms: Optional[int] = None,
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
        check_int(name='min_freezing_ms', value=min_freezing_ms, min_value=1)
        check_int(name='movement_threshold', value=movement_threshold, min_value=1)
        check_int(name='clean_ms', value=clean_ms, min_value=1)
        if min_immobility_ms is not None:
            check_int(name='min_immobility_ms', value=min_immobility_ms, min_value=1)
            if int(min_immobility_ms) < int(min_freezing_ms):
                raise InvalidInputError(msg=f'min_immobility_ms ({min_immobility_ms} ms) must be greater than or equal to min_freezing_ms ({min_freezing_ms} ms). At equal values there is no freezing band and every sustained bout is immobility.', source=self.__class__.__name__)
        self.min_immobility_ms = min_immobility_ms
        self.save_dir = save_dir
        if self.save_dir is None:
            self.save_dir = os.path.join(self.logs_path, f'freezing_data_minfreeze_{min_freezing_ms}ms_{movement_threshold}mm_{self.datetime}')
            os.makedirs(self.save_dir)
        else:
            check_if_dir_exists(in_dir=self.save_dir)
        self.min_freezing_ms, self.movement_threshold, self.clean_ms = min_freezing_ms, movement_threshold, clean_ms
    def run(self):
        freezing_cols = ['VIDEO', 'FREEZING FRAMES', 'FREEZING TIME (S)', 'FREEZING BOUT COUNTS', 'FREEZING PCT OF SESSION']
        immobility_cols = ['IMMOBILITY FRAMES', 'IMMOBILITY TIME (S)', 'IMMOBILITY BOUT COUNTS', 'IMMOBILITY PCT OF SESSION'] if self.min_immobility_ms is not None else []
        video_cols = ['VIDEO TOTAL FRAMES', 'VIDEO TOTAL TIME (S)']
        agg_results = pd.DataFrame(columns=freezing_cols + immobility_cols + video_cols)
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
            freezing_idx = np.argwhere((mm_s >= 0) & (mm_s <= self.movement_threshold)).astype(np.int32).flatten()
            df[f'Probability_{FREEZING}'] = 0
            df[FREEZING] = 0
            df.loc[freezing_idx, FREEZING] = 1
            df = plug_holes_shortest_bout(data_df=df, clf_name=FREEZING, fps=fps, shortest_bout=self.clean_ms)
            bouts = detect_bouts(data_df=df, target_lst=[FREEZING], fps=fps)
            bouts = bouts[bouts['Bout_time'] >= (self.min_freezing_ms / 1000)]
            if self.min_immobility_ms is not None:
                immobility_bouts = bouts[bouts['Bout_time'] >= (self.min_immobility_ms / 1000)]
                bouts = bouts[bouts['Bout_time'] < (self.min_immobility_ms / 1000)]
            if len(bouts) > 0:
                df[FREEZING] = 0
                freezing_idx = list(bouts.apply(lambda x: list(range(int(x["Start_frame"]), int(x["End_frame"]) + 1)), 1))
                freezing_idx = [x for xs in freezing_idx for x in xs]
                df.loc[freezing_idx, FREEZING] = 1
                df.loc[freezing_idx, f'Probability_{FREEZING}'] = 1
            else:
                df[FREEZING] = 0
                freezing_idx = []
            if self.min_immobility_ms is not None:
                df[f'Probability_{IMMOBILITY}'] = 0
                df[IMMOBILITY] = 0
                if len(immobility_bouts) > 0:
                    immobility_idx = list(immobility_bouts.apply(lambda x: list(range(int(x["Start_frame"]), int(x["End_frame"]) + 1)), 1))
                    immobility_idx = [x for xs in immobility_idx for x in xs]
                    df.loc[immobility_idx, IMMOBILITY] = 1
                    df.loc[immobility_idx, f'Probability_{IMMOBILITY}'] = 1
                else:
                    immobility_idx = []
            df.to_csv(save_file_path)
            agg_row = [video_name, len(freezing_idx), round(len(freezing_idx) / fps, 4), len(bouts), round((len(freezing_idx) / len(df)) * 100, 4)]
            if self.min_immobility_ms is not None:
                agg_row += [len(immobility_idx), round(len(immobility_idx) / fps, 4), len(immobility_bouts), round((len(immobility_idx) / len(df)) * 100, 4)]
            agg_row += [len(df), round(len(df)/fps, 2)]
            agg_results.loc[len(agg_results)] = agg_row
        agg_results.to_csv(agg_results_path)
        self.timer.stop_timer(); stdout_success(msg=f'Results saved in {self.save_dir} directory.', elapsed_time=self.timer.elapsed_time_str)



# FreezingDetector(
#     data_dir=r'H:\projects\brainwide_trap\brainwide_trap\project_folder\csv\outlier_corrected_movement_location',
#     config_path=r"H:\projects\brainwide_trap\brainwide_trap\project_folder\project_config.ini",
#     movement_threshold=3,
#     clean_ms=100,            # bridge jitter / drop blips shorter than this
#     min_freezing_ms=3000,    # freezing = stillness >= 3.0 s ...
#     min_immobility_ms=4500,  # ... until >= 6.0 s, where it becomes immobility (None = no immobility)
# ).run()


#
# FreezingDetector(
#     data_dir=r'G:\projects\jason_zhang\jason_project\project_folder\csv\outlier_corrected_movement_location',
#     config_path=r"G:\projects\jason_zhang\jason_project\project_folder\project_config.ini",
#     time_window=3,
#     movement_threshold=3,
#     shortest_bout=100
# ).run()



# FreezingDetector(
#     data_dir=r'F:\troubleshooting\sam\sam\project_folder\csv\outlier_corrected_movement_location',
#     config_path=r"F:\troubleshooting\sam\sam\project_folder\project_config.ini",
#     save_dir=r'F:\troubleshooting\sam\sam\project_folder\logs\freezing_500',
#     time_window=3,
#     movement_threshold=5,
#     shortest_bout=100
# ).run()


# FreezingDetector(
#     data_dir=r'E:\troubleshooting\mitra_emergence_hour\project_folder\csv\outlier_corrected_movement_location',
#     config_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini",
#     time_window=3,
#     movement_threshold=5,
#     shortest_bout=100
# ).run()