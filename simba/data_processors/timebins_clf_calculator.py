__author__ = "Simon Nilsson"

import glob
import os
from collections import defaultdict
from typing import List, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_filepath_list_is_empty, check_int,
    check_that_column_exist, check_valid_boolean, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import Options
from simba.utils.errors import (InvalidInputError, NoChoosenMeasurementError,
                                NoDataError)
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df

FIRST_OCCURRENCE = "First occurrence (s)"
EVENT_COUNT = "Event count"
TOTAL_EVENT_DURATION = "Total event duration (s)"
MEAN_EVENT_DURATION = "Mean event duration (s)"
MEDIAN_EVENT_DURATION = "Median event duration (s)"
MEAN_EVENT_INTERVAL = "Mean event interval (s)"
MEDIAN_EVENT_INTERVAL = "Median event interval (s)"
MEASUREMENT_NAMES = [FIRST_OCCURRENCE, EVENT_COUNT, TOTAL_EVENT_DURATION, MEAN_EVENT_DURATION, MEDIAN_EVENT_DURATION, MEAN_EVENT_INTERVAL, MEDIAN_EVENT_INTERVAL]

class TimeBinsClfCalculator(ConfigReader):
    """
    Computes aggregate classification results in user-defined time-bins. Results are stored in
    the ``project_folder/logs`` directory of the SimBA project`

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter int bin_length: Integer representing the time bin size in seconds
    :parameter List[str] measurements: Aggregate statistic measures calculated for each time bin. OPTIONS: ['First occurrence (s)', 'Event count',
        Total event duration (s)', 'Mean event duration (s)', 'Median event duration (s)', 'Mean event interval (s)',
        'Median event interval (s)']
    :parameter List[str] classifiers: Names of classifiers to calculate aggregate statistics in time-bins for. EXAMPLE: ['Attack', 'Sniffing']

    .. note::
    `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

    Example
    ----------
    >>> timebin_clf_analyzer = TimeBinsClfCalculator(config_path='MyConfigPath', bin_length=15, measurements=['Event count', 'Total event duration (s)'])
    >>> timebin_clf_analyzer.run()

    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bin_length: int,
                 classifiers: List[str],
                 first_occurrence: bool = True,
                 event_count: bool = True,
                 total_event_duration: bool = True,
                 mean_event_duration: bool = True,
                 median_event_duration: bool = True,
                 mean_interval_duration: bool = True,
                 median_interval_duration: bool = True):

        super().__init__(config_path=config_path)
        check_file_exist_and_readable(file_path=config_path)
        check_int(name=f'{self.__class__.__name__} bin_length', value=bin_length, min_value=1)
        check_valid_lst(data=classifiers, source=f'{self.__class__.__name__} classifiers', valid_dtypes=(str,), valid_values=self.clf_names, min_len=1)
        if len(self.machine_results_paths) == 0:
            raise NoDataError(msg=f'No data files found in {self.machine_results_dir} directory. Get classification data before analyzing classification time-bin data.', source=self.__class__.__name__)
        self.clfs, self.bin_length = classifiers, bin_length
        self.first_occurrence = first_occurrence
        self.event_count = event_count
        self.total_event_duration = total_event_duration
        self.mean_event_duration = mean_event_duration
        self.median_event_duration = median_event_duration
        self.mean_interval_duration = mean_interval_duration
        self.median_interval_duration = median_interval_duration
        self.measurements = []
        for i, j in zip([first_occurrence, event_count, total_event_duration, mean_event_duration, median_event_duration, mean_interval_duration, median_interval_duration], MEASUREMENT_NAMES):
            check_valid_boolean(value=i, source=f'{self.__class__.__name__} {j}', raise_error=True)
            if i: self.measurements.append(j)




    def run(self):
        video_dict = {}

        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.machine_results_paths)
        for file_cnt, file_path in enumerate(self.machine_results_paths):
            _, file_name, _ = get_fn_ext(file_path)
            video_dict[file_name] = {}
            print(f'Analyzing classification in time-bins ({self.bin_length}s) for video {file_name} ({file_cnt+1}/{len(self.machine_results_paths)})')
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clfs, file_name=file_path)
            video_settings, px_per_mm, fps = self.read_video_info(video_name=file_name)
            bin_frame_length = max(1, int(self.bin_length * fps))
            splits = np.arange(0, data_df.shape[0], bin_frame_length)
            data_df_lst = [data_df.iloc[start: start + bin_frame_length] for start in splits]
            for bin_cnt, df in enumerate(data_df_lst):
                video_dict[file_name][bin_cnt] = {}
                bouts_df = detect_bouts(data_df=df, target_lst=list(self.clf_names), fps=fps)
                bouts_df["Shifted start"] = bouts_df["Start_time"].shift(-1)
                bouts_df["Interval duration"] = (bouts_df["Shifted start"] - bouts_df["End Time"])
                for clf in self.clf_names:
                    video_dict[file_name][bin_cnt][clf] = defaultdict(list)
                    bout_df = bouts_df.loc[bouts_df["Event"] == clf]
                    if len(bouts_df) > 0:
                        video_dict[file_name][bin_cnt][clf][FIRST_OCCURRENCE] = (round(bout_df["Start_time"].min(), 3))
                        video_dict[file_name][bin_cnt][clf][EVENT_COUNT] = len(bout_df)
                        video_dict[file_name][bin_cnt][clf][TOTAL_EVENT_DURATION] = round(bout_df["Bout_time"].sum(), 3)
                        video_dict[file_name][bin_cnt][clf][MEAN_EVENT_DURATION] = round(bout_df["Bout_time"].mean(), 3)
                        video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_DURATION] = round(bout_df["Bout_time"].median(), 3)
                    else:
                        video_dict[file_name][bin_cnt][clf][FIRST_OCCURRENCE] = None
                        video_dict[file_name][bin_cnt][clf][EVENT_COUNT] = 0
                        video_dict[file_name][bin_cnt][clf][TOTAL_EVENT_DURATION] = 0
                        video_dict[file_name][bin_cnt][clf][MEAN_EVENT_DURATION] = 0
                        video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_DURATION] = 0
                    if len(bouts_df) > 1:
                        video_dict[file_name][bin_cnt][clf][MEAN_EVENT_INTERVAL] = round(bout_df[:-1]["Interval duration"].mean(), 3)
                        video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_INTERVAL] = round(bout_df[:-1]["Interval duration"].median(), 3)
                    else:
                        video_dict[file_name][bin_cnt][clf][MEAN_EVENT_INTERVAL] = None
                        video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_INTERVAL] = None

        print('Saving results...')
        self.out_df_lst = []
        for video_name, video_info in video_dict.items():
            for bin_number, bin_data in video_info.items():
                data_df = (pd.DataFrame.from_dict(bin_data).reset_index().rename(columns={"index": "MEASUREMENT"}))
                data_df = pd.melt(data_df, id_vars=["MEASUREMENT"]).rename(columns={"value": "VALUE", "variable": "CLASSIFIER"})
                data_df.insert(loc=0, column="TIME BIN #", value=bin_number)
                data_df.insert(loc=0, column="VIDEO", value=video_name)
                self.out_df_lst.append(data_df)

        out_df = pd.concat(self.out_df_lst, axis=0).sort_values(by=["VIDEO", "TIME BIN #"]).set_index("VIDEO")
        out_df = out_df[out_df["MEASUREMENT"].isin(self.measurements)]
        out_df = out_df[out_df["CLASSIFIER"].isin(self.clfs)]
        self.save_path = os.path.join(self.project_path, "logs", f"Time_bins_ML_results_{self.datetime}.csv")
        out_df.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(msg=f'Classification time-bins results saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

# test = TimeBinsClfCalculator(config_path=r'D:\troubleshooting\mitra\project_folder\project_config.ini',
#                              classifiers=['lay-on-belly'],
#                              bin_length=60)
#
# test.run()



# test = TimeBinsClf(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                    bin_length=2,
#                    measurements=['First occurrence (s)', 'Event count', 'Total event duration (s)', 'Mean event duration (s)'],
#                    classifiers=['Attack', 'Sniffing'])
# test.analyze_timebins_clf()


# test = TimeBinsClf(config_path='/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/project_config.ini',
#                    bin_length=2,
#                    measurements=['First occurrence (s)', 'Event count', 'Total event duration (s)', 'Mean event duration (s)'])
# test.analyze_timebins_clf()
