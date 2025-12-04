__author__ = "Simon Nilsson"

import os
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_int, check_that_column_exist,
    check_valid_boolean, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.errors import NoDataError
from simba.utils.lookups import get_current_time
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    find_time_stamp_from_frame_numbers,
                                    get_fn_ext, read_df)

FIRST_OCCURRENCE = "First occurrence (s)"
EVENT_COUNT = "Event count"
TOTAL_EVENT_DURATION = "Total event duration (s)"
MEAN_EVENT_DURATION = "Mean event duration (s)"
MEDIAN_EVENT_DURATION = "Median event duration (s)"
MEAN_EVENT_INTERVAL = "Mean event interval (s)"
MEDIAN_EVENT_INTERVAL = "Median event interval (s)"
START_TIME = "START TIME"
END_TIME = "END TIME"
MEASUREMENT = 'MEASUREMENT'
CLASSIFIER = 'CLASSIFIER'
TIME_BIN_ID = 'TIME BIN #'
VIDEO = "VIDEO"
MEASUREMENT_NAMES = [FIRST_OCCURRENCE, EVENT_COUNT, TOTAL_EVENT_DURATION, MEAN_EVENT_DURATION, MEDIAN_EVENT_DURATION, MEAN_EVENT_INTERVAL, MEDIAN_EVENT_INTERVAL]

class TimeBinsClfCalculator(ConfigReader):
    """
    Computes aggregate classification results in user-defined time-bins. Results are stored in
    the ``project_folder/logs`` directory of the SimBA project.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param int bin_length: Integer representing the time bin size in seconds.
    :param List[str] classifiers: Names of classifiers to calculate aggregate statistics in time-bins for. EXAMPLE: ['Attack', 'Sniffing']
    :param Optional[Union[str, os.PathLike]] data_path: Optional path to directory containing CSV files or single CSV file. If None, uses machine results from project. Default: None.
    :param bool first_occurrence: If True, calculate first occurrence time for each classifier in each time bin. Default: False.
    :param bool event_count: If True, calculate event count for each classifier in each time bin. Default: False.
    :param bool total_event_duration: If True, calculate total event duration for each classifier in each time bin. Default: True.
    :param bool mean_event_duration: If True, calculate mean event duration for each classifier in each time bin. Default: False.
    :param bool median_event_duration: If True, calculate median event duration for each classifier in each time bin. Default: False.
    :param bool mean_interval_duration: If True, calculate mean interval duration between events for each classifier in each time bin. Default: False.
    :param bool median_interval_duration: If True, calculate median interval duration between events for each classifier in each time bin. Default: False.
    :param bool include_timestamp: If True, include START TIME and END TIME (in HH:MM:SS format) columns in output. Default: False.
    :param bool transpose: If True, transpose results with MultiIndex columns (CLASSIFIER, TIME BIN #, MEASUREMENT) so one video per row Default: False.

    .. note::
        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

    :example:
    >>> timebin_clf_analyzer = TimeBinsClfCalculator(config_path='MyConfigPath', bin_length=15, classifiers=['Attack', 'Sniffing'], event_count=True, total_event_duration=True)
    >>> timebin_clf_analyzer.run()
    >>> timebin_clf_analyzer.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bin_length: int,
                 classifiers: List[str],
                 data_path: Optional[Union[str, os.PathLike]] = None,
                 first_occurrence: bool = False,
                 event_count: bool = False,
                 total_event_duration: bool = True,
                 mean_event_duration: bool = False,
                 median_event_duration: bool = False,
                 mean_interval_duration: bool = False,
                 median_interval_duration: bool = False,
                 include_timestamp: bool = False,
                 transpose: bool = False):

        super().__init__(config_path=config_path)
        check_file_exist_and_readable(file_path=config_path)
        check_int(name=f'{self.__class__.__name__} bin_length', value=bin_length, min_value=1)
        check_valid_lst(data=classifiers, source=f'{self.__class__.__name__} classifiers', valid_dtypes=(str,), valid_values=self.clf_names, min_len=1)
        if data_path is None:
            if len(self.machine_results_paths) == 0:
                raise NoDataError(msg=f'No data files found in {self.machine_results_dir} directory. Get classification data before analyzing classification time-bin data.',source=self.__class__.__name__)
            self.data_paths = deepcopy(self.machine_results_paths)
        elif os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=('.csv',), raise_warning=False, raise_error=True, as_dict=False)
        elif os.path.isfile(data_path):
            self.data_paths = [data_path]
        for file_path in self.data_paths:
            check_file_exist_and_readable(file_path=file_path, raise_error=True)
        self.clfs, self.bin_length = classifiers, bin_length
        self.event_count, self.total_event_duration, self.first_occurrence = event_count, total_event_duration, first_occurrence
        self.mean_event_duration, self.median_event_duration, self.transpose = mean_event_duration, median_event_duration, transpose
        self.mean_interval_duration,  self.median_interval_duration, self.include_timestamp = mean_interval_duration, median_interval_duration, include_timestamp
        check_valid_boolean(value=include_timestamp, source=f'{self.__class__.__name__} include_timestamp', raise_error=True)
        check_valid_boolean(value=transpose, source=f'{self.__class__.__name__} transpose', raise_error=True)
        self.measurements = []
        for i, j in zip([first_occurrence, event_count, total_event_duration, mean_event_duration, median_event_duration, mean_interval_duration, median_interval_duration], MEASUREMENT_NAMES):
            check_valid_boolean(value=i, source=f'{self.__class__.__name__} {j}', raise_error=True)
            if i: self.measurements.append(j)

    def _reformat_results(self):
        self.out_df_lst = []
        for video_name, video_info in self.video_dict.items():
            for bin_number, bin_data in video_info.items():
                start_time, end_time = bin_data[START_TIME], bin_data[END_TIME]
                data_df = (pd.DataFrame.from_dict(bin_data).reset_index().rename(columns={"index": MEASUREMENT}))
                data_df = pd.melt(data_df, id_vars=[MEASUREMENT]).rename(columns={"value": "VALUE", "variable": CLASSIFIER})
                if self.include_timestamp and not self.transpose:
                    data_df.insert(loc=0, column=START_TIME, value=start_time)
                    data_df.insert(loc=0, column=END_TIME, value=start_time)
                data_df.insert(loc=0, column=TIME_BIN_ID, value=bin_number)
                data_df.insert(loc=0, column=VIDEO, value=video_name)
                self.out_df_lst.append(data_df)
        self.out_df = pd.concat(self.out_df_lst, axis=0).sort_values(by=[VIDEO, TIME_BIN_ID])
        self.out_df = self.out_df[self.out_df[MEASUREMENT].isin(self.measurements)]
        self.out_df = self.out_df[self.out_df[CLASSIFIER].isin(self.clfs)]
        if self.transpose:
            self.out_df["mi"] = list(zip(self.out_df[CLASSIFIER], self.out_df[TIME_BIN_ID].astype(int), self.out_df[MEASUREMENT]))
            tmp = self.out_df.pivot_table(index="VIDEO", columns="mi", values="VALUE", aggfunc="first")
            tmp.columns = pd.MultiIndex.from_tuples(tmp.columns, names=[CLASSIFIER, TIME_BIN_ID, MEASUREMENT])
            tmp = tmp.sort_index(axis=1)
            self.out_df = tmp.reset_index()
        else:
            self.out_df = self.out_df.set_index(VIDEO)

    def run(self):
        self.video_dict = {}
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            _, file_name, _ = get_fn_ext(file_path)
            self.video_dict[file_name] = {}
            print(f'Analyzing classification in time-bins ({self.bin_length}s) for video {file_name} ({file_cnt+1}/{len(self.data_paths)}; {get_current_time()})...')
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clfs, file_name=file_path)
            video_settings, px_per_mm, fps = self.read_video_info(video_name=file_name)
            bin_frame_length = max(1, int(self.bin_length * fps))
            splits = np.arange(0, data_df.shape[0], bin_frame_length)
            data_df_lst = [data_df.iloc[start: start + bin_frame_length] for start in splits]
            for bin_cnt, df in enumerate(data_df_lst):
                self.video_dict[file_name][bin_cnt] = {}
                bin_times = find_time_stamp_from_frame_numbers(start_frame=int(bin_frame_length * bin_cnt),end_frame=min(int(bin_frame_length * (bin_cnt + 1)), len(data_df)), fps=fps)
                bouts_df = detect_bouts(data_df=df, target_lst=list(self.clf_names), fps=fps)
                bouts_df["Shifted start"] = bouts_df["Start_time"].shift(-1)
                bouts_df["Interval duration"] = (bouts_df["Shifted start"] - bouts_df["End Time"])
                for clf in self.clf_names:
                    self.video_dict[file_name][bin_cnt][clf] = defaultdict(list)
                    self.video_dict[file_name][bin_cnt][START_TIME], self.video_dict[file_name][bin_cnt][END_TIME] = bin_times[0], bin_times[1]
                    bout_df = bouts_df.loc[bouts_df["Event"] == clf]
                    if len(bouts_df) > 0:
                        self.video_dict[file_name][bin_cnt][clf][FIRST_OCCURRENCE] = (round(bout_df["Start_time"].min(), 3))
                        self.video_dict[file_name][bin_cnt][clf][EVENT_COUNT] = len(bout_df)
                        self.video_dict[file_name][bin_cnt][clf][TOTAL_EVENT_DURATION] = round(bout_df["Bout_time"].sum(), 3)
                        self.video_dict[file_name][bin_cnt][clf][MEAN_EVENT_DURATION] = round(bout_df["Bout_time"].mean(), 3)
                        self.video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_DURATION] = round(bout_df["Bout_time"].median(), 3)
                    else:
                        self.video_dict[file_name][bin_cnt][clf][FIRST_OCCURRENCE] = None
                        self.video_dict[file_name][bin_cnt][clf][EVENT_COUNT] = 0
                        self.video_dict[file_name][bin_cnt][clf][TOTAL_EVENT_DURATION] = 0
                        self.video_dict[file_name][bin_cnt][clf][MEAN_EVENT_DURATION] = 0
                        self.video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_DURATION] = 0
                    if len(bouts_df) > 1:
                        self.video_dict[file_name][bin_cnt][clf][MEAN_EVENT_INTERVAL] = round(bout_df[:-1]["Interval duration"].mean(), 3)
                        self.video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_INTERVAL] = round(bout_df[:-1]["Interval duration"].median(), 3)
                    else:
                        self.video_dict[file_name][bin_cnt][clf][MEAN_EVENT_INTERVAL] = None
                        self.video_dict[file_name][bin_cnt][clf][MEDIAN_EVENT_INTERVAL] = None

        self._reformat_results()
    def save(self):
        self.save_path = os.path.join(self.project_path, "logs", f"Time_bins_ML_results_{self.datetime}.csv")
        self.out_df.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(msg=f'Classification time-bins results saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

#
# test = TimeBinsClfCalculator(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini",
#                              classifiers=['attack'],
#                              bin_length=60,
#                              include_timestamp=True,
#                              transpose=True)
#
# test.run()
# test.save()
#


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
