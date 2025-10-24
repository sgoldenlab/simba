__author__ = "Simon Nilsson"

import argparse
import os
import sys
from copy import deepcopy
from typing import List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_dir_exists, check_valid_boolean,
    check_valid_dataframe, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import TagNames
from simba.utils.errors import NoChoosenMeasurementError
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df)
from simba.utils.warnings import NoDataFoundWarning

FIRST_OCCURRENCE = "First occurrence (s)"
EVENT_COUNT = "Event count"
TOTAL_EVENT_DURATION = "Total event duration (s)"
MEAN_EVENT_DURATION = "Mean event duration (s)"
MEDIAN_EVENT_DURATION = "Median event duration (s)"
MEAN_EVENT_INTERVAL = "Mean event interval (s)"
MEDIAN_EVENT_INTERVAL = "Median event interval (s)"
FRAME_COUNT = 'Frame count'
VIDEO_LENGTH = "Video length (s)"
MEASUREMENT_NAMES = [FIRST_OCCURRENCE, EVENT_COUNT, TOTAL_EVENT_DURATION, MEAN_EVENT_DURATION, MEDIAN_EVENT_DURATION, MEAN_EVENT_INTERVAL, MEDIAN_EVENT_INTERVAL]

class AggregateClfCalculator(ConfigReader):
    """
    Compute aggregate descriptive statistics from classification data.

    This class analyzes machine learning classifier results to calculate various descriptive
    statistics such as bout counts, durations, intervals, and first occurrences for each
    classifier in each video. Results can be saved in detailed or summary formats.

    .. note::
       `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.
       `Example expected output file <https://github.com/sgoldenlab/simba/blob/master/misc/detailed_bout_data_summary_20231011091832.csv>`__.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param List[str] classifiers: List of classifier names to calculate aggregate statistics for. Must be valid classifier names from the project.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory containing the machine results CSV files. If None, uses ``project_folder/csv/machine_results``.
    :param bool detailed_bout_data: If True, saves detailed bout data (start frame, end frame, bout time, etc.) for each bout in each video. Default: False.
    :param bool transpose: If True, creates output with one video per row. If False, one measurement per row. Default: False.
    :param bool first_occurrence: If True, calculates first occurrence time for each classifier. Default: True.
    :param bool event_count: If True, calculates total number of bouts for each classifier. Default: True.
    :param bool total_event_duration: If True, calculates total duration of all bouts for each classifier. Default: True.
    :param bool mean_event_duration: If True, calculates mean duration of bouts for each classifier. Default: True.
    :param bool median_event_duration: If True, calculates median duration of bouts for each classifier. Default: True.
    :param bool mean_interval_duration: If True, calculates mean interval between bouts for each classifier. Default: True.
    :param bool median_interval_duration: If True, calculates median interval between bouts for each classifier. Default: True.
    :param bool frame_count: If True, includes total frame count in the output. Default: False.
    :param bool video_length: If True, includes video length in seconds in the output. Default: False.

    :raises NoChoosenMeasurementError: If no measurement types are selected (all measurement booleans are False).

    :example:
    >>> clf_calculator = AggregateClfCalculator(
    ...     config_path="project_folder/project_config.ini",
    ...     classifiers=['Attack', 'Sniffing'],
    ...     detailed_bout_data=True,
    ...     transpose=True
    ... )
    >>> clf_calculator.run()
    >>> clf_calculator.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 classifiers: List[str],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 detailed_bout_data: bool = False,
                 transpose: bool = False,
                 first_occurrence: bool = True,
                 event_count: bool = True,
                 total_event_duration: bool = True,
                 mean_event_duration: bool = True,
                 median_event_duration: bool = True,
                 mean_interval_duration: bool = True,
                 median_interval_duration: bool = True,
                 frame_count: bool = False,
                 video_length: bool = False):

        super().__init__(config_path=config_path)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        check_valid_lst(data=classifiers, source=f'{self.__class__.__name__} classifiers', min_len=1, valid_dtypes=(str,), valid_values=self.clf_names)
        if data_dir is None:
            data_dir = self.machine_results_dir
        else:
            check_if_dir_exists(in_dir=data_dir)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f'.{self.file_type}'], raise_error=True)

        self.measurements = []
        for i, j  in zip([first_occurrence, event_count, total_event_duration, mean_event_duration, median_event_duration, mean_interval_duration, median_interval_duration], MEASUREMENT_NAMES):
            check_valid_boolean(value=i, source=f'{self.__class__.__name__} {j}', raise_error=True)
            if i: self.measurements.append(j)

        check_valid_boolean(value=frame_count, source=f'{self.__class__.__name__} frame_count', raise_error=True)
        check_valid_boolean(value=video_length, source=f'{self.__class__.__name__} video_length', raise_error=True)
        check_valid_boolean(value=transpose, source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=detailed_bout_data, source=f'{self.__class__.__name__} detailed_bout_data', raise_error=True)
        if not any([first_occurrence, event_count, total_event_duration, mean_event_duration, median_event_duration, mean_interval_duration, median_interval_duration]):
            raise NoChoosenMeasurementError(source=self.__class__.__name__)

        self.classifiers, self.detailed_bout_data, self.transpose = classifiers, detailed_bout_data, transpose
        self.first_occurrence = first_occurrence
        self.event_count = event_count
        self.total_event_duration = total_event_duration
        self.mean_event_duration = mean_event_duration
        self.median_event_duration = median_event_duration
        self.mean_interval_duration = mean_interval_duration
        self.median_interval_duration = median_interval_duration
        self.frame_count = frame_count
        self.video_length = video_length
        self.save_path = os.path.join(self.logs_path, f"data_summary_{self.datetime}.csv")
        self.detailed_save_path = os.path.join(self.logs_path, f"detailed_bout_data_summary_{self.datetime}.csv")

    def run(self):
        self.results_df, self.bouts_df_lst = pd.DataFrame(), []
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.machine_results_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            _, file_name, _ = get_fn_ext(file_path)
            print(f"Analyzing classifier descriptive statistics for video {file_name} ({file_cnt+1}/{len(self.machine_results_paths)})...")
            _, _, fps = self.read_video_info(video_name=file_name)
            check_file_exist_and_readable(file_path)
            data_df = read_df(file_path, self.file_type)
            check_valid_dataframe(df=data_df, required_fields=self.classifiers, source=file_path)
            bouts_df = detect_bouts(data_df=data_df, target_lst=self.classifiers, fps=fps)
            if self.detailed_bout_data and (len(bouts_df) > 0):
                bouts_df_for_detailes = deepcopy(bouts_df)
                bouts_df_for_detailes.insert(loc=0, column="Video", value=file_name)
                self.bouts_df_lst.append(bouts_df_for_detailes)
            bouts_df["Shifted start"] = bouts_df["Start_time"].shift(-1)
            bouts_df["Interval duration"] = (bouts_df["Shifted start"] - bouts_df["End Time"])
            for clf in self.classifiers:
                clf_results_dict = {}
                clf_data = bouts_df.loc[bouts_df["Event"] == clf]
                if len(clf_data) > 0:
                    clf_results_dict[FIRST_OCCURRENCE] = round(clf_data["Start_time"].min(), 3)
                    clf_results_dict[EVENT_COUNT] = len(clf_data)
                    clf_results_dict[TOTAL_EVENT_DURATION] = round(clf_data["Bout_time"].sum(), 3)
                    clf_results_dict[MEAN_EVENT_DURATION] = round(clf_data["Bout_time"].mean(), 3)
                    clf_results_dict[MEDIAN_EVENT_DURATION] = round(clf_data["Bout_time"].median(), 3)
                else:
                    clf_results_dict[FIRST_OCCURRENCE] = None
                    clf_results_dict[EVENT_COUNT] = 0
                    clf_results_dict[TOTAL_EVENT_DURATION] = 0
                    clf_results_dict[MEAN_EVENT_DURATION] = None
                    clf_results_dict[MEDIAN_EVENT_DURATION] = None
                if len(clf_data) > 1:
                    interval_df = clf_data[:-1].copy()
                    clf_results_dict[MEAN_EVENT_INTERVAL] = round(interval_df["Interval duration"].mean(), 3)
                    clf_results_dict[MEDIAN_EVENT_INTERVAL] = round(interval_df["Interval duration"].median(), 3)
                else:
                    clf_results_dict[MEAN_EVENT_INTERVAL] = None
                    clf_results_dict[MEDIAN_EVENT_INTERVAL] = None
                if self.frame_count:
                        clf_results_dict[FRAME_COUNT] = len(data_df)
                if self.video_length:
                    clf_results_dict["Video length (s)"] = round(len(data_df) / fps, 3)
                video_clf_pd = (pd.DataFrame.from_dict(clf_results_dict, orient="index").reset_index().rename(columns={"index": "MEASUREMENT", 0: "VALUE"}))
                video_clf_pd.insert(loc=0, column="CLASSIFIER", value=clf)
                video_clf_pd.insert(loc=0, column="VIDEO", value=file_name)
                self.results_df = pd.concat([self.results_df, video_clf_pd], axis=0)

    def save(self) -> None:
        """
        Method to save classifier aggregate statistics created in :meth:`~simba.ClfLogCreator.analyze_data` to disk.
        Results are stored in the `project_folder/logs` directory of the SimBA project
        """

        self.results_df = self.results_df[self.results_df["CLASSIFIER"].isin(self.classifiers)].set_index("VIDEO")
        self.results_df = self.results_df[self.results_df["MEASUREMENT"].isin(self.measurements + [FRAME_COUNT, VIDEO_LENGTH])]
        if not self.transpose:
            self.results_df.to_csv(self.save_path)
        else:
            self.results_df.loc[self.results_df["MEASUREMENT"].isin([FRAME_COUNT, VIDEO_LENGTH]), "CLASSIFIER"] = "Metadata"
            self.results_df = self.results_df.reset_index().drop_duplicates(subset=["VIDEO", "CLASSIFIER", "MEASUREMENT"], keep="first")
            self.results_df["clf_measure"] = (self.results_df["CLASSIFIER"] + " - " + self.results_df["MEASUREMENT"])
            self.results_df = self.results_df.drop(["CLASSIFIER", "MEASUREMENT"], axis=1).reset_index()
            self.results_df = self.results_df.pivot(index="VIDEO", columns="clf_measure", values="VALUE")
            self.results_df.to_csv(self.save_path)

        if self.detailed_bout_data:
            self.bouts_df = pd.concat(self.bouts_df_lst, axis=0)
            self.bouts_df = self.bouts_df[self.bouts_df["Event"].isin(self.classifiers)].set_index("Video")
            if len(self.bouts_df) == 0:
                NoDataFoundWarning(msg=f"No detailed bout data saved: No bouts detected for the selected classifiers: {self.classifiers}")
            else:
                self.bouts_df.to_csv(self.detailed_save_path)
                stdout_success(msg=f"Detailed bout data log saved at {self.detailed_save_path}", source=self.__class__.__name__)
        self.timer.stop_timer()
        stdout_success(msg=f"Data aggregate log saved at {self.save_path}", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description='Compute aggregate descriptive statistics from classification data.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to SimBA project config file')
    parser.add_argument('--classifiers', type=str, nargs='+', required=True, help='List of classifier names to analyze')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing machine results CSV files (default: project machine_results directory)')
    parser.add_argument('--detailed_bout_data', action='store_true', help='Save detailed bout data for each bout')
    parser.add_argument('--transpose', action='store_true', help='Create output with one video per row')
    parser.add_argument('--no_first_occurrence', action='store_true', help='Disable first occurrence calculation')
    parser.add_argument('--no_event_count', action='store_true', help='Disable event count calculation')
    parser.add_argument('--no_total_event_duration', action='store_true', help='Disable total event duration calculation')
    parser.add_argument('--no_mean_event_duration', action='store_true', help='Disable mean event duration calculation')
    parser.add_argument('--no_median_event_duration', action='store_true', help='Disable median event duration calculation')
    parser.add_argument('--no_mean_interval_duration', action='store_true', help='Disable mean interval duration calculation')
    parser.add_argument('--no_median_interval_duration', action='store_true', help='Disable median interval duration calculation')
    parser.add_argument('--frame_count', action='store_true', help='Include frame count in output')
    parser.add_argument('--video_length', action='store_true', help='Include video length in output')
    
    args = parser.parse_args()
    
    clf_calculator = AggregateClfCalculator(
        config_path=args.config_path,
        classifiers=args.classifiers,
        data_dir=args.data_dir,
        detailed_bout_data=args.detailed_bout_data,
        transpose=args.transpose,
        first_occurrence=not args.no_first_occurrence,
        event_count=not args.no_event_count,
        total_event_duration=not args.no_total_event_duration,
        mean_event_duration=not args.no_mean_event_duration,
        median_event_duration=not args.no_median_event_duration,
        mean_interval_duration=not args.no_mean_interval_duration,
        median_interval_duration=not args.no_median_interval_duration,
        frame_count=args.frame_count,
        video_length=args.video_length
    )
    clf_calculator.run()
    clf_calculator.save()





# test = AggregateClfCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                               classifiers=['straub_tail'],
#                               transpose=True,
#                               mean_event_duration = False,
#                               median_event_duration = False,
#                               mean_interval_duration = False,
#                               median_interval_duration = False)
# test.run()
# test.save()




# test = AggregateClfCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                               data_measures=["Bout count", "Total event duration (s)", "Mean event bout duration (s)", "Median event bout duration (s)", "First event occurrence (s)", "Mean event bout interval duration (s)", "Median event bout interval duration (s)"],
#                               classifiers=['straub_tail'],
#                               video_meta_data = ['Frame count', "Video length (s)"],#
#                               transpose=True)
# test.run()
# test.save()


# test = AggregateClfCalculator(config_path=r"/Users/simon/Desktop/envs/troubleshooting/raph/project_folder/project_config.ini",
#                               data_measures=['Total event duration (s)', 'Median event bout duration (s)'],
#                               classifiers=['walking'],
#                               video_meta_data =['Frame count'],
#                               transpose=True)
#
#
#
# test.run()
# test.save()