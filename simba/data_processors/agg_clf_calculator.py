__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import List, Optional, Union

import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_filepath_list_is_empty)
from simba.utils.data import detect_bouts
from simba.utils.enums import Options, TagNames
from simba.utils.errors import InvalidInputError
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import get_fn_ext, read_df
from simba.utils.warnings import NoDataFoundWarning


class AggregateClfCalculator(ConfigReader):
    """
    Compute aggregate descriptive statistics from classification data.

    :param str config_path: path to SimBA project config file in Configparser format
    :param List[str] data_measures: Aggregate statistics measures to calculate. OPTIONS: ['Bout count', 'Total event duration (s)',
        'Mean event bout duration (s)', 'Median event bout duration (s)', 'First event occurrence (s)',
        'Mean event bout interval duration (s)', 'Median event bout interval duration (s)']
    :param List[str] classifiers: Classifiers to calculate aggregate statistics for. E.g.,: ['Attack', 'Sniffing']
    :param Optional[List[str]] video_meta_data: Video metadata to include in the output. Options: 'Frame count', 'Video length (s)'.
    :param bool detailed_bout_data: If True, save detailed data for each bout in each video (start frame, end frame, bout time etc.)
    :param bool transpose: If True, then one video per row. Else, one meassure per row. Default: False.

    .. note::
       `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

       `Example expected ouput file <https://github.com/sgoldenlab/simba/blob/master/misc/detailed_bout_data_summary_20231011091832.csv>`__.

    Examples
    -----
    >>> clf_log_creator = AggregateClfCalculator(config_path="MyConfigPath", data_measures=['Bout count', 'Total event duration (s)'], classifiers=['Attack', 'Sniffing'])
    >>> clf_log_creator.run()
    >>> clf_log_creator.save()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_measures: List[
            Literal[
                "Bout count",
                "Total event duration (s)",
                "Mean event bout duration (s)",
                "Median event bout duration (s)",
                "First event occurrence (s)",
                "Mean event bout interval duration (s)",
                "Median event bout interval duration (s)",
            ]
        ],
        classifiers: List[str],
        detailed_bout_data: bool = False,
        transpose: bool = False,
        video_meta_data: Optional[
            List[Literal["Frame count", "Video length (s)"]]
        ] = None,
    ):

        super().__init__(config_path=config_path)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        (
            self.chosen_measures,
            self.classifiers,
            self.video_meta_data,
            self.detailed_bout_data,
        ) = (data_measures, classifiers, video_meta_data, detailed_bout_data)
        if (
            len(
                list(
                    set(self.chosen_measures).intersection(
                        Options.CLF_DESCRIPTIVES_OPTIONS.value
                    )
                )
            )
            == 0
        ):
            raise InvalidInputError(
                msg=f"No valid input data measures. OPTIONS: {Options.CLF_DESCRIPTIVES_OPTIONS.value}",
                source=self.__class__.__name__,
            )

        self.save_path = os.path.join(
            self.project_path, "logs", f"data_summary_{self.datetime}.csv"
        )
        self.detailed_save_path = os.path.join(
            self.project_path, "logs", f"detailed_bout_data_summary_{self.datetime}.csv"
        )
        self.transpose = transpose
        check_if_filepath_list_is_empty(
            filepaths=self.machine_results_paths,
            error_msg="SIMBA ERROR: No data files found in the project_folder/csv/machine_results directory. Run classifiers before analysing results.",
        )
        print(
            f"Analyzing {str(len(self.machine_results_paths))} file(s) for {str(len(self.clf_names))} classifiers..."
        )

    def run(self):
        self.results_df, self.bouts_df_lst = pd.DataFrame(), []
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.machine_results_paths
        )
        for file_cnt, file_path in enumerate(self.machine_results_paths):
            _, file_name, _ = get_fn_ext(file_path)
            print("Analyzing video {}...".format(file_name))
            _, _, fps = self.read_video_info(video_name=file_name)
            check_file_exist_and_readable(file_path)
            data_df = read_df(file_path, self.file_type)
            bouts_df = detect_bouts(data_df=data_df, target_lst=self.clf_names, fps=fps)
            if self.detailed_bout_data and (len(bouts_df) > 0):
                bouts_df_for_detailes = deepcopy(bouts_df)
                bouts_df_for_detailes.insert(loc=0, column="Video", value=file_name)
                self.bouts_df_lst.append(bouts_df_for_detailes)
            bouts_df["Shifted start"] = bouts_df["Start_time"].shift(-1)
            bouts_df["Interval duration"] = (
                bouts_df["Shifted start"] - bouts_df["End Time"]
            )
            for clf in self.clf_names:
                clf_results_dict = {}
                clf_data = bouts_df.loc[bouts_df["Event"] == clf]
                if len(clf_data) > 0:
                    clf_results_dict["First event occurrence (s)"] = round(
                        clf_data["Start_time"].min(), 3
                    )
                    clf_results_dict["Bout count"] = len(clf_data)
                    clf_results_dict["Total event duration (s)"] = round(
                        clf_data["Bout_time"].sum(), 3
                    )
                    clf_results_dict["Mean event bout duration (s)"] = round(
                        clf_data["Bout_time"].mean(), 3
                    )
                    clf_results_dict["Median event bout duration (s)"] = round(
                        clf_data["Bout_time"].median(), 3
                    )
                else:
                    clf_results_dict["First event occurrence (s)"] = None
                    clf_results_dict["Bout count"] = None
                    clf_results_dict["Total event duration (s)"] = None
                    clf_results_dict["Mean event bout duration (s)"] = None
                    clf_results_dict["Median event bout duration (s)"] = None
                if len(clf_data) > 1:
                    interval_df = clf_data[:-1].copy()
                    clf_results_dict["Mean event bout interval duration (s)"] = round(
                        interval_df["Interval duration"].mean(), 3
                    )
                    clf_results_dict["Median event bout interval duration (s)"] = round(
                        interval_df["Interval duration"].median(), 3
                    )
                else:
                    clf_results_dict["Mean event bout interval duration (s)"] = None
                    clf_results_dict["Median event bout interval duration (s)"] = None
                if self.video_meta_data is not None:
                    if "Frame count" in self.video_meta_data:
                        clf_results_dict["Frame count"] = len(data_df)
                    if "Video length (s)" in self.video_meta_data:
                        clf_results_dict["Video length (s)"] = round(
                            len(data_df) / fps, 3
                        )
                video_clf_pd = (
                    pd.DataFrame.from_dict(clf_results_dict, orient="index")
                    .reset_index()
                    .rename(columns={"index": "Measure", 0: "Value"})
                )
                video_clf_pd.insert(loc=0, column="Classifier", value=clf)
                video_clf_pd.insert(loc=0, column="Video", value=file_name)
                self.results_df = pd.concat([self.results_df, video_clf_pd], axis=0)

    def save(self) -> None:
        """
        Method to save classifier aggregate statistics created in :meth:`~simba.ClfLogCreator.analyze_data` to disk.
        Results are stored in the `project_folder/logs` directory of the SimBA project
        """

        self.results_df = (
            self.results_df[
                self.results_df["Measure"].isin(
                    self.chosen_measures + self.video_meta_data
                )
            ]
            .sort_values(by=["Video", "Classifier", "Measure"])
            .reset_index(drop=True)
        )
        self.results_df = self.results_df[
            self.results_df["Classifier"].isin(self.classifiers)
        ].set_index("Video")
        if not self.transpose:
            self.results_df.to_csv(self.save_path)
        else:
            self.results_df.loc[
                self.results_df["Measure"].isin(self.video_meta_data), "Classifier"
            ] = "Metadata"
            self.results_df = self.results_df.reset_index().drop_duplicates(
                subset=["Video", "Classifier", "Measure"], keep="first"
            )
            self.results_df["clf_measure"] = (
                self.results_df["Classifier"] + " - " + self.results_df["Measure"]
            )
            self.results_df = self.results_df.drop(
                ["Classifier", "Measure"], axis=1
            ).reset_index()
            self.results_df = self.results_df.pivot(
                index="Video", columns="clf_measure", values="Value"
            )
            self.results_df.to_csv(self.save_path)

        if self.detailed_bout_data:
            self.bouts_df = pd.concat(self.bouts_df_lst, axis=0)
            self.bouts_df = self.bouts_df[
                self.bouts_df["Event"].isin(self.classifiers)
            ].set_index("Video")
            if len(self.bouts_df) == 0:
                NoDataFoundWarning(
                    msg=f"No detailed bout data saved: No bouts detected for the selected classifiers: {self.classifiers}"
                )
            else:
                self.bouts_df.to_csv(self.detailed_save_path)
                stdout_success(
                    msg=f"Detailed bout data log saved at {self.detailed_save_path}",
                    source=self.__class__.__name__,
                )
        self.timer.stop_timer()
        stdout_success(
            msg=f"Data aggregate log saved at {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


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
