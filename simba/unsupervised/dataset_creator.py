__author__ = "Simon Nilsson"

import os
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.bout_aggregator import bout_aggregator
from simba.unsupervised.enums import Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_filepath_list_is_empty,
                                check_instance)
from simba.utils.errors import NoDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_pickle


class DatasetCreator(ConfigReader, UnsupervisedMixin):
    """
    Transform raw frame-wise supervised classification data into aggregated
    data for unsupervised analyses. Saves the aggergated data in to logs directory of the SimBa project.

    :param Union[str, os.PathLike] config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param Dict[str, Any] settings: Attributes for which data should be included and how the data should be aggregated.

    :example:
    >>> settings = {'data_slice': 'ALL FEATURES (EXCLUDING POSE)', 'clf_slice': 'Attack', 'bout_aggregation_type': 'MEDIAN', 'min_bout_length': 66, 'feature_path': '/Users/simon/Desktop/envs/simba_dev/simba/assets/unsupervised/features.csv'}
    >>> db_creator = DatasetCreator(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini', settings=settings)
    >>> db_creator.run()
    """

    def __init__(self, config_path: Union[str, os.PathLike], settings: Dict[str, Any]):
        check_file_exist_and_readable(file_path=config_path)
        check_instance(
            source=f"{self.__class__.__name__} settings",
            instance=settings,
            accepted_types=(dict,),
        )
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        print(
            f"Creating unsupervised learning dataset from {len(self.machine_results_paths)} data files..."
        )
        check_if_filepath_list_is_empty(
            filepaths=self.machine_results_paths,
            error_msg="NO MACHINE LEARNING DATA FOUND in project_folder/csv/machine_results directory",
        )
        self.settings = settings
        self.clf_type, self.feature_lst = None, None
        self.save_path = os.path.join(
            self.logs_path, f"unsupervised_data_{self.datetime}.pickle"
        )
        self.log_save_path = os.path.join(
            self.logs_path, f"unsupervised_data_log_{self.datetime}.csv"
        )
        self.clf_probability_cols = [f"Probability_{x}" for x in self.clf_names]
        self.clf_cols = self.clf_names + self.clf_probability_cols

    def run(self):
        if (
            self.settings[Unsupervised.DATA_SLICE_SELECTION.value]
            == Unsupervised.ALL_FEATURES_EX_POSE.value
        ):
            self._run(drop_bps=True, user_defined=False)
        elif (
            self.settings[Unsupervised.DATA_SLICE_SELECTION.value]
            == Unsupervised.USER_DEFINED_SET.value
        ):
            self._run(drop_bps=False, user_defined=True)
        else:
            self._run(drop_bps=False, user_defined=False)

    def _read_files(self):
        raw_x_df = []
        for cnt, file_path in enumerate(self.machine_results_paths):
            _, video_name, _ = get_fn_ext(filepath=file_path)
            print(
                f"Reading {video_name} (File {cnt + 1}/{len(self.machine_results_paths)})..."
            )
            df = read_df(file_path=file_path, file_type=self.file_type)
            df.insert(0, Unsupervised.FRAME.value, df.index)
            df.insert(0, Unsupervised.VIDEO.value, video_name)
            raw_x_df.append(df)
        return pd.concat(raw_x_df, axis=0).reset_index(drop=True)

    def _run(self, drop_bps: bool, user_defined: bool):
        timer = SimbaTimer(start=True)
        self.raw_x_df = self._read_files()
        self.raw_bp_df = self.raw_x_df[
            [Unsupervised.VIDEO.value, Unsupervised.FRAME.value] + self.bp_col_names
        ]
        self.raw_y_df = self.raw_x_df[
            [Unsupervised.FRAME.value, Unsupervised.VIDEO.value] + self.clf_cols
        ]
        self.raw_x_df = self.raw_x_df.drop(self.clf_cols, axis=1)
        if drop_bps:
            self.raw_x_df = self.raw_x_df.drop(self.bp_col_names, axis=1)
        if user_defined:
            self.raw_x_df = self.raw_x_df[
                self.feature_lst + [Unsupervised.FRAME.value, Unsupervised.VIDEO.value]
            ]
        self.feature_names = self.raw_x_df.columns[2:]
        timer.stop_timer()
        stdout_success(
            msg="Reading data complete!", elapsed_time=timer.elapsed_time_str
        )
        self._clf_slicer()
        self.save()

    def _clf_slicer(self):
        bout_data = pd.concat([self.raw_x_df, self.raw_y_df], axis=1)
        bout_data = bout_data.loc[:, ~bout_data.columns.duplicated()].copy()
        self.bouts_x_df = bout_aggregator(
            data=bout_data,
            clfs=self.clf_names,
            video_info=self.video_info_df,
            feature_names=self.feature_names,
            min_bout_length=int(self.settings[Unsupervised.MIN_BOUT_LENGTH.value]),
            aggregator=self.settings[Unsupervised.BOUT_AGGREGATION_TYPE.value],
        )
        if self.settings[Unsupervised.CLF_SLICE_SELECTION.value] in self.clf_names:
            self.bouts_x_df = self.bouts_x_df[
                self.bouts_x_df[Unsupervised.CLASSIFIER.value]
                == self.settings[Unsupervised.CLF_SLICE_SELECTION.value]
            ].reset_index(drop=True)
        self.bouts_y_df = self.bouts_x_df[
            [
                Unsupervised.VIDEO.value,
                Unsupervised.START_FRAME.value,
                Unsupervised.END_FRAME.value,
                Unsupervised.PROBABILITY.value,
                Unsupervised.CLASSIFIER.value,
            ]
        ]
        self.bouts_x_df = self.bouts_x_df.drop(
            [Unsupervised.PROBABILITY.value, Unsupervised.CLASSIFIER.value], axis=1
        )

    def __aggregate_dataset_stats(self):
        stats = {}
        stats["CREATION DATE"] = self.datetime
        stats["FRAME_COUNT"] = len(self.raw_x_df)
        stats["FEATURE_COUNT"] = len(self.feature_names)
        stats["BOUTS_COUNT"] = len(self.bouts_x_df)
        stats["CLASSIFIER_COUNT"] = len(
            self.bouts_y_df[Unsupervised.CLASSIFIER.value].unique()
        )
        for clf in self.bouts_y_df[Unsupervised.CLASSIFIER.value].unique():
            clf_bout_df = self.bouts_y_df[
                self.bouts_y_df[Unsupervised.CLASSIFIER.value] == clf
            ]
            clf_bout_df["LENGTH"] = (
                clf_bout_df[Unsupervised.END_FRAME.value]
                - clf_bout_df[Unsupervised.START_FRAME.value]
            )
            stats[f"{clf}_BOUT_COUNT"] = len(clf_bout_df)
            stats[f"{clf}_MEAN_BOUT_LENGTH (FRAMES)"] = clf_bout_df["LENGTH"].mean()
        stats_df = pd.DataFrame.from_dict(stats, orient="index", columns=["VALUE"])
        stats_df.to_csv(self.log_save_path)
        stdout_success(
            msg=f"Log for unsupervised learning dataset saved at {self.log_save_path}"
        )

    def save(self):
        print("Saving SimBA unsupervised dataset...")
        if len(self.bouts_x_df) == 0:
            raise NoDataError(
                msg="The data contains zero frames after the selected slice setting",
                source=self.__class__.__name__,
            )
        results = {}
        results["DATETIME"] = self.datetime
        results[Unsupervised.BOUT_AGGREGATION_TYPE.name] = self.settings[
            Unsupervised.BOUT_AGGREGATION_TYPE.value
        ]
        results[Unsupervised.MIN_BOUT_LENGTH.name] = self.settings[
            Unsupervised.MIN_BOUT_LENGTH.value
        ]
        results["FEATURE_NAMES"] = self.feature_names
        results["FRAME_FEATURES"] = self.raw_x_df.set_index(
            [Unsupervised.VIDEO.value, Unsupervised.FRAME.value]
        ).astype(np.float32)
        results["FRAME_POSE"] = self.raw_bp_df.set_index(
            [Unsupervised.VIDEO.value, Unsupervised.FRAME.value]
        ).astype(np.int64)
        results["FRAME_TARGETS"] = self.raw_y_df.set_index(
            [Unsupervised.VIDEO.value, Unsupervised.FRAME.value]
        ).astype(np.int8)
        results["BOUTS_FEATURES"] = self.bouts_x_df.set_index(
            [
                Unsupervised.VIDEO.value,
                Unsupervised.START_FRAME.value,
                Unsupervised.END_FRAME.value,
            ]
        ).astype(np.float32)
        results["BOUTS_TARGETS"] = self.bouts_y_df
        write_pickle(data=results, save_path=self.save_path)
        self.timer.stop_timer()
        self.__aggregate_dataset_stats()
        stdout_success(
            msg=f"Dataset for unsupervised learning saved at {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# settings = {"data_slice": "ALL FEATURES (EXCLUDING POSE)", "clf_slice": "Attack", "bout_aggregation_type": "MEDIAN", "min_bout_length": 66, "feature_path": "/Users/simon/Desktop/envs/simba_dev/simba/assets/unsupervised/features.csv",}
# db_creator = DatasetCreator(config_path="/Users/simon/Desktop/envs/simba/troubleshooting/unsupervised/project_folder/project_config.ini", settings=settings)
# db_creator.run()


# settings = {"data_slice": "ALL FEATURES (EXCLUDING POSE)", "clf_slice": "ALL CLASSIFIERS (6)", "bout_aggregation_type": "MEDIAN", "min_bout_length": 66, "feature_path": "/Users/simon/Desktop/envs/simba_dev/simba/assets/unsupervised/features.csv",}
# db_creator = DatasetCreator(config_path="/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini", settings=settings)
# db_creator.run()
