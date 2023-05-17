__author__ = "Simon Nilsson"

import os
import pandas as pd
import numpy as np
from simba.unsupervised.bout_aggregator import bout_aggregator
from simba.unsupervised.enums import Unsupervised
from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.utils.read_write import read_df, get_fn_ext
from simba.utils.checks import check_if_filepath_list_is_empty, check_file_exist_and_readable
from simba.utils.errors import NoDataError
from simba.utils.printing import stdout_success

class DatasetCreator(ConfigReader, UnsupervisedMixin):

    """
    Transform raw frame-wise supervised classification data into aggregated
    data for unsupervised analyses.

    :param config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param settings: user attributes for how the data should be aggregated.

    :example:
    >>> settings = {'data_slice': 'ALL FEATURES (EXCLUDING POSE)', 'clf_slice': 'Attack', 'bout_aggregation_type': 'MEDIAN', 'min_bout_length': 66, 'feature_path': '/Users/simon/Desktop/envs/simba_dev/simba/assets/unsupervised/features.csv'}
    >>> _ = DatasetCreator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini', settings=settings)
    """

    def __init__(self,
                 config_path: str,
                 settings: dict):



        print('Creating unsupervised learning dataset...')
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        check_if_filepath_list_is_empty(filepaths=self.machine_results_paths, error_msg='NO MACHINE LEARNING DATA FOUND')
        self.settings = settings
        self.clf_type, self.feature_lst = None, None
        self.save_path = os.path.join(self.logs_path, 'unsupervised_data_{}.pickle'.format(self.datetime))
        self.log_save_path = os.path.join(self.logs_path, 'unsupervised_data_log_{}.csv'.format(self.datetime))
        self.clf_probability_cols = ['Probability_' + x for x in self.clf_names]
        self.clf_cols = self.clf_names + self.clf_probability_cols
        self.bp_names = self.get_body_part_names()
        self.bp_names = [item for sublist in self.bp_names for item in sublist]
        if settings[Unsupervised.DATA_SLICE_SELECTION.value] == Unsupervised.ALL_FEATURES_EX_POSE.value:
            self.data_concatenator(drop_bps=True, user_defined=False)
        elif settings[Unsupervised.DATA_SLICE_SELECTION.value] == Unsupervised.USER_DEFINED_SET.value:
            check_file_exist_and_readable(self.settings[Unsupervised.FEATURE_PATH.value])
            self.feature_lst = list(pd.read_csv(self.settings[Unsupervised.FEATURE_PATH.value], header=None)[0])
            self.data_concatenator(drop_bps=False, user_defined=True)
        else:
            self.data_concatenator(drop_bps=False, user_defined=False)
        self.clf_slicer()
        self.save()

    def data_concatenator(self,
                          drop_bps: bool,
                          user_defined: bool):
        print('Reading in data...')
        self.raw_x_df = []
        for file_path in self.machine_results_paths:
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path,file_type=self.file_type)
            df.insert(0, Unsupervised.FRAME.value, df.index)
            df.insert(0, Unsupervised.VIDEO.value, video_name)
            self.raw_x_df.append(df)
        self.raw_x_df = pd.concat(self.raw_x_df, axis=0).reset_index(drop=True)
        self.raw_bp_df = self.raw_x_df[[Unsupervised.VIDEO.value, Unsupervised.FRAME.value] + self.bp_names]
        self.raw_y_df = self.raw_x_df[[Unsupervised.FRAME.value, Unsupervised.VIDEO.value] + self.clf_cols]
        self.raw_x_df = self.raw_x_df.drop(self.clf_cols, axis=1)
        if drop_bps:
            self.raw_x_df = self.raw_x_df.drop(self.bp_names, axis=1)
        if user_defined:
            self.raw_x_df = self.raw_x_df[self.feature_lst + [Unsupervised.FRAME.value, Unsupervised.VIDEO.value]]
        self.feature_names = self.raw_x_df.columns[2:]


    def clf_slicer(self):
        bout_data = pd.concat([self.raw_x_df, self.raw_y_df], axis=1)
        bout_data = bout_data.loc[:, ~bout_data.columns.duplicated()].copy()
        self.bouts_x_df = bout_aggregator(data=bout_data,
                                    clfs=self.clf_names,
                                    video_info=self.video_info_df,
                                    feature_names=self.feature_names,
                                    min_bout_length=int(self.settings[Unsupervised.MIN_BOUT_LENGTH.value]),
                                    aggregator=self.settings[Unsupervised.BOUT_AGGREGATION_TYPE.value]).reset_index(drop=True)
        if self.settings[Unsupervised.CLF_SLICE_SELECTION.value] in self.clf_names:
            self.bouts_x_df = self.bouts_x_df[self.bouts_x_df[Unsupervised.CLASSIFIER.value] == self.settings[Unsupervised.CLF_SLICE_SELECTION.value]].reset_index(drop=True)
        self.bouts_y_df = self.bouts_x_df[[Unsupervised.VIDEO.value, Unsupervised.START_FRAME.value, Unsupervised.END_FRAME.value, Unsupervised.PROBABILITY.value, Unsupervised.CLASSIFIER.value]]
        self.bouts_x_df = self.bouts_x_df.drop([Unsupervised.PROBABILITY.value, Unsupervised.CLASSIFIER.value], axis=1)


    def __aggregate_dataset_stats(self):
        stats = {}
        stats['FRAME_COUNT'] = len(self.raw_x_df)
        stats['FEATURE_COUNT'] = len(self.feature_names)
        stats['BOUTS_COUNT'] = len(self.bouts_x_df)
        stats['CLASSIFIER_COUNT'] = len(self.bouts_y_df[Unsupervised.CLASSIFIER.value].unique())
        for clf in self.bouts_y_df[Unsupervised.CLASSIFIER.value].unique():
            clf_bout_df = self.bouts_y_df[self.bouts_y_df[Unsupervised.CLASSIFIER.value] == clf]
            clf_bout_df['LENGTH'] = clf_bout_df[Unsupervised.END_FRAME.value] - clf_bout_df[Unsupervised.START_FRAME.value]
            stats[f'{clf}_BOUT_COUNT'] = len(clf_bout_df)
            stats[f'{clf}_MEAN_BOUT_LENGTH (FRAMES)'] = clf_bout_df['LENGTH'].mean()
        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['VALUE'])
        stats_df.to_csv(self.log_save_path)
        stdout_success(msg=f'Log for unsupervised learning saved at {self.log_save_path}')

    def save(self):
        if len(self.bouts_x_df) == 0:
            raise NoDataError(msg='The data contains zero frames after the chosen slice setting')
        results = {}
        results['DATETIME'] = self.datetime
        results['AGGREGATION_METHOD'] = self.settings[Unsupervised.BOUT_AGGREGATION_TYPE.value]
        results['MIN_BOUT'] = self.settings[Unsupervised.MIN_BOUT_LENGTH.value]
        results['FEATURE_NAMES'] = self.feature_names
        results['FRAME_FEATURES'] = self.raw_x_df.set_index([Unsupervised.VIDEO.value, Unsupervised.FRAME.value]).astype(np.float32)
        results['FRAME_POSE'] = self.raw_bp_df.set_index([Unsupervised.VIDEO.value, Unsupervised.FRAME.value]).astype(np.float32)
        results['FRAME_TARGETS'] = self.raw_y_df.set_index([Unsupervised.VIDEO.value, Unsupervised.FRAME.value]).astype(np.float32)
        results['BOUTS_FEATURES'] = self.bouts_x_df.set_index([Unsupervised.VIDEO.value, Unsupervised.START_FRAME.value, Unsupervised.END_FRAME.value]).astype(np.float32)
        results['BOUTS_TARGETS'] = self.bouts_y_df

        self.write_pickle(data=results, save_path=self.save_path)
        self.timer.stop_timer()
        self.__aggregate_dataset_stats()
        stdout_success(msg=f'Dataset for unsupervised learning saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

# settings = {'data_slice': 'ALL FEATURES (EXCLUDING POSE)',
#             'clf_slice': 'Attack',
#             'bout_aggregation_type': 'MEDIAN',
#             'min_bout_length': 66,
#             'feature_path': '/Users/simon/Desktop/envs/simba_dev/simba/assets/unsupervised/features.csv'}
# #
# _ = DatasetCreator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                    settings=settings)

