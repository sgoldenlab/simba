import os, glob
import pandas as pd
from simba.read_config_unit_tests import (read_project_path_and_file_type,
                                          read_config_file,
                                          read_config_entry,
                                          check_if_filepath_list_is_empty,
                                          check_file_exist_and_readable)
from simba.unsupervised.misc import bout_aggregator
from simba.train_model_functions import get_all_clf_names
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer)
from simba.features_scripts.unit_tests import read_video_info, read_video_info_csv
from simba.drop_bp_cords import getBpNames
from simba.rw_dfs import read_df
from simba.enums import Paths, ReadConfig, Dtypes
from datetime import datetime
import pickle


class DatasetCreator(object):
    def __init__(self,
                 config_path: str,
                 settings: dict):

        self.config, self.settings = read_config_file(ini_path=config_path), settings
        self.config_path, self.clf_type = config_path, None
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.save_path = os.path.join(self.logs_path, 'unsupervised_data_{}.pickle'.format(self.datetime))
        self.input_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.video_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.files_found = glob.glob(f'{self.input_dir}/*.{self.file_type}')
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        self.clf_probability_cols = ['Probability_' + x for x in self.clf_names]

        self.clf_cols = self.clf_names + self.clf_probability_cols
        self.bp_names = list(getBpNames(inifile=self.config_path))
        self.bp_names = [item for sublist in self.bp_names for item in sublist]
        check_if_filepath_list_is_empty(filepaths=self.files_found, error_msg='NO MACHINE LEARNING DATA FOUND')
        self.timer = SimbaTimer()
        self.timer.start_timer()
        print('Creating unsupervised learning dataset...')
        if settings['data_slice'] == 'ALL FEATURES (EXCLUDING POSE)':
            self.all_features_concatenator()
        elif settings['data_slice'] == 'USER-DEFINED FEATURE SET':
            self.user_defined_concatenator()
        else:
            self.all_data_concatenator()
        self.get_feature_names()
        self.clf_slicer()
        self.save()

    def all_data_concatenator(self):
        self.df = []
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path,file_type=self.file_type)
            df.insert(0, 'FRAME', df.index)
            df.insert(0, 'VIDEO', video_name)
            self.df.append(df)
        self.df = pd.concat(self.df, axis=0).reset_index(drop=True)
        self.df_bp = self.df[self.bp_names + ['FRAME', 'VIDEO']]

    def all_features_concatenator(self):
        self.df = []
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path, file_type=self.file_type)
            df.insert(0, 'FRAME', df.index)
            df.insert(0, 'VIDEO', video_name)
            self.df.append(df)
        self.df = pd.concat(self.df, axis=0).reset_index(drop=True)
        self.df_bp = self.df[self.bp_names + ['FRAME', 'VIDEO']]
        self.df = self.df.drop(self.bp_names, axis=1)

    def user_defined_concatenator(self):
        if not self.settings['feature_path']:
            raise FileNotFoundError('Select a feature file path')
        check_file_exist_and_readable(self.settings['feature_path'])
        feature_lst = list(pd.read_csv(self.settings['feature_path'], header=None)[0])
        self.df = []
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path, file_type=self.file_type, usecols=feature_lst + self.clf_cols + self.bp_names)
            df.insert(0, 'FRAME', df.index)
            df.insert(0, 'VIDEO', video_name)
            self.df.append(df)
        self.df = pd.concat(self.df, axis=0).reset_index(drop=True)
        self.df_bp = self.df[self.bp_names + ['FRAME', 'VIDEO']]
        self.df = self.df.drop(self.bp_names, axis=1)


    def clf_slicer(self):
        self.df = bout_aggregator(data=self.df,
                                  clfs=self.clf_names,
                                  video_info=self.video_info_df,
                                  feature_names=self.feature_names,
                                  min_bout_length=int(self.settings['min_bout_length']),
                                  aggregator=self.settings['bout_aggregation']).reset_index(drop=True)
        if self.settings['clf_slice'] in self.clf_names:
            self.df = self.df[self.df['CLASSIFIER'] == self.settings['clf_slice']].reset_index(drop=True)

    def get_feature_names(self):
        self.feature_names = [x for x in self.df.columns if x not in self.clf_cols]
        self.feature_names = self.feature_names[2:]

    def save(self):
        if len(self.df) == 0:
            print('SIMBA ERROR: The data contains zero frames after the chosen slice setting')
            raise ValueError()

        self.results = {}
        self.results['VIDEO_NAMES'] = self.df[['VIDEO']]
        self.results['DATA'] = self.df[self.feature_names]
        self.results['POSE'] = self.df_bp
        self.results['START_FRAME'] = self.df[['START_FRAME']]
        self.results['END_FRAME'] = self.df[['END_FRAME']]
        self.results['CLF'] = pd.get_dummies(self.df[["CLASSIFIER"]], prefix='', prefix_sep='')
        self.results['CLF_PROBABILITY'] = self.df[['PROBABILITY']]

        with open(self.save_path, 'wb') as f:
            pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: Dataset for unsupervised learning saved at {self.save_path}. The dataset contains {str(len(self.results["DATA"]))} bouts (elapsed time {self.timer.elapsed_time_str}s)')


#
# settings = {'data_slice': 'ALL FEATURES (EXCLUDING POSE)',
#             'clf_slice': 'Attack',
#             'bout_aggregation': 'MEDIAN',
#             'min_bout_length': 66,
#             'feature_path': '/Users/simon/Desktop/envs/simba_dev/simba/assets/unsupervised/features.csv'}
# #
# _ = DatasetCreator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                    settings=settings)
#
