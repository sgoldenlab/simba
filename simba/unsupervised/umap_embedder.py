import os
import random
from simba.enums import Paths
from simba.read_config_unit_tests import (read_config_file,
                                          check_file_exist_and_readable)
from simba.misc_tools import SimbaTimer

from simba.unsupervised.misc import (check_that_directory_is_empty,
                                     check_directory_exists,
                                     read_pickle,
                                     define_scaler,
                                     drop_low_variance_fields,
                                     scaler_transform,
                                     check_expected_fields)

from sklearn.feature_selection import VarianceThreshold
import itertools
import pickle
import pandas as pd
from datetime import datetime
import simba
try:
    from cuml import UMAP
    gpu_flag = True
except ModuleNotFoundError:
    from umap import UMAP

class UMAPEmbedder(object):
    def __init__(self,
                 config_path: str,
                 data_path: str,
                 save_dir: str):

        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.data_path = data_path
        self.config = read_config_file(ini_path=config_path)
        self.save_dir = save_dir
        check_file_exist_and_readable(file_path=self.data_path)
        check_directory_exists(directory=self.save_dir)
        check_that_directory_is_empty(directory=self.save_dir)
        model_names_dir = os.path.join(os.path.dirname(simba.__file__), Paths.UNSUPERVISED_MODEL_NAMES.value)
        self.model_names = list(pd.read_parquet(model_names_dir)['NAMES'])
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def fit(self,
            hyper_parameters: dict):
        self.hyp, self.low_var_cols = hyper_parameters, []
        self.search_space = list(itertools.product(*[self.hyp['n_neighbors'],
                                                     self.hyp['min_distance'],
                                                     self.hyp['spread']]))
        self.data = read_pickle(data_path=self.data_path)
        self.original_feature_names = self.data['DATA'].columns
        if self.hyp['variance']:
            self.find_low_variance_fields()
            self.drop_low_variance_fields()
        self.out_feature_names = [x for x in self.original_feature_names if x not in self.low_var_cols]
        self.scaler = define_scaler(scaler_name=self.hyp['scaler'])
        self.scaler.fit(self.data['DATA'])
        self.scaler_transform()
        self.fit_umap()

    def scaler_transform(self):
        if not hasattr(self, 'scaler'):
            raise KeyError('Fit or load scaler before scaler transform.')
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.data['DATA']), columns=self.out_feature_names)

    def find_low_variance_fields(self):
        feature_selector = VarianceThreshold(threshold=round((self.hyp['variance'] / 100), 2))
        feature_selector.fit(self.data['DATA'])
        self.low_var_cols = [c for c in self.data['DATA'].columns if c not in self.data['DATA'].columns[feature_selector.get_support()]]

    def drop_low_variance_fields(self):
        self.data['DATA'] = self.data['DATA'].drop(columns=self.low_var_cols)

    def fit_umap(self):
        self.model_timer = SimbaTimer()
        self.model_timer.start_timer()
        self.results = {}
        self.results['SCALER'] = self.scaler
        self.results['LOW_VARIANCE_FIELDS'] = self.low_var_cols
        self.results['ORIGINAL_FEATURE_NAMES'] = self.original_feature_names
        self.results['OUT_FEATURE_NAMES'] = self.out_feature_names
        self.results['VIDEO_NAMES'] = self.data['VIDEO_NAMES']
        self.results['START_FRAME'] = self.data['START_FRAME']
        self.results['END_FRAME'] = self.data['END_FRAME']
        self.results['POSE'] = self.data['POSE']
        self.results['CLF'] = self.data['CLF']
        self.results['CLF_PROBABILITY'] = self.data['CLF_PROBABILITY']

        for h_cnt, h in enumerate(self.search_space):
            self.h_cnt = h_cnt
            self.parameters = {'n_neighbors': h[0],
                               'min_distance': h[1],
                               'spread': h[2]}
            embedder = UMAP(min_dist=self.parameters['min_distance'],
                            n_neighbors=self.parameters['n_neighbors'],
                            spread=self.parameters['spread'],
                            metric='euclidean',
                            verbose=2)
            embedder.fit(self.scaled_data)
            self.results['PARAMETERS'] = self.parameters
            self.results['MODEL'] = embedder
            self.results['HASH'] = random.sample(self.model_names, 1)[0]
            self.__save()
        self.timer.stop_timer()
        print('SIMBA COMPLETE: {} umap models saved in {} (elapsed time: {}s)'.format(str(len(self.search_space)), self.save_dir, self.timer.elapsed_time_str))

    def __save(self):
        save_path = os.path.join(self.save_dir, '{}.pickle'.format(self.results['HASH']))
        with open(save_path, 'wb') as f:
            pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.model_timer.stop_timer()
        print('Fitted UMAP models saved at {} (elapsed time {}s)'.format(save_path, self.model_timer.elapsed_time_str))


def UMAPTransform(model_path: str,
                  data_path: str,
                  settings: dict,
                  save_dir: str or None=None):

        timer = SimbaTimer()
        timer.start_timer()
        if save_dir is not None:
            check_directory_exists(directory=save_dir)
        check_file_exist_and_readable(file_path=model_path)
        check_file_exist_and_readable(file_path=data_path)
        embedder = read_pickle(data_path=model_path)
        data = read_pickle(data_path=data_path)
        data_df = drop_low_variance_fields(data=data['DATA'], fields=embedder['LOW_VARIANCE_FIELDS'])
        check_expected_fields(data_fields=data_df.columns, expected_fields=embedder['OUT_FEATURE_NAMES'])
        scaled_data = scaler_transform(data=data_df, scaler=embedder['SCALER'])
        transformed_data = pd.DataFrame(embedder['MODEL'].transform(scaled_data), columns=['X', 'Y'])

        results = pd.concat([data['VIDEO_NAMES'],
                             data['START_FRAME'],
                             data['END_FRAME'],
                             data['CLF'],
                             data['CLF_PROBABILITY'],
                             transformed_data], axis=1)

        if settings['feature_values'] and not settings['scaled_features']:
            results = pd.concat([results, data], axis=1)
        elif settings['feature_values'] and settings['scaled_features']:
            results = pd.concat([results, scaled_data], axis=1)

        if not settings['save_format']:
            return results

        elif settings['save_format'] is 'csv':
            save_path = os.path.join(save_dir, f'transformed_{embedder["HASH"]}.csv')
            results.to_csv(save_path, index=False)
        elif settings['save_format'] is 'parquet':
            save_path = os.path.join(save_dir, f'transformed_{embedder["HASH"]}.parquet')
            results.to_parquet(save_path)
        timer.stop_timer()
        print('Transformed data saved at {} (elapsed time: {}s)'.format(save_dir, timer.elapsed_time_str))


# settings = {'feature_values': True, 'scaled_features': True, 'save_format': 'csv'}
# model_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/magical_darwin.pickle'
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230215093552.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/'
#
# _ = UMAPTransform(model_path=model_path, data_path=data_path, save_dir=save_dir, settings=settings)
# # #
#
#
#
#
#
# hyper_parameters = {'n_neighbors': [10, 20], 'min_distance': [0], 'spread': [1], 'scaler': 'MIN-MAX', 'variance': 0.25}
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230215100620.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/'
# config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
# embedder = UMAPEmbedder(config_path=config_path, data_path=data_path, save_dir=save_dir)
# embedder.fit(hyper_parameters=hyper_parameters)