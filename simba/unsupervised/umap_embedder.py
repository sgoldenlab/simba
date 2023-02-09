import os.path
import random
from simba.read_config_unit_tests import read_config_file
from simba.misc_tools import SimbaTimer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
import umap
import itertools
import pickle
import pandas as pd
from datetime import datetime
import simba

class UMAPEmbedder(object):
    def __init__(self,
                 config_path: str,
                 data_path: str,
                 save_dir: str):

        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.data_path = data_path
        self.config = read_config_file(ini_path=config_path)
        self.save_dir = save_dir
        model_names_dir = os.path.join(os.path.dirname(simba.__file__), 'assets', 'unsupervised', 'model_names.parquet')
        self.model_names = list(pd.read_parquet(model_names_dir)['NAMES'])
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def fit(self,
            hyper_parameters: dict):
        self.hyp, self.low_var_cols = hyper_parameters, []
        self.search_space = list(itertools.product(*[self.hyp['n_neighbors'],
                                                     self.hyp['min_distance']]))
        self.read_data()
        self.fit_scaler()
        self.scaler_transform()
        if self.hyp['variance']:
            self.find_low_variance_fields()
            self.drop_low_variance_fields()
        self.fit_umap()

    def read_data(self):
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.feature_names = self.data['DATA'].columns

    def fit_scaler(self):
        if self.hyp['scaler'] == 'MIN-MAX':
            self.scaler = MinMaxScaler()
        elif self.hyp['scaler'] == 'STANDARD':
            self.scaler = StandardScaler()
        elif self.hyp['scaler'] == 'QUANTILE':
            self.scaler = QuantileTransformer()
        self.scaler.fit(self.data['DATA'])

    def fit_standard_scaler(self):

        self.scaler.fit(self.data['DATA'])

    def scaler_transform(self):
        if not hasattr(self, 'scaler'):
            raise KeyError('Fit or load scaler before scaler transform.')
        self.scaled_data = pd.DataFrame(self.scaler.transform(self.data['DATA']), columns=self.feature_names)

    def find_low_variance_fields(self):
        feature_selector = VarianceThreshold(threshold=round((self.hyp['variance'] / 100), 2))
        feature_selector.fit(self.scaled_data)
        self.low_var_cols = [c for c in self.data['DATA'].columns if c not in self.data['DATA'].columns[feature_selector.get_support()]]

    def drop_low_variance_fields(self):
        self.scaled_data = self.scaled_data.drop(columns=self.low_var_cols)

    def fit_umap(self):
        self.model_timer = SimbaTimer()
        self.model_timer.start_timer()
        self.results = {}
        self.results['scaler'] = self.scaler
        self.results['low_var_cols'] = self.low_var_cols
        self.results['VIDEO_NAMES'] = self.data['VIDEO_NAMES']
        self.results['FRAME_IDS'] = self.data['FRAME_IDS']
        self.results['CLF'] = self.data['CLF']
        self.results['CLF_PROBABILITY'] = self.data['CLF_PROBABILITY']

        for h_cnt, h in enumerate(self.search_space):
            self.h_cnt = h_cnt
            self.parameters = {'n_neighbors': h[0],
                               'min_distance': h[1]}
            self.scaled_data = self.scaled_data.head(100)
            embedder = umap.UMAP(min_dist=self.parameters['min_distance'],
                                 n_neighbors=self.parameters['n_neighbors'],
                                 metric='euclidean',
                                 verbose=2)
            embedder.fit(self.scaled_data)
            self.results['parameters'] = self.parameters
            self.results['models'] = embedder
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


# hyper_parameters = {'n_neighbors': [10, 20], 'min_distance': [0], 'scaler': 'MIN-MAX', 'variance': 0.25}
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/input/unsupervised_data_20230208135227.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/'
# config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
# embedder = UMAPEmbedder(config_path=config_path, data_path=data_path, save_dir=save_dir)
# embedder.fit(hyper_parameters=hyper_parameters)