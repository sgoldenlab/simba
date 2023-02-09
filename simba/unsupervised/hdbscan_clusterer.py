from datetime import datetime
from simba.read_config_unit_tests import read_config_file
from simba.misc_tools import SimbaTimer
import hdbscan
import pandas as pd
import itertools
import pickle
import os, glob
import simba
import random

class HDBSCANClusterer(object):
    def __init__(self,
                 config_path: str,
                 data_path: str,
                 save_dir: str):

        self.datetime, self.save_dir = datetime.now().strftime('%Y%m%d%H%M%S'), save_dir
        self.data_paths = glob.glob(data_path + '/*.pickle')
        self.config = read_config_file(ini_path=config_path)
        model_names_dir = os.path.join(os.path.dirname(simba.__file__), 'assets', 'unsupervised', 'model_names.parquet')
        self.model_names = list(pd.read_parquet(model_names_dir)['NAMES'])
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def fit(self,
            hyper_parameters: dict):
        self.hyp = hyper_parameters
        self.search_space = list(itertools.product(*[self.hyp['alpha'],
                                                     self.hyp['min_cluster_size'],
                                                     self.hyp['min_samples'],
                                                     self.hyp['cluster_selection_epsilon']]))
        self.read_data()
        self.fit_hdbscan()

    def read_data(self):
        self.embeddings = {}
        for cnt, file_path in enumerate(self.data_paths):
            with open(file_path, 'rb') as f:
                self.embeddings[cnt] = pickle.load(f)

    def fit_hdbscan(self):
        self.model_cnt = 0
        for k, v in self.embeddings.items():
            embedding_data = v['models'].embedding_
            for h_cnt, h in enumerate(self.search_space):
                self.results = {}
                self.parameters = {'alpha': h[0],
                                   'min_cluster_size': h[1],
                                   'min_samples': h[2],
                                   'cluster_selection_epsilon': h[3]}
                self.clusterer = hdbscan.HDBSCAN(algorithm="best",
                                                 alpha=self.parameters['alpha'],
                                                 approx_min_span_tree=True,
                                                 gen_min_span_tree=True,
                                                 min_cluster_size=self.parameters['min_cluster_size'],
                                                 min_samples=self.parameters['min_samples'],
                                                 cluster_selection_epsilon=self.parameters['cluster_selection_epsilon'],
                                                 p=None,
                                                 prediction_data=True)
                self.clusterer.fit(embedding_data)
                self.results['model'] = self.clusterer
                self.results['parameters'] = self.parameters
                self.results['HASH'] = v['HASH']
                self.name = random.sample(self.model_names, 1)[0]
                self.results['NAME'] = self.name
                self.__save()
                self.model_cnt += 1

    def __save(self):
        save_path = os.path.join(self.save_dir, '{}.pickle'.format(self.name))
        with open(save_path, 'wb') as f:
            pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.timer.stop_timer()
        print('SIMBA COMPLETE: Fitted HDBSCAN models saved at {} (elapsed time {}s)'.format(save_path, self.timer.elapsed_time_str))

# hyper_parameters = {'alpha': [1.0], 'min_cluster_size': [20, 40], 'min_samples': [2], 'cluster_selection_epsilon': [20]}
# embedding_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models'
# config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
# clusterer = HDBSCANClusterer(config_path=config_path, data_path=embedding_dir, save_dir=save_dir)
# clusterer.fit(hyper_parameters=hyper_parameters)