from datetime import datetime
from simba.read_config_unit_tests import check_if_filepath_list_is_empty
from simba.misc_tools import SimbaTimer, check_file_exist_and_readable
from simba.unsupervised.misc import (check_directory_exists,
                                     check_that_directory_is_empty,
                                     read_pickle,
                                     write_pickle)
from simba.unsupervised.umap_embedder import UMAPTransform
from simba.enums import Paths
import pandas as pd
import itertools
import os, glob
import simba
import random
from copy import deepcopy
try:
    from cuml.cluster.hdbscan import HDBSCAN
    from cuml.cluster import hdbscan
    gpu_flag = True
except ModuleNotFoundError:
    from hdbscan import HDBSCAN
    import hdbscan


class HDBSCANClusterer(object):
    def __init__(self,
                 data_path: str,
                 save_dir: str):

        self.datetime, self.save_dir, self.data_path = datetime.now().strftime('%Y%m%d%H%M%S'), save_dir, data_path
        check_directory_exists(directory=data_path)
        check_directory_exists(directory=save_dir)
        check_that_directory_is_empty(directory=save_dir)
        self.data_paths = glob.glob(data_path + '/*.pickle')
        check_if_filepath_list_is_empty(filepaths=self.data_paths,
                                        error_msg=f'SIMBA ERROR: No pickle files in {data_path}')
        model_names_dir = os.path.join(os.path.dirname(simba.__file__), Paths.UNSUPERVISED_MODEL_NAMES.value)
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
        self.embeddings = read_pickle(self.data_path)
        self.model_cnt = str(len(self.search_space) * len(self.embeddings.keys()))
        print(f'Creating {self.model_cnt } HDBSCAN models...')
        self.fit_hdbscan()
        self.timer.stop_timer()
        print(f'SIMBA COMLETE: {self.model_cnt} saved in {self.save_dir} (elapsed time {self.timer.elapsed_time_str}s)')

    def fit_hdbscan(self):
        for k, v in self.embeddings.items():
            self.fit_timer = SimbaTimer()
            self.fit_timer.start_timer()
            embedding_data = v['MODEL'].embedding_
            for h_cnt, h in enumerate(self.search_space):
                self.results = {}
                self.parameters = {'alpha': h[0],
                                   'min_cluster_size': h[1],
                                   'min_samples': h[2],
                                   'cluster_selection_epsilon': h[3]}
                self.clusterer = HDBSCAN(algorithm="best",
                                         alpha=self.parameters['alpha'],
                                         approx_min_span_tree=True,
                                         gen_min_span_tree=True,
                                         min_cluster_size=self.parameters['min_cluster_size'],
                                         min_samples=self.parameters['min_samples'],
                                         cluster_selection_epsilon=self.parameters['cluster_selection_epsilon'],
                                         p=None,
                                         prediction_data=True)
                self.clusterer.fit(embedding_data)
                self.results['EMBEDDER'] = v
                self.results['MODEL'] = self.clusterer
                self.results['PARAMETERS'] = self.parameters
                self.results['HASH'] = v['HASH']
                self.name = random.sample(self.model_names, 1)[0]
                self.results['NAME'] = self.name
                self.__save()

    def __save(self):
        save_path = os.path.join(self.save_dir, '{}.pickle'.format(self.name))
        write_pickle(data=self.results, save_path=save_path)
        self.fit_timer.stop_timer()
        print(f'Fitted HDBSCAN models {self.name} (elapsed time {self.fit_timer.elapsed_time_str}s)...')


def HDBSCANTransform(clusterer_model_path: str,
                     data_path: str,
                     save_dir: str,
                     settings: dict):

    timer = SimbaTimer()
    timer.start_timer()
    check_directory_exists(directory=save_dir)
    check_file_exist_and_readable(file_path=clusterer_model_path)
    model = read_pickle(data_path=clusterer_model_path)

    results = {}
    embedding_data = UMAPTransform(model=model['EMBEDDER'], data_path=data_path, save_dir=None, settings=None)
    transform_labels, transform_strength = hdbscan.approximate_predict(model['MODEL'], embedding_data[['X', 'Y']].values)
    embedding_data['CLUSTER'] = transform_labels
    embedding_data['CLUSTER_STRENGTHS'] = transform_strength
    results['DATA'] = embedding_data
    results['POSE'] = model['EMBEDDER']['POSE']
    save_path = os.path.join(save_dir, f'{model["HASH"]}_{model["NAME"]}.pickle')
    write_pickle(data=results, save_path=save_path)
    timer.stop_timer()
    print(f'SIMBA COMPLETE: TRANSFORMED HDBSCAN results saved at {save_path} (elapsed time: {timer.elapsed_time_str}s)')

# hyper_parameters = {'alpha': [1.0, 0.5], 'min_cluster_size': [2, 10], 'min_samples': [1], 'cluster_selection_epsilon': [20, 10, 5]}
# embedding_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models'
# config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
# clusterer = HDBSCANClusterer(data_path=embedding_dir, save_dir=save_dir)
# clusterer.fit(hyper_parameters=hyper_parameters)
#

# settings = {'feature_values': True, 'scaled_features': True, 'save_format': 'csv'}
# clusterer_model_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/amazing_burnell.pickle'
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230215093552.pickle'
# save_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
#
# _ = HDBSCANTransform(clusterer_model_path=clusterer_model_path,
#                      data_path=data_path,
#                      save_dir=save_path,
#                      settings=settings)




