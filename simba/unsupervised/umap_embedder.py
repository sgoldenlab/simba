import os
import random
from simba.enums import Paths
from simba.read_config_unit_tests import check_file_exist_and_readable
from simba.misc_tools import SimbaTimer

from simba.unsupervised.misc import (check_that_directory_is_empty,
                                     check_directory_exists,
                                     read_pickle,
                                     write_pickle,
                                     define_scaler,
                                     find_low_variance_fields,
                                     drop_low_variance_fields,
                                     scaler_transform,
                                     check_expected_fields)

import itertools
import pandas as pd
from datetime import datetime
import simba
try:
    from cuml import UMAP
    gpu_flag = True
except ModuleNotFoundError:
    from umap import UMAP

class UMAPGridSearch(object):
    def __init__(self,
                 data_path: str,
                 save_dir: str):

        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.data_path = data_path
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
            self.low_var_cols = find_low_variance_fields(data=self.data['DATA'], variance=self.hyp['variance'])
            self.data['DATA'] = drop_low_variance_fields(data=self.data['DATA'], fields=self.low_var_cols)

        self.out_feature_names = [x for x in self.original_feature_names if x not in self.low_var_cols]
        self.scaler = define_scaler(scaler_name=self.hyp['scaler'])
        self.scaler.fit(self.data['DATA'])
        self.scaled_data = scaler_transform(data=self.data['DATA'],scaler=self.scaler)
        self.__fit_umap()

    def __fit_umap(self):
        self.model_timer = SimbaTimer()
        self.model_timer.start_timer()
        self.results = {}
        self.results['SCALER'] = self.scaler
        self.results['LOW_VARIANCE_FIELDS'] = self.low_var_cols
        self.results['ORIGINAL_FEATURE_NAMES'] = self.original_feature_names
        self.results['OUT_FEATURE_NAMES'] = self.out_feature_names
        self.results['VIDEO NAMES'] = self.data['VIDEO_NAMES']
        self.results['START FRAME'] = self.data['START_FRAME']
        self.results['END FRAME'] = self.data['END_FRAME']
        self.results['POSE'] = self.data['POSE']
        self.results['DATA'] = self.scaler
        self.results['CLASSIFIER'] = self.data['CLF']
        self.results['CLASSIFIER PROBABILITY'] = self.data['CLF_PROBABILITY']

        for h_cnt, h in enumerate(self.search_space):
            self.h_cnt = h_cnt
            self.parameters = {'n_neighbors': h[0],
                               'min_distance': h[1],
                               'spread': h[2]}
            embedder = UMAP(min_dist=self.parameters['min_distance'],
                            n_neighbors=int(self.parameters['n_neighbors']),
                            spread=self.parameters['spread'],
                            metric='euclidean',
                            verbose=2)
            embedder.fit(self.scaled_data.values)
            self.results['PARAMETERS'] = self.parameters
            self.results['MODEL'] = embedder
            self.results['TYPE'] = 'UMAP'
            self.results['HASH'] = random.sample(self.model_names, 1)[0]
            write_pickle(data=self.results, save_path=os.path.join(self.save_dir, '{}.pickle'.format(self.results['HASH'])))
        self.timer.stop_timer()
        print('SIMBA COMPLETE: {} umap models saved in {} (elapsed time: {}s)'.format(str(len(self.search_space)), self.save_dir, self.timer.elapsed_time_str))

def UMAPTransform(model: str or object,
                  data_path: str,
                  settings: dict or None=None,
                  save_dir: str or None=None):

        timer = SimbaTimer()
        timer.start_timer()
        if save_dir is not None:
            check_directory_exists(directory=save_dir)
        if type(model) == 'str':
            check_file_exist_and_readable(file_path=model)
            embedder = read_pickle(data_path=model)
        else:
            embedder = model
        check_file_exist_and_readable(file_path=data_path)
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
        if settings:
            if settings['features'] == 'INCLUDE: ORIGINAL':
                results = pd.concat([results, data_df], axis=1)
            if settings['features'] == 'INCLUDE: SCALED':
                results = pd.concat([results, scaled_data], axis=1)
            if settings['save_format'] is 'csv':
                save_path = os.path.join(save_dir, f'transformed_{embedder["HASH"]}.csv')
                results.to_csv(save_path, index=False)

        else:
            return transformed_data

        timer.stop_timer()
        print('Transformed data saved at {} (elapsed time: {}s)'.format(save_dir, timer.elapsed_time_str))


# settings = {'features': 'INCLUDE: SCALED', 'save': 'csv}
# model_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/funny_heisenberg.pickle'
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230222150701.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/'
# _ = UMAPTransform(model_path=model_path, data_path=data_path, save_dir=save_dir, settings=settings)

#
#
#
#
#
# hyper_parameters = {'n_neighbors': [10, 2], 'min_distance': [1.0], 'spread': [1.0], 'scaler': 'MIN-MAX', 'variance': 0.25}
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230222134410.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
# config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
# embedder = UMAPEmbedder(data_path=data_path, save_dir=save_dir)
# embedder.fit(hyper_parameters=hyper_parameters)