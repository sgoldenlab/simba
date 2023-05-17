__author__ = "Simon Nilsson"

import os
import random
from copy import deepcopy
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Unsupervised
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.checks import check_file_exist_and_readable

import itertools
import pandas as pd


try:
    from cuml import UMAP
    gpu_flag = True
except ModuleNotFoundError:
    from umap import UMAP

class UmapEmbedder(UnsupervisedMixin):
    """
    Methods for grid-search UMAP model fit and transform.
    Defaults to GPU and cuml.UMAP if GPU available. If GPU unavailable, then umap.UMAP.

    :param data_path: Path holding pickled data-set created by `simba.unsupervised.dataset_creator.DatasetCreator.
    :param save_dir: Empty directory where to save the UMAP results.
    :param hyper_parameters: dict holding UMAP hyperparameters in list format.

    :Example I: Fit.
    >>> hyper_parameters = {'n_neighbors': [10, 2], 'min_distance': [1.0], 'spread': [1.0], 'scaler': 'MIN-MAX', 'variance': 0.25}
    >>> data_path = 'unsupervised/project_folder/logs/unsupervised_data_20230416145821.pickle'
    >>> save_dir = 'unsupervised/dr_models'
    >>> config_path = 'unsupervised/project_folder/project_config.ini'
    >>> embedder = UmapEmbedder(data_path=data_path, save_dir=save_dir)
    >>> embedder.fit(hyper_parameters=hyper_parameters)
    """

    def __init__(self):
        super().__init__()


    def fit(self,
            data_path: str,
            save_dir: str,
            hyper_parameters: dict):

        self.data_path = data_path
        check_file_exist_and_readable(file_path=self.data_path)
        self.data = self.read_pickle(data_path=data_path)
        self.umap_df = deepcopy(self.data[Unsupervised.BOUTS_FEATURES.value]).set_index([Unsupervised.VIDEO.value, Unsupervised.START_FRAME.value, Unsupervised.END_FRAME.value])
        self.save_dir = save_dir
        self.check_that_directory_is_empty(directory=self.save_dir)
        self.low_var_cols, self.hyper_parameters = None, hyper_parameters
        self.check_umap_hyperparameters(hyper_parameters=hyper_parameters)
        self.search_space = list(itertools.product(*[hyper_parameters[Unsupervised.N_NEIGHBORS.value],
                                                     hyper_parameters[Unsupervised.MIN_DISTANCE.value],
                                                     hyper_parameters[Unsupervised.SPREAD.value]]))
        print(f'Building {len(self.search_space)} UMAP model(s)...')
        if hyper_parameters[Unsupervised.VARIANCE.value] > 0:
            self.low_var_cols = self.find_low_variance_fields(data=self.umap_df, variance=hyper_parameters[Unsupervised.VARIANCE.value])
            self.umap_df = self.drop_fields(data=self.umap_df, fields=self.low_var_cols)
        self.scaler = self.define_scaler(scaler_name=hyper_parameters[Unsupervised.SCALER.value])
        self.scaler.fit(self.umap_df)
        self.scaled_umap_data = self.scaler_transform(data=self.umap_df, scaler=self.scaler)
        self.__create_methods_log()
        self.__fit_umaps()
        self.timer.stop_timer()
        stdout_success(msg=f'{len(self.search_space)} models saved in {self.save_dir} directory', elapsed_time=self.timer.elapsed_time_str)

    def __create_methods_log(self):
        self.methods = {}
        self.methods[Unsupervised.SCALER.value] = self.scaler
        self.methods[Unsupervised.SCALER_TYPE.value] = self.hyper_parameters[Unsupervised.SCALER.value]
        self.methods[Unsupervised.SCALED_DATA.value] = self.scaled_umap_data
        self.methods[Unsupervised.VARIANCE.value] = self.hyper_parameters[Unsupervised.VARIANCE.value]
        self.methods[Unsupervised.LOW_VARIANCE_FIELDS.value] = self.low_var_cols
        self.methods[Unsupervised.FEATURE_NAMES.value] = self.scaled_umap_data.columns

    def __fit_umaps(self):
        for cnt, h in enumerate(self.search_space):
            self.model_count = cnt
            self.model = {}
            self.model_timer = SimbaTimer()
            self.model_timer.start_timer()
            self.model[Unsupervised.HASHED_NAME.value] = random.sample(self.model_names, 1)[0]
            self.model[Unsupervised.PARAMETERS.value] = {Unsupervised.N_NEIGHBORS.value: h[0],
                                                         Unsupervised.MIN_DISTANCE.value: h[1],
                                                         Unsupervised.SPREAD.value: h[2]}
            self.model[Unsupervised.MODEL.value] = UMAP(min_dist=self.model[Unsupervised.PARAMETERS.value][Unsupervised.MIN_DISTANCE.value],
                                                                 n_neighbors=int(self.model[Unsupervised.PARAMETERS.value][Unsupervised.N_NEIGHBORS.value]),
                                                                 spread=self.model[Unsupervised.PARAMETERS.value][Unsupervised.SPREAD.value],
                                                                 metric=Unsupervised.EUCLIDEAN.value,
                                                                 verbose=2)
            self.model[Unsupervised.MODEL.value].fit(self.scaled_umap_data.values)
            results = {}
            results[Unsupervised.DATA.value] = self.data
            results[Unsupervised.METHODS.value] = self.methods
            results[Unsupervised.DR_MODEL.value] = self.model
            self.__save(data=results)

    def __save(self, data: dict) -> None:
        self.write_pickle(data=data, save_path=os.path.join(self.save_dir, f'{self.model[Unsupervised.HASHED_NAME.value]}.pickle'))
        self.model_timer.stop_timer()
        stdout_success(msg=f'Model {self.model_count+1}/{len(self.search_space)} ({self.model[Unsupervised.HASHED_NAME.value]}) saved...', elapsed_time=self.model_timer.elapsed_time)


    def transform(self,
                  data_path: str,
                  model: str or dict,
                  settings: dict,
                  save_dir: str or None=None):

        """

        :param data_path:
        :param model:
        :param settings:
        :param save_dir:
        :return:

        :Example I: Transform.
        >>> data_path = 'unsupervised/project_folder/logs/unsupervised_data_20230416145821.pickle'
        >>> save_dir = 'unsupervised/transformed_umap'
        >>> settings = {'DATA': 'RAW', 'format': 'csv'}
        >>> embedder = UmapEmbedder(data_path=data_path, save_dir=save_dir)
        >>> embedder.transform(model='unsupervised/dr_models/boring_lederberg.pickle', settings=settings)

        """

        timer = SimbaTimer(start=True)
        if isinstance(model, str):
            check_file_exist_and_readable(file_path=model)
            model = self.read_pickle(data_path=model)

        check_file_exist_and_readable(file_path=data_path)
        data = self.read_pickle(data_path=data_path)
        self.umap_df = deepcopy(data[Unsupervised.BOUTS_FEATURES.value]).set_index([Unsupervised.VIDEO.value, Unsupervised.START_FRAME.value, Unsupervised.END_FRAME.value])
        self.umap_df = self.drop_fields(data=self.umap_df, fields=model[Unsupervised.METHODS.value][Unsupervised.LOW_VARIANCE_FIELDS.value])
        self.scaled_umap_data = self.scaler_transform(data=self.umap_df, scaler=model[Unsupervised.METHODS.value][Unsupervised.SCALER.value])
        self.check_expected_fields(data_fields=list(self.scaled_umap_data.columns), expected_fields=model[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value])
        self.results = pd.DataFrame(model[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].transform(self.scaled_umap_data), columns=['X', 'Y'], index=self.umap_df.index)
        if settings[Unsupervised.DATA.value] == Unsupervised.SCALED.value:
            self.results = pd.concat([self.scaled_umap_data, self.results], axis=1)
        elif settings[Unsupervised.DATA.value] == Unsupervised.RAW.value:
            self.results = pd.concat([self.umap_df, self.results], axis=1)
        if save_dir:
            save_path = os.path.join(save_dir, f'Transformed_{model[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}.csv')
            if settings[Unsupervised.FORMAT.value] is Unsupervised.CSV.value:
                self.results.to_csv(save_path)
            timer.stop_timer()
            print(f'Transformed data saved at {save_dir} (elapsed time: {timer.elapsed_time_str}s)')

# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230416145821.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/transformed_umap'
# settings = {'DATA': 'RAW', 'format': 'csv'}
# embedder = UmapEmbedder(data_path=data_path, save_dir=save_dir)
# embedder.transform(model='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/boring_lederberg.pickle', settings=settings)




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
# data_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/logs/unsupervised_data_20230416145821.pickle'
# save_dir = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models'
# config_path = '/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini'
# embedder = UmapEmbedder(data_path=data_path, save_dir=save_dir)
# embedder.fit(hyper_parameters=hyper_parameters)