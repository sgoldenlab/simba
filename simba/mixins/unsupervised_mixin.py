__author__ = "Simon Nilsson"

import os, glob
import pickle
import simba
from simba.utils.enums import Paths
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
from datetime import datetime
import numpy as np

from sklearn.preprocessing import (MinMaxScaler,
                                   StandardScaler,
                                   QuantileTransformer)
from simba.utils.errors import (InvalidInputError,
                                NoDataError,
                                NoFilesFoundError,
                                InvalidFileTypeError,
                                MissingColumnsError,
                                IntegerError,
                                DirectoryNotEmptyError)
from simba.utils.checks import check_float
from simba.utils.printing import SimbaTimer
from simba.unsupervised.enums import Unsupervised, UMLOptions


class UnsupervisedMixin(object):
    def __init__(self):

        """
        Methods for unsupervised ML.
        """

        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.timer = SimbaTimer(start=True)
        model_names_dir = os.path.join(os.path.dirname(simba.__file__), Paths.UNSUPERVISED_MODEL_NAMES.value)
        self.model_names = list(pd.read_parquet(model_names_dir)[Unsupervised.NAMES.value])

    def read_pickle(self,
                    data_path: str) -> dict:

        if os.path.isdir(data_path):
            data = {}
            files_found = glob.glob(data_path + '/*.pickle')
            if len(files_found) == 0:
                raise NoFilesFoundError(msg=f'SIMBA ERROR: Zero pickle files found in {data_path}.')
            for file_cnt, file_path in enumerate(files_found):
                with open(file_path, 'rb') as f:
                    data[file_cnt] = pickle.load(f)
        if os.path.isfile(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def write_pickle(self, data: dict, save_path: str):
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e.args[0])
            raise InvalidFileTypeError(msg='Data could not be saved as a pickle.')


    def check_umap_hyperparameters(self,
                                   hyper_parameters):
        for key in Unsupervised.HYPERPARAMETERS.value:
            if key not in hyper_parameters.keys():
                raise InvalidInputError(msg=f'Hyperparameter dictionary is missing {key} entry.')
        for key in [Unsupervised.N_NEIGHBORS.value, Unsupervised.MIN_DISTANCE.value, Unsupervised.SPREAD.value]:
            if len(hyper_parameters[key]) == 0:
                raise InvalidInputError(msg=f'Hyperparameter dictionary key {key} has 0 entries.')
        if hyper_parameters[Unsupervised.SCALER.value] not in UMLOptions.SCALER_OPTIONS.value:
            raise InvalidInputError(msg=f'Scaler {Unsupervised.SCALER.value} not supported. Opitions: {UMLOptions.SCALER_OPTIONS.value}')
        check_float('VARIANCE THRESHOLD', value=hyper_parameters[Unsupervised.VARIANCE.value])

    def find_low_variance_fields(self, data: pd.DataFrame, variance: float):
        feature_selector = VarianceThreshold(threshold=round((variance / 100), 2))
        feature_selector.fit(data)
        low_variance_fields = [c for c in data.columns if c not in data.columns[feature_selector.get_support()]]
        if len(low_variance_fields) == len(data.columns):
            raise NoDataError(msg=f'All feature columns show a variance below the {str(variance)} threshold. Thus, no data remain for analysis.')
        return low_variance_fields

    def drop_fields(self, data: pd.DataFrame, fields: list):
        return data.drop(columns=fields)

    def define_scaler(self, scaler_name: str):
        if scaler_name not in UMLOptions.SCALER_OPTIONS.value:
            raise InvalidInputError(msg=f'Scaler {scaler_name} not supported. Options: {UMLOptions.SCALER_OPTIONS.value}')
        if scaler_name == Unsupervised.MIN_MAX.value:
            return MinMaxScaler()
        elif scaler_name == Unsupervised.STANDARD.value:
            return StandardScaler()
        elif scaler_name == Unsupervised.QUANTILE.value:
            return QuantileTransformer()

    def scaler_transform(self, data: pd.DataFrame, scaler: MinMaxScaler or StandardScaler or QuantileTransformer):
        return pd.DataFrame(scaler.transform(data), columns=data.columns).set_index(data.index)

    def scaler_inverse_transform(self, data: pd.DataFrame, scaler=MinMaxScaler or StandardScaler or QuantileTransformer):
        return pd.DataFrame(scaler.inverse_transform(data), columns=data.columns).set_index(data.index)

    def check_expected_fields(self, data_fields: list, expected_fields: list):
        remaining_fields = [x for x in data_fields if x not in expected_fields]
        if len(remaining_fields) > 0:
            raise MissingColumnsError(f'The data contains {str(len(remaining_fields))} unexpected field(s): {str(remaining_fields)}')

    def check_key_exist_in_object(self, object: dict, key: str):
        if key not in object.keys():
            raise InvalidInputError(msg=f'{key} does not exist in object')

    def get_cluster_cnt(self,
                        data: np.array,
                        clusterer_name: str,
                        minimum_clusters: int = 1) -> int:
        cnt = np.unique(data).shape[0]
        if cnt < minimum_clusters:
            raise IntegerError(msg=f'Clustrer {clusterer_name} has {str(cnt)} clusters, but {str(minimum_clusters)} clusters is required for the operation.')
        return cnt

    def check_that_directory_is_empty(self, directory: str) -> None:
        try:
            all_files_in_folder = [f for f in next(os.walk(directory))[2] if not f[0] == '.']
        except StopIteration:
            return 0
        else:
            if len(all_files_in_folder) > 0:
                raise DirectoryNotEmptyError(msg=f'The {directory} is not empty and contains {str(len(all_files_in_folder))} files. Use a directory that is empty.')
