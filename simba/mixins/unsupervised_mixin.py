__author__ = "Simon Nilsson"

import os, glob
import pickle
import simba
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Union, List, Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal

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
from simba.utils.enums import Paths


class UnsupervisedMixin(object):
    """
    Methods for unsupervised ML.
    """
    def __init__(self):



        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.timer = SimbaTimer(start=True)
        model_names_dir = os.path.join(os.path.dirname(simba.__file__), Paths.UNSUPERVISED_MODEL_NAMES.value)
        self.model_names = list(pd.read_parquet(model_names_dir)[Unsupervised.NAMES.value])

    def read_pickle(self,
                    data_path: Union[str, os.PathLike]) -> dict:

        """
        Read a single or directory of pickled objects. If directory, returns dict with numerical sequential integer keys for
        each object.

        :param str data_path: Pickled file path, or directory of pickled files.
        :returns dict

        :example:
        >>> data = read_pickle(data_path='/test/unsupervised/cluster_models')
        """

        if os.path.isdir(data_path):
            data = {}
            files_found = glob.glob(data_path + '/*.pickle')
            if len(files_found) == 0:
                raise NoFilesFoundError(msg=f'SIMBA ERROR: Zero pickle files found in {data_path}.', source=self.__class__.__name__)
            for file_cnt, file_path in enumerate(files_found):
                with open(file_path, 'rb') as f:
                    data[file_cnt] = pickle.load(f)
        if os.path.isfile(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        return data

    def write_pickle(self,
                     data: dict,
                     save_path: Union[str, os.PathLike]) -> None:
        """
        Write a single object as pickle.

        :param str data_path: Pickled file path.
        :param str save_path: Location of saved pickle.

        :example:
        >>> data = test.write_pickle(data= my_model, save_path='/test/unsupervised/cluster_models/My_model.pickle')
        """

        try:
            with open(save_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e.args[0])
            raise InvalidFileTypeError(msg='Data could not be saved as a pickle.', source=self.__class__.__name__)


    def check_umap_hyperparameters(self,
                                   hyper_parameters: dict):
        """
        Checks if umap embedder hyperparameters are valid.

        :param dict hyper_parameters: Dictionary holding human hyerparameters.
        :raises InvalidInputError: If any input is invalid

        :example:
        >>> check_umap_hyperparameters(hyper_parameters={'n_neighbors': 2, 'min_distance': 0.1, 'spread': 1})
        """

        for key in Unsupervised.HYPERPARAMETERS.value:
            if key not in hyper_parameters.keys():
                raise InvalidInputError(msg=f'Hyperparameter dictionary is missing {key} entry.', source=self.__class__.__name__)
        for key in [Unsupervised.N_NEIGHBORS.value, Unsupervised.MIN_DISTANCE.value, Unsupervised.SPREAD.value]:
            if len(hyper_parameters[key]) == 0:
                raise InvalidInputError(msg=f'Hyperparameter dictionary key {key} has 0 entries.', source=self.__class__.__name__)
        if hyper_parameters[Unsupervised.SCALER.value] not in UMLOptions.SCALER_OPTIONS.value:
            raise InvalidInputError(msg=f'Scaler {Unsupervised.SCALER.value} not supported. Opitions: {UMLOptions.SCALER_OPTIONS.value}', source=self.__class__.__name__)
        check_float('VARIANCE THRESHOLD', value=hyper_parameters[Unsupervised.VARIANCE.value])

    def find_low_variance_fields(self, data: pd.DataFrame, variance: float) -> List[str]:
        """
        Finds fields with variance below provided threshold.

        :param pd.DataFrame data: Dataframe with continoues numerical features.
        :param float variance: Variance threshold (0.0-1.0).
        :return List[str]:
        """
        feature_selector = VarianceThreshold(threshold=round((variance / 100), 2))
        feature_selector.fit(data)
        low_variance_fields = [c for c in data.columns if c not in data.columns[feature_selector.get_support()]]
        if len(low_variance_fields) == len(data.columns):
            raise NoDataError(msg=f'All feature columns show a variance below the {str(variance)} threshold. Thus, no data remain for analysis.', source=self.find_low_variance_fields.__name__)
        return low_variance_fields

    def drop_fields(self, data: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        """
        Drops specified fields in dataframe.
        :param pd.DataFrame: Data in pandas format.
        :param  List[str] fields: Columns to drop.
        :return pd.DataFrame
        """
        return data.drop(columns=fields)

    def define_scaler(self, scaler_name: Literal['MIN-MAX', 'STANDARD', 'QUANTILE']) -> Union[MinMaxScaler, StandardScaler, QuantileTransformer]:
        """
        Creates sklearn scaler object. See ``UMLOptions.SCALER_OPTIONS.value`` for accepted scalers.

        :example:
        >>> define_scaler(scaler_name='MIN-MAX')
        """

        if scaler_name not in UMLOptions.SCALER_OPTIONS.value:
            raise InvalidInputError(msg=f'Scaler {scaler_name} not supported. Options: {UMLOptions.SCALER_OPTIONS.value}', source=self.__class__.__name__)
        if scaler_name == Unsupervised.MIN_MAX.value:
            return MinMaxScaler()
        elif scaler_name == Unsupervised.STANDARD.value:
            return StandardScaler()
        elif scaler_name == Unsupervised.QUANTILE.value:
            return QuantileTransformer()

    def scaler_transform(self,
                         data: pd.DataFrame,
                         scaler: Literal[MinMaxScaler or StandardScaler or QuantileTransformer]) -> pd.DataFrame:
        """
        Helper to run transform using fitted scaler.
        
        :param pd.DataFrame data: Data to transform.
        :param scaler: fitted scaler.
        """

        return pd.DataFrame(scaler.transform(data), columns=data.columns).set_index(data.index)

    def scaler_inverse_transform(self, data: pd.DataFrame, scaler=MinMaxScaler or StandardScaler or QuantileTransformer):
        return pd.DataFrame(scaler.inverse_transform(data), columns=data.columns).set_index(data.index)

    def check_expected_fields(self, data_fields: list, expected_fields: list):
        remaining_fields = [x for x in data_fields if x not in expected_fields]
        if len(remaining_fields) > 0:
            raise MissingColumnsError(f'The data contains {str(len(remaining_fields))} unexpected field(s): {str(remaining_fields)}', source=self.__class__.__name__)

    def check_key_exist_in_object(self, object: dict, key: str):
        if key not in object.keys():
            raise InvalidInputError(msg=f'{key} does not exist in object', source=self.__class__.__name__)

    def get_cluster_cnt(self,
                        data: np.array,
                        clusterer_name: str,
                        minimum_clusters: Optional[int] = 1) -> int:
        """
        Helper to check the number of unique observations in array. E.g., check the number of unique clusters.

        :param np.array data: 1D numpy array with cluster assignments.
        :param str clusterer_name: Arbitrary name of the fitted model used to generate ``data``.
        :param Optional[int] minimum_clusters: Minimum number of clusters allowed.
        :raises IntegerError: If unique cluster count is less than ``minimum_clusters``.
        """

        cnt = np.unique(data).shape[0]
        if cnt < minimum_clusters:
            raise IntegerError(msg=f'Clustrer {clusterer_name} has {str(cnt)} clusters, but {str(minimum_clusters)} clusters is required for the operation.', source=self.__class__.__name__)
        return cnt

    def check_that_directory_is_empty(self, directory: Union[str, os.PathLike]) -> None:
        """
        Checks if a directory is empty

        :param str directory: Directory to check.
        :raises DirectoryNotEmptyError: If ``directory`` contains files.
        """

        try:
            all_files_in_folder = [f for f in next(os.walk(directory))[2] if not f[0] == '.']
        except StopIteration:
            return 0
        else:
            if len(all_files_in_folder) > 0:
                raise DirectoryNotEmptyError(msg=f'The {directory} is not empty and contains {str(len(all_files_in_folder))} files. Use a directory that is empty.', source=self.__class__.__name__)