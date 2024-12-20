import datetime
import itertools
import os
import random
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
from numba import typed

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_keys_exist_in_dict, check_int,
                                check_str, check_valid_boolean,
                                check_valid_dict)
from simba.utils.enums import UML, Options
from simba.utils.lookups import get_model_names
from simba.utils.read_write import drop_df_fields, read_pickle, write_pickle

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.train_model_mixin import TrainModelMixin
from simba.mixins.unsupervised_mixin import UMLMixin
from simba.utils.printing import SimbaTimer, stdout_success


class SimBAUmap():

    def __init__(self):
        self.datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.mdl_names = get_model_names()

    def fit(self,
            data: Union[np.ndarray, pd.DataFrame],
            n_neighbors: int,
            min_distance: float,
            spread: float,
            gpu: bool = False,
            verbose: bool = True):

        check_int(name=f'{self.__class__.__name__} n_neighbors', value=n_neighbors, min_value=1)
        check_float(name=f'{self.__class__.__name__} min_distance', value=min_distance, min_value=0.0)
        check_float(name=f'{self.__class__.__name__} spread', value=spread, min_value=min_distance)
        check_valid_boolean(value=[gpu], source=f'{self.__class__.__name__} gpu')
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        verbose = [1 if verbose else 0][0]
        mdl = UMLMixin.umap_define(n_neighbors=n_neighbors, min_distance=min_distance, spread=spread, gpu=gpu, verbose=verbose)
        return UMLMixin.umap_fit(mdl=mdl, data=data)

    def fit_grid(self,
                 data_path: Union[str, os.PathLike],
                 save_dir: [str, os.PathLike],
                 hyperparameters: dict,
                 gpu: bool = False,
                 verbose: bool = True,
                 scaler: Literal['min-max', 'standard', 'quantile'] = 'min-max',
                 variance_threshold: Optional[float] = None,
                 multicollinearity_threshold: Optional[float] = None):

        """

        :param data_path:
        :param save_dir:
        :param hyperparameters:
        :param gpu:
        :param verbose:
        :param scaler:
        :param variance_threshold:
        :param multicollinearity_threshold:
        :return: None

        :example:
        >>> hyperparameters = {"n_neighbors": (5,), "min_distance": (0.1, 0.5, 0.0), "spread": (1.0,)}
        >>> variance_threshold = 0.05
        >>> multicollinearity = 0.9999
        >>> scaler = "min-max"
        >>> embedder = SimBAUmap()
        >>> embedder.fit_grid(data_path=r"C:\troubleshooting\nastacia_unsupervised\datasets\data.pickle", save_dir=r"C:\troubleshooting\nastacia_unsupervised\embedding_data", hyperparameters=hyperparameters, gpu=False, scaler=scaler, variance_threshold=variance_threshold, multicollinearity_threshold=multicollinearity, verbose=True)
        """

        timer = SimbaTimer(start=True)
        check_file_exist_and_readable(file_path=data_path)
        check_valid_dict(x=hyperparameters, valid_key_dtypes=(str,), valid_values_dtypes=(tuple,), required_keys=UML.FIT_KEYS.value)
        check_str(name=f'{self.__class__.__name__} scaler', value=scaler.upper(), options=Options.SCALER_OPTIONS.value)
        check_valid_boolean(value=[gpu], source=f'{self.__class__.__name__} gpu')
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        data = read_pickle(data_path=data_path)
        check_if_keys_exist_in_dict(data=data, key=['FEATURES'])
        X_data = deepcopy(data['FEATURES'])

        low_variance_fields, collinear_fields = None, None
        if variance_threshold is not None:
            check_float(name=f'{self.__class__.__name__} variance_threshold', value=variance_threshold, min_value=0, max_value=1.0)
            low_variance_fields = TrainModelMixin.find_low_variance_fields(data=X_data, variance_threshold=variance_threshold)
            X_data = drop_df_fields(data=X_data, fields=low_variance_fields)


        if multicollinearity_threshold is not None:
            check_float(name=f'{self.__class__.__name__} multicollinearity_threshold', value=multicollinearity_threshold, min_value=0, max_value=1.0)
            collinear_fields = TrainModelMixin.find_highly_correlated_fields(data=X_data.values, threshold=multicollinearity_threshold, field_names=typed.List(data['FEATURES'].columns))
            X_data = drop_df_fields(data=X_data, fields=collinear_fields)

        scaler = TrainModelMixin.define_scaler(scaler_name=scaler)
        scaler = TrainModelMixin.fit_scaler(scaler=scaler, data=X_data)
        X_data_scaled = TrainModelMixin.scaler_transform(data=X_data, scaler=scaler)
        search_spaces = list(itertools.product(*[hyperparameters['n_neighbors'], hyperparameters['min_distance'], hyperparameters['spread']]))
        for search_space in search_spaces:
            mdl_name = random.sample(self.mdl_names, 1)[0]
            save_path = os.path.join(save_dir, f'{mdl_name}.pickle')
            mdl = self.fit(data=X_data_scaled, n_neighbors=search_space[0], min_distance=search_space[1], spread=search_space[2], gpu=gpu, verbose=verbose)
            mdl_lk = {UML.DR_MODEL.value: {UML.HASHED_NAME.value: mdl_name,
                                           UML.PARAMETERS.value: {UML.N_NEIGHBORS.value: search_space[0], UML.MIN_DISTANCE.value: search_space[1], UML.SPREAD.value: search_space[2]},
                                           UML.MODEL.value: mdl},
                      UML.METHODS.value: {UML.SCALER.value: scaler,
                                          UML.SCALER_TYPE.value: type(scaler),
                                          UML.MULTICOLLINEARITY_THRESHOLD.value: multicollinearity_threshold,
                                          UML.VARIANCE_THRESHOLD.value: variance_threshold},
                      UML.DATA.value: {UML.COLLINEAR_FIELDS.value: collinear_fields,
                                       UML.LOW_VARIANCE_FIELDS.value: low_variance_fields,
                                       UML.RAW.value: data,
                                       UML.SCALED_TRAIN_DATA.value: X_data_scaled,
                                       UML.UNSCALED_TRAIN_DATA.value: X_data}}
            write_pickle(data=mdl_lk, save_path=save_path)
        timer.stop_timer()
        if verbose:
            stdout_success(msg=f'{len(search_spaces)} model(s) saved in {save_dir}', elapsed_time=timer.elapsed_time_str)



# hyperparameters = {"n_neighbors": (5,), "min_distance": (0.1, 0.5, 0.0), "spread": (1.0,)}
# variance_threshold = 0.05
# multicollinearity = 0.9999
# scaler = "min-max"
# embedder = SimBAUmap()
# embedder.fit_grid(data_path=r"C:\troubleshooting\nastacia_unsupervised\datasets\data.pickle", save_dir=r"C:\troubleshooting\nastacia_unsupervised\embedding_data", hyperparameters=hyperparameters, gpu=False, scaler=scaler, variance_threshold=variance_threshold, multicollinearity_threshold=multicollinearity, verbose=True)
