import os
import itertools
import datetime
import numpy as np
from typing import Union, Tuple
from simba.utils.checks import check_if_dir_exists, check_valid_tuple
from simba.utils.read_write import find_files_of_filetypes_in_directory, read_pickle, check_if_keys_exist_in_dict, check_valid_array, check_int, check_float, check_valid_boolean, write_pickle
from simba.utils.enums import UML, Formats
from simba.utils.lookups import get_model_names
from simba.mixins.unsupervised_mixin import UMLMixin
from simba.utils.printing import SimbaTimer, stdout_success
import random

class SimBAHdbscan():

    def __init__(self):
        self.datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.mdl_names = get_model_names()


    def fit(self,
            data: np.ndarray,
            alpha: float,
            min_cluster_size: int,
            min_samples: int,
            cluster_selection_epsilon: float,
            gpu: bool = False,
            verbose: bool = True):

        check_valid_array(data=data, source=f'{self.__class__.__name__} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_int(name=f'{self.__class__.__name__} min_cluster_size', value=min_cluster_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} min_samples', value=min_samples, min_value=1)
        check_float(name=f'{self.__class__.__name__} alpha', value=alpha, min_value=0.0)
        check_float(name=f'{self.__class__.__name__} cluster_selection_epsilon', value=cluster_selection_epsilon, min_value=0.0)
        check_valid_boolean(value=[gpu], source=f'{self.__class__.__name__} gpu')
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        mdl = UMLMixin.hdbscan_define(alpha=alpha, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, min_cluster_size=min_cluster_size, verbose=verbose, gpu=gpu)
        return UMLMixin.hdbscan_fit(mdl=mdl, data=data)

    def fit_grid(self,
                 data_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 alpha: Tuple[float, ...],
                 min_cluster_size: Tuple[int, ...],
                 min_samples: Tuple[int, ...],
                 cluster_selection_epsilon: Tuple[float, ...],
                 gpu: bool = False,
                 verbose: bool = True):

        """

        :param data_path:
        :param save_dir:
        :param alpha:
        :param min_cluster_size:
        :param min_samples:
        :param cluster_selection_epsilon:
        :param gpu:
        :param verbose:
        :return:

        :example:
        >>> alpha = (1.0,)
        >>> min_cluster_size = (15,)
        >>> min_samples = (1,)
        >>> cluster_selection_epsilon = (1.0, 0.5)
        >>> clusterer = SimBAHdbscan()
        >>> clusterer.fit_grid(data_path=r'C:\troubleshooting\nastacia_unsupervised\embedding_data', save_dir=r"C:\troubleshooting\nastacia_unsupervised\cluster_data", alpha=alpha, min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
        """

        timer = SimbaTimer(start=True)
        if os.path.isdir(data_path):
            file_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.pkl', '.pickle'], raise_error=True)
        else:
            file_paths = [data_path]
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)
        check_valid_tuple(x=alpha, source=f'{self.__class__.__name__} alpha', valid_dtypes=(float,), minimum_length=1)
        check_valid_tuple(x=min_cluster_size, source=f'{self.__class__.__name__} min_cluster_size', valid_dtypes=(int,), minimum_length=1)
        check_valid_tuple(x=min_samples, source=f'{self.__class__.__name__} min_samples', valid_dtypes=(int,), minimum_length=1)
        check_valid_tuple(x=cluster_selection_epsilon, source=f'{self.__class__.__name__} cluster_selection_epsilon', valid_dtypes=(float,), minimum_length=1)
        search_spaces = list(itertools.product(*[alpha, min_cluster_size, min_samples, cluster_selection_epsilon]))
        for file_path in file_paths:
            data = read_pickle(data_path=file_path)
            check_if_keys_exist_in_dict(data=data, key=UML.DR_MODEL.value)
            embedding = data[UML.DR_MODEL.value][UML.MODEL.value].embedding_
            for search_space in search_spaces:
                mdl_name = random.sample(self.mdl_names, 1)[0]
                save_path = os.path.join(save_dir, f'{mdl_name}.pickle')
                mdl = self.fit(data=embedding, alpha=search_space[0], min_cluster_size=search_space[1], min_samples=search_space[2], cluster_selection_epsilon=search_space[3], gpu=gpu, verbose=verbose)
                data[UML.CLUSTER_MODEL.value] = {UML.HASHED_NAME.value: mdl_name,
                                                 UML.PARAMETERS.value: {UML.ALPHA.value: search_space[0], UML.MIN_CLUSTER_SIZE.value: search_space[1], UML.MIN_SAMPLES.value:search_space[1], UML.EPSILON.value: search_space[3]},
                                                 UML.MODEL.value: mdl}
                write_pickle(data=data, save_path=save_path)
        timer.stop_timer()
        if verbose:
            stdout_success(msg=f'{len(search_spaces)*len(file_paths)} model(s) saved in {save_dir}',elapsed_time=timer.elapsed_time_str)

# alpha = (1.0,)
# min_cluster_size = (15,)
# min_samples = (1,)
# cluster_selection_epsilon = (1.0, 0.5)
#
# clusterer = SimBAHdbscan()
# clusterer.fit_grid(data_path=r'C:\troubleshooting\nastacia_unsupervised\embedding_data', save_dir=r"C:\troubleshooting\nastacia_unsupervised\cluster_data", alpha=alpha, min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
#
