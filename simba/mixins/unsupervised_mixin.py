__author__ = "Simon Nilsson"
from typing import Union
import pandas as pd
import numpy as np
import datetime
from simba.data_processors.cuda.utils import _cuda_available
from simba.utils.checks import check_float, check_int, check_valid_boolean, check_instance, check_valid_array
from simba.utils.errors import SimBAGPUError
from simba.utils.data import get_library_version
from simba.utils.lookups import get_model_names
from simba.utils.enums import Formats
try:
    import cuml.umap as cuml_umap
    import cuml.cluster.hdbscan as cuml_hdbscan
except ModuleNotFoundError:
    import umap as cuml_umap
    import hdbscan as cuml_hdbscan
    pass
import umap
import hdbscan

class UMLMixin(object):
    def __init__(self):
        self.datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.mdl_names = get_model_names()


    @staticmethod
    def umap_define(n_neighbors: int,
                    min_distance: float,
                    spread: float,
                    gpu: bool = False,
                    verbose: int = 1):

        check_int(name=f'umap_define n_neighbors', value=n_neighbors, min_value=1)
        check_float(name=f'umap_define min_distance', value=min_distance, min_value=0.0)
        check_float(name=f'umap_define spread', value=spread, min_value=min_distance)
        check_valid_boolean(value=[gpu], source=f'umap_define gpu')
        check_valid_boolean(value=[verbose], source=f'{UMLMixin.umap_fit.__name__} verbose')
        if gpu and not _cuda_available()[0]:
            raise SimBAGPUError(msg='No GPU detected and GPU as True passed', source=UMLMixin.umap_define.__name__)
        if gpu and not get_library_version(library_name='cuml'):
            raise SimBAGPUError(msg='cuML library not detected and GPU as True passed', source=UMLMixin.umap_define.__name__)
        if gpu:
            return cuml_umap.UMAP(min_dist=min_distance, n_neighbors=n_neighbors, spread=spread, metric='euclidean', verbose=verbose)
        else:
            return umap.UMAP(min_dist=min_distance, n_neighbors=n_neighbors, spread=spread, metric='euclidean', verbose=verbose)

    @staticmethod
    def umap_fit(mdl: Union[umap.UMAP, cuml_umap.UMAP], data: Union[np.ndarray, pd.DataFrame]) -> Union[umap.UMAP, cuml_umap.UMAP]:
        check_instance(source=f'{UMLMixin.umap_fit.__name__} mdl', instance=mdl, accepted_types=(umap.UMAP, cuml_umap.UMAP,))
        check_instance(source=f'{UMLMixin.umap_fit.__name__} data', instance=data, accepted_types=(pd.DataFrame, np.ndarray,))
        if isinstance(data, pd.DataFrame):
            data = data.values
        check_valid_array(data=data, source=f'{UMLMixin.umap_fit.__name__} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        return mdl.fit(data)


    @staticmethod
    def hdbscan_define(alpha: float,
                       min_cluster_size: int,
                       min_samples: int,
                       cluster_selection_epsilon: float,
                       gpu: bool = False,
                       verbose: int = 1) -> Union[hdbscan.HDBSCAN, cuml_hdbscan.HDBSCAN]:

        check_int(name=f'hdbscan_define min_cluster_size', value=min_cluster_size, min_value=1)
        check_int(name=f'hdbscan_define min_samples', value=min_samples, min_value=1)
        check_float(name=f'hdbscan_define alpha', value=alpha, min_value=0.0)
        check_float(name=f'hdbscan_define cluster_selection_epsilon', value=cluster_selection_epsilon, min_value=0.0)
        check_valid_boolean(value=[gpu], source=f'hdbscan_define gpu')
        check_valid_boolean(value=[verbose], source=f'hdbscan_define verbose')
        if gpu and not _cuda_available()[0]:
            raise SimBAGPUError(msg='No GPU detected and GPU as True passed', source=UMLMixin.hdbscan_define.__name__)
        if gpu and not get_library_version(library_name='cuml'):
            raise SimBAGPUError(msg='cuML library not detected and GPU as True passed', source=UMLMixin.hdbscan_define.__name__)
        if not gpu:
            return hdbscan.HDBSCAN(algorithm="best", alpha=alpha, approx_min_span_tree=True, gen_min_span_tree=True, min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, p=None, prediction_data=True)
        else:
            return cuml_hdbscan.HDBSCAN(algorithm="best", alpha=alpha, approx_min_span_tree=True, gen_min_span_tree=True, min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, p=None, prediction_data=True)

    @staticmethod
    def hdbscan_fit(mdl: Union[hdbscan.HDBSCAN, cuml_hdbscan.HDBSCAN], data: Union[np.ndarray, pd.DataFrame]) -> object:
        check_instance(source=f'{UMLMixin.umap_fit.__name__} mdl', instance=mdl, accepted_types=(hdbscan.HDBSCAN, cuml_hdbscan.HDBSCAN,))
        check_instance(source=f'{UMLMixin.umap_fit.__name__} data', instance=data, accepted_types=(pd.DataFrame, np.ndarray,))
        if isinstance(data, pd.DataFrame):
            data = data.values
        check_valid_array(data=data, source=f'{UMLMixin.umap_fit.__name__} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        return mdl.fit(data)