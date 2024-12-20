from sklearn.covariance import EllipticEnvelope
from typing import Optional
from simba.utils.read_write import read_pickle
import numpy as np
from simba.mixins.plotting_mixin import PlottingMixin

from simba.utils.checks import check_valid_array, check_float


def elliptic_envelope(data: np.ndarray,
                      contamination: Optional[float] = 1e-1,
                      normalize: Optional[bool] = True) -> np.ndarray:
    """
    Compute the Mahalanobis distances of each observation in the input array using Elliptic Envelope method.

    .. image:: _static/img/elliptic_envelope.png
       :width: 800
       :align: center

    :param data: Input data array of shape (n_samples, n_features).
    :param Optional[float] contamination: The proportion of outliers to be assumed in the data. Defaults to 0.1.
    :param Optional[bool] normalize: Whether to normalize the Mahalanobis distances between 0 and 1. Defaults to True.
    :return np.ndarray: The Mahalanobis distances of each observation in array. Larger values indicate outliers.

    :example:
    >>> data_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters/beautiful_beaver.pickle'
    >>> x = read_pickle(data_path=data_path)['DR_MODEL']['MODEL'].embedding_
    >>> y = elliptic_envelope(data=x, contamination=0.1)
    >>> data = np.hstack((x, y.reshape(-1, 1)))
    >>> img = PlottingMixin.continuous_scatter(data=data, columns=('X', 'Y', 'Mahalanobis distances'), size=20, palette='jet')
    """

    check_valid_array(data=data, accepted_ndims=(2,), accepted_dtypes=(np.float64, np.float32, np.int32, np.int64, float, int))
    check_float(name=f'{elliptic_envelope} contamination', value=contamination, min_value=0.0, max_value=1.0)
    mdl = EllipticEnvelope(contamination=contamination).fit(data)
    y = -mdl.score_samples(data)
    if normalize:
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return y




#
#     #img = PlottingMixin.categorical_scatter(data=data, columns=('X', 'Y', 'LOF'), size=20, palette='Dark2')
#
# data_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters/beautiful_beaver.pickle'
# #data_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/small_clusters/adoring_hoover.pickle'
# x = read_pickle(data_path=data_path)['DR_MODEL']['MODEL'].embedding_
# y = elliptic_envelope(data=x, contamination=0.1)
# data = np.hstack((x, y.reshape(-1, 1)))
# img = PlottingMixin.continuous_scatter(data=data, columns=('X', 'Y', 'Mahalanobis distances'), size=20, palette='jet')
#
#





