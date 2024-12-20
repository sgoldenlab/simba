from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from itertools import combinations

from simba.utils.checks import check_valid_array, check_float, check_int
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin

def local_outlier_factor(data: np.ndarray,
                         k: Union[int, float] = 5,
                         contamination: Optional[float] = 1e-10,
                         normalize: Optional[bool] = False,
                         groupby_idx: Optional[int] = None) -> np.ndarray:
    """
    Compute the local outlier factor of each observation.

    .. note::
       The final LOF scores are negated. Thus, higher values indicate more atypical (outlier) data points. Values
       Method calls ``sklearn.neighbors.LocalOutlierFactor`` directly. Attempted to use own jit compiled implementation,
       but runtime was 3x-ish slower than ``sklearn.neighbors.LocalOutlierFactor``.

       If groupby_idx is not None, then the index 1 of ``data`` array for which to group the data and compute LOF within each segment/cluster.
       E.g., can be field holding cluster identifier. Thus, outliers are computed within each segment/cluster, ensuring that other segments cannot affect
       outlier scores within each analyzing each cluster.

       If groupby_idx is provided, then all observations with cluster/segment variable ``-1`` will be treated as unclustered and assigned the max outlier score found withiin the clustered observations.

    .. image:: _static/img/local_outlier_factor.png
       :width: 700
       :align: center

    :param ndarray data: 2D array with feature values where rows represent frames and columns represent features.
    :param Union[int, float] k: Number of neighbors to evaluate for each observation. If the value is a float, then interpreted as the ratio of data.shape[0]. If the value is an integer, then it represent the number of neighbours to evaluate.
    :param Optional[float] contamination: Small pseudonumber to avoid DivisionByZero error.
    :param Optional[bool] normalize: Whether to normalize the distances between 0 and 1. Defaults to False.
    :param Optional[int] groupby_idx: If int, then the index 1 of ``data`` for which to group the data and compute LOF on each segment. E.g., can be field holding a cluster identifier.
    :returns np.ndarray: Array of size data.shape[0] with local outlier scores.

    :example:
    >>> data, lbls = make_blobs(n_samples=2000, n_features=2, centers=10, random_state=42)
    >>> data = np.hstack((data, lbls.reshape(-1, 1)))
    >>> lof = local_outlier_factor(data=data, groupby_idx=2, k=100, normalize=True)
    >>> results = np.hstack((data[:, 0:2], lof.reshape(lof.shape[0], 1)))
    >>> PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey',size=30)

    """
    def get_lof(data, k, contamination):
        check_float(name=f"{local_outlier_factor.__name__} k", value=k)
        if isinstance(k, int):
            k = min(k, data.shape[0])
        elif isinstance(k, float):
            k = int(data.shape[0] * k)
        lof_model = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
        _ = lof_model.fit_predict(data)
        y = -lof_model.negative_outlier_factor_.astype(np.float32)
        if normalize:
            return (y - np.min(y)) / (np.max(y) - np.min(y))
        else:
            return y

    if groupby_idx is not None:
        check_int(name=f'{local_outlier_factor.__name__} groupby_idx', value=groupby_idx, min_value=0, max_value=data.shape[1]-1)
        check_valid_array(source=f"{local_outlier_factor.__name__} local_outlier_factor", data=data, accepted_sizes=[2], min_axis_1=3)
    else:
        check_valid_array(source=f"{local_outlier_factor.__name__} data", data=data, accepted_sizes=[2], min_axis_1=2)
    check_float(name=f"{local_outlier_factor.__name__} contamination", value=contamination, min_value=0.0)

    if groupby_idx is None:
        return get_lof(data, k, contamination)
    else:
        results = []
        data_w_idx = np.hstack((np.arange(0, data.shape[0]).reshape(-1, 1), data))
        unique_c = np.unique(data[:, groupby_idx]).astype(np.float32)
        if -1.0 in unique_c:
            unique_c = unique_c[np.where(unique_c != -1)]
            unclustered_idx = np.argwhere(data[:, groupby_idx] == -1.0).flatten()
            unclustered = data_w_idx[unclustered_idx]
            data_w_idx = np.delete(data_w_idx, unclustered_idx, axis=0)
        else:
            unclustered = None
        for i in unique_c:
            s_data = data_w_idx[np.argwhere(data_w_idx[:, groupby_idx+1] == i)].reshape(-1, data_w_idx.shape[1])
            idx = s_data[:, 0].reshape(s_data.shape[0], 1)
            s_data = np.delete(s_data, [0, groupby_idx+1], 1)
            lof = get_lof(s_data, k, contamination).reshape(s_data.shape[0], 1)
            results.append(np.hstack((idx, lof)))
        x = np.concatenate(results, axis=0)
        if unclustered is not None:
            max_lof = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
            unclustered = np.hstack((unclustered, max_lof))[:, [0, -1]]
            x = np.vstack((x, unclustered))
        return x[np.argsort(x[:, 0])][:, -1]


def elliptic_envelope(data: np.ndarray,
                      contamination: Optional[float] = 1e-1,
                      normalize: Optional[bool] = False,
                      groupby_idx: Optional[int] = None) -> np.ndarray:
    """
    Compute the Mahalanobis distances of each observation in the input array using Elliptic Envelope method.

    .. image:: _static/img/EllipticEnvelope.png
       :width: 700
       :align: center

    .. image:: _static/img/elliptic_envelope.png
       :width: 700
       :align: center

    :param data: Input data array of shape (n_samples, n_features).
    :param Optional[float] contamination: The proportion of outliers to be assumed in the data. Defaults to 0.1.
    :param Optional[bool] normalize: Whether to normalize the Mahalanobis distances between 0 and 1. Defaults to True.
    :return np.ndarray: The Mahalanobis distances of each observation in array. Larger values indicate outliers.

    :example:
    >>> data, lbls = make_blobs(n_samples=2000, n_features=2, centers=1, random_state=42)
    >>> envelope_score = elliptic_envelope(data=data, normalize=True)
    >>> results = np.hstack((data[:, 0:2], envelope_score.reshape(lof.shape[0], 1)))
    >>> results = pd.DataFrame(results, columns=['X', 'Y', 'ENVELOPE SCORE'])
    >>> PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey', columns=['X', 'Y', 'ENVELOPE SCORE'],size=30)

    """

    def get_envelope(data, contamination) -> np.ndarray:
        mdl = EllipticEnvelope(contamination=contamination).fit(data)
        y = -mdl.score_samples(data)
        if normalize:
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        return y

    if groupby_idx is not None:
        check_int(name=f'{elliptic_envelope.__name__} groupby_idx', value=groupby_idx, min_value=0, max_value=data.shape[1]-1)
        check_valid_array(source=f"{elliptic_envelope.__name__} local_outlier_factor", data=data, accepted_sizes=[2], min_axis_1=3)
    else:
        check_valid_array(source=f"{elliptic_envelope.__name__} data", data=data, accepted_sizes=[2], min_axis_1=2)
    check_float(name=f"{elliptic_envelope.__name__} contamination", value=contamination, min_value=0.0, max_value=1.0)

    if groupby_idx is None:
        return get_envelope(data, contamination)
    else:
        results = []
        data_w_idx = np.hstack((np.arange(0, data.shape[0]).reshape(-1, 1), data))
        unique_c = np.unique(data[:, groupby_idx]).astype(np.float32)
        if -1.0 in unique_c:
            unique_c = unique_c[np.where(unique_c != -1)]
            unclustered_idx = np.argwhere(data[:, groupby_idx] == -1.0).flatten()
            unclustered = data_w_idx[unclustered_idx]
            data_w_idx = np.delete(data_w_idx, unclustered_idx, axis=0)
        else:
            unclustered = None
        for i in unique_c:
            s_data = data_w_idx[np.argwhere(data_w_idx[:, groupby_idx+1] == i)].reshape(-1, data_w_idx.shape[1])
            idx = s_data[:, 0].reshape(s_data.shape[0], 1)
            s_data = np.delete(s_data, [0, groupby_idx+1], 1)
            lof = get_envelope(s_data, contamination).reshape(s_data.shape[0], 1)
            results.append(np.hstack((idx, lof)))
        x = np.concatenate(results, axis=0)
        if unclustered is not None:
            max_lof = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
            unclustered = np.hstack((unclustered, max_lof))[:, [0, -1]]
            x = np.vstack((x, unclustered))
        return x[np.argsort(x[:, 0])][:, -1]



def angle_based_od(data: np.ndarray,
                   k: Union[int, float] = 5,
                   groupby_idx: Optional[int] = None,
                   normalize: Optional[bool] = False) -> np.ndarray:
    """

    :param data:
    :param k:
    :return:
    Adopted from https://pyod.readthedocs.io/en/latest/_modules/pyod/models/abod.html#ABOD
    """

    def _wcos(x: np.ndarray, nn_s: np.ndarray):
        nn_pair = list(combinations(list(range(nn_s.shape[0])), 2))


        # #wcos = np.full((nn_s.shape[0], nn_s.shape[0]), 0.0)
        w = []
        for l in nn_pair:
            a = nn_s[l[0]] - x
            b = nn_s[l[1]] - x
            val = np.dot(a, b) / (np.linalg.norm(a, 2) ** 2) / (np.linalg.norm(b, 2) ** 2)
            w.append(val)
        #     for g in range(i + 1, nn_s.shape[0]):
        #         if (np.array_equal(nn_s[p], x)) or (np.array_equal(nn_s[g], x)):
        #             continue
        #         else:
        #             a = nn_s[p] - x
        #             b = nn_s[g] - x
        #             val = np.dot(a, b) / (np.linalg.norm(a, 2) ** 2) / (np.linalg.norm(b, 2) ** 2)
        #             w.append(val)
        #print(w, nn_s.shape, x.shape, x)
        #print(w)
        return np.var(w)

    if groupby_idx is not None:
        check_int(name=f'{angle_based_od.__name__} groupby_idx', value=groupby_idx, min_value=0, max_value=data.shape[1]-1)
        check_valid_array(source=f"{angle_based_od.__name__} local_outlier_factor", data=data, accepted_sizes=[2], min_axis_1=3)
    else:
        check_valid_array(source=f"{angle_based_od.__name__} data", data=data, accepted_sizes=[2], min_axis_1=2)
    check_float(name=f"{local_outlier_factor.__name__} k", value=k)
    if groupby_idx is None:
        if isinstance(k, int):
            k = min(k, data.shape[0]-1)
        elif isinstance(k, float):
            k = int((data.shape[0]-1) * k)
        distances = FeatureExtractionMixin.cdist(array_1=data, array_2=data)
        #print(distances)
        results = np.full((data.shape[0],), np.nan)
        for i in range(distances.shape[0]):
            idx = np.argsort(distances[i])[1:k+1]
            nn_s = data[idx, :]
            #print(nn_s)
            results[i] = _wcos(data[i], nn_s)
            #print(results[i])
        if normalize:
            return (results - np.min(results)) / (np.max(results) - np.min(results))

        else:
            return results
    # else:
    #     results = []
    #     data_w_idx = np.hstack((np.arange(0, data.shape[0]).reshape(-1, 1), data))
    #     unique_c = np.unique(data[:, groupby_idx]).astype(np.float32)
    #     if -1.0 in unique_c:
    #         unique_c = unique_c[np.where(unique_c != -1)]
    #         unclustered_idx = np.argwhere(data[:, groupby_idx] == -1.0).flatten()
    #         unclustered = data_w_idx[unclustered_idx]
    #         data_w_idx = np.delete(data_w_idx, unclustered_idx, axis=0)
    #     else:
    #         unclustered = None
    #     for i in unique_c:
    #         c_data = data_w_idx[np.argwhere(data_w_idx[:, groupby_idx + 1] == i)].reshape(-1, data_w_idx.shape[1])
    #         c_data_idx = c_data[:, 0].reshape(c_data.shape[0], 1)
    #         c_data = np.delete(c_data, [0, groupby_idx + 1], 1)
    #         print(c_data)
    #         distances = FeatureExtractionMixin.cdist(array_1=c_data, array_2=c_data)
    #         c_results = np.full((c_data.shape[0], 1), np.nan)
    #         c_results_idx = np.full((c_data.shape[0], 1), np.nan)
    #         for j in range(distances.shape[0]):
    #             if isinstance(k, int):
    #                 c_k = min(k, c_data.shape[0]-1)
    #             elif isinstance(k, float):
    #                 c_k = int((c_data.shape[0] - 1) * k)
    #                 if c_k < 1:
    #                     c_k = 1
    #             nn_idx = np.argsort(distances[j])[1:c_k + 1]
    #             nn_s = c_data[nn_idx, :]
    #             c_results_idx[j] = c_data_idx[j]
    #             c_results[j] = _wcos(c_data[j], nn_s)
    #         if normalize:
    #             c_results = (c_results - np.min(c_results)) / (np.max(c_results) - np.min(c_results))
    #         results.append(np.hstack((c_results_idx, c_results)))
    #     results = np.concatenate(results, axis=0)
    #     if unclustered is not None:
    #         max_angle_od = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
    #         unclustered = np.hstack((unclustered, max_angle_od))[:, [0, -1]]
    #         results = np.vstack((results, unclustered))
    #     return results[np.argsort(results[:, 0])][:, -1]
    #     #



                # lof = get_envelope(s_data, contamination).reshape(s_data.shape[0], 1)
                # results.append(np.hstack((idx, lof)))


        #else:
            #return (wcos - np.min(wcos)) / (np.max(wcos) - np.min(wcos))
# data, lbls = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=42)
# #data = np.hstack((data, lbls.reshape(-1, 1)))
#
# score = angle_based_od(data=data, k=100, normalize=True)
# #print(score)
# results = np.hstack((data[:, 0:2], score.reshape(score.shape[0], 1)))
# #print(results)
# PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey', size=10)

# centroid_distances = FeatureExtractionMixin.cdist(
#             array_1=centroids.astype(np.float32), array_2=centroids.astype(np.float32)
#         )
#


#
#
#
#
# data, lbls = make_blobs(n_samples=2000, n_features=2, centers=1, random_state=42)
# #data = np.hstack((data, lbls.reshape(-1, 1)))
# #outliers = np.random.rand(50, 2) * 20 - 10
# #outliers = np.hstack((outliers, np.full((outliers.shape[0], 1), -1.0)))
# #data = np.concatenate([data, outliers])
# lof = local_outlier_factor(data=data, groupby_idx=2, k=100, normalize=True)
# results = np.hstack((data[:, 0:2], lof.reshape(lof.shape[0], 1)))
# results = pd.DataFrame(results, columns=['X', 'Y', 'LOF'])
# PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey', columns=['X', 'Y', 'LOF'],size=30)
#
# data, lbls = make_blobs(n_samples=2000, n_features=2, centers=1, random_state=42)
# envelope_score = elliptic_envelope(data=data, normalize=True)
# results = np.hstack((data[:, 0:2], envelope_score.reshape(lof.shape[0], 1)))
# results = pd.DataFrame(results, columns=['X', 'Y', 'ENVELOPE SCORE'])
# PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey', columns=['X', 'Y', 'ENVELOPE SCORE'],size=30)
#
#
#
#



# data = np.random.normal(loc=45, scale=1, size=(100, 2)).astype(np.float32)
# for i in range(5): data = np.vstack([data, np.random.normal(loc=45, scale=1, size=(100, 2)).astype(np.float32)])
# for i in range(2): data = np.vstack([data, np.random.normal(loc=90, scale=1, size=(100, 2)).astype(np.float32)])
# c = np.random.randint(-1, 5, (800,)).reshape(-1, 1)
# data = np.hstack((data, c))
# local_outlier_factor(data=data, k=5, groupby_idx=2)
#
#

#[1.004, 1.007, 0.986, 1.018, 0.986, 0.996, 24.067, 24.057]