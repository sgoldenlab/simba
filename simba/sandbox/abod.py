from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from pyod.models.abod import ABOD

from simba.utils.checks import check_valid_array, check_float, check_int
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin

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
        wcos = np.full((nn_s.shape[0],), np.nan)
        for j in range(nn_s.shape[0]):
            wcos[j] = np.dot(x, nn_s[j]) / (np.linalg.norm(x, 2) ** 2) / (np.linalg.norm(nn_s[j], 2) ** 2)
        return np.var(wcos)

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
        results = np.full((data.shape[0],), np.nan)
        for i in range(distances.shape[0]):
            idx = np.argsort(distances[i])[1:k+1]
            nn_s = data[idx, :]
            results[i] = _wcos(data[i], nn_s)
        if normalize:
            return (results - np.min(results)) / (np.max(results) - np.min(results))

        else:
            return results
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
            c_data = data_w_idx[np.argwhere(data_w_idx[:, groupby_idx + 1] == i)].reshape(-1, data_w_idx.shape[1])
            c_data_idx = c_data[:, 0].reshape(c_data.shape[0], 1)
            c_data = np.delete(c_data, [0, groupby_idx + 1], 1)
            print(c_data)
            distances = FeatureExtractionMixin.cdist(array_1=c_data, array_2=c_data)
            c_results = np.full((c_data.shape[0], 1), np.nan)
            c_results_idx = np.full((c_data.shape[0], 1), np.nan)
            for j in range(distances.shape[0]):
                if isinstance(k, int):
                    c_k = min(k, c_data.shape[0]-1)
                elif isinstance(k, float):
                    c_k = int((c_data.shape[0] - 1) * k)
                    if c_k < 1:
                        c_k = 1
                nn_idx = np.argsort(distances[j])[1:c_k + 1]
                nn_s = c_data[nn_idx, :]
                c_results_idx[j] = c_data_idx[j]
                c_results[j] = _wcos(c_data[j], nn_s)
            if normalize:
                c_results = (c_results - np.min(c_results)) / (np.max(c_results) - np.min(c_results))
            results.append(np.hstack((c_results_idx, c_results)))
        results = np.concatenate(results, axis=0)
        if unclustered is not None:
            max_angle_od = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
            unclustered = np.hstack((unclustered, max_angle_od))[:, [0, -1]]
            results = np.vstack((results, unclustered))
        return results[np.argsort(results[:, 0])][:, -1]


data, lbls = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=42)
abod_model = ABOD(contamination=0.1, method='fast', n_neighbors=10)
abod_model.fit(data)

pred = abod_model.predict(data)