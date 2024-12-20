from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Union, Optional
from sklearn.datasets import make_blobs
import pandas as pd

from simba.utils.checks import check_valid_array, check_float, check_int
from simba.mixins.plotting_mixin import PlottingMixin


def isolation_forest(x: np.ndarray,
                     estimators: Union[int, float] = 0.2,
                     groupby_idx: Optional[int] = None,
                     normalize: Optional[bool] = False):

    """
    An implementation of the Isolation Forest algorithm for outlier detection.

    .. image:: _static/img/isolation_forest.png.png
       :width: 700
       :align: center

    .. note::
       The isolation forest scores are negated. Thus, higher values indicate more atypical (outlier) data points.

    :param np.ndarray x: 2-D array with feature values.
    :param Union[int, float] estimators: Number of splits. If the value is a float, then interpreted as the ratio of x shape.
    :param Optional[int] groupby_idx: If int, then the index 1 of ``data`` for which to group the data and compute LOF on each segment. E.g., can be field holding a cluster identifier.
    :param Optional[bool] normalize: Whether to normalize the outlier score between 0 and 1. Defaults to False.
    :return:

    :example:
    >>> x, lbls = make_blobs(n_samples=10000, n_features=2, centers=10, random_state=42)
    >>> x = np.hstack((x, lbls.reshape(-1, 1)))
    >>> scores = isolation_forest(x=x, estimators=10, normalize=True)
    >>> results = np.hstack((x[:, 0:2], scores.reshape(scores.shape[0], 1)))
    >>> results = pd.DataFrame(results, columns=['X', 'Y', 'ISOLATION SCORE'])
    >>> PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey', columns=['X', 'Y', 'ISOLATION SCORE'],size=30)

    """

    def get_if_scores(x: np.ndarray, estimators: estimators):
        if isinstance(estimators, float):
            check_float(name=f'{isolation_forest.__name__} estimators', value=estimators, min_value=10e-6, max_value=1.0)
            estimators = x.shape[0] * estimators
            if estimators < 1: estimators = 1
        else:
            check_int(name=f'{isolation_forest.__name__} estimators', value=estimators, min_value=1)
        mdl = IsolationForest(n_estimators=estimators, n_jobs=-1, behaviour='new', contamination='auto')
        r = abs(mdl.fit(x).score_samples(x))
        if normalize:
            r = (r - np.min(r)) / (np.max(r) - np.min(r))
        return r

    if groupby_idx is None:
        check_valid_array(data=x, source=isolation_forest.__name__, accepted_ndims=(2,), min_axis_1=2, accepted_dtypes=(np.int64, np.int32,np.int8,np.float32,np.float64,int,float))
        return get_if_scores(x=x, estimators=estimators)

    else:
        check_valid_array(data=x, source=isolation_forest.__name__, accepted_ndims=(2,), min_axis_1=3, accepted_dtypes=(np.int64, np.int32,np.int8,np.float32,np.float64,int,float))
        results = []
        data_w_idx = np.hstack((np.arange(0, x.shape[0]).reshape(-1, 1), x))
        unique_c = np.unique(x[:, groupby_idx]).astype(np.float32)
        if -1.0 in unique_c:
            unique_c = unique_c[np.where(unique_c != -1)]
            unclustered_idx = np.argwhere(x[:, groupby_idx] == -1.0).flatten()
            unclustered = data_w_idx[unclustered_idx]
            data_w_idx = np.delete(data_w_idx, unclustered_idx, axis=0)
        else:
            unclustered = None
        for i in unique_c:
            s_data = data_w_idx[np.argwhere(data_w_idx[:, groupby_idx+1] == i)].reshape(-1, data_w_idx.shape[1])
            idx = s_data[:, 0].reshape(s_data.shape[0], 1)
            s_data = np.delete(s_data, [0, groupby_idx+1], 1)
            i_f = get_if_scores(s_data, estimators).reshape(s_data.shape[0], 1)
            results.append(np.hstack((idx, i_f)))
        x = np.concatenate(results, axis=0)
        if unclustered is not None:
            max_if = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
            unclustered = np.hstack((unclustered, max_if))[:, [0, -1]]
            x = np.vstack((x, unclustered))
        return x[np.argsort(x[:, 0])][:, -1]