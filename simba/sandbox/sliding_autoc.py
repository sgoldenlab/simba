import numpy as np
from numba import njit, prange
import time

@njit("(float32[:], float64, float64, float64)")
def sliding_autocorrelation(data: np.ndarray, max_lag: float, time_window: float, fps: float):
    """
    Jitted compute of sliding auto-correlations (the correlation of a feature with itself using lagged windows).

    :example:
    >>> data = np.array([0,1,2,3,4, 5,6,7,8,1,10,11,12,13,14]).astype(np.float32)
    >>> Statistics().sliding_autocorrelation(data=data, max_lag=0.5, time_window=1.0, fps=10)
    >>> [ 0., 0., 0.,  0.,  0., 0., 0.,  0. ,  0., -3.686, -2.029, -1.323, -1.753, -3.807, -4.634]
    """

    max_frm_lag, time_window_frms = int(max_lag * fps), int(time_window * fps)
    results = np.full((data.shape[0]), -1.0)
    for right in prange(time_window_frms - 1, data.shape[0]):
        left = right - time_window_frms + 1
        w_data = data[left: right + 1]
        corrcfs = np.full((max_frm_lag), np.nan)
        corrcfs[0] = 1
        for shift in range(1, max_frm_lag):
            c = np.corrcoef(w_data[:-shift], w_data[shift:])[0][1]
            if np.isnan(c):
                corrcfs[shift] = 1
            else:
                corrcfs[shift] = np.corrcoef(w_data[:-shift], w_data[shift:])[0][1]
        mat_ = np.zeros(shape=(corrcfs.shape[0], 2))
        const = np.ones_like(corrcfs)
        mat_[:, 0] = const
        mat_[:, 1] = corrcfs
        det_ = np.linalg.lstsq(mat_.astype(np.float32), np.arange(0, max_frm_lag).astype(np.float32))[0]
        results[right] = det_[::-1][0]
    return results


#data = np.array([0,1,2,3,4, 5,6,7,8,1,10,11,12,13,14]).astype(np.float32)
start = time.time()
data = np.random.randint(0, 100, (1000, )).astype(np.float32)
sliding_autocorrelation(data=data, max_lag=0.5, time_window=1.0, fps=10.0)
print(time.time() - start)