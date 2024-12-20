import numpy as np
from numba import cuda, njit



@njit("(float32[:], float64, float64)")
def sliding_iqr(x: np.ndarray, window_size: float, sample_rate: float) -> np.ndarray:
    """
    Compute the sliding interquartile range (IQR) for a 1D array of feature values.

    :param ndarray x: 1D array representing the feature values for which the IQR will be calculated.
    :param float window_size: Size of the sliding window, in seconds.  This value determines how many samples are included in each window.
    :param float sample_rate: The sampling rate in samples per second, e.g., fps.
    :returns : Sliding IQR values
    :rtype: np.ndarray

    :references:
        .. [1] Hession, Leinani E., Gautam S. Sabnis, Gary A. Churchill, and Vivek Kumar. “A Machine-Vision-Based Frailty Index for Mice.” Nature Aging 2, no. 8 (August 16, 2022): 756–66. https://doi.org/10.1038/s43587-022-00266-0.

    :example:
    >>> data = np.random.randint(0, 50, (90,)).astype(np.float32)
    >>> window_size = 0.5
    >>> sliding_iqr(x=data, window_size=0.5, sample_rate=10.0)
    """

    frm_win = max(1, int(window_size * sample_rate))
    results =np.full(shape=(x.shape[0],), dtype=np.float32, fill_value=-1.0)
    for r in range(frm_win, x.shape[0]+1):
        sorted_sample = np.sort(x[r - frm_win:r])
        lower_idx = sorted_sample.shape[0] // 4
        upper_idx = (3 * sorted_sample.shape[0]) // 4
        lower_val = sorted_sample[lower_idx]
        upper_val = sorted_sample[upper_idx]
        results[r-1] = upper_val - lower_val
    return results

data = np.random.randint(0, 50, (90,)).astype(np.float32)
window_size = 0.5
sliding_iqr(x=data, window_size=0.5, sample_rate=10.0)