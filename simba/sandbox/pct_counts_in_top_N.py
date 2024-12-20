import numpy as np

from simba.utils.checks import check_valid_array, check_int, check_float

def pct_in_top_n(x: np.ndarray, n: float) -> float:
    """
    Compute percentage of elements in the top 'n' frequencies in the input.

    :param x: Input array.
    :param n: Number of top frequencies.
    :return: Percentage of elements in the top 'n' frequencies.

    :example:
    >>> x = np.random.randint(0, 10, (100,))
    >>> pct_in_top_n(x=x, n=5)
    """

    check_valid_array(data=x, accepted_ndims=(1,), source=pct_in_top_n.__name__)
    check_int(name=pct_in_top_n.__name__, value=n, max_value=x.shape[0])
    cnts = np.sort(np.unique(x, return_counts=True)[1])[-n:]
    return np.sum(cnts) / x.shape[0]

def sliding_pct_in_top_n(x: np.ndarray, windows: np.ndarray, n: int, fps: float) -> np.ndarray:
    """
    Compute the percentage of elements in the top 'n' frequencies in sliding windows of the input array.

    .. note::
      To compute percentage of elements in the top 'n' frequencies in entire array, use ``simba.mixins.statistics_mixin.Statistics.pct_in_top_n``.

    :param np.ndarray x: Input 1D array.
    :param np.ndarray windows: Array of window sizes in seconds.
    :param int n: Number of top frequencies.
    :param float fps: Sampling frequency for time convesrion.
    :return np.ndarray: 2D array of computed percentages of elements in the top 'n' frequencies for each sliding window.

    :example:
    >>> x = np.random.randint(0, 10, (100000,))
    >>> results = sliding_pct_in_top_n(x=x, windows=np.array([1.0]), n=4, fps=10)
    """

    check_valid_array(data=x, source=f'{sliding_pct_in_top_n.__name__} x', accepted_ndims=(1,), accepted_dtypes=(np.float32, np.float64, np.int64, np.int32, int, float))
    check_valid_array(data=windows, source=f'{sliding_pct_in_top_n.__name__} windows', accepted_ndims=(1,), accepted_dtypes=(np.float32, np.float64, np.int64, np.int32, int, float))
    check_int(name=f'{sliding_pct_in_top_n.__name__} n', value=n, min_value=1)
    check_float(name=f'{sliding_pct_in_top_n.__name__} fps', value=n, min_value=10e-6)
    results = np.full((x.shape[0], windows.shape[0]), -1.0)
    for i in range(windows.shape[0]):
        W_s = int(windows[i] * fps)
        for cnt, (l, r) in enumerate(zip(range(0, x.shape[0] + 1), range(W_s, x.shape[0] + 1))):
            sample = x[l:r]
            cnts = np.sort(np.unique(sample, return_counts=True)[1])[-n:]
            results[int(r - 1), i] = np.sum(cnts) / sample.shape[0]
    return results


