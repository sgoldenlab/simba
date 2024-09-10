import numpy as np
from typing import Optional
from time import perf_counter
from simba.utils.enums import Formats
from simba.utils.checks import check_valid_array, check_float
from numba import cuda
from simba.data_processors.cuda.utils import _cuda_mean, _cuda_std


THREADS_PER_BLOCK = 1024

@cuda.jit(device=True)
def _count_at_threshold(x: np.ndarray, inverse: int, threshold: float):
    results = 0
    for i in range(x.shape[0]):
        if inverse[0] == 0:
            if x[i] > threshold[0]:
                results += 1
        else:
            if x[i] < threshold[0]:
                results += 1
    return results

@cuda.jit()
def _sliding_crossings_kernal(data, time, threshold, inverse, results):
    r = cuda.grid(1)
    l = int(r - time[0])
    if r > data.shape[0] or r < 0:
        return
    elif l > data.shape[0] or l < 0:
        return
    else:
        sample = data[l:r]
        results[r-1] = _count_at_threshold(sample, inverse, threshold)

def sliding_threshold(data: np.ndarray, time_window: float, sample_rate: float, value: float, inverse: Optional[bool] = False) -> np.ndarray:
    """
    Compute the count of observations above or below threshold crossings over a sliding window using GPU acceleration.

    :param np.ndarray data: Input data array.
    :param float time_window: Size of the sliding window in seconds.
    :param float sample_rate: Number of samples per second in the data.
    :param float value: Threshold value.
    :param Optional[bool] inverse: If False, counts values above the threshold. If True, counts values below.
    :return: Array containing count of threshold crossings per window.
    :rtype: np.ndarray
    """

    check_float(name='sample_rate', value=sample_rate, min_value=10e-6)
    check_float(name='sample_rate', value=sample_rate, min_value=10e-6)
    check_float(name='time_window', value=time_window, min_value=10e-6)
    check_valid_array(data=data, source=sliding_threshold.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)

    data_dev = cuda.to_device(data)
    time_window_frames = np.array([np.ceil(time_window * sample_rate)])
    time_window_frames_dev = cuda.to_device(time_window_frames)
    value = np.array([value])
    invert = np.array([0])
    if inverse: invert[0] = 1
    value_dev = cuda.to_device(value)
    inverse_dev = cuda.to_device(invert)
    bpg = (data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=(data.shape), dtype=np.int32)
    _sliding_crossings_kernal[bpg, THREADS_PER_BLOCK](data_dev, time_window_frames_dev, value_dev, inverse_dev, results)
    return results.copy_to_host()

@cuda.jit()
def _sliding_percent_beyond_n_std_kernel(data, time, std_n, results):
    r = cuda.grid(1)
    l = int(r - time[0])
    if r > data.shape[0] or r < 0:
        return
    elif l > data.shape[0] or l < 0:
        return
    else:
        sample = data[l:r]
        m = _cuda_mean(sample)
        std_val = _cuda_std(sample, m) * std_n[0]
        cnt = 0
        for i in range(sample.shape[0]):

            if (sample[i] > (m + std_val)) or (sample[i] < (m - std_val)):
                print(sample[i], m + std_val)
                cnt += 1
        results[r-1] = cnt

def sliding_percent_beyond_n_std(data: np.ndarray, time_window: float, sample_rate: float, value: float) -> np.ndarray:
    """
    Computes the percentage of points in each sliding window of `data` that fall beyond
    `n` standard deviations from the mean of that window.

    This function uses GPU acceleration via CUDA to efficiently compute the result over large datasets.

    :param np.ndarray data: The input 1D data array for which the sliding window computation is to be performed.
    :param float time_window: The length of the time window in seconds.
    :param float sample_rate: The sample rate of the data in Hz (samples per second).
    :param float value: The number of standard deviations beyond which to count data points.
    :return: An array containing the count of data points beyond `n` standard deviations for each window.
    :rtype: np.ndarray

    :example:
    >>> data = np.random.randint(0, 100, (100,))
    >>> results = sliding_percent_beyond_n_std(data=data, time_window=1, sample_rate=10, value=2)
    """

    data_dev = cuda.to_device(data)
    time_window_frames = np.array([np.ceil(time_window * sample_rate)])
    time_window_frames_dev = cuda.to_device(time_window_frames)
    value = np.array([value])
    value_dev = cuda.to_device(value)
    bpg = (data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=(data.shape), dtype=np.int32)
    _sliding_percent_beyond_n_std_kernel[bpg, THREADS_PER_BLOCK](data_dev, time_window_frames_dev, value_dev, results)
    return results.copy_to_host()



