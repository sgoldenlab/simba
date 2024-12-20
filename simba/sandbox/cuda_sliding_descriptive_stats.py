from typing import Tuple

import numpy as np
from numba import cuda

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import math

from simba.data_processors.cuda.utils import (_cuda_abs_energy,
                                              _cuda_bubble_sort, _cuda_mac,
                                              _cuda_mad, _cuda_max, _cuda_mean,
                                              _cuda_median, _cuda_min,
                                              _cuda_range, _cuda_rms,
                                              _cuda_standard_deviation,
                                              _cuda_sum, _cuda_variance)
from simba.utils.checks import (check_float, check_valid_array,
                                check_valid_tuple)
from simba.utils.enums import Formats

THREADS_PER_BLOCK = 512


# @cuda.jit(device=True)
# def _cuda_quartile(x: np.ndarray, y: float):
#     b = _cuda_bubble_sort(x)
#     idx = int(math.ceil(y*b.shape[0]))
#     for i in b:
#         print(i)
#     print('ssssssssssssssssssssssssssssss')
#     return idx


@cuda.jit(device=True)
def _bubble_sort(x):
    diff = cuda.local.array(shape=512, dtype=np.float32)
    for i in range(512):
        diff[i] = np.inf


@cuda.jit(device=True)
def _cuda_iqr(x):
    _cuda_bubble_sort(x)
    lower_idx = x.shape[0] // 4
    upper_idx = (3 * x.shape[0]) // 4
    lower_val = x[lower_idx]
    upper_val = x[upper_idx]
    cuda.syncthreads()

    return x[-1] - x[0]
    #return upper_val - lower_val

    #return sorted_arr

@cuda.jit()
def _cuda_descriptive_stats_kernel(x, win_size, sV, results):
    i = cuda.grid(1)
    if ((x.shape[0]) < i) or (i < win_size[0]):
        return
    else:
        sample = x[i - win_size[0]: i]
        if sV[0] == 1: results[i-1, 0] = _cuda_variance(sample)
        if sV[1] == 1: results[i-1, 1] = _cuda_mac(sample)
        if sV[2] == 1: results[i-1, 2] = _cuda_median(sample)
        if sV[3] == 1: results[i-1, 3] = _cuda_standard_deviation(sample)
        if sV[4] == 1: results[i-1, 4] = _cuda_mad(sample)
        if sV[5] == 1: results[i-1, 5] = _cuda_mean(sample)
        if sV[6] == 1: results[i-1, 6] = _cuda_min(sample)
        if sV[7] == 1: results[i-1, 7] = _cuda_max(sample)
        if sV[8] == 1: results[i-1, 8] = _cuda_sum(sample)
        if sV[9] == 1: results[i-1, 9] = _cuda_rms(sample)
        if sV[10] == 1: results[i-1, 10] = _cuda_abs_energy(sample)
        if sV[11] == 1: results[i - 1, 11] = _cuda_range(sample)
        if sV[12] == 1: results[i - 1, 0] = _cuda_iqr(sample)
        # val = _cuda_iqr(sample)
        # print(val)
        cuda.syncthreads()


def sliding_descriptive_statistics_cuda(data: np.ndarray,
                                        window_size: float,
                                        sample_rate: float,
                                        statistics: Tuple[Literal["var", "max", "min", "std"]]):

    STATISTICS = ('var', 'mac', 'median', 'std', 'mad', 'mean', 'min', 'max', 'sum', 'rms', 'abs_energy', 'range', 'iqr')
    check_valid_array(data=data, source=f'{sliding_descriptive_statistics_cuda.__name__} data', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_descriptive_statistics_cuda.__name__} window_size', value=window_size, min_value=10e-6)
    check_float(name=f'{sliding_descriptive_statistics_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    check_valid_tuple(x=statistics, source=f'{sliding_descriptive_statistics_cuda.__name__} statistics', valid_dtypes=(str,), accepted_values=STATISTICS)
    frm_win = np.array([max(1, int(window_size*sample_rate))])
    sV = np.zeros(shape=(len(STATISTICS),), dtype=np.uint8)
    for cnt, statistic in enumerate(STATISTICS):
        if statistic in statistics: sV[cnt] = 1
    results = np.full(shape=(data.shape[0], len(STATISTICS)), fill_value=-1.0, dtype=np.float32)
    x_dev = cuda.to_device(data)
    win_size_dev = cuda.to_device(frm_win)
    sv_dev = cuda.to_device(sV)
    results_dev = cuda.to_device(results)
    bpg = (data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_descriptive_stats_kernel[bpg, THREADS_PER_BLOCK](x_dev, win_size_dev, sv_dev, results_dev)
    results = results_dev.copy_to_host()
    print(results[:, 0])








data = np.random.randint(0, 50, (90,))
window_size = 1.5
sliding_descriptive_statistics_cuda(data=data, window_size=window_size, sample_rate=30, statistics=('iqr',))
sliding_iqr(x=data, window_size=window_size, sample_rate=30)




# arr = np.array([99, 2, 3, 5, 7, 9, 11])
# bubble_sort(arr)

#
#
# x = np.array([2.5, 5, 7.5, 10.0])
# _cuda_variance(x=x)

#np.mean([57, 42, 8, 136])
#np.mean(np.diff(np.abs(data))