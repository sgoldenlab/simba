from typing import Tuple

import numpy as np
from numba import cuda

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import math

from simba.data_processors.cuda.utils import _cuda_mean
from simba.utils.checks import (check_float, check_valid_array,
                                check_valid_tuple)
from simba.utils.enums import Formats


def _cuda_variance(x: np.ndarray):
    #mean = _cuda_mean(x=x)
    mean = np.mean(x)
    num = 0
    for i in range(x.shape[0]):
        num += abs(x[i] - mean)
    return num / (x.shape[0] - 1)





def sliding_descriptive_statistics_cuda(data: np.ndarray,
                                        window_size: float,
                                        sample_rate: float,
                                        statistics: Tuple[Literal["var", "max", "min", "std"]]):

    STATISTICS = ('var', 'max', 'min', 'std', 'median', 'mean', 'mad', 'sum', 'mac', 'rms', 'abs_energy')
    check_valid_array(data=data, source=f'{sliding_descriptive_statistics_cuda.__name__} data', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_descriptive_statistics_cuda.__name__} window_size', value=window_size, min_value=10e-6)
    check_float(name=f'{sliding_descriptive_statistics_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    check_valid_tuple(x=statistics, source=f'{sliding_descriptive_statistics_cuda.__name__} statistics', valid_dtypes=(str,)) #TODO: ACCESPTED ENTRIES
    frm_win = np.array([max(1, int(window_size*sample_rate))])
    sV = np.zeros(shape=(11,), dtype=np.uint8)
    for cnt, statistic in enumerate(STATISTICS):
        if statistic in statistics: sV[cnt] = 1
    print(sV)




data = np.random.randint(0, 500, (100,))
window_size = 1.5
sliding_descriptive_statistics_cuda(data=data, window_size=window_size, sample_rate=10, statistics=('max', 'min'))


x = np.array([2.5, 5, 7.5, 10.0])
_cuda_variance(x=x)
