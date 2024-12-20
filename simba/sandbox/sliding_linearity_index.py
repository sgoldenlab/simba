import numpy as np
from numba import cuda
import time
from simba.utils.enums import Formats
from simba.utils.checks import check_valid_array, check_float
from simba.data_processors.cuda.utils import _euclid_dist, _cuda_available
from simba.utils.errors import SimBAGPUError

THREADS_PER_BLOCK = 1024

@cuda.jit()
def _sliding_linearity_index_kernel(x, time_frms, results):
    r = cuda.grid(1)
    if r >= x.shape[0] or r < 0:
        return
    l = int(r - time_frms[0])
    if l < 0 or l >= r:
        return
    sample_x = x[l:r]
    straight_line_distance = _euclid_dist(sample_x[0], sample_x[-1])
    path_dist = 0
    for i in range(1, sample_x.shape[0]):
        path_dist +=  _euclid_dist(sample_x[i-1], sample_x[i])
    if path_dist == 0:
        results[r] = 0.0
    else:
        results[r] = straight_line_distance / path_dist
        

def sliding_linearity_index_cuda(x: np.ndarray,
                                 window_size: float,
                                 sample_rate: float) -> np.ndarray:
    """

    Calculates the straightness (linearity) index of a path using CUDA acceleration.

    The output is a value between 0 and 1, where 1 indicates a perfectly straight path.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_spatial_density_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    :param np.ndarray x: An (N, M) array representing the path, where N is the number of points and M is the number of spatial dimensions (e.g., 2 for 2D or 3 for 3D). Each row represents the coordinates of a point along the path.
    :param float x: The size of the sliding window in seconds. This defines the time window over which the linearity index is calculated. The window size should be specified in seconds.
    :param float sample_rate: The sample rate in Hz (samples per second), which is used to convert the window size from seconds to frames.
    :return: A 1D array of length N, where each element represents the linearity index of the path within a sliding  window. The value is a ratio between the straight-line distance and the actual path length for each window. Values range from 0 to 1, with 1 indicating a perfectly straight path.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 500, (100, 2)).astype(np.float32)
    >>> q = sliding_linearity_index_cuda(x=x, window_size=2, sample_rate=30)
    """

    check_valid_array(data=x, source=f'{sliding_linearity_index_cuda.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_linearity_index_cuda.__name__} window_size', value=window_size)
    check_float(name=f'{sliding_linearity_index_cuda.__name__} sample_rate', value=sample_rate)
    x = np.ascontiguousarray(x)
    time_window_frames = np.array([max(1.0, np.ceil(window_size * sample_rate))])
    if not _cuda_available()[0]:
        SimBAGPUError(msg='No GPU found', source=sliding_linearity_index_cuda.__name__)
    x_dev = cuda.to_device(x)
    time_window_frames_dev = cuda.to_device(time_window_frames)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=x.shape[0], dtype=np.float16)
    _sliding_linearity_index_kernel[bpg, THREADS_PER_BLOCK](x_dev, time_window_frames_dev, results)
    return results.copy_to_host()




for cnt in [1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 128000000, 256000000, 512000000, 1024000000]:
    times = []
    for i in range(3):
        start = time.perf_counter()
        x = np.random.randint(0, 500, (cnt, 2))
        results_cuda = sliding_linearity_index_cuda(x=x, window_size=2.5, sample_rate=30)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    print(cnt, '\t'*2, np.mean(times), np.std(times))

# x = np.random.randint(0, 500, (100, 2)).astype(np.float32)
# q = sliding_linearity_index_cuda(x=x, window_size=2, sample_rate=30)
