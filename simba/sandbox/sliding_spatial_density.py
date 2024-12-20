import time

import numpy as np
import pandas as pd
from simba.utils.checks import check_valid_array, check_float
from simba.utils.enums import Formats
from simba.data_processors.cuda.utils import _euclid_dist
from numba import cuda

THREADS_PER_BLOCK = 1024

@cuda.jit()
def _sliding_spatial_density_kernel(x, time_window, radius, results):
    r = cuda.grid(1)
    if r >= x.shape[0] or r < 0:
        return
    l = int(r - time_window[0])
    if l < 0 or l >= r:
        return
    total_neighbors = 0
    n_points = r - l
    if n_points <= 0:
        results[r] = 0
        return
    for i in range(l, r):
        for j in range(l, r):
            if i != j:
                dist = _euclid_dist(x[i], x[j])
                if dist <= radius[0]:
                    total_neighbors += 1

    results[r] = total_neighbors / n_points if n_points > 0 else 0


def sliding_spatial_density_cuda(x: np.ndarray,
                                 radius: float,
                                 pixels_per_mm: float,
                                 window_size: float,
                                 sample_rate: float) -> np.ndarray:
    """
    Computes the spatial density of points within a moving window along a trajectory using CUDA for acceleration.

    This function calculates a spatial density measure for each point along a 2D trajectory path by counting the number
    of neighboring points within a specified radius. The computation is performed within a sliding window that moves
    along the trajectory, using GPU acceleration to handle large datasets efficiently.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_spatial_density_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    :param np.ndarray x: A 2D array of shape (N, 2), where N is the number of points and each point has two spatial coordinates (x, y). The array represents the trajectory path of points in a 2D space (e.g., x and y positions in space).
    :param float radius: The radius (in millimeters) within which to count neighboring points around each trajectory point. Defines the area of interest around each point.
    :param float pixels_per_mm: The scaling factor that converts the physical radius (in millimeters) to pixel units for spatial density calculations.
    :param float window_size: The size of the sliding window (in seconds or points) to compute the density of points. A larger window size will consider more points in each density calculation.
    :param float sample_rate: The rate at which to sample the trajectory points (e.g., frames per second or samples per unit time). It adjusts the granularity of the sliding window.
    :return: A 1D numpy array where each element represents the computed spatial density for the trajectory at the corresponding point in time (or frame). Higher values indicate more densely packed points within the specified radius, while lower values suggest more sparsely distributed points.
    :rtype: np.ndarray

    :example:
    >>> df = pd.read_csv("/mnt/c/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Test_3.csv")
    >>> x = df[['Nose_1_x', 'Nose_1_y']].values
    >>> results_cuda = sliding_spatial_density_cuda(x=x, radius=10.0, pixels_per_mm=4.0, window_size=1, sample_rate=20)

    """

    check_valid_array(data=x, source=f'{sliding_spatial_density_cuda.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} radius', value=radius)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} window_size', value=window_size)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} sample_rate', value=sample_rate)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} pixels_per_mm', value=pixels_per_mm)

    x = np.ascontiguousarray(x)
    pixel_radius = np.array([np.ceil(max(1.0, (radius * pixels_per_mm)))]).astype(np.float64)
    time_window_frames = np.array([np.ceil(window_size * sample_rate)])
    x_dev = cuda.to_device(x)
    time_window_frames_dev = cuda.to_device(time_window_frames)
    radius_dev = cuda.to_device(pixel_radius)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=x.shape[0], dtype=np.float16)
    _sliding_spatial_density_kernel[bpg, THREADS_PER_BLOCK](x_dev, time_window_frames_dev, radius_dev, results)
    return results.copy_to_host()

def sliding_spatial_density(x: np.ndarray,
                                radius: float,
                                pixels_per_mm: float,
                                window_size: float,
                                sample_rate: float) -> np.ndarray:

    pixel_radius = np.ceil(max(1.0, (radius * pixels_per_mm)))
    frame_window_size = int(np.ceil(max(1.0, (window_size * sample_rate))))
    results = np.full(shape=(x.shape[0]), fill_value=np.nan, dtype=np.float32)
    for r in range(frame_window_size, x.shape[0] + 1):
        l = r - frame_window_size
        sample_x = x[l:r]
        n_points, total_neighbors = sample_x.shape[0], 0
        for i in range(n_points):
            distances = np.linalg.norm(sample_x - sample_x[i], axis=1)
            neighbors = np.sum(distances <= pixel_radius) - 1
            total_neighbors += neighbors
        results[r - 1] = total_neighbors / n_points
    return results



for cnt in [1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 128000000, 256000000, 512000000, 1024000000]:
    times = []
    for i in range(3):
        start = time.perf_counter()
        x = np.random.randint(0, 500, (cnt, 2))
        results_cuda = sliding_spatial_density(x=x, radius=10.0, pixels_per_mm=4.0, window_size=2.5, sample_rate=30)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    print(cnt, '\t'*2, np.mean(times), np.std(times))





# df = pd.read_csv("/mnt/c/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Test_3.csv")
# x = df[['Nose_1_x', 'Nose_1_y']].values
# results_cuda = sliding_spatial_density_cuda(x=x, radius=10.0, pixels_per_mm=4.0, window_size=1, sample_rate=20)
# results_numpy = sliding_spatial_density(x=x, radius=10.0, pixels_per_mm=4.0, window_size=1, sample_rate=20)
# print(results_cuda)
# print(results_numpy)




#
# x = np.array([[0, 100], [50, 98], [10, 872], [100, 27], [103, 2], [927, 286], [10, 10]])
#
#
# #x = np.random.randint(0, 20, (20, 2))  # Example trajectory with 100 points in 2D space

