from numba import jit, prange
import numpy as np
import time

@jit(nopython=True)
def count_values_in_range(data: np.array, ranges: np.array):
    results = np.full((data.shape[0], ranges.shape[0]), 0)
    for i in prange(data.shape[0]):
        for j in prange(ranges.shape[0]):
            lower_bound, upper_bound = ranges[j][0], ranges[j][1]
            results[i][j] = data[i][np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)].shape[0]
    return results


@jit(nopython=True)
def count_values_in_range_mp(data: np.array, ranges: np.array):
    results = np.full((data.shape[0], ranges.shape[0]), 0)
    for i in prange(data.shape[0]):
        for j in prange(ranges.shape[0]):
            lower_bound, upper_bound = ranges[j][0], ranges[j][1]



            results[i][j] = data[i][np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)].shape[0]
    return results


ranges = np.array([[0.0, 0.1], [0.0, 0.5], [0.0, 0.75]])
data = np.random.random((1000000, 100))

# start = time.time()
# results = count_values_in_range(data=data, ranges=ranges)
# print(time.time() - start)

start = time.time()
results = count_values_in_range_mp(data=data, ranges=ranges)
print(time.time() - start)