from numba import cuda, float32
import math
import numpy as np
from numba import cuda


@cuda.jit(device=True)
def bitonic_merge(arr, low, cnt, direction):
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            if (arr[i] > arr[i + k]) == direction:  # Ascending or Descending
                arr[i], arr[i + k] = arr[i + k], arr[i]
        # Recursively merge the two halves
        bitonic_merge(arr, low, k, direction)
        bitonic_merge(arr, low + k, k, direction)


@cuda.jit(device=True)
def bitonic_sort(arr, low, cnt, direction):
    if cnt > 1:
        k = cnt // 2
        # First sort in ascending order
        bitonic_sort(arr, low, k, 1)  # Ascending
        # Then sort in descending order
        bitonic_sort(arr, low + k, k, 0)  # Descending
        # Merge the results
        bitonic_merge(arr, low, cnt, direction)


@cuda.jit
def sort_kernel(arr):
    n = arr.shape[0]
    idx = cuda.grid(1)

    # We only launch the sorting process for thread 0
    if idx == 0:
        # Create arrays for low, cnt, and direction
        low_array = cuda.local.array(1, dtype=float32)
        cnt_array = cuda.local.array(1, dtype=float32)
        direction_array = cuda.local.array(1, dtype=float32)

        # Set values in the arrays
        low_array[0] = 0
        cnt_array[0] = n
        direction_array[0] = 1  # Ascending order

        # Start the bitonic sort with the full array
        bitonic_sort(arr, low_array[0], cnt_array[0], direction_array[0])  # Sorting in ascending order


# Example data
arr = np.array([5.0, 3.0, 8.0, 1.0, 9.0, 7.0, 2.0, 4.0], dtype=np.float32)

# Allocate device memory
d_arr = cuda.to_device(arr)

# Launch kernel with a single thread block
threads_per_block = 256
blocks_per_grid = 1
sort_kernel[blocks_per_grid, threads_per_block](d_arr)

# Copy result back to host
d_arr.copy_to_host(arr)
print("Sorted array:", arr)