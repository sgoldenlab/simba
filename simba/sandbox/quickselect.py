import numba
import numpy as np
from numba import cuda

@cuda.jit
def partition(arr, left, right, pivot_index, partition_index):
    tid = cuda.grid(1)

    if tid < (right - left + 1):
        pivot = arr[pivot_index]
        i = left - 1
        for j in range(left, right):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        partition_index[0] = i + 1

def quickselect(arr, left, right, k):
    # If the subarray has one element, return it
    if left == right:
        return arr[left]

    # Select pivot_index, can be improved with more sophisticated strategies
    pivot_index = (left + right) // 2

    # Convert left, right, pivot_index into arrays for CUDA kernel
    left_device = np.array([left], dtype=np.int32)
    right_device = np.array([right], dtype=np.int32)
    pivot_index_device = np.array([pivot_index], dtype=np.int32)
    partition_index_device = np.zeros(1, dtype=np.int32)

    # Transfer to device
    arr_device = cuda.to_device(arr)
    left_device_device = cuda.to_device(left_device)
    right_device_device = cuda.to_device(right_device)
    pivot_index_device_device = cuda.to_device(pivot_index_device)
    partition_index_device_device = cuda.to_device(partition_index_device)

    # Run the partition kernel on GPU
    partition[1, 32](arr_device, left_device_device, right_device_device, pivot_index_device_device, partition_index_device_device)

    # Get partition index after the kernel has finished
    partition_idx = partition_index_device_device[0]

    # Check which side to recurse
    if k == partition_idx:
        return arr[k]
    elif k < partition_idx:
        return quickselect(arr, left, partition_idx - 1, k)
    else:
        return quickselect(arr, partition_idx + 1, right, k)

# Testing the Quickselect with CUDA
arr = np.random.randint(0, 1000, size=1000)
k = 500  # Find the 500th smallest element

# Call the quickselect function
result = quickselect(arr, 0, arr.size - 1, k)
print(f"The {k+1}-th smallest element is: {result}")
