import numpy as np
from numba import njit, prange


@njit("(float32[:],)")
def circular_range(data: np.ndarray) -> float:
    n = len(data)
    if n < 2:
        return 0.0
    data_sorted = np.sort(data)
    diffs = np.empty(n)
    for i in range(n - 1):
        diffs[i] = data_sorted[i + 1] - data_sorted[i]
    diffs[n - 1] = (data_sorted[0] + 360) - data_sorted[-1]
    max_diff = np.max(diffs)
    return 360 - max_diff


# Test cases
data1 = np.array([0, 350, 180, 275]).astype(np.float32)
data2 = np.array([350, 0, 10, 20, 30, 90, 190, 220, 250, 290, 320, 349, 90, 10]).astype(np.float32)

data2 = np.array([0, 10, 350, 180, 45, 10, 300, 290, 100, 0]).astype(np.float32)

data2 = np.arange(0, 370, 10).astype(np.float32)

result1 = circular_range(data=data1)
result2 = circular_range(data=data2)

print(f"Leftward Transition {data1}:", result1)
print(f"Rightward Transition {data2}:", result2)
