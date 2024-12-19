import math
from typing import Any, Dict, Tuple

import numpy as np
from numba import cuda, float64, guvectorize


@cuda.jit(device=True)
def _cuda_sum(x: np.ndarray):
    s = 0
    for i in range(x.shape[0]):
        s += x[i]
    return s

@cuda.jit(device=True)
def _cuda_sin(x, t):
    for i in range(x.shape[0]):
        v = math.sin(x[i])
        t[i] = v
    return t

@cuda.jit(device=True)
def _cuda_cos(x, t):
    for i in range(x.shape[0]):
        v = math.cos(x[i])
        t[i] = v
    return t

@cuda.jit(device=True)
def _cuda_min(x: np.ndarray):
    return min(x)

@cuda.jit(device=True)
def _cuda_max(x: np.ndarray):
    return max(x)

@cuda.jit(device=True)
def _cuda_standard_deviation(x):
    m = _cuda_mean(x)
    std_sum = 0
    for i in range(x.shape[0]):
        std_sum += abs(x[i] - m)
    return math.sqrt(std_sum / x.shape[0])

@cuda.jit(device=True)
def _cuda_std(x: np.ndarray, x_hat: float):
    std = 0
    for i in range(x.shape[0]):
        std += (x[0] - x_hat) ** 2
    return math.sqrt(std / x.shape[0])

@cuda.jit(device=True)
def _rad2deg(x):
    return x * (180/math.pi)

@cuda.jit(device=True)
def _deg2rad(x):
    return x * (math.pi/180)

@cuda.jit(device=True)
def _cross_test(x, y, x1, y1, x2, y2):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return cross < 0


@cuda.jit(device=True)
def _cuda_mean(x):
    s = 0
    for i in range(x.shape[0]):
        s += x[i]
    return s / x.shape[0]

@cuda.jit(device=True)
def _cuda_mse(img_1, img_2):
    s = 0.0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            k = (img_1[i, j] - img_2[i, j]) ** 2
            s += k
    return s / (img_1.shape[0] * img_1.shape[1])


@cuda.jit(device=True)
def _cuda_luminance_pixel_to_grey(r: int, g: int, b: int):
    r = 0.2126* r
    g = 0.7152 * g
    b = 0.0722 * b
    return b + g + r

@cuda.jit(device=True)
def _cuda_digital_pixel_to_grey(r: int, g: int, b: int):
    r = 0.299 * r
    g = 0.587 * g
    b = 0.114 * b
    return b + g + r

@cuda.jit(device=True)
def _euclid_dist(x, y):
    return math.sqrt(((y[0] - x[0]) ** 2) + ((y[1] - x[1]) ** 2))

@cuda.jit(device=True)
def _cuda_matrix_multiplication(mA, mB, out):
    """ Matrix multiplication"""
    for i in range(mA.shape[0]):
        for j in range(mB.shape[1]):
            for k in range(mA.shape[1]):
                out[i][j] += mA[i][k] * mB[k][j]
    return out

@cuda.jit(device=True)
def _cuda_2d_transpose(x, y):
    """ Transpose a 2d array """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[j][i] = x[i][j]
    return y

@cuda.jit(device=True)
def _cuda_subtract_2d(x: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """ Subtract 1d array values for every row in a 2d array"""
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = x[i][j] - vals[j]
    return x


@cuda.jit(device=True)
def _cuda_add_2d(x: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """ Add 1d array values for every row in a 2d array"""
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = x[i][j] + vals[j]
    return x


@cuda.jit(device=True)
def _cuda_variance(x: np.ndarray):
    mean = _cuda_mean(x)
    num = 0
    for i in range(x.shape[0]):
        num += abs(x[i] - mean)
    return num / (x.shape[0] - 1)


@cuda.jit(device=True)
def _cuda_mac(x: np.ndarray):
    """ mean average change in 1d array (max size 512)"""
    diff = cuda.local.array(shape=512, dtype=np.float64)
    for i in range(512):
        diff[i] = np.inf
    for j in range(1, x.shape[0]):
        diff[j] = abs(x[j] - x[j-1])
    s, cnt = 0, 0
    for p in range(diff.shape[0]):
        if (diff[p] != np.inf):
            s += diff[p]
            cnt += 1
    val = s / cnt
    cuda.syncthreads()
    return val

def _cuda_available() -> Tuple[bool, Dict[int, Any]]:
    """
    Check if GPU available. If True, returns the GPUs, the model, physical slots and compute capabilitie(s).

    :return: Two-part tuple with first value indicating with the GPU is available (bool) and the second value denoting GPU attributes (dict).
    :rtype: Tuple[bool, Dict[int, Any]]
    """
    is_available = cuda.is_available()
    devices = None
    if is_available:
        devices = {}
        for gpu_cnt, gpu in enumerate(cuda.gpus):
            devices[gpu_cnt] = {'model': gpu.name.decode("utf-8"),
                                'compute_capability': float("{}.{}".format(*gpu.compute_capability)),
                                'id': gpu.id,
                                'PCI_device_id': gpu.PCI_DEVICE_ID,
                                'PCI_bus_id': gpu.PCI_BUS_ID}

    return is_available, devices



@cuda.jit(device=True)
def _cuda_bubble_sort(x):
    n = x.shape[0]
    for i in range(n - 1):
        for j in range(n - i - 1):
            if x[j] > x[j + 1]:
                x[j], x[j + 1] = x[j + 1], x[j]
    return x


@cuda.jit(device=True)
def _cuda_median(x):
    sorted_arr = _cuda_bubble_sort(x)
    if not x.shape[0] % 2 == 0:
        return sorted_arr[int(math.floor(x.shape[0] / 2))]
    else:
        loc_1, loc_2 = int((x.shape[0] / 2) - 1), int(x.shape[0] / 2)
        return (sorted_arr[loc_1] + sorted_arr[loc_2]) / 2


@cuda.jit(device=True)
def _cuda_mad(x):
    diff = cuda.local.array(shape=512, dtype=np.float32)
    for i in range(512):
        diff[i] = np.inf
    m = _cuda_median(x)
    for j in range(x.shape[0]):
       diff[j] = abs(x[j] - m)
    return _cuda_median(diff[0:x.shape[0]-1])

@cuda.jit(device=True)
def _cuda_rms(x: np.ndarray):
    squared = cuda.local.array(shape=512, dtype=np.float64)
    for i in range(512): squared[i] = np.inf
    for j in range(x.shape[0]):
        squared[j] = x[j] ** 2
    m = _cuda_mean(squared[0: x.shape[0]-1])
    return math.sqrt(m)


@cuda.jit(device=True)
def _cuda_range(x: np.ndarray):
    return _cuda_max(x) - _cuda_min(x)

@cuda.jit(device=True)
def _cuda_abs_energy(x):
    squared = cuda.local.array(shape=512, dtype=np.float64)
    for i in range(512): squared[i] = np.inf
    for j in range(x.shape[0]):
        squared[j] = x[j] ** 2
    m = _cuda_sum(squared[0: x.shape[0] - 1])
    return math.sqrt(m)