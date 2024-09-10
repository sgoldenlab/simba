import math

import numpy as np
from numba import cuda


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
def _cuda_std(x: np.ndarray, x_hat: float):
    std = 0
    for i in range(x.shape[0]):
        std += (x[0] - x_hat) ** 2
    return std

@cuda.jit(device=True)
def _rad2deg(x):
    return x * (180/math.pi)

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