import numpy as np
from numba import jit


def sliding_mode(x: np.ndarray, window_size: float, sample_rate: float):
    window_frm_size = np.max((1.0, window_size*sample_rate))

