import math

import numpy as np
from numba import cuda

from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError

THREADS_PER_BLOCK = 256


@cuda.jit()
def _border_distances_kernel(data, w, h, ppm, window_size, results):
    """One thread per output frame r. results[r] = windowed-mean distance to LEFT, RIGHT, TOP, BOTTOM edges / ppm (truncated to int)."""
    r = cuda.grid(1)
    n = data.shape[0]
    if r >= n or window_size < 1 or r < window_size - 1:
        return
    left = 0.0; right = 0.0; top = 0.0; bottom = 0.0
    for k in range(r - window_size + 1, r + 1):
        x = data[k, 0]; y = data[k, 1]
        left += math.fabs(x)
        right += math.fabs(w - x)
        top += math.fabs(y)
        bottom += math.fabs(h - y)
    results[r, 0] = np.int32((left / window_size) / ppm)
    results[r, 1] = np.int32((right / window_size) / ppm)
    results[r, 2] = np.int32((top / window_size) / ppm)
    results[r, 3] = np.int32((bottom / window_size) / ppm)


def border_distances_cuda(data: np.ndarray,
                          pixels_per_mm: float,
                          img_resolution: np.ndarray,
                          time_window: float,
                          fps: float) -> np.ndarray:
    """
    Compute the windowed-mean distance of a key-point to the left, right, top, and bottom image edges, on the GPU.

    .. note::
       Output is (n_frames, 4) int32 in millimeters (LEFT, RIGHT, TOP, BOTTOM). Frames before the first full
       window (``current_frame - window_size < 0``) are -1. Values are truncated to whole millimeters, matching
       the CPU version. Binding memory is the (n, 2) float64 input, so ~300,000,000 frames is the ceiling on a
       12 GB card.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/border_distances_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.border_distances`.

    :param np.ndarray data: 2D array (n_frames, 2) of body-part (x, y) coordinates.
    :param float pixels_per_mm: Pixels per millimeter of the recorded video.
    :param np.ndarray img_resolution: Video resolution as (width, height).
    :param float time_window: Rolling time-window in seconds (e.g. 0.2).
    :param float fps: Frames per second of the recorded video. May be fractional (e.g. 29.97).
    :return: (n_frames, 4) int32 array of millimeter distances from LEFT, RIGHT, TOP, BOTTOM.
    :rtype: np.ndarray

    :example:

    >>> data = np.array([[250, 250], [250, 250], [250, 250], [500, 500], [500, 500], [500, 500]]).astype(np.float32)
    >>> border_distances_cuda(data=data, img_resolution=np.array([500, 500]), time_window=1.0, fps=2.0, pixels_per_mm=1.0)
    """
    check_valid_array(data=data, source=f'{border_distances_cuda.__name__} data', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=img_resolution, source=f'{border_distances_cuda.__name__} img_resolution', accepted_ndims=(1,), accepted_shapes=[(2,)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{border_distances_cuda.__name__} pixels_per_mm', value=pixels_per_mm, min_value=10e-6)
    check_float(name=f'{border_distances_cuda.__name__} time_window', value=time_window, min_value=10e-6)
    check_float(name=f'{border_distances_cuda.__name__} fps', value=fps, min_value=10e-6)
    n = data.shape[0]
    window_size = int(time_window * fps)
    w, h = float(img_resolution[0]), float(img_resolution[1])
    data_dev = cuda.to_device(np.ascontiguousarray(data).astype(np.float64))
    results = cuda.to_device(np.full((n, 4), -1, dtype=np.int32))
    bpg = math.ceil(n / THREADS_PER_BLOCK)
    _border_distances_kernel[bpg, THREADS_PER_BLOCK](data_dev, w, h, float(pixels_per_mm), window_size, results)
    return results.copy_to_host()


@cuda.jit()
def _img_edge_distances_kernel(data, w, h, ppm, window_size, results):
    """One thread per output frame r. results[r] = windowed-mean distance of all body-parts to the 4 image CORNERS / ppm."""
    r = cuda.grid(1)
    n = data.shape[0]
    nbp = data.shape[1]
    if r >= n or window_size < 1 or r < window_size - 1:
        return
    d0 = 0.0; d1 = 0.0; d2 = 0.0; d3 = 0.0
    for k in range(r - window_size + 1, r + 1):
        for b in range(nbp):
            x = data[k, b, 0]; y = data[k, b, 1]
            dxw = x - w; dyh = y - h
            d0 += math.sqrt(x * x + y * y)
            d1 += math.sqrt(dxw * dxw + y * y)
            d2 += math.sqrt(dxw * dxw + dyh * dyh)
            d3 += math.sqrt(x * x + dyh * dyh)
    cnt = window_size * nbp
    results[r, 0] = (d0 / cnt) / ppm
    results[r, 1] = (d1 / cnt) / ppm
    results[r, 2] = (d2 / cnt) / ppm
    results[r, 3] = (d3 / cnt) / ppm


def img_edge_distances_cuda(data: np.ndarray,
                            pixels_per_mm: float,
                            img_resolution: np.ndarray,
                            time_window: float,
                            fps: float) -> np.ndarray:
    """
    Compute the windowed-mean distance of body-parts to the four image corners, on the GPU.

    .. note::
       Output is (n_frames, 4) float32 in millimeters (TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT). Each frame
       averages the corner distances over every body-part in every frame of the window (window_size * n_bodyparts
       points). Frames before the first full window are NaN. Both input memory and runtime scale with n_bodyparts.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/img_edge_distances_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.img_edge_distances`.

    :param np.ndarray data: 3D array (n_frames, n_bodyparts, 2) of body-part (x, y) coordinates.
    :param float pixels_per_mm: Pixels per millimeter of the recorded video.
    :param np.ndarray img_resolution: Video resolution as (width, height).
    :param float time_window: Rolling time-window in seconds (e.g. 0.2).
    :param float fps: Frames per second of the recorded video. May be fractional (e.g. 29.97).
    :return: (n_frames, 4) float32 array of millimeter distances from TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT.
    :rtype: np.ndarray

    :example:

    >>> data = np.random.randint(0, 748, (5000, 2, 2)).astype(np.float32)
    >>> img_edge_distances_cuda(data=data, pixels_per_mm=2.13, img_resolution=np.array([748, 540]), time_window=1.0, fps=30.0)
    """
    check_valid_array(data=data, source=f'{img_edge_distances_cuda.__name__} data', accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if data.shape[2] != 2:
        raise InvalidInputError(msg=f'data must be (n_frames, n_bodyparts, 2); got last dim {data.shape[2]}', source=img_edge_distances_cuda.__name__)
    check_valid_array(data=img_resolution, source=f'{img_edge_distances_cuda.__name__} img_resolution', accepted_ndims=(1,), accepted_shapes=[(2,)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{img_edge_distances_cuda.__name__} pixels_per_mm', value=pixels_per_mm, min_value=10e-6)
    check_float(name=f'{img_edge_distances_cuda.__name__} time_window', value=time_window, min_value=10e-6)
    check_float(name=f'{img_edge_distances_cuda.__name__} fps', value=fps, min_value=10e-6)
    n = data.shape[0]
    window_size = int(time_window * fps)
    w, h = float(img_resolution[0]), float(img_resolution[1])
    data_dev = cuda.to_device(np.ascontiguousarray(data).astype(np.float64))
    results = cuda.to_device(np.full((n, 4), np.nan, dtype=np.float32))
    bpg = math.ceil(n / THREADS_PER_BLOCK)
    _img_edge_distances_kernel[bpg, THREADS_PER_BLOCK](data_dev, w, h, float(pixels_per_mm), window_size, results)
    return results.copy_to_host()
