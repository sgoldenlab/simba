"""
Low-level CUDA/Numba GPU helpers.

This module holds the small ``@cuda.jit(device=True)`` primitives (means, medians,
standard deviation, MAD, RMS, ranges, Euclidean distance, matrix multiply/transpose,
RGB-to-grey conversions and NaN-aware reductions) that the other
:mod:`simba.data_processors.cuda` kernels compose, together with the public
:func:`get_nvc_decoder` factory for GPU (NVDEC) hardware video decoding.

How to use
----------
The ``_cuda_*`` functions are **CUDA device functions**: they execute *on* the GPU and
can only be called from inside another CUDA kernel (a function decorated with
``@cuda.jit``) or another device function — never directly from host/Python code.
They operate on data already resident in GPU memory (device arrays, or thread-local
scratch arrays), take one element/row per thread, and return plain scalars or arrays
for further use inside the kernel. Call them as ordinary functions within a kernel;
Numba inlines them at compile time (there is no kernel-launch overhead per call).

.. code-block:: python

    from numba import cuda
    from simba.data_processors.cuda.utils import _cuda_mean, _cuda_std

    @cuda.jit
    def zscore_rows(data, out):           # data: (n_rows, row_len) device array
        i = cuda.grid(1)
        if i < data.shape[0]:
            row = data[i]
            m = _cuda_mean(row)           # device helpers, called from within the kernel
            s = _cuda_std(row, m)
            out[i] = (row[0] - m) / s

    zscore_rows[blocks, threads](d_data, d_out)   # launch on device arrays

Notes:

- The reductions backed by a fixed thread-local scratch buffer
  (:func:`_cuda_mac`, :func:`_cuda_mad`, :func:`_cuda_rms`, :func:`_cuda_abs_energy`)
  assume inputs of **at most 512 elements**.
- Several helpers mutate their argument in place (e.g. :func:`_cuda_bubble_sort`,
  :func:`_cuda_median`, :func:`_cuda_subtract_2d`) — pass a copy if you need the input
  preserved.
- The **host-callable** entry points are :func:`get_nvc_decoder` and :func:`get_nvc_encoder`,
  which build NVDEC/NVENC hardware video decoder/encoder objects from ordinary Python code.
"""

import math
import os
import subprocess
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numba import cuda, float64

try:
    import PyNvVideoCodec as nvc
except ImportError:
    nvc = None

@cuda.jit(device=True)
def _cuda_sum(x: np.ndarray):
    """
    Device: sum of a 1D array.

    :param np.ndarray x: Input 1D array.
    :return: Sum of all elements of ``x``.
    """
    s = 0
    for i in range(x.shape[0]):
        s += x[i]
    return s

@cuda.jit(device=True)
def _cuda_sin(x, t):
    """
    Device: element-wise sine of ``x`` written into ``t``.

    :param np.ndarray x: Input 1D array of angles in radians.
    :param np.ndarray t: Output 1D array (same length as ``x``) that receives the sines.
    :return: The output array ``t``.
    """
    for i in range(x.shape[0]):
        v = math.sin(x[i])
        t[i] = v
    return t

@cuda.jit(device=True)
def _cuda_cos(x, t):
    """
    Device: element-wise cosine of ``x`` written into ``t``.

    :param np.ndarray x: Input 1D array of angles in radians.
    :param np.ndarray t: Output 1D array (same length as ``x``) that receives the cosines.
    :return: The output array ``t``.
    """
    for i in range(x.shape[0]):
        v = math.cos(x[i])
        t[i] = v
    return t

@cuda.jit(device=True)
def _cuda_min(x: np.ndarray):
    """
    Device: minimum value of a 1D array.

    :param np.ndarray x: Input 1D array.
    :return: Smallest element of ``x``.
    """
    return min(x)

@cuda.jit(device=True)
def _cuda_max(x: np.ndarray):
    """
    Device: maximum value of a 1D array.

    :param np.ndarray x: Input 1D array.
    :return: Largest element of ``x``.
    """
    return max(x)

@cuda.jit(device=True)
def _cuda_standard_deviation(x):
    """
    Device: population standard deviation of a 1D array (divides by N).

    :param np.ndarray x: Input 1D array.
    :return: Standard deviation of ``x``.
    """
    m = _cuda_mean(x)
    std_sum = 0
    for i in range(x.shape[0]):
        std_sum += (x[i] - m) ** 2
    return math.sqrt(std_sum / x.shape[0])

@cuda.jit(device=True)
def _cuda_std(x: np.ndarray, x_hat: float):
    """
    Device: standard deviation of ``x`` about a precomputed mean.

    :param np.ndarray x: Input 1D array.
    :param float x_hat: Precomputed mean of ``x``.
    :return: Standard deviation of ``x`` about ``x_hat``.
    """
    std = 0
    for i in range(x.shape[0]):
        std += (x[i] - x_hat) ** 2
    return math.sqrt(std / x.shape[0])

@cuda.jit(device=True)
def _rad2deg(x):
    """
    Device: convert an angle from radians to degrees.

    :param float x: Angle in radians.
    :return: Angle in degrees.
    """
    return x * (180/math.pi)

@cuda.jit(device=True)
def _deg2rad(x):
    """
    Device: convert an angle from degrees to radians.

    :param float x: Angle in degrees.
    :return: Angle in radians.
    """
    return x * (math.pi/180)

@cuda.jit(device=True)
def _cross_test(x, y, x1, y1, x2, y2):
    """
    Device: which side of a directed segment a point lies on (2D cross-product sign).

    :param float x: X-coordinate of the test point.
    :param float y: Y-coordinate of the test point.
    :param float x1: X-coordinate of the segment start.
    :param float y1: Y-coordinate of the segment start.
    :param float x2: X-coordinate of the segment end.
    :param float y2: Y-coordinate of the segment end.
    :return: True if the point lies to the right of the directed segment (x1,y1)->(x2,y2).
    """
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return cross < 0


@cuda.jit(device=True)
def _cuda_mean(x):
    """
    Device: arithmetic mean of a 1D array.

    :param np.ndarray x: Input 1D array.
    :return: Mean of ``x``.
    """
    s = 0
    for i in range(x.shape[0]):
        s += x[i]
    return s / x.shape[0]

@cuda.jit(device=True)
def _cuda_mse(img_1, img_2):
    """
    Device: mean squared error between two equally-shaped 2D images.

    :param np.ndarray img_1: First 2D image.
    :param np.ndarray img_2: Second 2D image (same shape as ``img_1``).
    :return: Mean squared error between the two images.
    """
    s = 0.0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            k = (img_1[i, j] - img_2[i, j]) ** 2
            s += k
    return s / (img_1.shape[0] * img_1.shape[1])


@cuda.jit(device=True)
def _cuda_luminance_pixel_to_grey(r: int, g: int, b: int):
    """
    Device: Rec.709 luminance greyscale value of an RGB pixel.

    :param int r: Red channel value.
    :param int g: Green channel value.
    :param int b: Blue channel value.
    :return: Greyscale (Rec.709 luminance) value of the pixel.
    """
    r = 0.2126 * r
    g = 0.7152 * g
    b = 0.0722 * b
    return b + g + r

@cuda.jit(device=True)
def _cuda_digital_pixel_to_grey(r: int, g: int, b: int):
    """
    Device: Rec.601 digital greyscale value of an RGB pixel.

    :param int r: Red channel value.
    :param int g: Green channel value.
    :param int b: Blue channel value.
    :return: Greyscale (Rec.601 digital) value of the pixel.
    """
    r = 0.299 * r
    g = 0.587 * g
    b = 0.114 * b
    return b + g + r

@cuda.jit(device=True)
def _euclid_dist_2d(x, y):
    """
    Device: Euclidean distance between two 2D points.

    :param np.ndarray x: First point as a length-2 array ``(x, y)``.
    :param np.ndarray y: Second point as a length-2 array ``(x, y)``.
    :return: Euclidean distance between the two points.
    """
    return math.sqrt(((y[0] - x[0]) ** 2) + ((y[1] - x[1]) ** 2))

@cuda.jit(device=True)
def _cuda_matrix_multiplication(mA, mB, out):
    """
    Device: matrix multiplication ``mA @ mB`` accumulated into ``out``.

    :param np.ndarray mA: Left 2D matrix of shape (m, k).
    :param np.ndarray mB: Right 2D matrix of shape (k, n).
    :param np.ndarray out: Pre-zeroed output 2D matrix of shape (m, n) that receives the product.
    :return: The output matrix ``out``.
    """
    for i in range(mA.shape[0]):
        for j in range(mB.shape[1]):
            for k in range(mA.shape[1]):
                out[i][j] += mA[i][k] * mB[k][j]
    return out

@cuda.jit(device=True)
def _cuda_2d_transpose(x, y):
    """
    Device: transpose a 2D array.

    :param np.ndarray x: Input 2D array of shape (m, n).
    :param np.ndarray y: Output 2D array of shape (n, m) that receives the transpose.
    :return: The output array ``y``.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[j][i] = x[i][j]
    return y

@cuda.jit(device=True)
def _cuda_subtract_2d(x: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """
    Device: subtract a per-column 1D vector from every row of a 2D array (in place).

    :param np.ndarray x: Input 2D array, modified in place.
    :param np.ndarray vals: 1D array with one value per column of ``x``.
    :return: The modified array ``x``.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = x[i][j] - vals[j]
    return x


@cuda.jit(device=True)
def _cuda_add_2d(x: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """
    Device: add a per-column 1D vector to every row of a 2D array (in place).

    :param np.ndarray x: Input 2D array, modified in place.
    :param np.ndarray vals: 1D array with one value per column of ``x``.
    :return: The modified array ``x``.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = x[i][j] + vals[j]
    return x


@cuda.jit(device=True)
def _cuda_variance(x: np.ndarray):
    """
    Device: sample variance of a 1D array (divides by N-1).

    :param np.ndarray x: Input 1D array.
    :return: Variance of ``x``.
    """
    mean = _cuda_mean(x)
    num = 0
    for i in range(x.shape[0]):
        num += (x[i] - mean) ** 2
    return num / (x.shape[0] - 1)


@cuda.jit(device=True)
def _cuda_mac(x: np.ndarray):
    """
    Device: mean absolute change (mean of ``|x[i] - x[i-1]|``) of a 1D array.

    :param np.ndarray x: Input 1D array (maximum length 512).
    :return: Mean of the absolute first-order differences of ``x``.
    """
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

def _is_cuda_available() -> Tuple[bool, Dict[int, Any]]:
    """
    Check if GPU available **to numba**. If True, returns the GPUs, the model, physical slots and compute capabilitie(s).

    Use this check for code paths that run ``numba`` CUDA kernels. ``numba.cuda.is_available()`` requires both the
    NVIDIA driver *and* ``libNVVM`` (numba's JIT compiler backend, shipped with the CUDA toolkit), so it returns False
    on machines where the GPU is otherwise usable - do not use it to gate ``torch`` code paths.

    .. seealso::
       For the ``torch`` equivalent, required by YOLO and SAM, see :func:`~simba.utils.checks.is_torch_cuda_available`.

    :return: Two-part tuple with first value indicating with the GPU is available (bool) and the second value denoting GPU attributes (dict).
    :rtype: Tuple[bool, Dict[int, Any]]
    """

    is_available, devices = cuda.is_available(), None
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
    """
    Device: in-place bubble sort of a 1D array (ascending).

    :param np.ndarray x: Input 1D array, sorted in place.
    :return: The sorted array ``x``.
    """
    n = x.shape[0]
    for i in range(n - 1):
        for j in range(n - i - 1):
            if x[j] > x[j + 1]:
                x[j], x[j + 1] = x[j + 1], x[j]
    return x


@cuda.jit(device=True)
def _cuda_median(x):
    """
    Device: median of a 1D array.

    :param np.ndarray x: Input 1D array (sorted in place as a side effect).
    :return: Median value of ``x``.
    """
    sorted_arr = _cuda_bubble_sort(x)
    if not x.shape[0] % 2 == 0:
        return sorted_arr[int(math.floor(x.shape[0] / 2))]
    else:
        loc_1, loc_2 = int((x.shape[0] / 2) - 1), int(x.shape[0] / 2)
        return (sorted_arr[loc_1] + sorted_arr[loc_2]) / 2


@cuda.jit(device=True)
def _cuda_mad(x):
    """
    Device: median absolute deviation (MAD) of a 1D array.

    :param np.ndarray x: Input 1D array (maximum length 512).
    :return: Median absolute deviation of ``x``.
    """
    diff = cuda.local.array(shape=512, dtype=np.float32)
    for i in range(512):
        diff[i] = np.inf
    m = _cuda_median(x)
    for j in range(x.shape[0]):
       diff[j] = abs(x[j] - m)
    return _cuda_median(diff[0:x.shape[0]])

@cuda.jit(device=True)
def _cuda_rms(x: np.ndarray):
    """
    Device: root-mean-square of a 1D array.

    :param np.ndarray x: Input 1D array (maximum length 512).
    :return: Root-mean-square of ``x``.
    """
    squared = cuda.local.array(shape=512, dtype=np.float64)
    for i in range(512): squared[i] = np.inf
    for j in range(x.shape[0]):
        squared[j] = x[j] ** 2
    m = _cuda_mean(squared[0: x.shape[0]])
    return math.sqrt(m)


@cuda.jit(device=True)
def _cuda_range(x: np.ndarray):
    """
    Device: range (max minus min) of a 1D array.

    :param np.ndarray x: Input 1D array.
    :return: Difference between the maximum and minimum of ``x``.
    """
    return _cuda_max(x) - _cuda_min(x)

@cuda.jit(device=True)
def _cuda_abs_energy(x):
    """
    Device: absolute energy (sum of squared elements) of a 1D array.

    :param np.ndarray x: Input 1D array (maximum length 512).
    :return: Sum of the squared elements of ``x``.
    """
    squared = cuda.local.array(shape=512, dtype=np.float64)
    for i in range(512): squared[i] = np.inf
    for j in range(x.shape[0]):
        squared[j] = x[j] ** 2
    m = _cuda_sum(squared[0: x.shape[0]])
    return m


@cuda.jit(device=True)
def _cuda_nanmean(x, N) -> float:
    """
    Compute the mean of `x` ignoring NaN values.

    :param np.ndarray x: Input array of length N.
    :param int N: Number of elements in `x` to consider.
    :return: Mean of non-NaN elements in `x`. Returns 0.0 if no valid elements found.
    """

    s, count = 0.0, 0
    for i in range(N):
        if not math.isnan(x[i]):
            s += x[i]
            count += 1
    if count == 0:
        return 0.0
    return s / count

@cuda.jit(device=True)
def _cuda_nanvariance(x, N) -> float:
    """
    Compute the variance of `x` ignoring NaN values using the unbiased estimator.

    :param np.ndarray x: Input array of length N.
    :param int N: Number of elements in `x` to consider. Note this is
    :return: Variance of non-NaN elements in `x`. Returns 0.0 if count <= 1.
    """

    mean = _cuda_nanmean(x, N)
    num, count = 0.0, 0
    for i in range(N):
        if not math.isnan(x[i]):
            diff = x[i] - mean
            num += diff * diff
            count += 1
    if count <= 1:
        return 0.0
    return num / (count - 1)

@cuda.jit(device=True)
def _cuda_diff(x, start, end, diff):
    """
    Compute first-order differences of a slice of `x` and store in `diff`.

    :param np.ndarray x: Input array.
    :param int start: Start index (inclusive) of the slice
    :param int end: End index (exclusive) of the slice.
    :param np.ndarray diff: Output array to store differences of x[start:end].

    :example:

    >>> diff = cuda.local.array(shape=512, dtype=float64)  # pre-allocated local array
    >>> cuda_diff(x, 0, 5, diff)
    """
    for i in range(end - start):
        diff[i] = math.nan
    for j in range(start + 1, end):
        diff[j - start] = x[j] - x[j - 1]

@cuda.jit(device=True)
def _cuda_are_rows_equal(x, y, idx_1, idx_2):
    """
    Device: check whether a row in one 2D array equals a row in another.

    :param np.ndarray x: First 2D array.
    :param np.ndarray y: Second 2D array (same number of columns as ``x``).
    :param int idx_1: Row index into ``x``.
    :param int idx_2: Row index into ``y``.
    :return: True if row ``idx_1`` of ``x`` equals row ``idx_2`` of ``y``.
    """
    for i in range(x.shape[1]):
        if x[idx_1, i] != y[idx_2, i]:
            return False
    return True


@cuda.jit()
def rgb_to_grey_argb_kernel(rgb: np.ndarray, argb: np.ndarray) -> None:
    """
    CUDA kernel: decoded RGB frame (H, W, 3) -> greyscale frame in the ARGB layout NVENC takes with ``fmt='ARGB'``
    (bytes [B, G, R, A] = [L, L, L, 255] per pixel). L is the Rec.709 luminance from :func:`_cuda_luminance_pixel_to_grey`.
    One thread per pixel (2D grid over W x H).

    :param np.ndarray rgb: Device array (H, W, 3) uint8, e.g. a decoded NVDEC frame. May be larger than ``argb`` (NVDEC pads odd widths); pixels outside ``argb`` are ignored.
    :param np.ndarray argb: Device array (H, W, 4) uint8, written in place.

    :example:
    >>> import math, torch
    >>> from numba import cuda
    >>> argb_t = torch.empty((1, 960, 1280, 4), dtype=torch.uint8, device='cuda:0')
    >>> rgb = cuda.as_cuda_array(torch.from_dlpack(decoded_frame))          # (960, 1280, 3)
    >>> rgb_to_grey_argb_kernel[(math.ceil(1280 / 16), math.ceil(960 / 16)), (16, 16)](rgb, cuda.as_cuda_array(argb_t)[0])
    >>> bitstream = encoder.Encode(argb_t[0])                               # encoder from get_nvc_encoder(..., fmt='ARGB')
    """
    x, y = cuda.grid(2)
    if y < rgb.shape[0] and x < rgb.shape[1] and y < argb.shape[0] and x < argb.shape[1]:
        lum = _cuda_luminance_pixel_to_grey(rgb[y][x][0], rgb[y][x][1], rgb[y][x][2])
        argb[y][x][0] = lum
        argb[y][x][1] = lum
        argb[y][x][2] = lum
        argb[y][x][3] = 255


@cuda.jit()
def rgb_to_yuv444_kernel(rgb: np.ndarray, yuv: np.ndarray, h: int) -> None:
    """
    CUDA kernel: decoded RGB frame (H, W, 3) -> planar YUV444 frame (3H, W), the layout NVENC takes with
    ``fmt='YUV444'``. BT.601 limited range (the same conversion as :func:`simba.utils.read_write.bgr_to_yuv`).
    One thread per pixel (2D grid over W x H).

    Full-resolution chroma keeps small coloured overlays (e.g. pose circles) crisp; 4:2:0 output subsamples the
    chroma and can blur or erase them. To encode 4:2:0, convert the result with :func:`yuv444_to_nv12_kernel`.

    :param np.ndarray rgb: Device array (H, W, 3) uint8, e.g. a decoded NVDEC frame. May be larger than ``yuv`` (NVDEC pads odd widths); pixels outside ``yuv`` are ignored.
    :param np.ndarray yuv: Device array (3H, W) uint8, written in place: planar [Y; Cb; Cr].
    :param int h: Frame height H.

    :example:
    >>> import math, torch
    >>> from numba import cuda
    >>> yuv_t = torch.empty((3 * 960, 1280), dtype=torch.uint8, device='cuda:0')
    >>> rgb = cuda.as_cuda_array(torch.from_dlpack(decoded_frame))          # (960, 1280, 3)
    >>> rgb_to_yuv444_kernel[(math.ceil(1280 / 32), math.ceil(960 / 8)), (32, 8)](rgb, cuda.as_cuda_array(yuv_t), 960)
    >>> bitstream = encoder.Encode(yuv_t)                                   # encoder from get_nvc_encoder(..., fmt='YUV444')
    """
    x, y = cuda.grid(2)
    if y >= h or x >= yuv.shape[1] or y >= rgb.shape[0] or x >= rgb.shape[1]:
        return
    r = rgb[y, x, 0]; g = rgb[y, x, 1]; b = rgb[y, x, 2]
    yy = 16.0 + 0.256788 * r + 0.504129 * g + 0.097906 * b
    cb = 128.0 - 0.148223 * r - 0.290993 * g + 0.439216 * b
    cr = 128.0 + 0.439216 * r - 0.367788 * g - 0.071427 * b
    yuv[y, x] = np.uint8(min(255.0, max(0.0, yy)))
    yuv[h + y, x] = np.uint8(min(255.0, max(0.0, cb)))
    yuv[2 * h + y, x] = np.uint8(min(255.0, max(0.0, cr)))


@cuda.jit()
def yuv444_to_nv12_kernel(yuv: np.ndarray, nv12: np.ndarray, h: int) -> None:
    """
    CUDA kernel: planar YUV444 frame (3H, W) -> NV12 (3H/2, W), the 4:2:0 layout NVENC takes with ``fmt='NV12'``.
    Copies the Y plane and averages each 2x2 block of Cb and Cr into the interleaved UV plane. One thread per pixel
    (2D grid over W x H); H and W must be even.

    :param np.ndarray yuv: Device array (3H, W) uint8: planar [Y; Cb; Cr].
    :param np.ndarray nv12: Device array (3H/2, W) uint8, written in place: [Y; interleaved UV].
    :param int h: Frame height H.
    """
    x, y = cuda.grid(2)
    if y >= h or x >= yuv.shape[1]:
        return
    nv12[y, x] = yuv[y, x]
    if (x % 2 == 0) and (y % 2 == 0):
        u = (int(yuv[h + y, x]) + int(yuv[h + y, x + 1]) + int(yuv[h + y + 1, x]) + int(yuv[h + y + 1, x + 1]) + 2) // 4
        v = (int(yuv[2 * h + y, x]) + int(yuv[2 * h + y, x + 1]) + int(yuv[2 * h + y + 1, x]) + int(yuv[2 * h + y + 1, x + 1]) + 2) // 4
        nv12[h + y // 2, x] = np.uint8(u)
        nv12[h + y // 2, x + 1] = np.uint8(v)


@cuda.jit(device=True)
def _cuda_point_to_segment_dist(px, py, ax, ay, bx, by):
    """
    Device: shortest Euclidean distance from point (px, py) to the line segment (ax, ay)-(bx, by).

    Used e.g. to rasterise polygon/line outlines on the GPU (a pixel is "on" an outline of thickness
    ``t`` when this distance is <= ``t / 2``).

    :return: Distance from the point to the nearest location on the segment.
    """
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom == 0.0:
        dx, dy = px - ax, py - ay
        return math.sqrt(dx * dx + dy * dy)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    dx, dy = px - (ax + t * abx), py - (ay + t * aby)
    return math.sqrt(dx * dx + dy * dy)


def get_nvc_decoder(video_path: Union[str, os.PathLike],
                    output_color_type=None,
                    gpu_id: int = 0,
                    use_device_memory: bool = False,
                    threaded: bool = False,
                    buffer_size: int = 16,
                    start_frame: int = 0):
    """
    Create an NVDEC hardware video decoder for GPU-accelerated frame reading.

    With ``threaded=False`` a random-access :class:`PyNvVideoCodec.SimpleDecoder` is returned
    (index/seek per frame). With ``threaded=True`` a :class:`PyNvVideoCodec.ThreadedDecoder` is
    returned, which decodes ``buffer_size`` frames ahead on a background thread and is streamed
    with ``get_batch_frames(n)``; this overlaps decoding on the NVDEC engine with downstream GPU
    compute/encode, and is the faster choice for sequential full-video passes.

    :param Union[str, os.PathLike] video_path: Path to the video file to decode.
    :param PyNvVideoCodec.OutputColorType output_color_type: Colour format of the decoded frames. Pass a member of the ``PyNvVideoCodec.OutputColorType`` enum, typically ``nvc.OutputColorType.RGB`` (interleaved HWC, shape ``(H, W, 3)``) or ``nvc.OutputColorType.RGBP`` (planar CHW, shape ``(3, H, W)``). Defaults to ``None``, which is resolved to ``nvc.OutputColorType.RGB`` inside the function - the default is ``None`` rather than the enum itself because ``PyNvVideoCodec`` is an optional dependency (``nvc`` may be ``None`` at import), so the enum cannot be referenced in the signature.
    :param int gpu_id: Index of the GPU to decode on. Default 0.
    :param bool use_device_memory: If True, keep decoded frames in GPU device memory; otherwise copy to host.
    :param bool threaded: If True, return a background-decoding ``ThreadedDecoder`` (sequential streaming); if False, a random-access ``SimpleDecoder``. Default False.
    :param int buffer_size: Number of frames the ``ThreadedDecoder`` decodes ahead (ignored when ``threaded=False``). Default 16.
    :param int start_frame: Frame index the ``ThreadedDecoder`` starts decoding from, via seeking (ignored when ``threaded=False``). Used to split a video into chunks across multiple NVDEC engines. Default 0.
    :raises SimBAGPUError: If PyNvVideoCodec is not installed, or if no CUDA GPU is available.
    :return: A configured ``PyNvVideoCodec.SimpleDecoder`` or ``PyNvVideoCodec.ThreadedDecoder``.

    :example:
    >>> import PyNvVideoCodec as nvc
    >>> from numba import cuda
    >>> import torch
    >>> # Random-access decode (default RGB output), read one frame into a CUDA device array:
    >>> decoder = get_nvc_decoder(video_path='video.mp4', use_device_memory=True)
    >>> frame = cuda.as_cuda_array(torch.from_dlpack(decoder[0]))   # (H, W, 3) uint8 on GPU
    >>> # Sequential streaming decode across a background thread, from frame 500:
    >>> decoder = get_nvc_decoder(video_path='video.mp4', output_color_type=nvc.OutputColorType.RGB, use_device_memory=True, threaded=True, buffer_size=32, start_frame=500)
    >>> batch = decoder.get_batch_frames(16)
    """

    from simba.utils.checks import (check_file_exist_and_readable,
                                    check_instance, check_int)
    from simba.utils.errors import SimBAGPUError

    if nvc is None:
        raise SimBAGPUError(msg='PyNvVideoCodec is not installed. Install it to use GPU accelerated video decoding.', source=get_nvc_decoder.__name__)
    if not _is_cuda_available()[0]:
        raise SimBAGPUError(msg='No GPU detected.', source=get_nvc_decoder.__name__)
    if output_color_type is None:
        output_color_type = nvc.OutputColorType.RGB
    check_file_exist_and_readable(file_path=video_path)
    check_int(name=f'{get_nvc_decoder.__name__} gpu_id', value=gpu_id, min_value=0)
    check_int(name=f'{get_nvc_decoder.__name__} buffer_size', value=buffer_size, min_value=1)
    check_int(name=f'{get_nvc_decoder.__name__} start_frame', value=start_frame, min_value=0)
    check_instance(source=f'{get_nvc_decoder.__name__} use_device_memory', instance=use_device_memory, accepted_types=(bool,))
    check_instance(source=f'{get_nvc_decoder.__name__} threaded', instance=threaded, accepted_types=(bool,))
    check_nvdec_decodable(video_path=video_path, gpu_id=gpu_id)
    if threaded:
        return nvc.CreateThreadedDecoder(encSource=str(video_path), bufferSize=buffer_size, gpuid=gpu_id, useDeviceMemory=use_device_memory, outputColorType=output_color_type, startFrame=start_frame)
    return nvc.SimpleDecoder(video_path, gpu_id=gpu_id, use_device_memory=use_device_memory, output_color_type=output_color_type)


_NVDEC_DECODABLE = set()
NVDEC_CODECS = ('h264', 'hevc', 'av1', 'vp8', 'vp9', 'mpeg1video', 'mpeg2video', 'mpeg4', 'mjpeg', 'vc1')   # ffprobe codec names NVDEC can handle (pixel format permitting)


def check_nvdec_decodable(video_path: Union[str, os.PathLike], gpu_id: int = 0) -> None:
    """
    Check that the GPU's NVDEC engine can decode a video, and raise a SimBA error if it cannot.

    Called by :func:`get_nvc_decoder` before every decoder is created. Without it, a video NVDEC does not support
    makes the threaded decoder abort the whole Python process (no exception, so the SimBA GUI closes without a
    message). Unsupported on most GPUs: H.264 that is not 8-bit 4:2:0 (e.g. 10-bit, 4:2:2, 4:4:4, and monochrome
    H.264 such as ``ffmpeg -vf format=gray -c:v libx264`` output), HEVC 4:2:2, VP9 4:4:4, and codecs NVDEC has no
    decoder for (raw video, PNG, ProRes, FFV1).

    The check: (i) rejects VP9 that is not 4:2:0 (read with ffprobe, as decoding it can crash even a probe), then
    (ii) opens the video with a non-threaded ``SimpleDecoder`` and decodes frame 0, which raises a catchable error
    for unsupported streams. Takes ~0.1 s. The result is cached per file (path, size, modification time).

    :param Union[str, os.PathLike] video_path: Path to the video.
    :param int gpu_id: Index of the GPU to decode on. Default 0.
    :raises SimBAGPUError: If the video cannot be decoded on this GPU.
    :return: None

    :example:
    >>> check_nvdec_decodable(video_path='project_folder/videos/Video_1.mp4')
    """
    from simba.utils.errors import SimBAGPUError

    if nvc is None:
        raise SimBAGPUError(msg='PyNvVideoCodec is not installed. Install it to use GPU accelerated video decoding.', source=check_nvdec_decodable.__name__)
    stat = os.stat(video_path)
    key = (os.path.abspath(str(video_path)), stat.st_size, stat.st_mtime)
    if key in _NVDEC_DECODABLE:
        return
    name = os.path.basename(str(video_path))
    fix = 'Use the CPU option, or re-encode the video to H.264 8-bit 4:2:0 (e.g. ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4).'
    try:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,pix_fmt', '-of', 'csv=p=0', str(video_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        fields = probe.stdout.decode('utf-8', errors='replace').strip().split(',')
        codec, pix_fmt = (fields + ['', ''])[:2]
    except (OSError, subprocess.SubprocessError):
        codec, pix_fmt = '', ''
    if codec == 'vp9' and pix_fmt and not pix_fmt.startswith('yuv420'):
        raise SimBAGPUError(msg=f'The GPU video decoder (NVDEC) cannot decode {name} (VP9, {pix_fmt}). {fix}', source=check_nvdec_decodable.__name__)
    n_gpus = len(_is_cuda_available()[1] or {})
    if gpu_id >= n_gpus:
        raise SimBAGPUError(msg=f'gpu_id={gpu_id} was requested but {n_gpus} GPU(s) were found (gpu_id counts from 0).', source=check_nvdec_decodable.__name__)
    try:
        decoder = nvc.SimpleDecoder(str(video_path), gpu_id=gpu_id, use_device_memory=False, output_color_type=nvc.OutputColorType.RGB)
        frame = decoder[0]
    except Exception as e:
        detail = f'{codec}, {pix_fmt}' if codec else 'unknown format'
        error_type = [line.split(':', 1)[1].strip() for line in str(e).splitlines() if line.strip().startswith('Error Type')]
        reason = error_type[0] if error_type else (str(e).strip().splitlines() or [type(e).__name__])[0]
        if 'Unsupported' in type(e).__name__ or (codec and codec not in NVDEC_CODECS):
            raise SimBAGPUError(msg=f'The GPU video decoder (NVDEC) cannot decode {name} ({detail}): this codec / pixel format is not supported by the GPU. {fix}', source=check_nvdec_decodable.__name__)
        raise SimBAGPUError(msg=f'The GPU video decoder (NVDEC) failed to open {name} ({detail}): {reason}. Possible causes: the GPU is out of memory or busy, or the file is damaged. If the problem persists, use the CPU option.', source=check_nvdec_decodable.__name__)
    if frame is None:
        raise SimBAGPUError(msg=f'The GPU video decoder (NVDEC) could not decode the first frame of {name}. {fix}', source=check_nvdec_decodable.__name__)
    del frame, decoder
    _NVDEC_DECODABLE.add(key)


def get_nvc_encoder(width: int,
                    height: int,
                    codec: str = 'h264',
                    fmt: str = 'ARGB',
                    use_cpu_input_buffer: bool = False,
                    fps: Optional[float] = None,
                    qp: Optional[int] = None):
    """
    Create an NVENC hardware video encoder for GPU-accelerated frame encoding.

    The counterpart of :func:`get_nvc_decoder`. The returned encoder's ``Encode(frame)`` takes one
    frame (a GPU device tensor when ``use_cpu_input_buffer=False``) and returns an elementary-stream
    bitstream chunk (empty until the encoder has buffered enough frames); call ``EndEncode()`` at the
    end to flush the tail. The raw elementary stream is then muxed into a container with
    ``ffmpeg -c copy``.

    :param int width: Frame width in pixels.
    :param int height: Frame height in pixels.
    :param str codec: NVENC codec, one of 'h264', 'hevc', 'av1'. Default 'h264'.
    :param str fmt: Input surface format the encoder expects (e.g. 'ARGB', 'NV12'). Default 'ARGB'.
    :param bool use_cpu_input_buffer: If True, frames are supplied from host memory; if False (default), from GPU device memory.
    :param Optional[float] fps: Frame rate written into the stream. Pass the source video's fps: if None, the stream says 30 fps, which overrides ``ffmpeg -framerate`` when muxing. Default None (kept for backwards compatibility).
    :param Optional[int] qp: If not None, constant-quality encoding (``rc='constqp'``, ``tuning_info='high_quality'``) at this quantization parameter (0-51; lower = higher quality, larger files). None: NVENC's default rate control. Default None.
    :raises SimBAGPUError: If PyNvVideoCodec is not installed, or if no CUDA GPU is available.
    :return: A configured ``PyNvVideoCodec`` encoder (B-frames off: with B-frames, muxing the raw stream with ``ffmpeg -c copy`` drops the first 3-4 frames).
    """

    from simba.utils.checks import (check_float, check_instance, check_int,
                                    check_str)
    from simba.utils.errors import SimBAGPUError

    if nvc is None:
        raise SimBAGPUError(msg='PyNvVideoCodec is not installed. Install it to use GPU accelerated video encoding.', source=get_nvc_encoder.__name__)
    if not _is_cuda_available()[0]:
        raise SimBAGPUError(msg='No GPU detected.', source=get_nvc_encoder.__name__)
    check_int(name=f'{get_nvc_encoder.__name__} width', value=width, min_value=1)
    check_int(name=f'{get_nvc_encoder.__name__} height', value=height, min_value=1)
    check_str(name=f'{get_nvc_encoder.__name__} codec', value=codec, options=('h264', 'hevc', 'av1'))
    check_str(name=f'{get_nvc_encoder.__name__} fmt', value=fmt)
    check_instance(source=f'{get_nvc_encoder.__name__} use_cpu_input_buffer', instance=use_cpu_input_buffer, accepted_types=(bool,))
    if fps is not None:
        check_float(name=f'{get_nvc_encoder.__name__} fps', value=fps, min_value=10e-6)
    if qp is not None:
        check_int(name=f'{get_nvc_encoder.__name__} qp', value=qp, min_value=0, max_value=51)
    if fps is None and qp is None:
        return nvc.CreateEncoder(width, height, fmt, use_cpu_input_buffer, codec=codec, bf=0)
    if qp is None:
        return nvc.CreateEncoder(width, height, fmt, use_cpu_input_buffer, codec=codec, bf=0, fps=fps)
    if fps is None:
        return nvc.CreateEncoder(width, height, fmt, use_cpu_input_buffer, codec=codec, bf=0, tuning_info='high_quality', rc='constqp', qp=qp)
    return nvc.CreateEncoder(width, height, fmt, use_cpu_input_buffer, codec=codec, bf=0, fps=fps, tuning_info='high_quality', rc='constqp', qp=qp)





NVC_SEEK_MARGIN = 32   # frames: a multi-worker chunk seeks this far before its start frame, then skips forward by timestamp


class NVCFrameAlignmentError(Exception):
    """
    A multi-worker NVDEC chunk could not find its exact start frame (irregular frame timestamps, or a seek that landed
    past the start). Deliberately not a SimBA error (those print when created): callers catch it and render the video
    with one worker instead.
    """
    pass


def get_nvc_frame_timing(video_path: Union[str, os.PathLike], gpu_id: int = 0) -> Optional[Tuple[int, float]]:
    """
    Get the timestamp of frame 0 and the exact timestamp step per frame of a video, in the units of the frame timestamps
    returned by the NVDEC decoder (the stream time base).

    Used by multi-worker NVENC classes: seeking a ``ThreadedDecoder`` to a start frame can land a few frames late
    (e.g. MPEG-4 with B-frames), so chunks identify frames by timestamp instead (see :func:`iter_nvc_frames`). The step
    is computed from the container's frame rate and time base (ffprobe), not from decoded timestamps, as those are
    rounded in some containers (e.g. MKV at 29.97 fps: 0, 33, 67, 100, ...). It is checked against the first two
    decoded frames.

    :param Union[str, os.PathLike] video_path: Path to the video file.
    :param int gpu_id: Index of the GPU to decode on. Default 0.
    :return: ``(timestamp of frame 0, timestamp step)``, or None if it cannot be determined (ffprobe fails, fewer than two frames, decoded frames without timestamps, or timestamps that do not match the frame rate). None means the video should be rendered by a single worker.
    :rtype: Optional[Tuple[int, float]]

    :example:
    >>> get_nvc_frame_timing(video_path='video.mp4')        # 25 fps, time base 1/12800
    (0, 512.0)
    """
    try:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate,time_base', '-of', 'default=noprint_wrappers=1', str(video_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        fields = dict(line.split('=', 1) for line in probe.stdout.decode('utf-8', errors='replace').splitlines() if '=' in line)
        fps, time_base = Fraction(fields['r_frame_rate']), Fraction(fields['time_base'])
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, ZeroDivisionError):
        return None
    if fps <= 0 or time_base <= 0:
        return None
    step = float(1 / (fps * time_base))
    decoder = get_nvc_decoder(video_path=video_path, gpu_id=gpu_id, use_device_memory=True, threaded=True, buffer_size=2, start_frame=0)
    try:
        frames = decoder.get_batch_frames(2)
        timestamps = [int(fr.timestamp) for fr in frames] if frames else []
    except (AttributeError, TypeError, ValueError):                       # PyNvVideoCodec version without frame timestamps
        timestamps = []
    finally:
        try:
            decoder.end()
        except Exception:
            pass
    if len(timestamps) < 2 or abs((timestamps[1] - timestamps[0]) - step) > 1:
        return None
    return timestamps[0], step


def iter_nvc_frames(decoder, start: int, end: int, buffer_size: int, timing: Optional[Tuple[int, float]] = None):
    """
    Yield ``(frame index, decoded frame)`` for frames ``[start, end)`` from a ``ThreadedDecoder``.

    Without ``timing`` the decoder must have been created with ``start_frame=start`` and frames are counted from
    ``start``. With ``timing`` (from :func:`get_nvc_frame_timing`) the decoder should be created with
    ``start_frame=max(0, start - NVC_SEEK_MARGIN)``: each frame's index is read from its timestamp, and frames before
    ``start`` are skipped, so the chunk starts on exactly frame ``start`` even when the seek lands late. From ``start``
    on, every timestamp must be exactly the next frame: a dropped or repeated frame raises, so the caller can fall
    back to counting frames with one worker (as the single-worker path and the pose data do).

    Each yielded frame lives in the decoder's buffer, which is recycled when the next batch is fetched: finish all GPU
    work on a frame (e.g. ``cuda.synchronize()``) before asking for the next one.

    :param decoder: A ``PyNvVideoCodec.ThreadedDecoder`` from :func:`get_nvc_decoder` with ``threaded=True``.
    :param int start: First frame index to yield.
    :param int end: Frame index to stop before.
    :param int buffer_size: Maximum number of frames fetched per batch.
    :param Optional[Tuple[int, float]] timing: ``(timestamp of frame 0, timestamp step)``, or None to count frames.
    :raises NVCFrameAlignmentError: With ``timing``: no frames after seeking, a timestamp off the frame grid, a first decoded frame after ``start``, or a dropped / repeated frame.
    :return: Generator of ``(int frame index, decoded frame)``. Stops early if the decoder runs out of frames (it then returns an empty batch).

    :example:
    >>> timing = get_nvc_frame_timing(video_path='video.mp4')
    >>> decoder = get_nvc_decoder(video_path='video.mp4', use_device_memory=True, threaded=True, start_frame=max(0, 500 - NVC_SEEK_MARGIN))
    >>> for frm_idx, frame in iter_nvc_frames(decoder=decoder, start=500, end=1000, buffer_size=16, timing=timing):
    ...     pass
    """
    i, first = start, True
    while i < end:
        frames = decoder.get_batch_frames(min(buffer_size, max(1, end - i)))
        if not frames:
            if timing is not None and first:                             # e.g. MKV: the decoder cannot seek, returns nothing
                raise NVCFrameAlignmentError(f'seeking to frame {start} returned no frames')
            return                                                       # decoder ran out (header frame count too high)
        for fr in frames:
            if timing is not None:
                pos = (int(fr.timestamp) - timing[0]) / timing[1]
                idx = int(round(pos))
                if abs(pos - idx) > 0.25:
                    raise NVCFrameAlignmentError(f'irregular frame timestamps (frame position {pos:.2f})')
                if first and idx > start:
                    raise NVCFrameAlignmentError(f'seeking to frame {start} landed on frame {idx}')
                first = False
                if idx < start:
                    continue                                             # before the chunk start (seek margin)
                if idx != i:
                    raise NVCFrameAlignmentError(f'expected frame {i} but decoded frame {idx} (dropped or repeated frame)')
            if i >= end:
                return
            yield i, fr
            i += 1


def check_nvc_frame_size(frame: np.ndarray, width: int, height: int, video_path: Union[str, os.PathLike], source: str = '') -> None:
    """
    Check that a decoded NVDEC frame is at least as large as the video size the output buffers were allocated for.

    The decoder can return frames of a different shape than the video metadata (e.g. phone videos with rotation
    metadata: metadata reports the rotated size, NVDEC returns the stored, unrotated frames). Drawing kernels index the
    frame by the metadata size, so a smaller frame would be read out of bounds. Larger frames are fine (NVDEC pads
    odd widths, e.g. 641 -> 642). Call on the first decoded frame.

    :param np.ndarray frame: Decoded frame as a (CUDA) array of shape (H, W, 3).
    :param int width: Video width from the metadata.
    :param int height: Video height from the metadata.
    :param Union[str, os.PathLike] video_path: Path to the video, for the error message.
    :param str source: Name of the caller, for the error message.
    :raises InvalidVideoFileError: If the frame is narrower or shorter than ``width`` x ``height``.

    :example:
    >>> check_nvc_frame_size(frame=rgb, width=1280, height=960, video_path='video.mp4')
    """
    from simba.utils.errors import InvalidVideoFileError
    if frame.shape[0] < height or frame.shape[1] < width:
        raise InvalidVideoFileError(msg=f'{os.path.basename(str(video_path))}: the GPU decoder returned {frame.shape[1]}x{frame.shape[0]} frames but the video metadata says {width}x{height} (e.g. rotation metadata). Use the CPU option for this video.', source=source)


def encode_nvc_frame(encoder, frame) -> bytes:
    """
    Encode one frame with an NVENC encoder from :func:`get_nvc_encoder`, safely.

    GPU work is asynchronous on both sides of ``encoder.Encode()``: the kernels that drew ``frame`` may still be
    running when Encode is called, and Encode returns before NVENC has finished copying ``frame``. A caller that then
    draws the next frame into the same buffer can overwrite it mid-copy, and the encoded frame shows (part of) the
    next frame - intermittently, more often when the GPU is busy. This waits for all GPU work before and after Encode.

    :param encoder: A ``PyNvVideoCodec`` encoder from :func:`get_nvc_encoder`.
    :param frame: Input frame on the GPU in the encoder's format (e.g. ARGB (H, W, 4) or planar YUV444 (3H, W) torch tensor).
    :return: Encoded bitstream for this call (may be empty: the encoder can buffer frames until ``EndEncode()``).
    :rtype: bytes

    :example:
    >>> encoder = get_nvc_encoder(width=1280, height=960, fps=25, fmt='ARGB')
    >>> raw.write(encode_nvc_frame(encoder=encoder, frame=argb_t[0]))
    """
    cuda.synchronize()                                                   # finish the kernels that drew the frame
    bitstream = encoder.Encode(frame)
    cuda.synchronize()                                                   # finish NVENC's copy of the frame before it is reused
    return bytes(bitstream) if bitstream else b''


def mux_nvc_raw_stream(raw_path: Union[str, os.PathLike], fps: float, save_path: Union[str, os.PathLike], source: str = '') -> None:
    """
    Wrap a raw NVENC elementary stream (e.g. ``.h264``) in an MP4 container with ``ffmpeg -c copy`` (no re-encode).

    :param Union[str, os.PathLike] raw_path: Path to the raw stream written from the encoder's bitstream.
    :param float fps: Frame rate of the stream.
    :param Union[str, os.PathLike] save_path: Output MP4 path. Overwritten if it exists.
    :param str source: Name of the caller, for error messages.
    :raises FFMPEGCodecGPUError: If ffmpeg fails or writes no file.

    :example:
    >>> mux_nvc_raw_stream(raw_path='video.mp4.raw.h264', fps=25, save_path='video.mp4')
    """
    from simba.utils.errors import FFMPEGCodecGPUError
    result = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(raw_path), '-c', 'copy', str(save_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.isfile(save_path):
        raise FFMPEGCodecGPUError(msg=f"FFmpeg could not write {save_path}: {result.stderr.decode('utf-8', errors='replace').strip()[-300:]}", source=source)


def concat_nvc_chunks(chunk_paths: List[str], save_path: Union[str, os.PathLike], source: str = '') -> None:
    """
    Join video chunks encoded with identical settings into one video with ffmpeg's concat demuxer (``-c copy``, no re-encode).
    The chunk list file is written next to the first chunk, so put the chunks in their own folder
    (e.g. ``tempfile.mkdtemp()``) and remove the folder afterwards.

    :param List[str] chunk_paths: Paths of the chunk videos, in playback order.
    :param Union[str, os.PathLike] save_path: Output video path. Overwritten if it exists.
    :param str source: Name of the caller, for error messages.
    :raises FFMPEGCodecGPUError: If ffmpeg fails or writes no file.

    :example:
    >>> concat_nvc_chunks(chunk_paths=['tmp/chunk_0.mp4', 'tmp/chunk_1.mp4'], save_path='video.mp4')
    """
    from simba.utils.errors import FFMPEGCodecGPUError
    list_path = os.path.join(os.path.dirname(chunk_paths[0]), 'chunks.txt')
    with open(list_path, 'w', encoding='utf-8') as fh:
        for p in chunk_paths:
            fh.write("file '{}'\n".format(str(p).replace('\\', '/').replace("'", "'\\''")))
    result = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', str(save_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.isfile(save_path):
        raise FFMPEGCodecGPUError(msg=f"FFmpeg could not join the video chunks into {save_path}: {result.stderr.decode('utf-8', errors='replace').strip()[-300:]}", source=source)
