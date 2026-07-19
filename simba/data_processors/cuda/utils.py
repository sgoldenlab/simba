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
from typing import Any, Dict, Tuple, Union

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
    if threaded:
        return nvc.CreateThreadedDecoder(encSource=str(video_path), bufferSize=buffer_size, gpuid=gpu_id, useDeviceMemory=use_device_memory, outputColorType=output_color_type, startFrame=start_frame)
    return nvc.SimpleDecoder(video_path, gpu_id=gpu_id, use_device_memory=use_device_memory, output_color_type=output_color_type)


def get_nvc_encoder(width: int,
                    height: int,
                    codec: str = 'h264',
                    fmt: str = 'ARGB',
                    use_cpu_input_buffer: bool = False):
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
    :raises SimBAGPUError: If PyNvVideoCodec is not installed, or if no CUDA GPU is available.
    :return: A configured ``PyNvVideoCodec`` encoder.
    """

    from simba.utils.checks import check_instance, check_int, check_str
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
    return nvc.CreateEncoder(width, height, fmt, use_cpu_input_buffer, codec=codec)



