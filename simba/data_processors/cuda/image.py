__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"


import math
import os
import time
from typing import Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal
try:
    import cupy as cp
    from cupyx.scipy.ndimage import rotate
except:
    import numpy as cp
    from scipy.ndimage import rotate

import warnings
from copy import deepcopy

import cv2
import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from simba.data_processors.cuda.utils import (_cuda_luminance_pixel_to_grey,
                                              _cuda_mse, _is_cuda_available)
from simba.mixins.image_mixin import ImageMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_string_value_is_valid_video_timestamp,
                                check_if_valid_img, check_if_valid_rgb_tuple,
                                check_instance, check_int,
                                check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_array, check_valid_boolean,
                                is_video_color)
from simba.utils.data import (create_color_palette,
                              find_frame_numbers_from_time_stamp)
from simba.utils.enums import Formats
from simba.utils.errors import (FFMPEGCodecGPUError, FrameRangeError,
                                InvalidInputError, SimBAGPUError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video,
    concatenate_videos_in_folder, create_directory, get_fn_ext,
    get_memory_usage_array, get_video_meta_data, read_df,
    read_img_batch_from_video, read_img_batch_from_video_gpu)
from simba.video_processors.async_frame_reader import (AsyncVideoFrameReader,
                                                       get_async_frame_batch)

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


PHOTOMETRIC = 'photometric'
DIGITAL = 'digital'
THREADS_PER_BLOCK = 2024

def create_average_frm_cupy(video_path: Union[str, os.PathLike],
                            start_frm: Optional[int] = None,
                            end_frm: Optional[int] = None,
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None,
                            save_path: Optional[Union[str, os.PathLike]] = None,
                            batch_size: Optional[int] = 3000,
                            verbose: Optional[bool] = False) -> Union[None, np.ndarray]:

    """
    Computes the average frame using GPU acceleration from a specified range of frames or time interval in a video file.
    This average frame typically used for background substraction.

    The function reads frames from the video, calculates their average, and optionally saves the result
    to a specified file. If `save_path` is provided, the average frame is saved as an image file;
    otherwise, the average frame is returned as a NumPy array.

    .. seealso::
       For CPU function see :func:`~simba.video_processors.video_processing.create_average_frm`.
       For CUDA function see :func:`~simba.data_processors.cuda.image.create_average_frm_cuda`

    :param Union[str, os.PathLike] video_path:  The path to the video file from which to extract frames.
    :param Optional[int] start_frm: The starting frame number (inclusive). Either `start_frm`/`end_frm` or `start_time`/`end_time` must be provided, but not both.
    :param Optional[int] end_frm:  The ending frame number (exclusive).
    :param Optional[str] start_time: The start time in the format 'HH:MM:SS' from which to begin extracting frames.
    :param Optional[str] end_time: The end time in the format 'HH:MM:SS' up to which frames should be extracted.
    :param Optional[Union[str, os.PathLike]] save_path: The path where the average frame image will be saved. If `None`, the average frame is returned as a NumPy array.
    :param Optional[int] batch_size: The number of frames to process in each batch. Default is 3000. Increase if your RAM allows it.
    :param Optional[bool] verbose:  If `True`, prints progress and informational messages during execution.
    :return: Returns `None` if the result is saved to `save_path`. Otherwise, returns the average frame as a NumPy array.

    :example:
    >>> create_average_frm_cupy(video_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_DOT_4_downsampled.mp4", verbose=True, start_frm=0, end_frm=9000)

    """

    def average_3d_stack(image_stack: np.ndarray) -> np.ndarray:
        num_frames, height, width, _ = image_stack.shape
        image_stack = cp.array(image_stack).astype(cp.float32)
        img = cp.clip(cp.sum(image_stack, axis=0) / num_frames, 0, 255).astype(cp.uint8)
        return img.get()

    if not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=create_average_frm_cupy.__name__)


    if ((start_frm is not None) or (end_frm is not None)) and ((start_time is not None) or (end_time is not None)):
        raise InvalidInputError(msg=f'Pass start_frm and end_frm OR start_time and end_time', source=create_average_frm_cupy.__name__)
    elif type(start_frm) != type(end_frm):
        raise InvalidInputError(msg=f'Pass start frame and end frame', source=create_average_frm_cupy.__name__)
    elif type(start_time) != type(end_time):
        raise InvalidInputError(msg=f'Pass start time and end time', source=create_average_frm_cupy.__name__)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=create_average_frm_cupy.__name__)
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    video_name = get_fn_ext(filepath=video_path)[1]
    if verbose:
        print(f'Getting average frame from {video_name}...')
    if (start_frm is not None) and (end_frm is not None):
        check_int(name='start_frm', value=start_frm, min_value=0, max_value=video_meta_data['frame_count'])
        check_int(name='end_frm', value=end_frm, min_value=0, max_value=video_meta_data['frame_count'])
        if start_frm > end_frm:
            raise InvalidInputError(msg=f'Start frame ({start_frm}) has to be before end frame ({end_frm}).', source=create_average_frm_cupy.__name__)
        frame_ids = list(range(start_frm, end_frm))
    elif (start_time is not None) and (end_time is not None):
        check_if_string_value_is_valid_video_timestamp(value=start_time, name=create_average_frm_cupy.__name__)
        check_if_string_value_is_valid_video_timestamp(value=end_time, name=create_average_frm_cupy.__name__)
        check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time, name=create_average_frm_cupy.__name__)
        check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start_time, video_path=video_path)
        frame_ids = find_frame_numbers_from_time_stamp(start_time=start_time, end_time=end_time, fps=video_meta_data['fps'])
    else:
        frame_ids = list(range(0, video_meta_data['frame_count']))
    frame_ids = [frame_ids[i:i+batch_size] for i in range(0,len(frame_ids),batch_size)]
    avg_imgs = []
    for batch_cnt in range(len(frame_ids)):
        start_idx, end_idx = frame_ids[batch_cnt][0], frame_ids[batch_cnt][-1]
        if start_idx == end_idx:
            continue
        imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=start_idx, end_frm=end_idx, verbose=verbose)
        imgs = np.stack(list(imgs.values()), axis=0)
        avg_imgs.append(average_3d_stack(image_stack=imgs))
    avg_img = average_3d_stack(image_stack=np.stack(avg_imgs, axis=0))
    if save_path is not None:
        cv2.imwrite(save_path, avg_img)
        if verbose:
            stdout_success(msg=f'Saved average frame at {save_path}', source=create_average_frm_cupy.__name__)
    else:
        return avg_img

def average_3d_stack_cupy(image_stack: np.ndarray) -> np.ndarray:
    num_frames, height, width, _ = image_stack.shape
    image_stack = cp.array(image_stack).astype(cp.float32)
    img = cp.clip(cp.sum(image_stack, axis=0) / num_frames, 0, 255).astype(cp.uint8)
    return img.get()

@cuda.jit()
def _average_3d_stack_cuda_kernel(data, results):
    x, y, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > data.shape[0] - 1 or y > data.shape[1] - 1 or x > data.shape[2] - 1:
        return
    else:
        sum_value = 0.0
        for n in range(data.shape[0]):
            sum_value += data[n, y, x, i]
        results[y, x, i] = sum_value / data.shape[0]

def _average_3d_stack_cuda(image_stack: np.ndarray) -> np.ndarray:
    check_instance(source=_average_3d_stack_cuda.__name__, instance=image_stack, accepted_types=(np.ndarray,))
    check_if_valid_img(data=image_stack[0], source=_average_3d_stack_cuda.__name__)
    if image_stack.ndim != 4:
        return image_stack
    x = np.ascontiguousarray(image_stack)
    x_dev = cuda.to_device(x)
    results = cuda.device_array((x.shape[1], x.shape[2], x.shape[3]), dtype=np.float32)
    grid_x = (x.shape[1] + 16 - 1) // 16
    grid_y = (x.shape[2] + 16 - 1) // 16
    grid_z = 3
    threads_per_block = (16, 16, 1)
    blocks_per_grid = (grid_y, grid_x, grid_z)
    _average_3d_stack_cuda_kernel[blocks_per_grid, threads_per_block](x_dev, results)
    results = results.copy_to_host()
    return results



def create_average_frm_cuda(video_path: Union[str, os.PathLike],
                            start_frm: Optional[int] = None,
                            end_frm: Optional[int] = None,
                            start_time: Optional[str] = None,
                            end_time: Optional[str] = None,
                            save_path: Optional[Union[str, os.PathLike]] = None,
                            batch_size: Optional[int] = 6000,
                            verbose: Optional[bool] = False) -> Union[None, np.ndarray]:
    """
    Computes the average frame using GPU acceleration from a specified range of frames or time interval in a video file.
    This average frame typically used for background substraction.


    The function reads frames from the video, calculates their average, and optionally saves the result
    to a specified file. If `save_path` is provided, the average frame is saved as an image file;
    otherwise, the average frame is returned as a NumPy array.

    .. seealso::
       For CuPy function see :func:`~simba.data_processors.cuda.image.create_average_frm_cupy`.
       For CPU function see :func:`~simba.video_processors.video_processing.create_average_frm`.

    :param Union[str, os.PathLike] video_path:  The path to the video file from which to extract frames.
    :param Optional[int] start_frm: The starting frame number (inclusive). Either `start_frm`/`end_frm` or `start_time`/`end_time` must be provided, but not both.
    :param Optional[int] end_frm:  The ending frame number (exclusive).
    :param Optional[str] start_time: The start time in the format 'HH:MM:SS' from which to begin extracting frames.
    :param Optional[str] end_time: The end time in the format 'HH:MM:SS' up to which frames should be extracted.
    :param Optional[Union[str, os.PathLike]] save_path: The path where the average frame image will be saved. If `None`, the average frame is returned as a NumPy array.
    :param Optional[int] batch_size: The number of frames to process in each batch. Default is 3000. Increase if your RAM allows it.
    :param Optional[bool] verbose:  If `True`, prints progress and informational messages during execution.
    :return: Returns `None` if the result is saved to `save_path`. Otherwise, returns the average frame as a NumPy array.

    :example:
    >>> create_average_frm_cuda(video_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_DOT_4_downsampled.mp4", verbose=True, start_frm=0, end_frm=9000)

    """

    if not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)",
                                  source=create_average_frm_cuda.__name__)

    if ((start_frm is not None) or (end_frm is not None)) and ((start_time is not None) or (end_time is not None)):
        raise InvalidInputError(msg=f'Pass start_frm and end_frm OR start_time and end_time',
                                source=create_average_frm_cuda.__name__)
    elif type(start_frm) != type(end_frm):
        raise InvalidInputError(msg=f'Pass start frame and end frame', source=create_average_frm_cuda.__name__)
    elif type(start_time) != type(end_time):
        raise InvalidInputError(msg=f'Pass start time and end time', source=create_average_frm_cuda.__name__)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=create_average_frm_cuda.__name__)
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    video_name = get_fn_ext(filepath=video_path)[1]
    if verbose:
        print(f'Getting average frame from {video_name}...')
    if (start_frm is not None) and (end_frm is not None):
        check_int(name='start_frm', value=start_frm, min_value=0, max_value=video_meta_data['frame_count'])
        check_int(name='end_frm', value=end_frm, min_value=0, max_value=video_meta_data['frame_count'])
        if start_frm > end_frm:
            raise InvalidInputError(msg=f'Start frame ({start_frm}) has to be before end frame ({end_frm}).', source=create_average_frm_cuda.__name__)
        frame_ids = list(range(start_frm, end_frm))
    elif (start_time is not None) and (end_time is not None):
        check_if_string_value_is_valid_video_timestamp(value=start_time, name=create_average_frm_cuda.__name__)
        check_if_string_value_is_valid_video_timestamp(value=end_time, name=create_average_frm_cuda.__name__)
        check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time, name=create_average_frm_cuda.__name__)
        check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start_time, video_path=video_path)
        frame_ids = find_frame_numbers_from_time_stamp(start_time=start_time, end_time=end_time, fps=video_meta_data['fps'])
    else:
        frame_ids = list(range(0, video_meta_data['frame_count']))
    frame_ids = [frame_ids[i:i + batch_size] for i in range(0, len(frame_ids), batch_size)]
    avg_imgs = []
    for batch_cnt in range(len(frame_ids)):
        start_idx, end_idx = frame_ids[batch_cnt][0], frame_ids[batch_cnt][-1]
        if start_idx == end_idx:
            continue
        imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=start_idx, end_frm=end_idx, verbose=verbose)
        avg_imgs.append(_average_3d_stack_cuda(image_stack=np.stack(list(imgs.values()), axis=0)))
    avg_img = average_3d_stack_cupy(image_stack=np.stack(avg_imgs, axis=0))
    if save_path is not None:
        cv2.imwrite(save_path, avg_img)
        if verbose:
            stdout_success(msg=f'Saved average frame at {save_path}', source=create_average_frm_cuda.__name__)
    else:
        return avg_img



@cuda.jit()
def _photometric(data, results):
    y, x, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > results.shape[0] - 1 or x > results.shape[1]  - 1 or y > results.shape[2] - 1:
        return
    else:
        r, g, b = data[i][x][y][0], data[i][x][y][1], data[i][x][y][2]
        results[i][x][y] = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)

@cuda.jit()
def _digital(data, results):
    y, x, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > results.shape[0] - 1 or x > results.shape[1]  - 1 or y > results.shape[2] - 1:
        return
    else:
        r, g, b = data[i][x][y][0], data[i][x][y][1], data[i][x][y][2]
        results[i][x][y] = (0.299 * r) + (0.587 * g) + (0.114 * b)

def img_stack_brightness(x: np.ndarray,
                         method: Optional[Literal['photometric', 'digital']] = 'digital',
                         ignore_black: Optional[bool] = True) -> np.ndarray:
    """
    Calculate the average brightness of a stack of images using a specified method.


    - **Photometric Method**: The brightness is calculated using the formula:

    .. math::
       \text{brightness} = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B

    - **Digital Method**: The brightness is calculated using the formula:

    .. math::
       \text{brightness} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B

    .. selalso::
       For CPU function see :func:`~simba.mixins.image_mixin.ImageMixin.brightness_intensity`.

    :param np.ndarray x: A 4D array of images with dimensions (N, H, W, C), where N is the number of images, H and W are the height and width, and C is the number of channels (RGB).
    :param Optional[Literal['photometric', 'digital']] method:  The method to use for calculating brightness. It can be 'photometric' for  the standard luminance calculation or 'digital' for an alternative set of coefficients. Default is 'digital'.
    :param Optional[bool] ignore_black: If True, black pixels (i.e., pixels with brightness value 0) will be ignored in the calculation of the average brightness. Default is True.
    :return np.ndarray: A 1D array of average brightness values for each image in the stack. If `ignore_black` is True, black pixels are ignored in the averaging process.


    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4_downsampled.mp4", start_frm=0, end_frm=5000)
    >>> imgs = np.stack(list(imgs.values()), axis=0)
    >>> x = img_stack_brightness(x=imgs)
    """

    check_instance(source=img_stack_brightness.__name__, instance=x, accepted_types=(np.ndarray,))
    check_if_valid_img(data=x[0], source=img_stack_brightness.__name__)
    x = np.ascontiguousarray(x).astype(np.uint8)
    if x.ndim == 4:
        grid_x = (x.shape[1] + 16 - 1) // 16
        grid_y = (x.shape[2] + 16 - 1) // 16
        grid_z = x.shape[0]
        threads_per_block = (16, 16, 1)
        blocks_per_grid = (grid_y, grid_x, grid_z)
        x_dev = cuda.to_device(x)
        results = cuda.device_array((x.shape[0], x.shape[1], x.shape[2]), dtype=np.uint8)
        if method == PHOTOMETRIC:
            _photometric[blocks_per_grid, threads_per_block](x_dev, results)
        else:
            _digital[blocks_per_grid, threads_per_block](x_dev, results)
        results = results.copy_to_host()
        if ignore_black:
            masked_array = np.ma.masked_equal(results, 0)
            results = np.mean(masked_array, axis=(1, 2)).filled(0)
    else:
        results = deepcopy(x)
        results = np.mean(results, axis=(1, 2))

    return results



@cuda.jit()
def _grey_mse(data, ref_img, stride, batch_cnt, mse_arr):
    y, x, i = cuda.grid(3)
    stride = stride[0]
    batch_cnt = batch_cnt[0]
    if batch_cnt == 0:
        if (i - stride) < 0 or x < 0 or y < 0:
            return
        else:
            if i < 0 or x < 0 or y < 0:
                return
    if i > mse_arr.shape[0] - 1 or x > mse_arr.shape[1]  - 1 or y > mse_arr.shape[2] - 1:
        return
    else:
        img_val = data[i][x][y]
        if i == 0:
            prev_val = ref_img[x][y]
        else:
            img_val = data[i][x][y]
            prev_val = data[i - stride][x][y]
        mse_arr[i][x][y] = (img_val - prev_val) ** 2


@cuda.jit()
def _rgb_mse(data, ref_img, stride, batch_cnt, mse_arr):
    y, x, i = cuda.grid(3)
    stride = stride[0]
    batch_cnt = batch_cnt[0]
    if batch_cnt == 0:
        if (i - stride) < 0 or x < 0 or y < 0:
            return
        else:
            if i < 0 or x < 0 or y < 0:
                return
    if i > mse_arr.shape[0] - 1 or x > mse_arr.shape[1]  - 1 or y > mse_arr.shape[2] - 1:
        return
    else:
        img_val = data[i][x][y]
        if i != 0:
            prev_val = data[i - stride][x][y]
        else:
            prev_val = ref_img[x][y]
        r_diff = (img_val[0] - prev_val[0]) ** 2
        g_diff = (img_val[1] - prev_val[1]) ** 2
        b_diff = (img_val[2] - prev_val[2]) ** 2
        mse_arr[i][x][y] = r_diff + g_diff + b_diff

def stack_sliding_mse(x: np.ndarray,
                      stride: Optional[int] = 1,
                      batch_size: Optional[int] = 1000) -> np.ndarray:
    """
    Computes the Mean Squared Error (MSE) between each image in a stack and a reference image,
    where the reference image is determined by a sliding window approach with a specified stride.
    The function is optimized for large image stacks by processing them in batches.

    .. seealso::
       For CPU function see :func:`~simba.mixins.image_mixin.ImageMixin.img_stack_mse` and
       :func:`~simba.mixins.image_mixin.ImageMixin.img_sliding_mse`.

    .. math::
       \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

    :param np.ndarray x: Input array of images, where the first dimension corresponds to the stack of images. The array should be either 3D (height, width, channels) or 4D (batch, height, width, channels).
    :param Optional[int] stride: The stride or step size for the sliding window that determines the reference image. Defaults to 1, meaning the previous image in the stack is used as the reference.
    :param Optional[int] batch_size: The number of images to process in a single batch. Larger batch sizes may improve performance but require more GPU memory.  Defaults to 1000.
    :return: A 1D NumPy array containing the MSE for each image in the stack compared to its corresponding reference image. The length of the array is equal to the number of images in the input stack.
    :rtype: np.ndarray

    """

    check_instance(source=stack_sliding_mse.__name__, instance=x, accepted_types=(np.ndarray,))
    check_if_valid_img(data=x[0], source=stack_sliding_mse.__name__)
    check_valid_array(data=x, source=stack_sliding_mse.__name__, accepted_ndims=[3, 4])
    stride = np.array([stride], dtype=np.int32)
    stride_dev = cuda.to_device(stride)
    out = np.full((x.shape[0]), fill_value=0.0, dtype=np.float32)
    for batch_cnt, l in enumerate(range(0, x.shape[0], batch_size)):
        r = l + batch_size
        batch_x = x[l:r]
        if batch_cnt != 0:
            if x.ndim == 3:
                ref_img = x[l-stride].astype(np.uint8).reshape(x.shape[1], x.shape[2])
            else:
                ref_img = x[l-stride].astype(np.uint8).reshape(x.shape[1], x.shape[2], 3)
        else:
            ref_img = np.full_like(x[l], dtype=np.uint8, fill_value=0)
        ref_img = ref_img.astype(np.uint8)
        grid_x = (batch_x.shape[1] + 16 - 1) // 16
        grid_y = (batch_x.shape[2] + 16 - 1) // 16
        grid_z = batch_x.shape[0]
        threads_per_block = (16, 16, 1)
        blocks_per_grid = (grid_y, grid_x, grid_z)
        ref_img_dev = cuda.to_device(ref_img)
        x_dev = cuda.to_device(batch_x)
        results = cuda.device_array((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]), dtype=np.uint8)
        batch_cnt_dev = np.array([batch_cnt], dtype=np.int32)
        if x.ndim == 3:
            _grey_mse[blocks_per_grid, threads_per_block](x_dev, ref_img_dev, stride_dev, batch_cnt_dev,  results)
        else:
            _rgb_mse[blocks_per_grid, threads_per_block](x_dev, ref_img_dev, stride_dev, batch_cnt_dev,  results)
        results = results.copy_to_host()
        results = np.mean(results, axis=(1, 2))
        out[l:r] = results
    return out


def img_stack_to_grayscale_cupy(imgs: Union[np.ndarray, cp.ndarray],
                                batch_size: Optional[int] = 250) -> np.ndarray:
    """
    Converts a stack of color images to grayscale using GPU acceleration with CuPy.

    .. seealso::
       For CPU function single images :func:`~simba.mixins.image_mixin.ImageMixin.img_to_greyscale` and
       :func:`~simba.mixins.image_mixin.ImageMixin.img_stack_to_greyscale` for stack. For CUDA JIT, see
       :func:`~simba.data_processors.cuda.image.img_stack_to_grayscale_cuda`.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/img_stack_to_grayscale_cupy.csv
       :widths: 10, 90
       :align: center
       :class: simba-table
       :header-rows: 1

    :param np.ndarray imgs: A 4D NumPy or CuPy array representing a stack of images with shape (num_images, height, width, channels). The images are expected to have 3 channels (RGB).
    :param Optional[int] batch_size: The number of images to process in each batch. Defaults to 250. Adjust this parameter to fit your GPU's memory capacity.
    :return np.ndarray: m A 3D NumPy or CuPy array of shape (num_images, height, width) containing the grayscale images. If the input array is not 4D, the function returns the input as is.

    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_IOT_1_cropped.mp4", verbose=False, start_frm=0, end_frm=i)
    >>> imgs = np.stack(list(imgs.values()), axis=0).astype(np.uint8)
    >>> gray_imgs = img_stack_to_grayscale_cupy(imgs=imgs)
    """


    check_instance(source=img_stack_to_grayscale_cupy.__name__, instance=imgs, accepted_types=(np.ndarray, cp.ndarray))
    check_if_valid_img(data=imgs[0], source=img_stack_to_grayscale_cupy.__name__)
    if imgs.ndim != 4:
        return imgs
    results = cp.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
    n = int(np.ceil((imgs.shape[0] / batch_size)))
    imgs = np.array_split(imgs, n)
    start = 0
    for i in range(len(imgs)):
        img_batch = cp.array(imgs[i])
        batch_cnt = img_batch.shape[0]
        end = start + batch_cnt
        vals = (0.07 * img_batch[:, :, :, 2] + 0.72 * img_batch[:, :, :, 1] + 0.21 * img_batch[:, :, :, 0])
        results[start:end] = vals.astype(cp.uint8)
        start = end
    if isinstance(imgs, np.ndarray):
        return results.get()
    else:
        return results



@cuda.jit()
def _img_stack_to_grayscale(data, results):
    y, x, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > results.shape[0] - 1 or x > results.shape[1] - 1 or y > results.shape[2] - 1:
        return
    else:
        b = 0.07 * data[i][x][y][2]
        g = 0.72 * data[i][x][y][1]
        r = 0.21 * data[i][x][y][0]
        val = b + g + r
        results[i][x][y] = val

def img_stack_to_grayscale_cuda(x: np.ndarray) -> np.ndarray:
    """
    Convert image stack to grayscale using CUDA.

    .. seealso::
       For CPU function single images :func:`~simba.mixins.image_mixin.ImageMixin.img_to_greyscale` and
       :func:`~simba.mixins.image_mixin.ImageMixin.img_stack_to_greyscale` for stack. For CuPy, see
       :func:`~simba.data_processors.cuda.image.img_stack_to_grayscale_cupy`.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/img_stack_to_grayscale_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :class: simba-table
       :header-rows: 1

    :param np.ndarray x: 4d array of color images in numpy format.
    :return np.ndarray: 3D array of greyscaled images.

    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"/mnt/c/troubleshooting/mitra/project_folder/videos/temp_2/592_MA147_Gq_Saline_0516_downsampled.mp4", verbose=False, start_frm=0, end_frm=i)
    >>> imgs = np.stack(list(imgs.values()), axis=0).astype(np.uint8)
    >>> grey_images = img_stack_to_grayscale_cuda(x=imgs)
    """
    check_instance(source=img_stack_to_grayscale_cuda.__name__, instance=x, accepted_types=(np.ndarray,))
    check_if_valid_img(data=x[0], source=img_stack_to_grayscale_cuda.__name__)
    if x.ndim != 4:
        return x
    x = np.ascontiguousarray(x).astype(np.uint8)
    x_dev = cuda.to_device(x)
    results = cuda.device_array((x.shape[0], x.shape[1], x.shape[2]), dtype=np.uint8)
    grid_x = (x.shape[1] + 16 - 1) // 16
    grid_y = (x.shape[2] + 16 - 1) // 16
    grid_z = x.shape[0]
    threads_per_block = (16, 16, 1)
    blocks_per_grid = (grid_y, grid_x, grid_z)
    _img_stack_to_grayscale[blocks_per_grid, threads_per_block](x_dev, results)
    results = results.copy_to_host()
    return results


def img_stack_to_bw(imgs: np.ndarray,
                    lower_thresh: Optional[int] = 100,
                    upper_thresh: Optional[int] = 100,
                    invert: Optional[bool] = True,
                    batch_size: Optional[int] = 1000) -> np.ndarray:
    """

    Converts a stack of RGB images to binary (black and white) images based on given threshold values using GPU acceleration.

    This function processes a 4D stack of images, converting each RGB image to a binary image using
    specified lower and upper threshold values. The conversion can be inverted if desired, and the
    processing is done in batches for efficiency.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/img_stack_to_bw.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    .. seealso::
       :func:`simba.mixins.image_mixin.ImageMixin.img_to_bw`
       :func:`simba.mixins.image_mixin.ImageMixin.img_stack_to_bw`

    :param np.ndarray imgs:  A 4D NumPy array representing a stack of RGB images, with shape (N, H, W, C).
    :param Optional[int]  lower_thresh: The lower threshold value. Pixel values below this threshold are set to 0 (or 1 if `invert` is True). Default is 100.
    :param Optional[int]  upper_thresh: The upper threshold value. Pixel values above this threshold are set to 1 (or 0 if `invert` is True). Default is 100.
    :param Optional[bool] invert: If True, the binary conversion is inverted, meaning that values below `lower_thresh` become 1, and values above `upper_thresh` become 0. Default is True.
    :param Optional[int]  batch_size: The number of images to process in a single batch. This helps manage memory usage for large stacks of images. Default is 1000.
    :return: A 3D NumPy array of shape (N, H, W), where each image has been converted to a binary format with pixel values of either 0 or 1.
    :rtype: np.ndarray
    """

    check_valid_array(data=imgs, source=img_stack_to_bw.__name__, accepted_ndims=(4,))
    check_int(name='lower_thresh', value=lower_thresh, max_value=255, min_value=0)
    check_int(name='upper_thresh', value=upper_thresh, max_value=255, min_value=0)
    check_int(name='batch_size', value=batch_size, min_value=1)
    results = cp.full((imgs.shape[0], imgs.shape[1], imgs.shape[2]), fill_value=cp.nan, dtype=cp.uint8)

    for l in range(0, imgs.shape[0], batch_size):
        r = l + batch_size
        batch_imgs = cp.array(imgs[l:r]).astype(cp.uint8)
        img_mean = cp.sum(batch_imgs, axis=3) / 3
        if not invert:
            batch_imgs = cp.where(img_mean < lower_thresh, 0, img_mean)
            batch_imgs = cp.where(batch_imgs > upper_thresh, 1, batch_imgs).astype(cp.uint8)
        else:
            batch_imgs = cp.where(img_mean < lower_thresh, 1, img_mean)
            batch_imgs = cp.where(batch_imgs > upper_thresh, 0, batch_imgs).astype(cp.uint8)

        results[l:r] = batch_imgs

    return results.get()

def segment_img_stack_vertical(imgs: np.ndarray,
                               pct: float,
                               left: bool,
                               right: bool) -> np.ndarray:
    """
    Segment a stack of images vertically based on a given percentage using GPU acceleration. For example, return the left half, right half, or senter half of each image in the stack.

    .. note::
       If both left and right are true, the center portion is returned.

    .. seealso::
       :func:`simba.mixins.image_mixin.ImageMixin.segment_img_vertical`

    :param np.ndarray imgs: A 3D or 4D NumPy array representing a stack of images. The array should have shape (N, H, W) for grayscale images or (N, H, W, C) for color images.
    :param float pct: The percentage of the image width to be used for segmentation.  This value should be between a small positive value (e.g., 10e-6) and 0.99.
    :param bool left: If True, the left side of the image stack will be segmented.
    :param bool right: If True, the right side of the image stack will be segmented.
    :return: A NumPy array containing the segmented images, with the same number of dimensions as the input.
    :rtype: np.ndarray
    """

    check_valid_boolean(value=[left, right], source=segment_img_stack_vertical.__name__)
    check_float(name=f'{segment_img_stack_vertical.__name__} pct', value=pct, min_value=10e-6, max_value=0.99)
    check_valid_array(data=imgs, source=f'{segment_img_stack_vertical.__name__} imgs', accepted_ndims=(3, 4,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if not left and not right:
        raise InvalidInputError(msg='left are right argument are both False. Set one or both to True.', source=segment_img_stack_vertical.__name__)
    imgs = cp.array(imgs).astype(cp.uint8)
    h, w = imgs[0].shape[0], imgs[0].shape[1]
    px_crop = int(w * pct)
    if left and not right:
        imgs = imgs[:, :, :px_crop]
    elif right and not left:
        imgs = imgs[:, :, imgs.shape[2] - px_crop:]
    else:
        imgs = imgs[:, :, int(px_crop/2):int(imgs.shape[2] - (px_crop/2))]
    return imgs.get()


def segment_img_stack_horizontal(imgs: np.ndarray,
                                 pct: float,
                                 upper: Optional[bool] = False,
                                 lower: Optional[bool] = False) -> np.ndarray:

    """
    Segment a stack of images horizontally based on a given percentage using GPU acceleration. For example, return the top half, bottom half, or center half of each image in the stack.

    .. note::
       If both top and bottom are true, the center portion is returned.

    .. seealso::
       :func:`simba.mixins.image_mixin.ImageMixin.segment_img_stack_horizontal`

    :param np.ndarray imgs: A 3D or 4D NumPy array representing a stack of images. The array should have shape (N, H, W) for grayscale images or (N, H, W, C) for color images.
    :param float pct: The percentage of the image width to be used for segmentation.  This value should be between a small positive value (e.g., 10e-6) and 0.99.
    :param bool upper: If True, the top part of the image stack will be segmented.
    :param bool lower: If True, the bottom part of the image stack will be segmented.
    :return: A NumPy array containing the segmented images, with the same number of dimensions as the input.
    :rtype: np.ndarray
    """

    check_valid_boolean(value=[upper, lower], source=segment_img_stack_horizontal.__name__)
    check_float(name=f'{segment_img_stack_horizontal.__name__} pct', value=pct, min_value=10e-6, max_value=0.99)
    check_valid_array(data=imgs, source=f'{segment_img_stack_vertical.__name__} imgs', accepted_ndims=(3, 4,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if not upper and not lower:
        raise InvalidInputError(msg='upper and lower argument are both False. Set one or both to True.', source=segment_img_stack_horizontal.__name__)
    imgs = cp.array(imgs).astype(cp.uint8)
    h, w = imgs[0].shape[0], imgs[0].shape[1]
    px_crop = int(h * pct)
    if upper and not lower:
        imgs = imgs[: , :px_crop, :]
    elif not upper and lower:
        imgs = imgs[:, imgs.shape[0] - px_crop :, :]
    else:
        imgs = imgs[:, int(px_crop/2):int((imgs.shape[0] - px_crop) / 2), :]

    return imgs.get()



@cuda.jit(device=True)
def _cuda_is_inside_polygon(x, y, polygon_vertices):
    """
    Checks if the pixel location is inside the polygon.

    :param int x: Pixel x location.
    :param int y: Pixel y location.
    :param np.ndarray polygon_vertices: 2-dimensional array representing the x and y coordinates of the polygon vertices.
    :return: Boolean representing if the x and y are located in the polygon.
    """

    n = len(polygon_vertices)
    p2x, p2y, xints, inside = 0.0, 0.0, 0.0, False
    p1x, p1y = polygon_vertices[0]
    for j in range(n + 1):
        p2x, p2y = polygon_vertices[j % n]
        if ((y > min(p1y, p2y)) and (y <= max(p1y, p2y)) and (x <= max(p1x, p2x))):
            if p1y != p2y:
                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xints:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside



@cuda.jit(device=True)
def _cuda_is_inside_circle(x, y, circle_x, circle_y, circle_r):
    """
    Device func to check if the pixel location is inside a circle.

    :param int x: Pixel x location.
    :param int y: Pixel y location.
    :param int circle_x: Center of circle x coordinate.
    :param int circle_y: Center of circle y coordinate.
    :param int y: Circle radius.
    :return: Boolean representing if the x and y are located in the circle.
    """

    p = (math.sqrt((x - circle_x) ** 2 + (y - circle_y) ** 2))
    if p <= circle_r:
        return True
    else:
        return False
@cuda.jit()
def _cuda_create_rectangle_masks(shapes, imgs, results, bboxes):
    """
    CUDA kernel to apply rectangular masks to a batch of images.
    """
    n, y, x = cuda.grid(3)
    if n >= imgs.shape[0]:
        return

    x_min = bboxes[n, 0]
    y_min = bboxes[n, 1]
    x_max = bboxes[n, 2]
    y_max = bboxes[n, 3]

    max_w = x_max - x_min
    max_h = y_max - y_min

    if x >= max_w or y >= max_h:
        return

    x_input = x + x_min
    y_input = y + y_min

    polygon = shapes[n]

    if _cuda_is_inside_polygon(x_input, y_input, polygon):
        if imgs.ndim == 4:
            for c in range(imgs.shape[3]):
                results[n, y, x, c] = imgs[n, y_input, x_input, c]
        else:
            results[n, y, x] = imgs[n, y_input, x_input]

@cuda.jit()
def _cuda_create_circle_masks(shapes, imgs, results, bboxes):
    """
    CUDA kernel to apply circular masks to a batch of images.
    """
    n, y, x = cuda.grid(3)
    if n >= imgs.shape[0]:
        return

    x_min = bboxes[n, 0]
    y_min = bboxes[n, 1]
    x_max = bboxes[n, 2]
    y_max = bboxes[n, 3]

    max_w = x_max - x_min
    max_h = y_max - y_min

    if x >= max_w or y >= max_h:
        return

    x_input = x + x_min
    y_input = y + y_min

    circle_x = shapes[n, 0]
    circle_y = shapes[n, 1]
    circle_r = shapes[n, 2]


    if _cuda_is_inside_circle(x_input, y_input, circle_x, circle_y, circle_r):
        if imgs.ndim == 4:
            for c in range(imgs.shape[3]):
                results[n, y, x, c] = imgs[n, y_input, x_input, c]
        else:
            results[n, y, x] = imgs[n, y_input, x_input]


def _get_bboxes(shapes):
    """
    Helper to get geometries in :func:`simba.data_processors.cuda.image.slice_imgs`.
    """
    bboxes = []
    for shape in shapes:
        if shape.shape[0] == 3:  # circle: [cx, cy, r]
            cx, cy, r = shape
            x_min = int(np.floor(cx - r))
            y_min = int(np.floor(cy - r))
            x_max = int(np.ceil(cx + r))
            y_max = int(np.ceil(cy + r))
        else:
            xs = shape[:, 0]
            ys = shape[:, 1]
            x_min = int(np.floor(xs.min()))
            y_min = int(np.floor(ys.min()))
            x_max = int(np.ceil(xs.max()))
            y_max = int(np.ceil(ys.max()))
        bboxes.append([x_min, y_min, x_max, y_max])
    return np.array(bboxes, dtype=np.int32)

def slice_imgs(video_path: Union[str, os.PathLike],
               shapes: np.ndarray,
               batch_size: int = 1000,
               verbose: bool = True,
               save_dir: Optional[Union[str, os.PathLike]] = None):
    """
    Slice frames from a video based on given polygon or circle coordinates, and return or save masked/cropped frame regions using GPU acceleration.

     This function supports two types of shapes:
    - Polygon: array of shape (N, M, 2), where N = number of frames, M = number of polygon vertices.
    - Circle: array of shape (N, 3), where each row represents [center_x, center_y, radius].

    :param Union[str, os.PathLike] video_path: Path to the input video file.
    :param np.ndarray shapes: Array of polygon coordinates or circle parameters for each frame. - Polygon: shape = (n_frames, n_vertices, 2) - Circle: shape = (n_frames, 3)
    :param int batch_size: Number of frames to process per batch during GPU processing. Default 1000.
    :param bool verbose: Whether to print progress and status messages. Default True.
    :param Optional[Union[str, os.PathLike]] save_dir: If provided, the masked/cropped video will be saved in this directory. Otherwise, the cropped image stack will be returned.

    .. video:: _static/img/simba.sandbox.cuda_slice_w_crop.slice_imgs.webm
       :width: 900
       :loop:

    .. video:: _static/img/slice_imgs_gpu.webm
       :width: 800
       :autoplay:
       :loop:

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/slice_imgs.csv
       :widths: 10, 90
       :align: center
       :class: simba-table
       :header-rows: 1

    .. note::
       For CPU multicore implementation, see :func:`simba.mixins.image_mixin.ImageMixin.slice_shapes_in_imgs`.
       For single core process, see :func:`simba.mixins.image_mixin.ImageMixin.slice_shapes_in_img`

    :example I:
    Example 1: Mask video using circular regions derived from body part center positions
    >>> video_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/03152021_NOB_IOT_8.mp4"
    >>> data_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/03152021_NOB_IOT_8.csv"
    >>> save_dir = '/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508'
    >>> nose_arr = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y']).values.reshape(-1, 2).astype(np.int32)
    >>> polygons = GeometryMixin().multiframe_bodyparts_to_circle(data=nose_arr, parallel_offset=60)
    >>> polygon_lst = []
    >>> center = GeometryMixin.get_center(polygons)
    >>> polygons = np.hstack([center, np.full(shape=(len(center), 1), fill_value=60)])
    >>> slice_imgs(video_path=video_path, shapes=polygons, batch_size=500, save_dir=save_dir)

    :example II:
     Example 2: Mask video using minimum rotated rectangles from polygon hulls
    >>> video_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/03152021_NOB_IOT_8.mp4"
    >>> data_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/03152021_NOB_IOT_8.csv"
    >>> save_dir = '/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508'
    >>> nose_arr = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Lat_left_x', 'Lat_left_y', 'Lat_right_x', 'Lat_right_y']).values.reshape(-1, 4, 2).astype(np.int32) ## READ THE BODY-PART THAT DEFINES THE HULL AND CONVERT TO ARRAY
    >>> polygons = GeometryMixin().multiframe_bodyparts_to_polygon(data=nose_arr, parallel_offset=60)
    >>> polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=polygons)
    >>> polygon_lst = []
    >>> for i in polygons:
    >>> polygon_lst.append(np.array(i.exterior.coords).astype(np.int32))
    >>> polygons = np.stack(polygon_lst, axis=0)
    >>> sliced_imgs = slice_imgs(video_path=video_path, shapes=polygons, batch_size=500, save_dir=save_dir)
    """

    THREADS_PER_BLOCK = (16, 8, 8)
    video_meta_data = get_video_meta_data(video_path=video_path, fps_as_int=False)
    video_meta_data['frame_count'] = shapes.shape[0]
    n, w, h = video_meta_data['frame_count'], video_meta_data['width'], video_meta_data['height']
    is_color = ImageMixin.is_video_color(video=video_path)
    timer, save_temp_dir, results, video_out_path = SimbaTimer(start=True), None, None, None
    bboxes = _get_bboxes(shapes)
    crop_heights = bboxes[:, 3] - bboxes[:, 1]
    crop_widths = bboxes[:, 2] - bboxes[:, 0]

    max_h = int(np.max(crop_heights))
    max_w = int(np.max(crop_widths))

    if save_dir is None:
        if not is_color:
            results = np.zeros((n, max_h, max_w), dtype=np.uint8)
        else:
            results = np.zeros((n, max_h, max_w, 3), dtype=np.uint8)
    else:
        save_temp_dir = os.path.join(save_dir, f'temp_{video_meta_data["video_name"]}')
        create_directory(paths=save_temp_dir, overwrite=True)
        video_out_path = os.path.join(save_dir, f'{video_meta_data["video_name"]}.mp4')

    frm_reader = AsyncVideoFrameReader(video_path=video_path, batch_size=batch_size, verbose=True, max_que_size=2)
    frm_reader.start()

    for batch_cnt in range(frm_reader.batch_cnt):
        start_img_idx, end_img_idx, batch_imgs = get_async_frame_batch(batch_reader=frm_reader, timeout=10)
        if verbose:
            print(f'Processing images {start_img_idx} - {end_img_idx} (of {n}; batch count: {batch_cnt+1}/{frm_reader.batch_cnt})...')

        batch_save_path = os.path.join(save_temp_dir, f'{batch_cnt}.mp4') if save_dir is not None else None

        batch_shapes = shapes[start_img_idx:end_img_idx].astype(np.int32)
        batch_bboxes = bboxes[start_img_idx:end_img_idx]

        x_dev = cuda.to_device(batch_shapes)
        bboxes_dev = cuda.to_device(batch_bboxes)
        batch_img_dev = cuda.to_device(batch_imgs)

        if not is_color:
            batch_results = np.zeros((batch_imgs.shape[0], max_h, max_w), dtype=np.uint8)
        else:
            batch_results = np.zeros((batch_imgs.shape[0], max_h, max_w, 3), dtype=np.uint8)
        batch_results_dev = cuda.to_device(batch_results)
        grid_n = math.ceil(batch_imgs.shape[0] / THREADS_PER_BLOCK[0])
        grid_y = math.ceil(max_h / THREADS_PER_BLOCK[1])
        grid_x = math.ceil(max_w / THREADS_PER_BLOCK[2])
        bpg = (grid_n, grid_y, grid_x)
        if batch_shapes.shape[1] == 3:
            _cuda_create_circle_masks[bpg, THREADS_PER_BLOCK](x_dev, batch_img_dev, batch_results_dev, bboxes_dev)
        else:
            _cuda_create_rectangle_masks[bpg, THREADS_PER_BLOCK](x_dev, batch_img_dev, batch_results_dev, bboxes_dev)
        if save_dir is None:
            results[start_img_idx:end_img_idx] = batch_results_dev.copy_to_host()
        else:
            frame_results = batch_results_dev.copy_to_host()
            results = {k: v for k, v in enumerate(frame_results)}
            ImageMixin().img_stack_to_video(imgs=results, fps=video_meta_data['fps'], save_path=batch_save_path, verbose=False)

    frm_reader.kill()
    timer.stop_timer()

    if save_dir:
        concatenate_videos_in_folder(in_folder=save_temp_dir, save_path=video_out_path, remove_splits=True, gpu=True)
        if verbose:
            stdout_success(msg=f'Shapes sliced in video saved at {video_out_path}.', elapsed_time=timer.elapsed_time_str)
        return None
    else:
        if verbose:
            stdout_success(msg='Shapes sliced in video.', elapsed_time=timer.elapsed_time_str)
        return results


@cuda.jit()
def _sliding_psnr(data, stride, results):
    r = cuda.grid(1)
    l = int(r - stride[0])
    if (r < 0) or (r > data.shape[0] -1):
        return
    if l < 0:
        return
    else:
        img_1, img_2 = data[r], data[l]
        mse = _cuda_mse(img_1, img_2)
        if mse == 0:
            results[r] = 0.0
        else:
            results[r] = 20 * math.log10(255 / math.sqrt(mse))

def sliding_psnr(data: np.ndarray,
                 stride_s: int,
                 sample_rate: float) -> np.ndarray:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between pairs of images in a stack using a sliding window approach.

    This function calculates PSNR for each image in a stack compared to another image in the stack that is separated by a specified stride.
    The sliding window approach allows for the comparison of image quality over a sequence of images.

    .. note::
       - PSNR values are measured in decibels (dB).
       - Higher PSNR values indicate better quality with minimal differences from the reference image.
       - Lower PSNR values indicate higher distortion or noise.

    .. math::

       \text{PSNR} = 20 \log_{10} \left( \frac{\text{MAX}}{\sqrt{\text{MSE}}} \right)

    where:
    - :math:`\text{MAX}` is the maximum possible pixel value (255 for 8-bit images)
    - :math:`\text{MSE}` is the Mean Squared Error between the two images

    :param data:  A 4D NumPy array of shape (N, H, W, C) representing a stack of images, where N is the number of images, H is the height, W is the width, and C is the number of color channels.
    :param stride_s: The base stride length in terms of the number of images between the images being compared. Determines the separation between images for comparison in the stack.
    :param sample_rate: The sample rate to scale the stride length. This allows for adjusting the stride dynamically based on the sample rate.
    :return: A 1D NumPy array of PSNR values, where each element represents the PSNR between the image at index `r` and the  image at index `l = r - stride`, for all valid indices `r`.
    :rtype: np.ndarray

    :example:
    >>> data = ImageMixin().read_img_batch_from_video(video_path =r"/mnt/c/troubleshooting/mitra/project_folder/videos/clipped/501_MA142_Gi_CNO_0514_clipped.mp4", start_frm=0, end_frm=299)
    >>> data = np.stack(list(data.values()), axis=0).astype(np.uint8)
    >>> data = ImageMixin.img_stack_to_greyscale(imgs=data)
    >>> p = sliding_psnr(data=data, stride_s=1, sample_rate=1)
    """

    results = np.full(data.shape[0], fill_value=255.0, dtype=np.float32)
    stride = np.array([stride_s * sample_rate], dtype=np.int32)
    if stride[0] < 1: stride[0] = 1
    stride_dev = cuda.to_device(stride)
    results_dev = cuda.to_device(results)
    data_dev = cuda.to_device(data)
    bpg = (data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _sliding_psnr[bpg, THREADS_PER_BLOCK](data_dev, stride_dev, results_dev)
    return results_dev.copy_to_host()

def rotate_img_stack_cupy(imgs: np.ndarray,
                          rotation_degrees: Optional[float] = 180,
                          batch_size: Optional[int] = 500) -> np.ndarray:
    """
    Rotates a stack of images by a specified number of degrees using GPU acceleration with CuPy.

    Accepts a 3D (single-channel images) or 4D (multichannel images) NumPy array, rotates each image in the stack by the specified degree around the center, and returns the result as a NumPy array.

    :param np.ndarray imgs: The input stack of images to be rotated. Expected to be a NumPy array with 3 or 4 dimensions.  3D shape: (num_images, height, width) - 4D shape: (num_images, height, width, channels)
    :param Optional[float] rotation_degrees: The angle by which the images should be rotated, in degrees. Must be between 1 and 359 degrees. Defaults to 180 degrees.
    :param Optional[int] batch_size: Number of images to process on GPU in each batch. Decrease if data can't fit on GPU RAM.
    :returns: A NumPy array containing the rotated images with the same shape as the input.
    :rtype: np.ndarray

    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/F0_gq_Saline_0626_clipped.mp4"
    >>> imgs = read_img_batch_from_video_gpu(video_path=video_path)
    >>> imgs = np.stack(np.array(list(imgs.values())), axis=0)
    >>> imgs = rotate_img_stack_cupy(imgs=imgs, rotation=50)
    """

    check_valid_array(data=imgs, source=f'{rotate_img_stack_cupy.__name__} imgs', accepted_ndims=(3, 4))
    check_int(name=f'{rotate_img_stack_cupy.__name__} rotation', value=rotation_degrees, min_value=1, max_value=359)
    results = cp.full_like(imgs, fill_value=np.nan, dtype=np.uint8)
    for l in range(0, imgs.shape[0], batch_size):
        r = l + batch_size
        batch_imgs = cp.array(imgs[l:r])
        results[l:r] = rotate(input=batch_imgs, angle=rotation_degrees, axes=(2, 1), reshape=True)
    return results.get()

def rotate_video_cupy(video_path: Union[str, os.PathLike],
                      save_path: Optional[Union[str, os.PathLike]] = None,
                      rotation_degrees: Optional[float] = 180,
                      batch_size: Optional[int] = None,
                      verbose: Optional[bool] = True) -> None:
    """
    Rotates a video by a specified angle using GPU acceleration and CuPy for image processing.

    :param Union[str, os.PathLike] video_path: Path to the input video file.
    :param Optional[Union[str, os.PathLike]] save_path: Path to save the rotated video. If None, saves the video in the same directory as the input with '_rotated_<rotation_degrees>' appended to the filename.
    :param nptional[float] rotation_degrees:  Degrees to rotate the video. Must be between 1 and 359 degrees. Default is 180.
    :param Optional[int] batch_size: The number of frames to process in each batch. Deafults to None meaning all images will be processed in a single batch.
    :returns: None.

    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/F0_gq_Saline_0626_clipped.mp4"
    >>> rotate_video_cupy(video_path=video_path, rotation_degrees=45)
    """

    timer = SimbaTimer(start=True)
    check_int(name=f'{rotate_img_stack_cupy.__name__} rotation', value=rotation_degrees, min_value=1, max_value=359)
    check_valid_boolean(source=f'{rotate_img_stack_cupy.__name__} verbose', value=verbose)
    if save_path is None:
        video_dir, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(video_dir, f'{video_name}_rotated_{rotation_degrees}.mp4')
    video_meta_data = get_video_meta_data(video_path=video_path)
    if batch_size is not None:
        check_int(name=f'{rotate_img_stack_cupy.__name__} batch_size', value=batch_size, min_value=1)
    else:
        batch_size = video_meta_data['frame_count']
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    is_clr = ImageMixin.is_video_color(video=video_path)
    frm_reader = AsyncVideoFrameReader(video_path=video_path, batch_size=batch_size, max_que_size=3, verbose=False)
    frm_reader.start()
    for batch_cnt in range(frm_reader.batch_cnt):
        start_idx, end_idx, imgs = get_async_frame_batch(batch_reader=frm_reader, timeout=10)
        if verbose:
            print(f'Rotating frames {start_idx}-{end_idx}... (of {video_meta_data["frame_count"]}, video: {video_meta_data["video_name"]})')
        imgs = rotate_img_stack_cupy(imgs=imgs, rotation_degrees=rotation_degrees, batch_size=batch_size)
        if batch_cnt == 0:
            writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (imgs.shape[2], imgs.shape[1]), isColor=is_clr)
        for img in imgs: writer.write(img)
    writer.release()
    timer.stop_timer()
    frm_reader.kill()
    if verbose:
        stdout_success(f'Rotated video saved at {save_path}', source=rotate_video_cupy.__name__)


@cuda.jit()
def _bg_subtraction_cuda_kernel(imgs, avg_img, results, is_clr, fg_clr, threshold):
    x, y, n = cuda.grid(3)
    if n < 0 or n > (imgs.shape[0] -1):
        return
    if y < 0 or y > (imgs.shape[1] -1):
        return
    if x < 0 or x > (imgs.shape[2] -1):
        return
    if is_clr[0] == 1:
        r1, g1, b1 = imgs[n][y][x][0],imgs[n][y][x][1], imgs[n][y][x][2]
        r2, g2, b2 = avg_img[y][x][0], avg_img[y][x][1], avg_img[y][x][2]
        r_diff, g_diff, b_diff = abs(r1-r2), abs(g1-g2), abs(b1-b2)
        grey_diff = _cuda_luminance_pixel_to_grey(r_diff, g_diff, b_diff)
        if grey_diff > threshold[0]:
            if fg_clr[0] != -1:
                r_out, g_out, b_out = fg_clr[0], fg_clr[1], fg_clr[2]
            else:
                r_out, g_out, b_out = r1, g1, b1
        else:
            r_out, g_out, b_out = results[n][y][x][0], results[n][y][x][1], results[n][y][x][2]
        results[n][y][x][0], results[n][y][x][1], results[n][y][x][2] = r_out, g_out, b_out

    else:
        val_1, val_2 = imgs[n][y][x][0], avg_img[y][x][0]
        grey_diff = abs(val_1-val_2)
        if grey_diff > threshold[0]:
            if fg_clr[0] != -1:
                val_out = val_1
            else:
                val_out = 255
        else:
            val_out = 0
        results[n][y][x] = val_out


def bg_subtraction_cuda(video_path: Union[str, os.PathLike],
                        avg_frm: np.ndarray,
                        save_path: Optional[Union[str, os.PathLike]] = None,
                        bg_clr: Optional[Tuple[int, int, int]] = (0, 0, 0),
                        fg_clr: Optional[Tuple[int, int, int]] = None,
                        batch_size: Optional[int] = 500,
                        threshold: Optional[int] = 50):
    """
    Remove background from videos using GPU acceleration.

    .. note::
       To create an `avg_frm`, use :func:`simba.video_processors.video_processing.create_average_frm`, :func:`simba.data_processors.cuda.image.create_average_frm_cupy`, or :func:`~simba.data_processors.cuda.image.create_average_frm_cuda`

    .. seealso::
       For CPU-based alternative, see :func:`simba.video_processors.video_processing.video_bg_subtraction` or :func:`~simba.video_processors.video_processing.video_bg_subtraction_mp`
       For GPU-based alternative, see :func:`~simba.data_processors.cuda.image.bg_subtraction_cupy`. Needs work, CPU/multicore appears faster.

    .. seealso::
       To create average frame on the CPU, see :func:`simba.video_processors.video_processing.create_average_frm`.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/bg_subtraction_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :class: simba-table
       :header-rows: 1

    :param Union[str, os.PathLike] video_path: The path to the video to remove the background from.
    :param np.ndarray avg_frm: Average frame of the video. Can be created with e.g., :func:`simba.video_processors.video_processing.create_average_frm`.
    :param Optional[Union[str, os.PathLike]] save_path: Optional location to store the background removed video. If None, then saved in the same directory as the input video with the `_bg_removed` suffix.
    :param Optional[Tuple[int, int, int]] bg_clr: Tuple representing the background color of the video.
    :param Optional[Tuple[int, int, int]] fg_clr: Tuple representing the foreground color of the video (e.g., the animal). If None, then the original pixel colors will be used. Default: 50.
    :param Optional[int] batch_size: Number of frames to process concurrently. Use higher values of RAM memory allows. Default: 500.
    :param Optional[int] threshold: Value between 0-255 representing the difference threshold between the average frame subtracted from each frame. Higher values and more pixels will be considered background. Default: 50.

    :example:
    >>> video_path = "/mnt/c/troubleshooting/mitra/project_folder/videos/clipped/592_MA147_Gq_CNO_0515.mp4"
    >>> avg_frm = create_average_frm(video_path=video_path)
    >>> bg_subtraction_cuda(video_path=video_path, avg_frm=avg_frm, fg_clr=(255, 255, 255))
    """

    check_if_valid_img(data=avg_frm, source=f'{bg_subtraction_cuda}')
    check_if_valid_rgb_tuple(data=bg_clr)
    check_int(name=f'{bg_subtraction_cuda.__name__} batch_size', value=batch_size, min_value=1)
    check_int(name=f'{bg_subtraction_cuda.__name__} threshold', value=threshold, min_value=0, max_value=255)
    THREADS_PER_BLOCK = (32, 32, 1)
    timer = SimbaTimer(start=True)
    video_meta = get_video_meta_data(video_path=video_path)
    batch_cnt = int(max(1, np.ceil(video_meta['frame_count'] / batch_size)))
    frm_batches = np.array_split(np.arange(0, video_meta['frame_count']), batch_cnt)
    n, w, h = video_meta['frame_count'], video_meta['width'], video_meta['height']
    avg_frm = cv2.resize(avg_frm, (w, h))
    if is_video_color(video_path): is_color = np.array([1])
    else: is_color = np.array([0])
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    if save_path is None:
        in_dir, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(in_dir, f'{video_name}_bg_removed.mp4')
    if fg_clr is not None:
        check_if_valid_rgb_tuple(data=fg_clr)
        fg_clr = np.array(fg_clr)
    else:
        fg_clr = np.array([-1])
    threshold = np.array([threshold]).astype(np.int32)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta['fps'], (w, h))
    y_dev = cuda.to_device(avg_frm.astype(np.float32))
    fg_clr_dev = cuda.to_device(fg_clr)
    is_color_dev = cuda.to_device(is_color)
    for frm_batch_cnt, frm_batch in enumerate(frm_batches):
        print(f'Processing frame batch {frm_batch_cnt+1} / {len(frm_batches)} (complete: {round((frm_batch_cnt / len(frm_batches)) * 100, 2)}%)')
        batch_imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=frm_batch[0], end_frm=frm_batch[-1])
        batch_imgs = np.stack(list(batch_imgs.values()), axis=0).astype(np.float32)
        batch_n = batch_imgs.shape[0]
        results = np.zeros_like(batch_imgs).astype(np.uint8)
        results[:] = bg_clr
        results = cuda.to_device(results)
        grid_x = math.ceil(w / THREADS_PER_BLOCK[0])
        grid_y = math.ceil(h / THREADS_PER_BLOCK[1])
        grid_z = math.ceil(batch_n / THREADS_PER_BLOCK[2])
        bpg = (grid_x, grid_y, grid_z)
        x_dev = cuda.to_device(batch_imgs)
        _bg_subtraction_cuda_kernel[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results, is_color_dev, fg_clr_dev, threshold)
        results = results.copy_to_host()
        for img_cnt, img in enumerate(results):
            writer.write(img)
    writer.release()
    timer.stop_timer()
    stdout_success(msg=f'Video saved at {save_path}', elapsed_time=timer.elapsed_time_str)


def bg_subtraction_cupy(video_path: Union[str, os.PathLike],
                        avg_frm: np.ndarray,
                        save_path: Optional[Union[str, os.PathLike]] = None,
                        bg_clr: Optional[Tuple[int, int, int]] = (0, 0, 0),
                        fg_clr: Optional[Tuple[int, int, int]] = None,
                        batch_size: Optional[int] = 500,
                        threshold: Optional[int] = 50):
    """
    Remove background from videos using GPU acceleration through CuPY.

    .. seealso::
       For CPU-based alternative, see :func:`simba.video_processors.video_processing.video_bg_subtraction` or :func:`~simba.video_processors.video_processing.video_bg_subtraction_mp`
       For GPU-based alternative, see :func:`~simba.data_processors.cuda.image.bg_subtraction_cuda`.
       Needs work, CPU/multicore appears faster.

    :param Union[str, os.PathLike] video_path: The path to the video to remove the background from.
    :param np.ndarray avg_frm: Average frame of the video. Can be created with e.g., :func:`simba.video_processors.video_processing.create_average_frm`.
    :param Optional[Union[str, os.PathLike]] save_path: Optional location to store the background removed video. If None, then saved in the same directory as the input video with the `_bg_removed` suffix.
    :param Optional[Tuple[int, int, int]] bg_clr: Tuple representing the background color of the video.
    :param Optional[Tuple[int, int, int]] fg_clr: Tuple representing the foreground color of the video (e.g., the animal). If None, then the original pixel colors will be used. Default: 50.
    :param Optional[int] batch_size: Number of frames to process concurrently. Use higher values of RAM memory allows. Default: 500.
    :param Optional[int] threshold: Value between 0-255 representing the difference threshold between the average frame subtracted from each frame. Higher values and more pixels will be considered background. Default: 50.


    :example:
    >>> avg_frm = create_average_frm(video_path="/mnt/c/troubleshooting/mitra/project_folder/videos/temp/temp_ex_bg_subtraction/original/844_MA131_gq_CNO_0624.mp4")
    >>> video_path = "/mnt/c/troubleshooting/mitra/project_folder/videos/temp/temp_ex_bg_subtraction/844_MA131_gq_CNO_0624_7.mp4"
    >>> bg_subtraction_cupy(video_path=video_path, avg_frm=avg_frm, batch_size=500)
    """

    if not _is_cuda_available()[0]:
        raise SimBAGPUError('NP GPU detected using numba.cuda', source=bg_subtraction_cupy.__name__)
    check_if_valid_img(data=avg_frm, source=f'{bg_subtraction_cupy}')
    avg_frm = cp.array(avg_frm)
    check_if_valid_rgb_tuple(data=bg_clr)
    check_int(name=f'{bg_subtraction_cupy.__name__} batch_size', value=batch_size, min_value=1)
    check_int(name=f'{bg_subtraction_cupy.__name__} threshold', value=threshold, min_value=0, max_value=255)
    timer = SimbaTimer(start=True)
    video_meta = get_video_meta_data(video_path=video_path)
    #print(avg_frm, avg_frm.shape, video_meta['width'], video_meta['height'])
    #avg_frm = cv2.resize(avg_frm, (video_meta['width'], video_meta['height'], 3))
    batch_cnt = int(max(1, np.ceil(video_meta['frame_count'] / batch_size)))
    frm_batches = np.array_split(np.arange(0, video_meta['frame_count']), batch_cnt)
    n, w, h = video_meta['frame_count'], video_meta['width'], video_meta['height']
    if is_video_color(video_path):
        is_color = np.array([1])
    else:
        is_color = np.array([0])
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    if save_path is None:
        in_dir, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(in_dir, f'{video_name}_bg_removed_ppp.mp4')
    if fg_clr is not None:
        check_if_valid_rgb_tuple(data=fg_clr)
        fg_clr = np.array(fg_clr)
    else:
        fg_clr = np.array([-1])
    writer = cv2.VideoWriter(save_path, fourcc, video_meta['fps'], (w, h))
    for frm_batch_cnt, frm_batch in enumerate(frm_batches):
        print(f'Processing frame batch {frm_batch_cnt + 1} / {len(frm_batches)} (complete: {round((frm_batch_cnt / len(frm_batches)) * 100, 2)}%)')
        batch_imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=frm_batch[0], end_frm=frm_batch[-1])
        batch_imgs = cp.array(np.stack(list(batch_imgs.values()), axis=0).astype(np.float32))
        img_diff = cp.abs(batch_imgs - avg_frm)
        if is_color:
            img_diff = img_stack_to_grayscale_cupy(imgs=img_diff, batch_size=img_diff.shape[0])
            mask = cp.where(img_diff > threshold, 1, 0).astype(cp.uint8)
            batch_imgs[mask == 0] = bg_clr
            if fg_clr[0] != -1:
                batch_imgs[mask == 1] = fg_clr
        batch_imgs = batch_imgs.astype(cp.uint8).get()
        for img_cnt, img in enumerate(batch_imgs):
            writer.write(img)
    writer.release()
    timer.stop_timer()
    stdout_success(msg=f'Video saved at {save_path}', elapsed_time=timer.elapsed_time_str)


@cuda.jit(max_registers=None)
def _pose_plot_kernel(imgs, data, circle_size, resolution, colors):
    bp_n, img_n = cuda.grid(2)
    if img_n < 0 or img_n > (imgs.shape[0] -1):
        return
    if bp_n < 0 or bp_n > (data[0].shape[0] -1):
        return

    img, bp_loc, color  = imgs[img_n], data[img_n][bp_n], colors[bp_n]
    for x1 in range(bp_loc[0]-circle_size[0], bp_loc[0]+circle_size[0]):
        for y1 in range(bp_loc[1]-circle_size[0], bp_loc[1]+circle_size[0]):
            if (x1 > 0) and (x1 < resolution[0]):
                if (y1 > 0) and (y1 < resolution[1]):
                    b = (x1 - bp_loc[0]) ** 2
                    c = (y1 - bp_loc[1]) ** 2
                    if (b + c) < (circle_size[0] ** 2):
                        imgs[img_n][y1][x1][0] = int(color[0])
                        imgs[img_n][y1][x1][1] = int(color[1])
                        imgs[img_n][y1][x1][2] = int(color[2])


def pose_plotter(data: Union[str, os.PathLike, np.ndarray],
                 video_path: Union[str, os.PathLike],
                 save_path: Union[str, os.PathLike],
                 circle_size: Optional[int] = None,
                 colors: Optional[str] = 'Set1',
                 batch_size: int = 750,
                 verbose: bool = True) -> None:

    """
    Creates a video overlaying pose-estimation data on frames from a given video using GPU acceleration.

    .. video:: _static/img/pose_plotter_cuda.mp4
       :width: 800
       :autoplay:
       :loop:

    .. seealso::
       For CPU based methods, see :func:`~simba.plotting.path_plotter.PathPlotterSingleCore` and :func:`~simba.plotting.path_plotter_mp.PathPlotterMulticore`.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/pose_plotter.csv
       :widths: 10, 90
       :align: center
       :class: simba-table
       :header-rows: 1

    :param Union[str, os.PathLike, np.ndarray] data: Path to a CSV file with pose-estimation data or a 3d numpy array (n_images, n_bodyparts, 2) with pose-estimated locations.
    :param Union[str, os.PathLike] video_path: Path to a video file where the ``data`` has been pose-estimated.
    :param Union[str, os.PathLike] save_path: Location where to store the output visualization.
    :param Optional[int] circle_size: The size of the circles representing the location of the pose-estimated locations. If None, the optimal size will be inferred as a 100th of the max(resultion_w, h).
    :param int batch_size: The number of frames to process concurrently on the GPU. Default: 750. Increase of host and device RAM allows it to improve runtime. Decrease if you hit memory errors.

    :example:
    >>> DATA_PATH = "/mnt/c/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/501_MA142_Gi_CNO_0521.csv"
    >>> VIDEO_PATH = "/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
    >>> SAVE_PATH = "/mnt/c/troubleshooting/mitra/project_folder/frames/output/pose_ex/test.mp4"
    >>> pose_plotter(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH, circle_size=10, batch_size=1000)
    """

    THREADS_PER_BLOCK = (32, 32, 1)
    if isinstance(data, str):
        check_file_exist_and_readable(file_path=data)
        df = read_df(file_path=data, file_type='csv')
        cols = [x for x in df.columns if not x.lower().endswith('_p')]
        data = df[cols].values
        data = np.ascontiguousarray(data.reshape(data.shape[0], int(data.shape[1] / 2), 2).astype(np.int32))
    elif isinstance(data, np.ndarray):
        check_valid_array(data=data, source=pose_plotter.__name__, accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)

    check_int(name=f'{pose_plotter.__name__} batch_size', value=batch_size, min_value=1)
    check_valid_boolean(value=[verbose], source=f'{pose_plotter.__name__} verbose')
    video_meta_data = get_video_meta_data(video_path=video_path)
    n, w, h = video_meta_data['frame_count'], video_meta_data['width'], video_meta_data['height']
    check_if_dir_exists(in_dir=os.path.dirname(save_path))
    if data.shape[0] != video_meta_data['frame_count']:
        raise FrameRangeError(msg=f'The data contains {data.shape[0]} frames while the video contains {video_meta_data["frame_count"]} frames')
    if circle_size is None:
        circle_size = np.array([PlottingMixin().get_optimal_circle_size(frame_size=(w, h))]).astype(np.int32)
    else:
        check_int(name=f'{pose_plotter.__name__} circle_size', value=circle_size, min_value=1)
        circle_size = np.array([circle_size]).astype(np.int32)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    video_writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (w, h))
    colors = np.array(create_color_palette(pallete_name=colors, increments=data[0].shape[0])).astype(np.int32)
    circle_size_dev = cuda.to_device(circle_size)
    colors_dev = cuda.to_device(colors)
    resolution_dev = cuda.to_device(np.array([video_meta_data['width'], video_meta_data['height']]))
    data = np.ascontiguousarray(data, dtype=np.int32)
    img_dev = cuda.device_array((batch_size, h, w, 3), dtype=np.int32)
    data_dev = cuda.device_array((batch_size, data.shape[1], 2), dtype=np.int32)
    total_timer, video_start_time = SimbaTimer(start=True), time.time()
    frm_reader = AsyncVideoFrameReader(video_path=video_path, batch_size=batch_size, max_que_size=3, verbose=False)
    frm_reader.start()
    for batch_cnt in range(frm_reader.batch_cnt):
        start_img_idx, end_img_idx, batch_frms = get_async_frame_batch(batch_reader=frm_reader, timeout=10)
        video_elapsed_time = str(round(time.time() - video_start_time, 4)) + 's'
        if verbose: print(f'Processing images {start_img_idx} - {end_img_idx} (of {n}; batch count: {batch_cnt+1}/{frm_reader.batch_cnt}, video: {video_meta_data["video_name"]}, elapsed video processing time: {video_elapsed_time})...')
        batch_data = data[start_img_idx:end_img_idx + 1]
        batch_n = batch_frms.shape[0]
        if verbose: print(f'Moving frames {start_img_idx}-{end_img_idx} to device...')
        img_dev[:batch_n].copy_to_device(batch_frms[:batch_n].astype(np.int32))
        data_dev[:batch_n] = cuda.to_device(batch_data[:batch_n])
        del batch_frms; del batch_data
        bpg = (math.ceil(batch_n / THREADS_PER_BLOCK[0]), math.ceil(batch_n / THREADS_PER_BLOCK[2]))
        if verbose: print(f'Creating frames {start_img_idx}-{end_img_idx} ...')
        _pose_plot_kernel[bpg, THREADS_PER_BLOCK](img_dev, data_dev, circle_size_dev, resolution_dev, colors_dev)
        if verbose: print(f'Moving frames to host {start_img_idx}-{end_img_idx} ...')
        batch_frms = img_dev.copy_to_host()
        if verbose: print(f'Writing frames to host {start_img_idx}-{end_img_idx} ...')
        for img_idx in range(0, batch_n):
            video_writer.write(batch_frms[img_idx].astype(np.uint8))
    video_writer.release()
    total_timer.stop_timer()
    frm_reader.kill()
    if verbose:
        stdout_success(msg=f'Pose-estimation video saved at {save_path}.', elapsed_time=total_timer.elapsed_time_str)






# DATA_PATH = "/mnt/c/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/501_MA142_Gi_CNO_0521.csv"
# VIDEO_PATH = "/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
# SAVE_PATH = "/mnt/c/troubleshooting/mitra/project_folder/frames/output/pose_ex/test.mp4"
# pose_plotter(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH, circle_size=10, batch_size=1000)
# # VIDEO_PATH = "/mnt/c/troubleshooting/mitra/project_folder/frames/output/pose_ex/test.mp4"
# # SAVE_PATH = "/mnt/c/troubleshooting/mitra/project_folder/frames/output/pose_ex/test_ROTATED.mp4"
# #
# # rotate_video_cupy(video_path=VIDEO_PATH, save_path=SAVE_PATH, batch_size=1000)
#
# #"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\501_MA142_Gi_CNO_0521.csv"
# pose_plotter(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH, circle_size=10, batch_size=1000)







# from simba.mixins.geometry_mixin import GeometryMixin
#
# video_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/03152021_NOB_IOT_8.mp4"
# data_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/03152021_NOB_IOT_8.csv"
# save_dir = '/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508'
#
# get_video_meta_data(video_path)
#
# nose_arr = read_df(file_path=data_path, file_type='csv', usecols=['Nose_x', 'Nose_y', 'Tail_base_x', 'Tail_base_y', 'Lat_left_x', 'Lat_left_y', 'Lat_right_x', 'Lat_right_y']).values.reshape(-1, 4, 2).astype(np.int32) ## READ THE BODY-PART THAT DEFINES THE HULL AND CONVERT TO ARRAY
#
# polygons = GeometryMixin().multiframe_bodyparts_to_polygon(data=nose_arr, parallel_offset=60) ## CONVERT THE BODY-PART TO POLYGONS WITH A LITTLE BUFFER
# polygons = GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=polygons) # CONVERT THE POLYGONS TO RECTANGLES (I.E., WITH 4 UNIQUE POINTS).
# polygon_lst = [] # GET THE POINTS OF THE RECTANGLES
# for i in polygons: polygon_lst.append(np.array(i.exterior.coords))
# polygons = np.stack(polygon_lst, axis=0)
# sliced_imgs = slice_imgs(video_path=video_path, shapes=polygons, batch_size=500, save_dir=save_dir) #SLICE THE RECTANGLES IN THE VIDEO.

#sliced_imgs =  {k: v for k, v in enumerate(sliced_imgs)}

#ImageMixin().img_stack_to_video(imgs=sliced_imgs, fps=29.97, save_path=r'/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/stacked.mp4')

#get_video_meta_data("/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/03152021_NOB_IOT_8.mp4")
# cv2.imshow('asdasdas', sliced_imgs[500])
# cv2.waitKey(0)

# DATA_PATH = "/mnt/c/troubleshooting/RAT_NOR/project_folder/csv/outlier_corrected_movement_location/03152021_NOB_IOT_8.csv"
# VIDEO_PATH = "/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/03152021_NOB_IOT_8.mp4"
# SAVE_PATH = "/mnt/c/troubleshooting/mitra/project_folder/frames/output/pose_ex/test.mp4"
#
#
# # DATA_PATH = "/mnt/c/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/501_MA142_Gi_CNO_0514.csv"
# # VIDEO_PATH = "/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0514.mp4"
# # SAVE_PATH = "/mnt/c/troubleshooting/mitra/project_folder/frames/output/pose_ex/test.mp4"
# pose_plotter(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH, circle_size=10, batch_size=100)



#
# #from simba.data_processors.cuda.image import create_average_frm_cupy
# SAVE_PATH = "/mnt/c/Users/sroni/Downloads/bg_remove_nb/bg_removed_ex_7.mp4"
# VIDEO_PATH = "/mnt/c/Users/sroni/Downloads/bg_remove_nb/open_field.mp4"
# avg_frm = create_average_frm_cuda(video_path=VIDEO_PATH)
# #
# get_video_meta_data(VIDEO_PATH)
# #
# bg_subtraction_cuda(video_path=VIDEO_PATH, avg_frm=avg_frm, save_path=SAVE_PATH, threshold=70)
