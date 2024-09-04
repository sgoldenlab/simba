__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"


import os
from typing import Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal
try:
    import cupy as cp
except:
    import numpy as cp

from copy import deepcopy

import cv2
import numpy as np
from numba import cuda

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_string_value_is_valid_video_timestamp,
                                check_if_valid_img, check_instance, check_int,
                                check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_array, check_valid_boolean)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import Formats
from simba.utils.errors import FFMPEGCodecGPUError, InvalidInputError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video, get_fn_ext,
    get_video_meta_data, read_img_batch_from_video_gpu)

PHOTOMETRIC = 'photometric'
DIGITAL = 'digital'

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
def _average_3d_stack_cuda(data, results):
    y, x, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > data.shape[0] - 1 or x > data.shape[1] - 1 or y > data.shape[2] - 1:
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
    _average_3d_stack_cuda[blocks_per_grid, threads_per_block](x_dev, results)
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

    .. selalso::
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
            raise InvalidInputError(msg=f'Start frame ({start_frm}) has to be before end frame ({end_frm}).',
                                    source=create_average_frm_cuda.__name__)
        frame_ids = list(range(start_frm, end_frm))
    elif (start_time is not None) and (end_time is not None):
        check_if_string_value_is_valid_video_timestamp(value=start_time, name=create_average_frm_cuda.__name__)
        check_if_string_value_is_valid_video_timestamp(value=end_time, name=create_average_frm_cuda.__name__)
        check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time,
                                              name=create_average_frm_cuda.__name__)
        check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start_time, video_path=video_path)
        frame_ids = find_frame_numbers_from_time_stamp(start_time=start_time, end_time=end_time,
                                                       fps=video_meta_data['fps'])
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

    .. seelalso::
       For CPU function see :func:`~simba.mixins.image_mixin.ImageMixin.img_stack_mse` and
       :func:`~simba.mixins.image_mixin.ImageMixin.img_sliding_mse`.


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


def img_stack_to_grayscale_cupy(imgs: np.ndarray,
                                batch_size: Optional[int] = 250) -> np.ndarray:
    """
    Converts a stack of color images to grayscale using GPU acceleration with CuPy.

    .. seelalso::
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

    :param np.ndarray imgs: A 4D NumPy array representing a stack of images with shape (num_images, height, width, channels). The images are expected to have 3 channels (RGB).
    :param Optional[int] batch_size: The number of images to process in each batch. Defaults to 250. Adjust this parameter to fit your GPU's memory capacity.
    :return np.ndarray: m A 3D NumPy array of shape (num_images, height, width) containing the grayscale images. If the input array is not 4D, the function returns the input as is.

    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_IOT_1_cropped.mp4", verbose=False, start_frm=0, end_frm=i)
    >>> imgs = np.stack(list(imgs.values()), axis=0).astype(np.uint8)
    >>> gray_imgs = img_stack_to_grayscale_cupy(imgs=imgs)
    """


    check_instance(source=img_stack_to_grayscale_cupy.__name__, instance=imgs, accepted_types=(np.ndarray,))
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
    return results.get()



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

    .. seelalso::
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
    check_instance(source=img_stack_to_grayscale_cuda.__name__, instance=imgs, accepted_types=(np.ndarray,))
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