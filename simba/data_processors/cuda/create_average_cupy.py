__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"


import os
from typing import Optional, Union

import cupy as cp
import cv2
import numpy as np

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.errors import FFMPEGCodecGPUError, InvalidInputError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video, get_fn_ext,
    get_video_meta_data, read_img_batch_from_video_gpu)


def create_average_frm(video_path: Union[str, os.PathLike],
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
    >>> create_average_frm(video_path=r"C:\troubleshooting\RAT_NOR\project_folder\videos\2022-06-20_NOB_DOT_4_downsampled.mp4", verbose=True, start_frm=0, end_frm=9000)
    """

    def average_3d_stack(image_stack: np.ndarray) -> np.ndarray:
        num_frames, height, width, _ = image_stack.shape
        image_stack = cp.array(image_stack).astype(cp.float32)
        img = cp.clip(cp.sum(image_stack, axis=0) / num_frames, 0, 255).astype(cp.uint8)
        return img.get()

    if not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=create_average_frm.__name__)


    if ((start_frm is not None) or (end_frm is not None)) and ((start_time is not None) or (end_time is not None)):
        raise InvalidInputError(msg=f'Pass start_frm and end_frm OR start_time and end_time', source=create_average_frm.__name__)
    elif type(start_frm) != type(end_frm):
        raise InvalidInputError(msg=f'Pass start frame and end frame', source=create_average_frm.__name__)
    elif type(start_time) != type(end_time):
        raise InvalidInputError(msg=f'Pass start time and end time', source=create_average_frm.__name__)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=create_average_frm.__name_)
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    video_name = get_fn_ext(filepath=video_path)[1]
    if verbose:
        print(f'Getting average frame from {video_name}...')
    if (start_frm is not None) and (end_frm is not None):
        check_int(name='start_frm', value=start_frm, min_value=0, max_value=video_meta_data['frame_count'])
        check_int(name='end_frm', value=end_frm, min_value=0, max_value=video_meta_data['frame_count'])
        if start_frm > end_frm:
            raise InvalidInputError(msg=f'Start frame ({start_frm}) has to be before end frame ({end_frm}).', source=create_average_frm.__name__)
        frame_ids = list(range(start_frm, end_frm))
    elif (start_time is not None) and (end_time is not None):
        check_if_string_value_is_valid_video_timestamp(value=start_time, name=create_average_frm.__name__)
        check_if_string_value_is_valid_video_timestamp(value=end_time, name=create_average_frm.__name__)
        check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time, name=create_average_frm.__name__)
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
            stdout_success(msg=f'Saved average frame at {save_path}', source=create_average_frm.__name__)
    else:
        return avg_img