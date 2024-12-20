import functools
import multiprocessing
import os
from copy import deepcopy
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries, slic

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_img, check_int,
                                check_valid_array)
from simba.utils.enums import Defaults, Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_frm_of_video)


def get_img_slic(img: np.ndarray,
                 n_segments: Optional[int] = 50,
                 compactness: Optional[int] = 50,
                 sigma: Optional[float] = 1) -> np.ndarray:

    """
    Simplify an image into superpixels using SLIC (Simple Linear Iterative Clustering).

    :param np.ndarray img: Image to segment.
    :param n_segments: Number of segments to produce.
    :param compactness: How compact ("square") the output segments are.
    :param np.ndarray sigma: Amount of Gaussian smoothing.
    :return: Smoothened version of the input image.
    :rtype: np.ndarray

    :example:
    >>> img = read_frm_of_video(video_path=r"C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0626.mp4", frame_index=0)
    >>> sliced_img = get_img_slic(img=img)
    """

    check_if_valid_img(data=img, source=f'{get_img_slic.__name__} img', raise_error=True)
    check_int(name=f'{get_img_slic.__name__} n_segments', value=n_segments, min_value=2)
    check_int(name=f'{get_img_slic.__name__} compactness', value=compactness, min_value=1)
    check_int(name=f'{get_img_slic.__name__} sigma', value=compactness, min_value=0)
    segments = slic(image=img, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=0)
    segmented_image = label2rgb(segments, img, kind='avg', bg_label=0)

    return segmented_image

def _slic_helper(frm_range: np.ndarray,
                 n_segments: int,
                 sigma: float,
                 compactness: int,
                 save_dir: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike]):

    """ SLIC multiprocess helper called by slic.get_video_slic"""

    video_cap = cv2.VideoCapture(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    batch, start_frm, end_frm = frm_range[0], frm_range[1][0], frm_range[1][-1]
    save_path = os.path.join(save_dir, f'{batch}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    for frm_idx in range(start_frm, end_frm):
        print(f'Frame {frm_idx}/{end_frm}, Batch {batch}...')
        img = read_frm_of_video(video_path=video_cap, frame_index=frm_idx)
        img = get_img_slic(img=img, n_segments=n_segments, compactness=compactness, sigma=sigma)
        writer.write(img)
    writer.release()
    return batch


def get_video_slic(video_path: Union[str, os.PathLike],
                   save_path: Union[str, os.PathLike],
                   n_segments: Optional[int] = 50,
                   compactness: Optional[int] = 50,
                   sigma: Optional[int] = 1,
                   core_cnt: Optional[int] = -1) -> None:

    """
    Apply SLIC superpixel segmentation to all frames of a video and save the output as a new video.

    .. video:: _static/img/get_video_slic.webm
       :width: 800
       :autoplay:
       :loop:


    :param Union[str, os.PathLike] video_path: Path to the input video file.
    :param Union[str, os.PathLike] save_path: Path to save the processed video with SLIC superpixel segmentation.
    :param Optional[int] n_segments: Approximate number of superpixels for each frame. Defaults to 50.
    :param Optional[int] compactness: Balance of color and spatial proximity.  Higher values result in more uniformly shaped superpixels. Defaults to 50.
    :param Optional[int] sigma: Standard deviation for Gaussian smoothing applied to each frame before segmentation. Defaults to 1.
    :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. Set to -1 to use all available cores. Defaults to -1.
    :return: None. The segmented video is saved to `save_path`.

    :example:
    >>> #video_path = r"C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0626.mp4"
    """
    timer = SimbaTimer(start=True)
    check_int(name=f'{get_img_slic.__name__} n_segments', value=n_segments, min_value=2)
    check_int(name=f'{get_img_slic.__name__} compactness', value=compactness, min_value=1)
    check_int(name=f'{get_img_slic.__name__} sigma', value=sigma, min_value=1)
    check_int(name=f'{get_img_slic.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    if core_cnt == -1 or core_cnt > find_core_cnt()[0]: core_cnt = find_core_cnt()[0]
    frm_ranges = np.array_split(np.arange(0, video_meta_data['frame_count'] + 1), core_cnt)
    frm_ranges = [(y, x) for y, x in enumerate(frm_ranges)]
    out_dir, out_name, _= get_fn_ext(filepath=save_path)
    temp_folder = os.path.join(out_dir, "temp")
    if not os.path.isdir(temp_folder): os.makedirs(temp_folder)
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_slic_helper,
                                      video_path=video_path,
                                      save_dir=temp_folder,
                                      n_segments=n_segments,
                                      compactness=compactness,
                                      sigma=sigma)
        for cnt, core_batch in enumerate(pool.map(constants, frm_ranges, chunksize=1)):
            print(f'Core batch {core_batch} complete...')
    pool.join()
    pool.terminate()
    timer.stop_timer()
    concatenate_videos_in_folder(in_folder=temp_folder, save_path=save_path)
    stdout_success(msg=f'SLIC video saved at {save_path}', elapsed_time=timer.elapsed_time_str)


if __name__=='__main__':
    video_path = r"C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0626.mp4"
    get_video_slic(video_path=video_path, save_path=r"C:\Users\sroni\OneDrive\Desktop\test.mp4")

#
# img = read_frm_of_video(video_path=r"C:\troubleshooting\mitra\project_folder\videos\FRR_gq_Saline_0626.mp4", frame_index=0)
# sliced_img = get_img_slic(img=img)
# cv2.imshow('sasdasd', sliced_img)
# cv2.waitKey(5000)



