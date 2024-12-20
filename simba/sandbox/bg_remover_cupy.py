import os
import cupy as cp
import numpy as np
from typing import Union, Optional, Tuple
import cv2
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_video_meta_data, read_img_batch_from_video_gpu, get_fn_ext
from simba.video_processors.video_processing import create_average_frm, video_bg_subtraction, video_bg_subtraction_mp
from simba.utils.checks import is_video_color, check_if_valid_rgb_tuple, check_if_valid_img, check_int
from simba.utils.enums import Formats
from simba.data_processors.cuda.utils import _cuda_luminance_pixel_to_grey, _cuda_available
from simba.utils.errors import SimBAGPUError
from simba.data_processors.cuda.image import img_stack_to_grayscale_cupy

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
       For CPU-based alternative, see :func:`simba.video_processors.video_processing.video_bg_subtraction`. Needs work, multi-core is faster.

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

    if not _cuda_available()[0]: raise SimBAGPUError('NP GPU detected using numba.cuda', source=bg_subtraction_cupy.__name__)
    check_if_valid_img(data=avg_frm, source=f'{bg_subtraction_cupy}')
    avg_frm = cp.array(avg_frm)
    check_if_valid_rgb_tuple(data=bg_clr)
    check_int(name=f'{bg_subtraction_cupy.__name__} batch_size', value=batch_size, min_value=1)
    check_int(name=f'{bg_subtraction_cupy.__name__} threshold', value=threshold, min_value=0, max_value=255)
    timer = SimbaTimer(start=True)
    video_meta = get_video_meta_data(video_path=video_path)
    batch_cnt = int(max(1, np.ceil(video_meta['frame_count'] / batch_size)))
    frm_batches = np.array_split(np.arange(0, video_meta['frame_count']), batch_cnt)
    n, w, h = video_meta['frame_count'], video_meta['width'], video_meta['height']
    if is_video_color(video_path): is_color = np.array([1])
    else: is_color = np.array([0])
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    if save_path is None:
        in_dir, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(in_dir, f'{video_name}_bg_removed_ppp.mp4')
    if fg_clr is not None:
        check_if_valid_rgb_tuple(data=fg_clr)
        fg_clr = np.array(fg_clr)
    else:
        fg_clr = -1
    writer = cv2.VideoWriter(save_path, fourcc, video_meta['fps'], (w, h))
    for frm_batch_cnt, frm_batch in enumerate(frm_batches):
        print(f'Processing frame batch {frm_batch_cnt+1} / {len(frm_batches)} (complete: {round((frm_batch_cnt / len(frm_batches)) * 100, 2)}%)')
        batch_imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=frm_batch[0], end_frm=frm_batch[-1])
        batch_imgs = cp.array(np.stack(list(batch_imgs.values()), axis=0).astype(np.float32))
        img_diff = cp.abs(batch_imgs - avg_frm)
        if is_color:
            img_diff = img_stack_to_grayscale_cupy(imgs=img_diff, batch_size=img_diff.shape[0])
            mask = cp.where(img_diff > threshold, 1, 0).astype(cp.uint8)
            batch_imgs[mask == 0] = bg_clr
            if fg_clr != -1:
                batch_imgs[mask == 1] = fg_clr
        batch_imgs = batch_imgs.astype(cp.uint8).get()
        for img_cnt, img in enumerate(batch_imgs):
            writer.write(img)
    writer.release()
    timer.stop_timer()
    stdout_success(msg=f'Video saved at {save_path}', elapsed_time=timer.elapsed_time_str)



avg_frm = create_average_frm(video_path="/mnt/c/troubleshooting/mitra/project_folder/videos/temp/temp_ex_bg_subtraction/original/844_MA131_gq_CNO_0624.mp4")
video_path = "/mnt/c/troubleshooting/mitra/project_folder/videos/temp/temp_ex_bg_subtraction/844_MA131_gq_CNO_0624_7.mp4"
bg_subtraction_cupy(video_path=video_path, avg_frm=avg_frm, batch_size=500)