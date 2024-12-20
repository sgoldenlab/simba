import os
import time

import numpy as np
import math
from typing import Union, Optional, Tuple
from numba import cuda
import cv2
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_video_meta_data, read_img_batch_from_video_gpu, get_fn_ext
from simba.video_processors.video_processing import create_average_frm, video_bg_subtraction, video_bg_subtraction_mp
from simba.utils.checks import is_video_color, check_if_valid_rgb_tuple, check_if_valid_img, check_int
from simba.utils.enums import Formats
from simba.data_processors.cuda.utils import _cuda_luminance_pixel_to_grey


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
       For CPU-based alternative, see :func:`simba.video_processors.video_processing.video_bg_subtraction`. Needs work, multi-core is faster.

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
    :param Optional[int] threshold: Value between 0-255 representing the difference threshold between the average frame subtracted from each frame. Higher values and more pixels will be considered background. Default: 100.

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



# video_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/clipped/03152021_NOB_IOT_8_clipped.mp4"
# video_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat7_8(2).mp4"
# video_path = "/mnt/c/troubleshooting/mitra/project_folder/videos/clipped/592_MA147_Gq_CNO_0515.mp4"
#
# avg_frm = create_average_frm(video_path="/mnt/c/troubleshooting/mitra/project_folder/videos/temp/temp_ex_bg_subtraction/original/844_MA131_gq_CNO_0624.mp4")
# video_path = "/mnt/c/troubleshooting/mitra/project_folder/videos/temp/temp_ex_bg_subtraction/844_MA131_gq_CNO_0624_7.mp4"
# timer = []
# for i in range(1):
#     start = time.perf_counter()
#     bg_subtraction_cuda(video_path=video_path, avg_frm=avg_frm, batch_size=3000)
#     end = time.perf_counter()
#     timer.append(end-start)
# print(np.mean(timer), np.std(timer))
