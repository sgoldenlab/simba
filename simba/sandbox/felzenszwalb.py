import functools
#from simba.utils.checks import
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb

from simba.utils.enums import Defaults, Formats
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_frm_of_video)


def _felzenszwalb_helper(frm_range: Tuple[int, List[int]],
                         scale: int,
                         min_size: int,
                         sigma: float,
                         save_dir: Union[str, os.PathLike],
                         video_path: Union[str, os.PathLike]):

    video_cap = cv2.VideoCapture(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    batch, start_frm, end_frm = frm_range[0], frm_range[1][0], frm_range[1][-1]
    save_path = os.path.join(save_dir, f'{batch}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    for frm_idx in range(start_frm, end_frm):
        print(f'Frame {frm_idx}/{end_frm}, Batch {batch}...')
        img = read_frm_of_video(video_path=video_cap, frame_index=frm_idx)
        segments = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
        compressed_frame = label2rgb(segments, img, kind='avg')
        compressed_frame_bgr = (compressed_frame * 255).astype(np.uint8)
        writer.write(compressed_frame_bgr)
    writer.release()
    return batch

    #imgs = ImageMixin.read_img_batch_from_video(video_path=video_path, start_frm=start_frm, end_frm=end_frm, core_cnt=1)


def felzenszwalb_video(video_path: Union[str, os.PathLike],
                       save_path: Optional[Union[str, os.PathLike]] = None,
                       scale: Optional[int] = 100,
                       sigma: Optional[float] = 0.5,
                       min_size: Optional[int] = 100,
                       core_cnt: Optional[int] = -1):

    """
    :param Optional[int] scale: Path to the video file.

    :example:
    >>> felzenszwalb_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/videos/704_MA115_Gi_CNO_0521_clipped.mp4', save_path='/Users/simon/Desktop/feltz/test.mp4')
    """

    timer = SimbaTimer(start=True)
    video_meta_data = get_video_meta_data(video_path=video_path)
    if core_cnt == -1: core_cnt = find_core_cnt()[0]
    frm_ranges = np.array_split(np.arange(0, video_meta_data['frame_count']+1), core_cnt)
    frm_ranges = [(y, x) for y, x in enumerate(frm_ranges)]
    out_dir, out_name, _= get_fn_ext(filepath=save_path)
    temp_folder = os.path.join(out_dir, "temp")
    if not os.path.isdir(temp_folder): os.makedirs(temp_folder)
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_felzenszwalb_helper,
                                      video_path=video_path,
                                      save_dir=temp_folder,
                                      scale=scale,
                                      sigma=sigma,
                                      min_size=min_size)
        for cnt, core_batch in enumerate(pool.map(constants, frm_ranges, chunksize=1)):
            print(f'Core batch {core_batch} complete...')
            pass
    timer.stop_timer()
    concatenate_videos_in_folder(in_folder=temp_folder, save_path=save_path)


felzenszwalb_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/videos/704_MA115_Gi_CNO_0521_clipped.mp4', save_path='/Users/simon/Desktop/feltz/test.mp4')