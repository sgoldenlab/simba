import os
from copy import deepcopy
from typing import Optional, Tuple, Union

import cv2
import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, get_video_meta_data
from simba.video_processors.video_processing import create_average_frm


def video_bg_subtraction(video_path: Union[str, os.PathLike],
                         bg_video_path: Optional[Union[str, os.PathLike]] = None,
                         bg_start_frm: Optional[int] = None,
                         bg_end_frm: Optional[int] = None,
                         bg_start_time: Optional[str] = None,
                         bg_end_time: Optional[str] = None,
                         bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
                         fg_color: Optional[Tuple[int, int, int]] = None,
                         save_path: Optional[Union[str, os.PathLike]] = None,
                         threshold: Optional[int] = 50,
                         verbose: Optional[bool] = True) -> None:
    """
    Subtract the background from a video.

    .. video:: _static/img/video_bg_subtraction.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/bg_remover_example_1.webm
       :width: 800
       :autoplay:
       :loop:

    .. video:: _static/img/bg_remover_example_2.webm
       :width: 800
       :autoplay:
       :loop:

    .. note::
       If  ``bg_video_path`` is passed, that video will be used to parse the background. If None, ``video_path`` will be use dto parse background.
       Either pass ``start_frm`` and ``end_frm`` OR ``start_time`` and ``end_time`` OR pass all four arguments as None.
       Those two arguments will be used to slice the background video, and the sliced part is used to parse the background.

       For example, in the scenario where there is **no** animal in the ``video_path`` video for the first 20s, then the first 20s can be used to parse the background.
       In this scenario, ``bg_video_path`` can be passed as ``None`` and bg_start_time and bg_end_time can be ``00:00:00`` and ``00:00:20``, repectively.

       In the scenario where there **is** animal(s) in the entire ``video_path`` video, pass ``bg_video_path`` as a path to a video recording the arena without the animals.

    :param Union[str, os.PathLike] video_path: The path to the video to remove the background from.
    :param Optional[Union[str, os.PathLike]] bg_video_path: Path to the video which contains a segment with the background only. If None, then ``video_path`` will be used.
    :param Optional[int] bg_start_frm: The first frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[int] bg_end_frm: The last frame in the background video to use when creating a representative background image. Default: None.
    :param Optional[str] bg_start_time: The start timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[str] bg_end_time: The end timestamp in `HH:MM:SS` format in the background video to use to create a representative background image. Default: None.
    :param Optional[Tuple[int, int, int]] bg_color: The RGB color of the moving objects in the output video. Defaults to None, which represents the original colors of the moving objects.
    :param Optional[Tuple[int, int, int]] fg_color: The RGB color of the background output video. Defaults to black (0, 0, 0).
    :param Optional[Union[str, os.PathLike]] save_path: The patch to where to save the output video where the background is removed. If None, saves the output video in the same directory as the input video with the ``_bg_subtracted`` suffix. Default: None.
    :return: None.

    :example:
    >>> video_bg_subtraction(video_path='/Users/simon/Downloads/1_LH_cropped.mp4', bg_start_time='00:00:00', bg_end_time='00:00:10', bg_color=(0, 106, 167), fg_color=(254, 204, 2))
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=video_path)
    if bg_video_path is None:
        bg_video_path = deepcopy(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    dir, video_name, ext = get_fn_ext(filepath=video_path)
    if save_path is None:
        save_path = os.path.join(dir, f'{video_name}_bg_subtracted{ext}')
    else:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=video_bg_subtraction.__name__)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'],(video_meta_data['width'], video_meta_data['height']))
    bg_frm = create_average_frm(video_path=bg_video_path, start_frm=bg_start_frm, end_frm=bg_end_frm, start_time=bg_start_time, end_time=bg_end_time)
    bg_frm = cv2.resize(bg_frm, (video_meta_data['width'], video_meta_data['height']))
    cap = cv2.VideoCapture(video_path)
    frm_cnt = 0
    while True:
        ret, frm = cap.read()
        if ret:
            out_img = np.full_like(frm, fill_value=bg_color)
            if not ret:
                break
            img_diff = np.abs(frm - bg_frm)
            gray_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
            mask = np.where(gray_diff < threshold, 0, 1)
            if fg_color is None:
                out_img[mask == 1] = frm[mask == 1]
            else:
                out_img[mask == 1] = fg_color
            writer.write(out_img)
            frm_cnt += 1
            if verbose:
                print(f'Background subtraction frame {frm_cnt}/{video_meta_data["frame_count"]} (Video: {video_name})')
        else:
            break

    writer.release()
    cap.release()
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'Background subtracted from {video_name} and saved at {save_path}', elapsed_time=timer.elapsed_time)



video_bg_subtraction(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0514_clipped.mp4',
                     fg_color=(255, 0, 0), threshold=255)