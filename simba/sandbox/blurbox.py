from typing import Union, Optional
import os
import subprocess
from simba.utils.read_write import get_fn_ext, find_all_videos_in_directory, get_video_meta_data
from simba.video_processors.roi_selector import ROISelector
from simba.utils.checks import check_ffmpeg_available, check_float, check_if_dir_exists, check_file_exist_and_readable
from simba.utils.printing import SimbaTimer, stdout_success


def roi_blurbox(video_path: Union[str, os.PathLike],
                blur_level: Optional[float] = 0.99,
                invert: Optional[bool] = False,
                save_path: Optional[Union[str, os.PathLike]] = None) -> None:

    """
    Blurs either the selected or unselected portion of a region-of-interest according to the passed blur level.
    Higher blur levels produces more opaque regions.

    .. video:: _static/img/roi_blurbox.webm
       :loop:

    :param Union[str, os.PathLike] video_path: The path to the video on disk
    :param Optional[float] blur_level: The level of the blur as a ratio 0-1.0.
    :param Optional[bool] invert: If True, blurs the unselected region. If False, blurs the selected region.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to save the blurred video. If None, then saves the blurred video in the same directory as the input video with the ``_blurred`` suffix.
    :return: None

    :example:
    >>> roi_blurbox(video_path='/Users/simon/Downloads/1_LH_clipped_downsampled.mp4', blur_level=0.2, invert=True)
    """
    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    check_float(name=f'{roi_blurbox.__name__} blur_level', value=blur_level, min_value=0.001, max_value=1.0)
    check_file_exist_and_readable(file_path=video_path)
    dir, video_name, ext = get_fn_ext(video_path)
    _ = get_video_meta_data(video_path=video_path)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=roi_blurbox.__name__)
    else:
        save_path = os.path.join(dir, f'{video_name}_blurred{ext}')
    roi_selector = ROISelector(path=video_path)
    roi_selector.run()
    w, h = roi_selector.width, roi_selector.height
    top_left_x, top_left_y = roi_selector.top_left
    max_blur_value = int(min(w, h) / 2) / 2
    blur_level = int(max_blur_value * blur_level)
    if not invert:
        cmd = f'ffmpeg -i "{video_path}" -filter_complex "[0:v]crop={w}:{h}:{top_left_x}:{top_left_y},boxblur={int(blur_level)}:10[fg]; [0:v][fg]overlay={top_left_x}:{top_left_y}[v]" -map "[v]" "{save_path}" -loglevel error -stats -hide_banner -y'
    else:
        cmd = f'ffmpeg -i "{video_path}" -filter_complex "[0:v]boxblur={blur_level}[bg];[0:v]crop={w}:{h}:{top_left_x}:{top_left_y}[fg];[bg][fg]overlay={top_left_x}:{top_left_y}" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Blurred {video_name} video saved in {save_path}', elapsed_time=timer.elapsed_time_str)

#roi_blurbox(video_path='/Users/simon/Downloads/1_LH_clipped_downsampled.mp4', blur_level=0.7, invert=False)
