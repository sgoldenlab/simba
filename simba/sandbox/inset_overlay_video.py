from typing import Union, Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import os
import subprocess
from simba.utils.read_write import get_fn_ext, find_all_videos_in_directory, get_video_meta_data
from simba.utils.checks import check_float, check_str, check_if_dir_exists, check_file_exist_and_readable
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success

def inset_overlay_video(video_path: Union[str, os.PathLike],
                        overlay_video_path: Union[str, os.PathLike],
                        position: Optional[Literal['top_left', 'bottom_right', 'top_right', 'bottom_left', 'center']] = 'top_left',
                        opacity: Optional[float] = 0.5,
                        scale: Optional[float] = 0.05,
                        save_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    """
    Inset a video overlay on a second video with specified size, opacity, and location.

    .. video:: _static/img/inset_overlay_video.webm
       :loop:

    :param Union[str, os.PathLike] video_path: The path to the video file.
    :param Union[str, os.PathLike] overlay_video_path: The path to video to be inserted into the ``video_path`` video.
    :param Optional[str] position: The position of the inset overlay video. Options: 'top_left', 'bottom_right', 'top_right', 'bottom_left', 'center'
    :param Optional[float] opacity: The opacity of the inset overlay video as a value between 0-1.0. 1.0 meaning the same as input image. Default: 0.5.
    :param Optional[float] scale: The scale of the inset overlay video as a ratio os the image size. Default: 0.05.
    :param Optional[Union[str, os.PathLike]] save_dir: The location where to save the output video. If None, then saves the video in the same directory as the first video.
    :return: None

    :example:
    >>> inset_overlay_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/multi_animal_dlc_two_c57/project_folder/videos/watermark/Together_1_powerpointready.mp4', overlay_video_path='/Users/simon/Desktop/splash.png', position='top_left', opacity=1.0, scale=0.2)
    """

    timer = SimbaTimer(start=True)
    POSITIONS = ['top_left', 'bottom_right', 'top_right', 'bottom_left', 'center']
    check_float(name=f'{inset_overlay_video.__name__} opacity', value=opacity, min_value=0.001, max_value=1.0)
    check_float(name=f'{inset_overlay_video.__name__} scale', value=scale, min_value=0.001, max_value=0.999)
    check_str(name=f'{inset_overlay_video.__name__} position', value=position, options=POSITIONS)
    check_file_exist_and_readable(file_path=video_path)
    check_file_exist_and_readable(file_path=overlay_video_path)

    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=inset_overlay_video.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, video_ext = get_fn_ext(video_path)
        _ = get_video_meta_data(video_path=video_path)
        print(f'Inserting overlay onto {video_name} (Video {file_cnt+1}/{len(video_paths)})...')
        out_path = os.path.join(save_dir, f'{video_name}_inset_overlay{video_ext}')
        print(out_path)
        if position == POSITIONS[0]:
            cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=0:0" "{out_path}" -y'
        elif position == POSITIONS[1]:
            cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=W-w:H-h" "{out_path}" -y'
        elif position == POSITIONS[2]:
            cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=W-w:0" "{out_path}" -y'
        elif position == POSITIONS[3]:
            cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=0:H-h" "{out_path}" -y'
        else:
            cmd = f'ffmpeg -i "{video_path}" -i "{overlay_video_path}" -filter_complex "[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[inset];[0:v][inset]overlay=(W-w)/2:(H-h)/2" "{out_path}" -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'{len(video_paths)} overlay video(s) saved in {save_dir}', elapsed_time=timer.elapsed_time_str)


# inset_overlay_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4.mp4',
#                     overlay_video_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/frames/output/heatmaps_locations/2022-06-20_NOB_DOT_4.mp4',
#                     position='top_left', opacity=0.5, scale=0.3)
