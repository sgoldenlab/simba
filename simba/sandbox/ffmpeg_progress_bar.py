from typing import Union, Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal
import os
import subprocess

from simba.utils.checks import check_file_exist_and_readable, check_int, check_if_dir_exists
from simba.utils.read_write import get_video_meta_data, get_fn_ext, find_all_videos_in_directory
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success


def overlay_video_progressbar(video_path: Union[str, os.PathLike],
                              bar_height: Optional[int] = 10,
                              color: Optional[str] = 'red',
                              position: Optional[str] = 'top',
                              save_dir: Optional[ Union[str, os.PathLike]] = None) -> None:

    """
    Overlay a progress bar on a directory of videos or a single video.

    .. video:: _static/img/overlay_video_progressbar.webm
       :loop:

    :param Union[str, os.PathLike] video_path: Directory containing video files or a single video file
    :param Optional[int] bar_height: The height of the progressbar in percent of the video height.
    :param Optional[str] color: The color of the progress bar. See simba.utils.lookups.get_color_dict keys for accepted names.
    :param Optional[Union[str, os.PathLike]] save_dir: If not None, then saves the videos in the passed directory. Else, in the same directry with the ``_progress_bar`` suffix.
    :return: None.
    """

    timer = SimbaTimer(start=True)
    color = ''.join(filter(str.isalnum, color)).lower()
    if os.path.isfile(video_path):
        check_file_exist_and_readable(file_path=video_path)
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_path = find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True)
        video_paths = list(video_path.values())
    else:
        raise InvalidInputError(msg='{} is not a valid file path or directory path.', source=overlay_video_progressbar.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir, _, _ = get_fn_ext(filepath=video_paths[0])
    for cnt, video_path in enumerate(video_paths):
        video_meta_data = get_video_meta_data(video_path=video_path)
        video_length = video_meta_data['video_length_s']
        width, height = video_meta_data['width'], video_meta_data['height']
        bar_height = int(height * (bar_height/100))
        _, video_name, ext = get_fn_ext(filepath=video_path)
        print(f'Inserting progress bar on video {video_name}...')
        save_path = os.path.join(save_dir, f'{video_name}_progress_bar{ext}')
        check_int(name=f'{overlay_video_progressbar} height', value=bar_height, max_value=height, min_value=1)
        if position == 'bottom':
            cmd = f'ffmpeg -i "{video_path}" -filter_complex "color=c={color}:s={width}x{bar_height}[bar];[0][bar]overlay=-w+(w/{video_length})*t:H-h:shortest=1" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        elif position == 'top':
            cmd = f'ffmpeg -i "{video_path}" -filter_complex "color=c={color}:s={width}x{bar_height}[bar];[0][bar]overlay=-w+(w/{video_length})*t:{bar_height}-h:shortest=1" -c:a copy "{save_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) saved with progressbar in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=overlay_video_progressbar.__name__, )

overlay_video_progressbar(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/reptile/AGGRESSIVITY_4_11_21_Trial_2_camera1_clipped.mp4',
                          bar_height=50,
                          color='green',
                          position='right')