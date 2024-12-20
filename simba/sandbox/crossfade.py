from typing import Union, Optional
import os.path
import subprocess
from simba.utils.read_write import get_fn_ext, get_video_meta_data
from simba.utils.checks import check_int, check_str, check_if_dir_exists
from simba.utils.lookups import get_ffmpeg_crossfade_methods
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.errors import InvalidInputError


def crossfade_two_videos(video_path_1: Union[str, os.PathLike],
                         video_path_2: Union[str, os.PathLike],
                         crossfade_duration: Optional[int] = 2,
                         crossfade_method: Optional[str] = 'fade',
                         crossfade_offset: Optional[int] = 2,
                         save_path: Optional[Union[str, os.PathLike]] = None):
    """
    Cross-fade two videos.

    .. video:: _static/img/overlay_video_progressbar.webm
       :loop:

    .. note::
       See ``simba.utils.lookups.get_ffmpeg_crossfade_methods`` for named crossfade methods.
       See `https://trac.ffmpeg.org/wiki/Xfade <https://trac.ffmpeg.org/wiki/Xfade>`__. for visualizations of named crossfade methods,

    :param Union[str, os.PathLike] video_path_1: Path to the first video on disk.
    :param Union[str, os.PathLike] video_path_2: Path to the second video on disk.
    :param Optional[int] crossfade_duration: The duration of the crossfade.
    :param Optional[str] crossfade_method: The crossfade method. For accepted methods, see ``simba.utils.lookups.get_ffmpeg_crossfade_methods``.
    :param Optional[int] crossfade_offset: The time in seconds into the first video before the crossfade duration begins.
    :param Optional[Union[str, os.PathLike]] save_path: The location where to save the crossfaded video. If None, then saves the video in the same directory as ``video_path_1`` with ``_crossfade`` suffix.

    :return: None.

    :example:
    >>> crossfade_two_videos(video_path_1='/Users/simon/Desktop/envs/simba/troubleshooting/reptile/1.mp4', video_path_2='/Users/simon/Desktop/envs/simba/troubleshooting/reptile/1.mp4', crossfade_duration=5, crossfade_method='zoomin', save_path='/Users/simon/Desktop/cross_test.mp4')
    """

    timer = SimbaTimer(start=True)
    video_1_meta = get_video_meta_data(video_path=video_path_1)
    video_2_meta = get_video_meta_data(video_path=video_path_2)
    if video_1_meta['resolution_str'] != video_1_meta['resolution_str']:
        raise InvalidInputError(msg=f'Video 1 and Video 2 needs to be the same resolution, got {video_2_meta["resolution_str"]} and {video_1_meta["resolution_str"]}', source=crossfade_two_videos.__name__)
    crossfade_offset_methods = get_ffmpeg_crossfade_methods()
    check_str(name=f'{crossfade_method} crossfade_method', value=crossfade_method, options=crossfade_offset_methods)
    check_int(name=f'{crossfade_two_videos.__name__} crossfade_duration', value=crossfade_duration, min_value=1, max_value=video_2_meta['video_length_s'])
    check_int(name=f'{crossfade_two_videos.__name__} crossfade_offset', value=crossfade_offset, min_value=0, max_value=video_1_meta['video_length_s'])
    dir_1, video_name_1, ext_1 = get_fn_ext(filepath=video_path_1)
    dir_2, video_name_2, ext_2 = get_fn_ext(filepath=video_path_2)
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
    else:
        save_path = os.path.join(dir_1, f'{video_name_1}_{video_name_2}_crossfade{ext_1}')
    cmd = f'ffmpeg -i "{video_path_1}" -i "{video_path_2}" -filter_complex "xfade=transition={crossfade_method}:offset={crossfade_offset}:duration={crossfade_duration}" "{save_path}" -loglevel error -stats -hide_banner -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f'Cross-faded video saved at {save_path}', elapsed_time=timer.elapsed_time_str)


