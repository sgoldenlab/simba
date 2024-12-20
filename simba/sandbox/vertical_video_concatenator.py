import os
from typing import List, Union, Optional
from simba.utils.read_write import get_video_meta_data
from simba.utils.checks import check_valid_lst, check_if_dir_exists, check_int, check_ffmpeg_available, check_nvidea_gpu_available
from simba.utils.errors import InvalidInputError, FFMPEGCodecGPUError
from simba.utils.printing import SimbaTimer
import subprocess

def vertical_video_concatenator(video_paths: List[Union[str, os.PathLike]],
                                save_path: Union[str, os.PathLike],
                                width_px: Optional[int] = None,
                                width_idx: Optional[int] = None,
                                gpu: Optional[bool] = False,
                                verbose: Optional[bool] = True) -> None:

    """
    Concatenates multiple videos vertically.

    :param List[Union[str, os.PathLike]] video_paths: List of input video file paths.
    :param Union[str, os.PathLike] save_path: File path to save the concatenated video.
    :param Optional[int] width_px: Width of the output video in pixels.
    :param Optional[int] width_idx: Index of the video to use for determining width.
    :param Optional[bool] gpu: Whether to use GPU-accelerated codec (default: False).
    :param Optional[bool] verbose:Whether to print progress messages (default: True).
    :raises FFMPEGCodecGPUError: If GPU is requested but not available.
    :raises InvalidInputError: If both or neither width_px and width_idx are provided.

    :example:
    >>> video_paths = ['/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat7_8(2).mp4',
    >>>                '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12.mp4',
    >>>                '/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/08102021_DOT_Rat11_12_1.mp4']
    >>> _ = vertical_video_concatenator(video_paths=video_paths, width_idx=1, save_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/test/new/08102021_DOT_Rat7_8(2)_.mp4', gpu=False)


    """

    check_ffmpeg_available()
    if gpu and not check_nvidea_gpu_available(): raise FFMPEGCodecGPUError(msg="NVIDIA GPU not available", source=vertical_video_concatenator.__name__)
    timer = SimbaTimer(start=True)
    check_valid_lst(data=video_paths, source=vertical_video_concatenator.__name__, min_len=2)
    check_if_dir_exists(in_dir=os.path.dirname(save_path), source=vertical_video_concatenator.__name__)

    if ((width_px is None) and (width_idx is None)) or ((width_px is not None) and (width_idx is not None)):
        raise InvalidInputError(msg='Provide a width_px OR width_idx', source=vertical_video_concatenator.__name__)
    if width_idx is not None:
        check_int(name=f'{vertical_video_concatenator.__name__} width index', value=width_idx, min_value=1, max_value=len(video_paths)-1)
        video_meta_data = [get_video_meta_data(video_path=video_path) for video_path in video_paths]
        width = int(video_meta_data[width_idx]['width'])
    else:
        check_int(name=f'{vertical_video_concatenator.__name__} width', value=width_px, min_value=1)
        width = int(width_px)
    video_path_str = " ".join([f'-i "{path}"' for path in video_paths])
    codec = 'h264_nvenc' if gpu else 'libx264'
    filter_complex = ";".join([f"[{idx}:v]scale={width}:-1[v{idx}]" for idx in range(len(video_paths))])
    filter_complex += f";{''.join([f'[v{idx}]' for idx in range(len(video_paths))])}"
    filter_complex += f"vstack=inputs={len(video_paths)}[v]"
    if verbose:
        print(f'Concatenating {len(video_paths)} video(s) vertically with a {width} pixel width...')
    cmd = f'ffmpeg {video_path_str} -filter_complex "{filter_complex}" -map "[v]" -c:v {codec} -loglevel error -stats "{save_path}" -y'
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    if verbose:
        print(f'Vertical concatenation complete. Saved at {save_path} (Elapsed time: {timer.elapsed_time_str}s.)')



