import os
from typing import Union, Optional
import subprocess

from simba.utils.checks import check_ffmpeg_available, check_nvidea_gpu_available, check_if_dir_exists
from simba.utils.errors import FFMPEGCodecGPUError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import find_all_videos_in_directory, get_fn_ext

def batch_video_to_greyscale(directory: Union[str, os.PathLike], gpu: Optional[bool] = False) -> None:
    """
    Convert a directory of video file to greyscale mp4 format. The results are stored in the same directory as the
    input files with the ``_grayscale.mp4`` suffix.

    :parameter Union[str, os.PathLike] directory: Path to directory holding video files in color.
    :parameter Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.
    :raise FFMPEGCodecGPUError: If no GPU is found and ``gpu == True``.

    :example:
    >>> _ = batch_video_to_greyscale(directory='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test_2')
    """
    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=batch_video_to_greyscale.__name__)
    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=directory, source=batch_video_to_greyscale.__name__)
    video_paths = find_all_videos_in_directory(directory=directory, as_dict=True, raise_error=True)
    for file_cnt, (file_name, file_path) in enumerate(video_paths.items()):
        video_timer = SimbaTimer(start=True)
        in_dir, _, _ = get_fn_ext(filepath=file_path)
        save_name = os.path.join(in_dir, f"{file_name}_grayscale.mp4")
        if gpu:
            command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -i "{file_path}" -vf "hwupload_cuda,hwdownload,format=nv12,format=gray" -c:v h264_nvenc -c:a copy "{save_name}" -y'
        else:
            command = f'ffmpeg -i "{file_path}" -vf format=gray -c:v libx264 "{save_name}" -hide_banner -loglevel error -y'
        print(f"Converting {file_name} to greyscale (Video {file_cnt+1}/{len(list(video_paths.keys()))})... ")
        subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        video_timer.stop_timer()
        print(f'Video {save_name} complete, (elapsed time: {video_timer.elapsed_time_str}s)')
    timer.stop_timer()
    stdout_success(msg=f"{len(list(video_paths.keys()))} video(s) converted to gresyscale! Saved in {directory} with '_greyscale' suffix", elapsed_time=timer.elapsed_time_str, source=batch_video_to_greyscale.__name__,)


#_ = batch_video_to_greyscale(directory='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/videos/test_2')