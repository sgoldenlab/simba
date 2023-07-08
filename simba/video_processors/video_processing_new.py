import os
import multiprocessing
import shutil
from typing import Union, Optional
import subprocess

from simba.utils.checks import check_file_exist_and_readable, check_int
from simba.utils.read_write import get_fn_ext, get_video_meta_data, concatenate_videos_in_folder
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.warnings import FileExistWarning, SameInputAndOutputWarning
from simba.mixins.video_processing_mixin import VideoProcessingMixin
from simba.utils.data import re

def change_single_video_fps(file_path: Union[str, os.PathLike],
                            fps: int,
                            gpu: Optional[bool] = False,
                            core_cnt: Optional[bool] = 1) -> None:
    """
    Change the fps of a single video file. Results are stored in the same directory as in the input file with
    the suffix ``_fps_new_fps``.

    :parameter Union[str, os.PathLike] file_path: Path to video file
    :parameter int fps: Fps of the new video file.

    :example:
    >>> _ = change_single_video_fps(file_path='project_folder/videos/Video_1.mp4', fps=15)
    """

    check_file_exist_and_readable(file_path=file_path)
    check_int(name='New fps', value=fps)
    check_int(name='CORE COUNT', value=core_cnt)
    video_processor = VideoProcessingMixin()
    video_meta_data = get_video_meta_data(video_path=file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if int(fps) == int(video_meta_data['fps']):
        SameInputAndOutputWarning(msg=f'The new fps is the same as the input fps for video {file_name} ({str(fps)})')
    save_path = os.path.join(dir_name, file_name + '_fps_{}{}'.format(str(fps), str(ext)))
    if os.path.isfile(save_path):
        FileExistWarning(msg=f'Overwriting existing file at {save_path}')

    timer = SimbaTimer(start=True)
    if core_cnt > 1:
        video_input_paths = video_processor.split_video_into_n_cunks(video_path=file_path, n=core_cnt)
        video_output_paths = video_processor.create_ouput_paths(video_input_paths=video_input_paths)
        cmds = []
        for input_path, output_path in zip(video_input_paths, video_output_paths):
            if gpu:
                cmds.append('ffmpeg -hwaccel auto -i "{}" -filter:v fps=fps={} "{}"'.format(input_path, fps, output_path))
            else:
                cmds.append('ffmpeg -i "{}" -filter:v fps=fps={} "{}" -threads 8'.format(input_path, fps, output_path))
        cmds = [x for x in cmds]
        with multiprocessing.Pool(core_cnt) as pool:
            for cnt, result in enumerate(pool.map(video_processor._ffmpeg_cmd_multiprocessor, cmds, chunksize=1)):
                print(f'Video {file_name} {round((((cnt+1)/len(cmds)) * 100), 2)}% complete...')
            pool.terminate()
            pool.join()
    #     concatenate_videos_in_folder(in_folder=os.path.dirname(video_output_paths[0]), save_path=save_path, remove_splits=True)
    #     shutil.rmtree(os.path.dirname(video_input_paths[0]))
    else:
        command = str('ffmpeg -i ') + '"' + str(file_path) + '"' + ' -filter:v fps=fps=' + str(fps) + ' ' + '"' + save_path + '" -y'
        subprocess.call(command, shell=True)

    timer.stop_timer()
    stdout_success(msg=f'SIMBA COMPLETE: FPS of video {file_name} changed from {str(video_meta_data["fps"])} to {str(fps)} and saved in directory {save_path}', elapsed_time=timer.elapsed_time_str)

        #
        #
        # else:
        #     cmd =
        #
        # with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
        #     video_paths = pool.map(video_processor._ffmpeg_cmd_multiprocessor, video_path)

change_single_video_fps(file_path='/Users/simon/Downloads/A_Ko15 8 PushOff.avi', fps=2, gpu=False, core_cnt=1)

