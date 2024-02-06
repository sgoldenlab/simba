import multiprocessing
import os
import subprocess
import functools
import shutil
try:
    from typing import List
except:
    from typing_extensions import List

from simba.utils.checks import check_int, check_file_exist_and_readable
from simba.utils.read_write import get_fn_ext

class VideoProcessingMixin(object):
    """
    Methods for videos processing
    """
    def __init__(self):
        pass

    @staticmethod
    def _chunk_video_helper(chunk_range, video_path, chunk_ranges, temp_dir):
        start_time, end_time = chunk_range
        chunk_index = chunk_ranges.index(chunk_range)
        output_file = os.path.join(temp_dir, f'{chunk_index}.mp4')
        command = 'ffmpeg -i "{}" -ss {} -to {} -c copy "{}" -y'.format(video_path, start_time, end_time, output_file)
        subprocess.call(command, shell=True)
        return output_file

    @staticmethod
    def _ffmpeg_cmd_multiprocessor(command: str):
        print(command)
        subprocess.call(command, shell=True)


    def split_video_into_n_cunks(self,
                                 video_path: str,
                                 n: int):

        dir, video_name, _ = get_fn_ext(filepath=video_path)
        temp_dir = os.path.join(dir, video_name + '_temp')
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        os.makedirs(dir, exist_ok=True)
        duration = float(subprocess.check_output('ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{}" -hide_banner -loglevel error'.format(video_path), shell=True))
        chunk_duration = duration / n
        chunk_ranges = [(i * chunk_duration, (i + 1) * chunk_duration) for i in range(n)]
        file_paths = []
        with multiprocessing.Pool(n, maxtasksperchild=10) as pool:
            constants = functools.partial(self._chunk_video_helper,
                                          video_path=video_path,
                                          chunk_ranges=chunk_ranges,
                                          temp_dir=temp_dir)
            for cnt, result in enumerate(pool.imap(constants, chunk_ranges, chunksize=1)):
                 file_paths.append(result)
        pool.terminate()
        pool.join()
        return file_paths


    def create_ouput_paths(self,
                           video_input_paths: List[os.PathLike]):

        temp_folder = os.path.join(os.path.dirname(video_input_paths[0]), 'temp')
        if os.path.isdir(temp_folder): shutil.rmtree(temp_folder)
        if not os.path.isdir(temp_folder): os.makedirs(temp_folder)
        output_paths =[os.path.join(temp_folder, os.path.basename(x)) for x in video_input_paths]
        return output_paths





# video_processor = VideoProcessingMixin()
#
# video_processor.change_single_video_fps(video_path='/Users/simon/Desktop/Example_1_frame_no.mp4', core_cnt=5)
#
#



