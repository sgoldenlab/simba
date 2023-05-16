__author__ = "Simon Nilsson"

import os
import subprocess, shutil
from datetime import datetime
from typing import Dict, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from simba.utils.read_write import get_fn_ext, remove_a_folder, get_video_meta_data, read_config_file, read_config_entry
from simba.utils.printing import stdout_success, SimbaTimer
from simba.utils.enums import Paths
from simba.mixins.config_reader import ConfigReader


class FrameMergererFFmpeg(ConfigReader):
    """
    Merge separate visualizations of classifications, descriptive statistics etc., into  single
    video mosaic.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str concat_type: Type of concatenation. E.g. ``vertical``, ``horizontal``
    :parameter dict frame_types: Dict holding video path to videos to concatenate. E.g., {'Video 1': path, 'Video 2': path}
    :parameter int video_height: Output video height.
    :parameter int video_width: Output video width.

    .. note:
       `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-12-merge-frames>`_.

    Example
    ----------
    >>> frame_types={'Video 1': 'project_folder/videos/Video_1.avi', 'Video 2': 'project_folder/videos/Video_2.avi'}
    >>> video_height, video_width, concat_type = 640, 480, 'vertical'
    >>> FrameMergererFFmpeg(config_path='MySimBaConfigPath', frame_types=frame_types, video_height=video_height, video_width=video_width, concat_type=concat_type)
    """

    def __init__(self,
                 concat_type: Literal['horizontal', 'vertical', 'mosaic', 'mixed_mosaic'],
                 frame_types: Dict[str, str],
                 video_height: int,
                 video_width: int,
                 config_path: Optional[str] = None):

        self.timer = SimbaTimer(start=True)
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        if config_path is not None:
            ConfigReader.__init__(self, config_path=config_path)
            self.output_dir = os.path.join(self.project_path, Paths.CONCAT_VIDEOS_DIR.value)
            self.temp_dir = os.path.join(self.project_path, Paths.CONCAT_VIDEOS_DIR.value, 'temp')
            self.output_path = os.path.join(self.project_path, Paths.CONCAT_VIDEOS_DIR.value, f'merged_video_{self.datetime}.mp4')
        else:
            self.file_path = list(frame_types.values())[0]
            self.output_dir, ss, df = get_fn_ext(filepath=self.file_path)
            self.temp_dir = os.path.join(self.output_dir, 'temp')
            self.output_path = os.path.join(self.output_dir, f'merged_video_{self.datetime}.mp4')

        self.video_height, self.video_width = video_height, video_width
        self.frame_types, self.concat_type = frame_types, concat_type
        self.video_cnt = len(self.frame_types.keys())
        self.even_bool = (len(self.frame_types.keys()) % 2) == 0
        self.blank_path = os.path.join(self.temp_dir, 'blank.mp4')
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        if not os.path.exists(self.temp_dir): os.makedirs(self.temp_dir)
        if concat_type == 'horizontal':
            self.__horizontal_concatenator(out_path=self.output_path, frames_dict=self.frame_types, final_img=True)
        elif concat_type == 'vertical':
            self.__vertical_concatenator(out_path=self.output_path, frames_dict=self.frame_types, final_img=True)
        elif concat_type == 'mosaic':
            self.__mosaic_concatenator(output_path=self.output_path, frames_dict=self.frame_types, final_img=True)
        elif concat_type == 'mixed_mosaic':
            self.__mixed_mosaic_concatenator()
        remove_a_folder(folder_dir=self.temp_dir)

    def __resize_height(self, new_height: int):
        for video_cnt, (video_type, video_path) in enumerate(self.frame_types.items()):
            video_meta_data = get_video_meta_data(video_path=video_path)
            out_path = os.path.join(self.temp_dir, video_type + '.mp4')
            if video_meta_data['height'] != new_height:
                print('Resizing {}...'.format(video_type))
                command = 'ffmpeg -y -i "{}" -vf scale=-2:{} "{}" -hide_banner -loglevel error'.format(video_path, new_height,  out_path)
                subprocess.call(command, shell=True, stdout=subprocess.PIPE)
            else:
                shutil.copy(video_path, out_path)

    def __resize_width(self, new_width: int):
        """ Helper to change the width of videos"""

        for video_cnt, (video_type, video_path) in enumerate(self.frame_types.items()):
            video_meta_data = get_video_meta_data(video_path=video_path)
            out_path = os.path.join(self.temp_dir, video_type + '.mp4')
            if video_meta_data['height'] != new_width:
                print('Resizing {}...'.format(video_type))
                command = 'ffmpeg -y -i "{}" -vf scale={}:-2 "{}" -hide_banner -loglevel error'.format(video_path, new_width,  out_path)
                subprocess.call(command, shell=True, stdout=subprocess.PIPE)
            else:
                shutil.copy(video_path, out_path)

    def __resize_width_and_height(self,
                                new_width: int,
                                new_height: int):
        """ Helper to change the width and height videos"""

        for video_cnt, (video_type, video_path) in enumerate(self.frame_types.items()):
            video_meta_data = get_video_meta_data(video_path=video_path)
            out_path = os.path.join(self.temp_dir, video_type + '.mp4')
            if video_meta_data['height'] != new_width:
                print('Resizing {}...'.format(video_type))
                command = 'ffmpeg -y -i "{}" -vf scale={}:{} "{}" -hide_banner -loglevel error'.format(video_path, new_width, new_height,  out_path)
                subprocess.call(command, shell=True, stdout=subprocess.PIPE)
            else:
                shutil.copy(video_path, out_path)

    def __create_blank_video(self):
        """ Helper to create a blank (black) video """

        video_meta_data = get_video_meta_data(list(self.frame_types.values())[0])
        cmd = 'ffmpeg -y -t {} -f lavfi -i color=c=black:s={}x{} -c:v libx264 -tune stillimage -pix_fmt yuv420p "{}" -hide_banner -loglevel error'.format(str(video_meta_data['video_length_s']), str(self.video_width), str(self.video_height), self.blank_path)
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        self.frame_types['blank'] = self.blank_path

    def __horizontal_concatenator(self,
                                  frames_dict: dict,
                                  out_path: str,
                                  include_resize=True,
                                  final_img=True):
        """ Helper to horizontally concatenate N videos """

        if include_resize: self.__resize_height(new_height=self.video_height)
        video_path_str = ''
        for video_type in frames_dict.keys():
            video_path_str += ' -i "{}"'.format(os.path.join(self.temp_dir, video_type + '.mp4'))
        cmd = 'ffmpeg -y{} -filter_complex hstack=inputs={} -vsync 2 "{}"'.format(video_path_str, str(len(frames_dict.keys())), out_path)
        print('Concatenating (horizontal) {} videos...'.format(str(len(frames_dict.keys()))))
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        if final_img:
            self.timer.stop_timer()
            stdout_success(msg=f'Merged video saved at {out_path}', elapsed_time=self.timer.elapsed_time_str)

    def __vertical_concatenator(self,
                                frames_dict: dict,
                                out_path: str,
                                include_resize=True,
                                final_img=True):
        """ Helper to vertically concatenate N videos """

        if include_resize: self.__resize_width(new_width=self.video_width)
        video_path_str = ''
        for video_type in frames_dict.keys():
            video_path_str += ' -i "{}"'.format(os.path.join(self.temp_dir, video_type + '.mp4'))
        cmd = 'ffmpeg -y{} -filter_complex vstack=inputs={} -vsync 2 "{}"'.format(video_path_str, str(len(frames_dict.keys())), out_path)
        print('Concatenating (vertical) {} videos...'.format(str(len(frames_dict.keys()))))
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        if final_img:
            self.timer.stop_timer()
            stdout_success(msg=f'Merged video saved at {out_path}', elapsed_time=self.timer.elapsed_time_str)

    def __mosaic_concatenator(self,
                            frames_dict: dict,
                            output_path: str,
                            final_img:bool):
        self.__resize_width_and_height(new_width=self.video_width, new_height=self.video_height)
        if not self.even_bool:
            self.__create_blank_video()
        lower_dict = dict(list(frames_dict.items())[len(frames_dict)//2:])
        upper_dict = dict(list(frames_dict.items())[:len(frames_dict)//2])
        if (len(upper_dict.keys()) == 1) & (len(lower_dict.keys()) != 1): upper_dict['blank'] = self.blank_path
        if (len(lower_dict.keys()) == 1) & (len(upper_dict.keys()) != 1): lower_dict['blank'] = self.blank_path
        self.__horizontal_concatenator(frames_dict=upper_dict, out_path=os.path.join(self.temp_dir, 'upper.mp4'), include_resize=False, final_img=False)
        self.__horizontal_concatenator(frames_dict=lower_dict, out_path=os.path.join(self.temp_dir, 'lower.mp4'), include_resize=False, final_img=False)
        frames_dict = {'upper': os.path.join(self.temp_dir, 'upper.mp4'), 'lower': os.path.join(self.temp_dir, 'lower.mp4')}
        self.__vertical_concatenator(frames_dict=frames_dict, out_path=output_path, include_resize=False, final_img=final_img)

    def __mixed_mosaic_concatenator(self):
        large_mosaic_dict = {str(list(self.frame_types.keys())[0]): str(list(self.frame_types.values())[0])}
        del self.frame_types[str(list(self.frame_types.keys())[0])]
        self.even_bool = (len(self.frame_types.keys()) % 2) == 0
        output_path = os.path.join(self.temp_dir, 'mosaic.mp4')
        self.__mosaic_concatenator(frames_dict=self.frame_types, output_path=output_path, final_img=False)
        self.frame_types = large_mosaic_dict
        self.__resize_height(new_height=get_video_meta_data(video_path=output_path)['height'])
        self.frame_types = {str(list(large_mosaic_dict.keys())[0]): os.path.join(self.temp_dir, str(list(large_mosaic_dict.keys())[0]) + '.mp4'), 'mosaic': os.path.join(self.temp_dir, 'mosaic.mp4')}
        self.__horizontal_concatenator(frames_dict=self.frame_types, out_path=self.output_path, include_resize=False, final_img=True)



# FrameMergererFFmpeg(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                     frame_types={'Video 1': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                                  'Video 2': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi'},
#                     video_height=640,
#                     video_width=480,
#                     concat_type='vertical') #horizontal, vertical, mosaic, mixed_mosaic
#
#
# FrameMergererFFmpeg(config_path=None,
#                     frame_types={'Video 1': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                                  'Video 2': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi'},
#                     video_height=640,
#                     video_width=480,
#                     concat_type='vertical') #horizontal, vertical, mosaic, mixed_mosaic
#
#


# FrameMergererFFmpeg(config_path=None,
#                     frame_types={'Video 1': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                                  'Video 2': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_2.avi',
#                                  'Video 3': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_3.avi',
#                                  'Video 4': '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_3.avi'},
#                     video_height=640,
#                     video_width=480,
#                     concat_type='mixed_mosaic') #horizontal, vertical, mosaic, mixed_mosaic


