__author__ = "Simon Nilsson"

from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry)
import os
import subprocess, shutil
from simba.misc_tools import (get_video_meta_data,
                              remove_a_folder,
                              get_fn_ext,
                              SimbaTimer)
from datetime import datetime
from simba.enums import Paths, ReadConfig


class FrameMergererFFmpeg(object):
    """
    Class for merging separate visualizations of classifications, descriptive statistics etc., into  single
    video mosaic.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    concat_type: str
        Type of concatenation. E.g. ``vertical``, ``horizontal``
    frame_types: dict
        Dict holding video  path to videos to concatenate. E.g., {'Video 1': path, 'Video 2': path}
    video_height: int
        Output video height (width depends on frame_types count)

    Notes
    -----
    `GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-12-merge-frames>`__.


    """

    def __init__(self,
                 config_path: str or None,
                 concat_type: str,
                 frame_types: dict,
                 video_height: int or None,
                 video_width: int or None):

        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        if config_path is not None:
            self.config_path, self.config = config_path, read_config_file(ini_path=config_path)
            self.project_path = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.PROJECT_PATH.value, data_type=ReadConfig.FOLDER_PATH.value)
            self.output_dir = os.path.join(self.project_path, Paths.CONCAT_VIDEOS_DIR.value)
            self.temp_dir = os.path.join(self.project_path, Paths.CONCAT_VIDEOS_DIR.value, 'temp')
            self.output_path = os.path.join(self.project_path, Paths.CONCAT_VIDEOS_DIR.value, 'merged_video_{}.mp4'.format(str(self.datetime)))
        else:
            self.file_path = list(frame_types.values())[0]
            self.output_dir, ss, df = get_fn_ext(filepath=self.file_path)
            self.temp_dir = os.path.join(self.output_dir, 'temp')
            self.output_path = os.path.join(self.output_dir, 'merged_video_{}.mp4'.format(str(self.datetime)))

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
            print('SIMBA COMPLETE: Merged video saved at {} (elapsed time: {}s)'.format(out_path, self.timer.elapsed_time_str))

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
            print('SIMBA COMPLETE: Merged video saved at {} (elapsed time: {}s)'.format(out_path, self.timer.elapsed_time_str))

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



# FrameMergererFFmpeg(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                     frame_types={'Video 1': '/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/videos/Together_1.mp4', 'Video 2': '/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/videos/Together_2.avi'},
#                     video_height=640,
#                     video_width=480,
#                     concat_type='vertical') #horizontal, vertical, mosaic, mixed_mosaic
#
#


