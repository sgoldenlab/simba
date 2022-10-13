import os, glob
import json
import shutil
import subprocess
import cv2
from simba.misc_tools import get_video_meta_data
from simba.read_config_unit_tests import check_file_exist_and_readable
import datetime
import simba
import pathlib

class FFMPEGCommandCreator(object):

    """
    Class for executing FFmpeg commands from instructions stored in json format.

    Parameters
    ----------
    json_path: str
        path to json file storing FFmpeg instructions

    Notes
    ----------
    `Batch pre-process tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`__.

    Examples
    ----------
    >>> ffmpeg_executor = FFMPEGCommandCreator(json_path='MyJsonFilePath')
    >>> ffmpeg_executor.crop_videos()
    >>> ffmpeg_executor.clip_videos()
    >>> ffmpeg_executor.downsample_videos()
    >>> ffmpeg_executor.apply_fps()
    >>> ffmpeg_executor.apply_grayscale()
    >>> ffmpeg_executor.apply_frame_count()
    >>> ffmpeg_executor.apply_clahe()
    >>> ffmpeg_executor.move_all_processed_files_to_output_folder()
    """

    def __init__(self,
                 json_path: str=None):

        check_file_exist_and_readable(json_path)
        with open(json_path, 'r') as fp:
            self.video_dict = json.load(fp)
        self.input_dir = self.video_dict['meta_data']['in_dir']
        self.out_dir = self.video_dict['meta_data']['out_dir']
        self.temp_dir = os.path.join(self.out_dir, 'temp')
        self.time_format = '%H:%M:%S'
        if os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        self.copy_videos_to_temp_dir()

    def replace_files_in_temp(self):
        processed_files = glob.glob(self.process_dir + '/*')
        for file_path in processed_files:
            file_path_basename = os.path.basename(file_path)
            shutil.move(file_path, os.path.join(self.temp_dir, file_path_basename))

    def create_process_dir(self):
        self.process_dir = os.path.join(self.temp_dir, 'process_dir')
        if os.path.exists(self.process_dir): shutil.rmtree(self.process_dir)
        os.makedirs(self.process_dir)

    def find_relevant_videos(self, variable=None):
        videos = {}
        for video, video_data in self.video_dict['video_data'].items():
            if video_data[variable]:
                video_basename = video_data['video_info']['video_name'] + video_data['video_info']['extension']
                videos[video_data['video_info']['video_name']] = {}
                videos[video_data['video_info']['video_name']]['path'] = os.path.join(self.temp_dir, video_basename)
                videos[video_data['video_info']['video_name']]['settings'] = video_data[variable + '_settings']
        return videos

    def downsample_videos(self):
        self.videos_to_downsample = self.find_relevant_videos(variable='downsample')
        self.create_process_dir()
        for video, video_data in self.videos_to_downsample.items():
            print('Down-sampling {}...'.format(video))
            in_path, out_path = video_data['path'], os.path.join(self.process_dir, os.path.basename(video_data['path']))
            width, height = str(video_data['settings']['width']), str(video_data['settings']['height'])
            command = 'ffmpeg -i {} -vf scale={}:{} {} -hide_banner -loglevel error'.format(in_path, width, height, out_path)
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print('Downsampling complete...')

    def clip_videos(self):
        self.videos_to_clip = self.find_relevant_videos(variable='clip')
        self.create_process_dir()
        for video, video_data in self.videos_to_clip.items():
            print('Clipping {}...'.format(video))
            in_path, out_path = video_data['path'], os.path.join(self.process_dir, os.path.basename(video_data['path']))
            start_time, end_time = str(video_data['settings']['start']).replace(' ', ''), str(video_data['settings']['stop']).replace(' ', '')
            start_time_shift, end_time = datetime.datetime.strptime(start_time, self.time_format), datetime.datetime.strptime(end_time, self.time_format)
            time_difference = str(end_time - start_time_shift)
            command = 'ffmpeg -i {} -ss {} -t {} -async 1 -qscale 0 -c:a copy {} -hide_banner -loglevel error'.format(in_path, start_time, time_difference, out_path)
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print('Clipping complete...')

    def apply_fps(self):
        self.videos_to_change_fps = self.find_relevant_videos(variable='fps')
        self.create_process_dir()
        for video, video_data in self.videos_to_change_fps.items():
            print('Changing FPS {}...'.format(video))
            in_path, out_path = video_data['path'], os.path.join(self.process_dir, os.path.basename(video_data['path']))
            fps = str(video_data['settings']['fps'])
            command = 'ffmpeg -i {} -filter:v fps=fps={} {} -hide_banner -loglevel error'.format(in_path, fps, out_path)
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print('FPS conversion complete...')


    def apply_grayscale(self):
        self.videos_to_greyscale = self.find_relevant_videos(variable='grayscale')
        self.create_process_dir()
        for video, video_data in self.videos_to_greyscale.items():
            print('Applying grayscale {}...'.format(video))
            in_path, out_path = video_data['path'], os.path.join(self.process_dir, os.path.basename(video_data['path']))
            command = 'ffmpeg -i {} -vf hue=s=0 {} -hide_banner -loglevel error'.format(in_path, out_path)
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print('Grayscale complete...')

    def apply_frame_count(self):
        self.videos_to_frm_cnt = self.find_relevant_videos(variable='frame_cnt')
        self.create_process_dir()
        simba_cw = os.path.dirname(simba.__file__)
        #simba_font_path = os.path.join(simba_cw, 'assets', 'UbuntuMono-Regular.ttf')
        simba_font_path = pathlib.Path(simba_cw, 'assets', 'UbuntuMono-Regular.ttf')
        for video, video_data in self.videos_to_frm_cnt.items():
            print('Applying frame count print {}...'.format(video))
            in_path, out_path = video_data['path'], os.path.join(self.process_dir, os.path.basename(video_data['path']))
            #command = 'ffmpeg -i ' + in_path + ' -vf "drawtext=: ' + "text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" + '" ' + "-c:a copy " + out_path + ' -hide_banner -loglevel error'
            command = 'ffmpeg -i ' + in_path + ' -vf "drawtext=fontfile={}:'.format(simba_font_path) + "text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" + '" ' + "-c:a copy " + out_path + ' -hide_banner -loglevel error'
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print('Applying frame count complete...')

    def apply_clahe(self):
        self.videos_to_frm_cnt = self.find_relevant_videos(variable='clahe')
        self.create_process_dir()
        for video, video_data in self.videos_to_frm_cnt.items():
            clahe_filter = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
            print('Applying CLAHE {}... (note: process can be slow for long videos)'.format(video))
            in_path, out_path = video_data['path'], os.path.join(self.process_dir, os.path.basename(video_data['path']))
            video_info = get_video_meta_data(in_path)
            cap = cv2.VideoCapture(in_path)
            fps, width, height = video_info['fps'], video_info['width'], video_info['height']
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height), 0)
            while True:
                ret, image = cap.read()
                if ret == True:
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe_img = clahe_filter.apply(img)
                    writer.write(clahe_img)
                else:
                    break
            cap.release()
            writer.release()
        self.replace_files_in_temp()
        print('Applying CLAHE complete...')

    def crop_videos(self):
        self.videos_to_crop = self.find_relevant_videos(variable='crop')
        self.create_process_dir()
        for video, video_data in self.videos_to_crop.items():
            print('Applying crop {}...'.format(video))
            in_path, out_path = video_data['path'], os.path.join(self.process_dir, os.path.basename(video_data['path']))
            crop_settings = self.video_dict['video_data'][video]['crop_settings']
            width, height = str(crop_settings['width']), str(crop_settings['height'])
            top_left_x, top_left_y = str(crop_settings['top_left_x']), str(crop_settings['top_left_y'])
            command = 'ffmpeg -i {} -vf "crop={}:{}:{}:{}" -c:v libx264 -crf 0 -c:a copy {} -hide_banner -loglevel error'.format(in_path, width, height, top_left_x, top_left_y, out_path)
            subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        self.replace_files_in_temp()
        print('Applying crop complete...')

    def copy_videos_to_temp_dir(self):
        for video, video_data in self.video_dict['video_data'].items():
            source = video_data['video_info']['file_path']
            print('Making a copy of {} ...'.format(os.path.basename(source)))
            destination = os.path.join(self.temp_dir, os.path.basename(source))
            shutil.copyfile(source, destination)

    def move_all_processed_files_to_output_folder(self):
        final_videos_path_lst = [f for f in glob.glob(self.temp_dir + '/*') if os.path.isfile(f)]
        for file_path in final_videos_path_lst:
            shutil.copy(file_path, os.path.join(self.out_dir, os.path.basename(file_path)))
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.process_dir):
            shutil.rmtree(self.process_dir)

# test = FFMPEGCommandCreator(json_path='/Users/simon/Desktop/train_model_project/project_folder/videos_2/batch_process.json_log')
# test.crop_videos()
# test.clip_videos()
# test.downsample_videos()
# test.apply_fps()
# test.apply_grayscale()
# test.apply_frame_count()
# test.apply_clahe()
# test.move_all_processed_files_to_output_folder()