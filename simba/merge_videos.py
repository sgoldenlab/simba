__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file)
import os, glob
from collections import Counter, defaultdict
from simba.misc_tools import get_video_meta_data, get_fn_ext
from copy import deepcopy
import numpy as np
import cv2
import imutils

class FrameMergerer(object):
    """
    Class for merging separate visualizations on classifications, descriptive statistics etc., into  single
    video(s) or frames.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    frame_types: list
        list of the categories of videos to merge (e.g., 'Classifications', 'Gantt', 'Path', 'Heatmaps',
        'Probability', 'Data table')
    video_setting: bool
        If True, SimBA will create compressed videos.
    frame_setting: bool
        If True, SimBA will create individual frames
    output_height: int
        The height of the output video in pixels (e.g., 640, 1280, 1920, 2560).

    Notes
    ----------
    Output video width is determined by len(frame_types).

    Examples
    ----------
    >>> FrameMergerer(config_path='MyConfigPath', frame_types=['Classifications', 'Gantt', 'Path'], frame_setting=False, video_setting=True, output_height=1280)

    """

    def __init__(self,
                 config_path: str,
                 frame_types: list,
                 video_setting: bool,
                 frame_setting: bool,
                 output_height: int):


        if (frame_setting is False) & (video_setting is False):
            print('SIMBA ERROR: Please select frames and/or video output')
            raise ValueError('SIMBA ERROR: Please select frames and/or video output')
        self.config = read_config_file(config_path)
        self.frame_types = frame_types
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='str')
        self.clf_path = os.path.join(self.project_path, 'frames', 'output', 'sklearn_results')
        self.gantt_path = os.path.join(self.project_path, 'frames', 'output', 'gantt_plots')
        self.path_path = os.path.join(self.project_path, 'frames', 'output', 'path_plots')
        self.data_path = os.path.join(self.project_path, 'frames', 'output', 'live_data_table')
        self.distance_path = os.path.join(self.project_path, 'frames', 'output', 'line_plot')
        self.save_dir = os.path.join(self.project_path, 'frames', 'output', 'merged')
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.probability_path = os.path.join(self.project_path, 'frames', 'output', 'probability_plots')
        self.heatmaps_path = os.path.join(self.project_path, 'frames', 'output', 'heatmaps_classifier_locations')
        self.find_videos_in_each_dir()
        self.plot_cnt = len(frame_types)
        self.non_clf_plots = len(frame_types)
        self.video_height = output_height
        self.mosaic_size = (int(self.video_height / 2), int(self.video_height / 2))
        if 'Classifications' in frame_types:
            self.non_clf_plots -= 1
        if not (self.non_clf_plots % 2) == 0:
            self.mosaic_window_cnt = self.non_clf_plots + 1
        else:
            self.mosaic_window_cnt = self.non_clf_plots
        self.mosaic_window_columns = int(self.mosaic_window_cnt / 2)
        self.mosaic_window_rows = 2
        self.check_which_videos_exist_in_all_categories()
        self.check_frm_lenth_of_each_video()
        self.find_viable_video_names()
        self.create_videos()


    def find_videos_in_each_dir(self):
        self.video_in_each_dir = {}
        if 'Classifications' in self.frame_types:
            self.video_in_each_dir['Classifications'] = glob.glob(self.clf_path + '/*.mp4')
        if 'Gantt' in self.frame_types:
            self.video_in_each_dir['Gantt'] = glob.glob(self.gantt_path + '/*.mp4')
        if 'Path' in self.frame_types:
            self.video_in_each_dir['Path'] = glob.glob(self.path_path + '/*.mp4')
        if 'Data table' in self.frame_types:
            self.video_in_each_dir['Data table'] = glob.glob(self.data_path + '/*.mp4')
        if 'Distance' in self.frame_types:
            self.video_in_each_dir['Distance'] = glob.glob(self.distance_path + '/*.mp4')
        if 'Probability' in self.frame_types:
            self.video_in_each_dir['Probability'] = glob.glob(self.probability_path + '/*.mp4')
        if 'Heatmaps' in self.frame_types:
            self.video_in_each_dir['Heatmaps'] = glob.glob(self.heatmaps_path + '/*.mp4')

    def create_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = os.path.join(self.save_dir, self.video_name + '.mp4')
        self.writer = cv2.VideoWriter(save_path, fourcc, self.video_fps, (self.out_image.shape[1], self.out_image.shape[0]))

    def check_which_videos_exist_in_all_categories(self):
        self.videos = []
        videos_cnt = Counter()
        for frame_type in self.frame_types:
            entries = [os.path.basename(x) for x in self.video_in_each_dir[frame_type]]
            for entry in entries:
                videos_cnt[entry] += 1
        for video_name in videos_cnt:
            if videos_cnt[video_name] != self.plot_cnt:
                print('SIMBA WARNING: Not all user-specified video types selected have been created for video {}. Video {} is therefore '
                      'omitted from merged video creation.'.format(video_name, video_name))
            else:
                self.videos.append(video_name)
        if len(self.videos) == 0:
            print('SIMBA ERROR: None of your videos have pre-generated visualizations for ALL the user-specified video types.')
            raise ValueError('SIMBA ERROR: None of your videos have pre-generated visualizations for ALL the user-specified video types.')
        else:
            for k, v in self.video_in_each_dir.items():
                for entry in v:
                    if not os.path.basename(entry) in self.videos:
                        self.video_in_each_dir[k].remove(entry)

    def check_frm_lenth_of_each_video(self):
        self.frame_counts = {}
        videos = []
        for video_name in self.videos:
            self.frame_counts[video_name] = {}
            for video_type, video_path_lst in self.video_in_each_dir.items():
                video_path = [x for x in video_path_lst if os.path.basename(x) == video_name][0]
                self.frame_counts[video_name][video_type] = get_video_meta_data(video_path)['frame_count']
            results_dict = defaultdict(list)
            for video_name, video_frm_cnts in self.frame_counts.items():
                for entry, value in video_frm_cnts.items():
                    results_dict[value].append(entry)
            if len(results_dict.keys()) > 1:
                print('SIMBA WARNING: Not all user-specified video types has the same number of frames for video {}. Video {} is therefore '
                    'omitted from merged video creation:'.format(video_name, video_name))
                for k, values in results_dict.items():
                    print('{} frames found in {} plots for video {}'.format(str(k), values, video_name))
            else:
                videos.append(video_name)
        if len(videos) == 0:
            print('SIMBA ERROR: None of your videos have pre-generated visualizations with the same number of frames for all the user-specified video types.')
            raise ValueError('SIMBA ERROR: None of your videos have pre-generated visualizations with the same number of frames for all the user-specified video types.')
        else:
            for k, v in self.video_in_each_dir.items():
                for entry in v:
                    if not os.path.basename(entry) in self.videos:
                        self.video_in_each_dir[k].remove(entry)

    def find_viable_video_names(self):
        self.out_video_names = set()
        for k, v in self.video_in_each_dir.items():
            for entry in v:
                self.out_video_names.add(os.path.basename(entry))
        self.out_video_names = list(set(self.out_video_names))

    def create_base_mosaic(self):
        self.video_width = int(int(self.video_height / 2) * self.mosaic_window_columns)
        self.moisac = np.uint8(np.zeros((self.video_height, self.video_width, 3)))

    def create_videos(self):
        print('Creating {} merged video(s)...'.format(str(len(self.out_video_names))))
        for video_cnt, video_name in enumerate(self.out_video_names):
            _, self.video_name, ext = get_fn_ext(video_name)
            if self.frame_setting:
                self.frame_save_dir = os.path.join(self.project_path, 'frames', 'output', 'merged', self.video_name)
                if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            self.video_path_dict = deepcopy(self.video_in_each_dir)
            for category, video_path in self.video_in_each_dir.items():
                self.video_path_dict[category] = [x for x in video_path if os.path.basename(x) == video_name][0]
            self.non_clf_video_path_dict = {x: self.video_path_dict[x] for x in self.video_path_dict if x != 'Classifications'}
            self.video_frm_cnt = get_video_meta_data(self.non_clf_video_path_dict[list(self.non_clf_video_path_dict.keys())[0]])['frame_count']
            self.video_fps = get_video_meta_data(self.non_clf_video_path_dict[list(self.non_clf_video_path_dict.keys())[0]])['fps']
            self.caps = {}
            for video_type, video_path in self.non_clf_video_path_dict.items():
                self.caps[video_type] = cv2.VideoCapture(video_path)
            if 'Classifications' in self.frame_types:
                self.scikit_cap = cv2.VideoCapture(self.video_path_dict['Classifications'])
            for frm_cnt in range(self.video_frm_cnt-1):
                images = []
                for cap in self.caps.values():
                    cap.set(1, frm_cnt)
                    _, img = cap.read()
                    img = cv2.resize(img, self.mosaic_size, interpolation=cv2.INTER_CUBIC)
                    images.append(img)
                mosaic_img_cnt = 0
                self.create_base_mosaic()
                for i in range(self.mosaic_window_rows):
                    for j in range(self.mosaic_window_columns):
                        try:
                            img = images[mosaic_img_cnt]
                            top_left_x, bottom_right_x = int(self.mosaic_size[0] * i), int(self.mosaic_size[0] * i) + self.mosaic_size[0]
                            top_left_y, bottom_right_y = int(self.mosaic_size[1] * j), int(self.mosaic_size[1] * j) + self.mosaic_size[1]
                            self.moisac[top_left_x:bottom_right_x,top_left_y:bottom_right_y] = np.uint8(img)
                            mosaic_img_cnt += 1
                        except IndexError:
                            pass
                if 'Classifications' in self.frame_types:
                    self.scikit_cap.set(1, frm_cnt)
                    _, scikit_img = self.scikit_cap.read()
                    scikit_img = imutils.resize(scikit_img, height=self.video_height)
                    self.out_image = np.uint8(np.concatenate((scikit_img, self.moisac), axis=1))
                else:
                    self.out_image = np.uint8(self.moisac)
                if (frm_cnt == 0) and (self.video_setting):
                    self.create_writer()
                else:
                    pass
                if self.video_setting:
                    self.writer.write(self.out_image)
                if self.frame_setting:
                    img_save_path = os.path.join(self.frame_save_dir, str(frm_cnt) + '.png')
                    cv2.imwrite(img_save_path)
                print('Frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt+1), str(self.video_frm_cnt),
                                                                 self.video_name, str(video_cnt + 1),
                                                                 len(self.videos)))
            if self.video_setting:
                self.writer.release()

        print('SIMBA COMPLETE: All visualizations created in project_folder/frames/output/merged directory')



# test = FrameMergerer(config_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                      frame_types=['Classifications', 'Gantt', 'Path', 'Heatmaps', 'Probability', 'Data table'],
#                      frame_setting=False,
#                      video_setting=True,
#                      output_height=1280)







