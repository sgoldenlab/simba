__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import read_config_entry, check_that_column_exist, read_config_file
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
import os, glob
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
import matplotlib.pyplot as plt
import cv2
import PIL
import io
import numpy as np
from copy import deepcopy

class TresholdPlotCreator(object):
    def __init__(self,
                 config_path: str,
                 clf_name: str,
                 frame_setting: bool,
                 video_setting: bool):

        '''
        Class for line chart visualizations displaying the classification probabilities of a single classifier.

        Parameters
        ----------
        config_path: str
            path to SimBA project config file in Configparser format
        clf_name: str
            Name of the classifier to create visualizations for
        frame_setting: bool
           When True, SimBA creates indidvidual frames in png format
        video_setting: bool
           When True, SimBA creates compressed video in mp4 format

        '''




        self.frame_setting = frame_setting
        self.video_setting = video_setting
        if (not self.frame_setting) and (not self.video_setting):
            raise ValueError('SIMBA ERROR: Please choose to select either videos, frames, or both frames and videos.')

        self.config = read_config_file(config_path)
        self.animal_cnt = read_config_entry(self.config, 'Path plot settings', 'no_animal_pathplot', 'int', 1)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'machine_results')
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.out_parent_dir = os.path.join(self.project_path, 'frames', 'output', 'probability_plots')
        self.clf_name = clf_name
        self.clf_name = 'Probability_' + self.clf_name
        self.fontsize = 8
        self.out_width, self.out_height = 640, 480
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))


    def create_plot(self):
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, fps = read_video_info(vidinfDf=self.vid_info_df, currVidName=self.video_name)
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clf_name, file_name=self.video_name)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.save_video_path = os.path.join(self.out_parent_dir, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.save_video_path, self.fourcc, fps, (self.out_width, self.out_height))
            data_df = data_df[self.clf_name]
            highest_p = float(data_df.max())
            p_list = []
            x_ticks = []
            x_labels = []
            for index, row in data_df.iteritems():
                p_list.append(row)
                plt.plot(p_list, color="m", linewidth=6)
                plt.plot(index, p_list[-1], "o", markersize=20, color="m")
                plt.ylim([0, highest_p])
                plt.ylabel(self.clf_name + ' probability', fontsize=10)
                x_ticks.append(index)
                x_labels.append(str(round((index / fps), 1)))
                if len(x_labels) > fps * 60:
                    x_labels_show, x_ticks_show = x_labels[0::250], x_ticks[0::250]
                elif len(x_labels) > ((fps * 60) * 10):
                    x_labels_show, x_ticks_show = x_labels[0::150], x_ticks[0::150]
                elif len(x_labels) < fps * 60:
                    x_labels_show, x_ticks_show = x_labels[0::75], x_ticks[0::75]
                else:
                    x_labels_show, x_ticks_show = deepcopy(x_labels), deepcopy(x_ticks)
                plt.xlabel('Time (s)', fontsize=self.fontsize)
                plt.grid()
                plt.xticks(x_ticks_show, x_labels_show, rotation='vertical', fontsize=8)
                plt.suptitle(self.clf_name, x=0.5, y=0.92, fontsize=12)

                buffer_ = io.BytesIO()
                plt.savefig(buffer_, format="png")
                buffer_.seek(0)
                image = PIL.Image.open(buffer_)
                ar = np.asarray(image)
                open_cv_image = ar[:, :, ::-1]
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                open_cv_image = cv2.resize(open_cv_image, (self.out_width, self.out_height))
                frame = np.uint8(open_cv_image)
                buffer_.close()
                plt.close()
                if self.frame_setting:
                    frame_save_path = os.path.join(self.save_frame_folder_dir, str(index) + '.png')
                    cv2.imwrite(frame_save_path, frame)
                if self.video_setting:
                    self.writer.write(frame)
                print('Probability frame: {} / {}. Video: {} ({}/{})'.format(str(index+1), str(len(data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))

            if self.video_setting:
                self.writer.release()
            print('Probability plot for video {} saved...'.format(self.video_name))

        print('SIMBA COMPLETE: All probability visualizations created in project_folder/frames/output/probability_plots directory')

# test = TresholdPlotCreator(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini',
#                            frame_setting=False, video_setting=True)
# test.create_plot()











