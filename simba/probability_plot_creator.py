__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          check_that_column_exist,
                                          read_config_file,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.misc_tools import SimbaTimer
from simba.misc_visualizations import make_probability_plot
import os
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df
from simba.enums import Paths, Formats, ReadConfig
import matplotlib.pyplot as plt
import cv2
import PIL
import io
import numpy as np

class TresholdPlotCreatorSingleProcess(object):
    '''
    Class for creating line chart visualizations displaying the classification probabilities of a single classifier.

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
    files_found: list
        File paths to create probability plots for, e.g., ['project_folder/csv/machine_results/MyVideo.csv]
    style_attr: dict
        Output image style attributes, e.g., {'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'magneta', 'circle size': 20}


    Examples
    -----
    >>> style_attr = {'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20}
    >>> clf_name='Attack'
    >>> files_found=['/_test/project_folder/csv/machine_results/Together_1.csv']

    >>> threshold_plot_creator = TresholdPlotCreatorSingleProcess(config_path='/_test/project_folder/project_config.ini', frame_setting=False, video_setting=True, last_frame=True, clf_name=clf_name, files_found=files_found, style_attr=style_attr)
    >>> threshold_plot_creator.create_plots()
    '''

    def __init__(self,
                 config_path: str,
                 clf_name: str,
                 frame_setting: bool,
                 video_setting: bool,
                 last_image: bool,
                 style_attr: dict,
                 files_found: list):

        self.frame_setting, self.video_setting, self.style_attr, self.last_image = frame_setting, video_setting, style_attr, last_image
        if (not self.frame_setting) and (not self.video_setting) and (not self.last_image):
            print('SIMBA ERROR: Please choose to either probability videos, probability frames, or both probability frames and videos.')
            raise ValueError('SIMBA ERROR: Please choose to either probability videos, probability frames, or both probability frames and videos.')

        self.config = read_config_file(config_path)
        self.animal_cnt = read_config_entry(self.config, ReadConfig.PATH_PLOT_SETTINGS.value, 'no_animal_pathplot', 'int', 1)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found = files_found
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.out_parent_dir = os.path.join(self.project_path, Paths.PROBABILITY_PLOTS_DIR.value)
        self.orginal_clf_name = clf_name
        self.clf_name = 'Probability_' + self.orginal_clf_name
        self.out_width, self.out_height = self.style_attr['width'], self.style_attr['height']
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        if not os.path.exists(self.out_parent_dir):
            os.makedirs(self.out_parent_dir)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))
        self.timer = SimbaTimer()
        self.timer.start_timer()


    def create_plots(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, self.video_name, _ = get_fn_ext(file_path)
            video_info, self.px_per_mm, fps = read_video_info(vid_info_df=self.vid_info_df, video_name=self.video_name)
            data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=data_df, column_name=self.clf_name, file_name=self.video_name)
            if self.frame_setting:
                self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name + '_' + self.orginal_clf_name)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.save_video_path = os.path.join(self.out_parent_dir, '{}_{}.mp4'.format(self.video_name, self.orginal_clf_name))
                self.writer = cv2.VideoWriter(self.save_video_path, self.fourcc, fps, (self.out_width, self.out_height))

            data_df = data_df[self.clf_name]


            if self.last_image:
                make_probability_plot(data=data_df,
                                      style_attr=self.style_attr,
                                      clf_name=self.clf_name,
                                      fps=fps,
                                      save_path=os.path.join(self.out_parent_dir, self.video_name + '_{}_{}.png'.format(self.orginal_clf_name, 'final_image')))

            if self.video_setting or self.frame_setting:
                if self.style_attr['y_max'] == 'auto':
                    max_y = np.amax(data_df)
                else:
                    max_y = float(self.style_attr['y_max'])

                y_ticks_locs = y_lbls = np.round(np.linspace(0, max_y, 10), 2)
                for i in range(len(data_df)):
                    p_values = list(data_df.loc[0:i])
                    plt.plot(p_values, color=self.style_attr['color'], linewidth=self.style_attr['line width'])
                    plt.plot(i, p_values[-1], "o", markersize=self.style_attr['circle size'], color=self.style_attr['color'])
                    plt.ylim([0, max_y])
                    plt.ylabel('{} {}'.format(self.orginal_clf_name, 'probability'), fontsize=self.style_attr['font size'])
                    x_ticks_locs = x_lbls = np.linspace(0, len(p_values), 5)
                    x_lbls = np.round((x_lbls / fps), 1)
                    plt.xlabel('Time (s)', fontsize=self.style_attr['font size'] + 4)
                    plt.grid()
                    plt.xticks(x_ticks_locs, x_lbls, rotation='horizontal', fontsize=self.style_attr['font size'])
                    plt.yticks(y_ticks_locs, y_lbls, fontsize=self.style_attr['font size'])
                    plt.suptitle(self.orginal_clf_name, x=0.5, y=0.92, fontsize=self.style_attr['font size'] + 4)
                    buffer_ = io.BytesIO()
                    plt.savefig(buffer_, format="png")
                    buffer_.seek(0)
                    image = PIL.Image.open(buffer_)
                    ar = np.asarray(image)
                    open_cv_image = cv2.cvtColor(ar, cv2.COLOR_RGB2BGR)
                    open_cv_image = cv2.resize(open_cv_image, (self.out_width, self.out_height))
                    frame = np.uint8(open_cv_image)
                    buffer_.close()
                    plt.close()
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_frame_folder_dir, str(i) + '.png')
                        cv2.imwrite(frame_save_path, frame)
                    if self.video_setting:
                        self.writer.write(frame)
                    print('Probability frame: {} / {}. Video: {} ({}/{})'.format(str(i+1), str(len(data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))
                if self.video_setting:
                    self.writer.release()
                video_timer.stop_timer()
                print('Probability plot for video {} saved (elapsed time: {}s)...'.format(self.video_name, video_timer.elapsed_time_str))
        self.timer.stop_timer()
        print('SIMBA COMPLETE: All probability visualizations created in project_folder/frames/output/probability_plots directory (elapsed time: {}s)'.format(self.timer.elapsed_time_str))


#
# test = TresholdPlotCreatorSingleProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                         frame_setting=False,
#                                         video_setting=True,
#                                         last_image=True,
#                                         clf_name='Attack',
#                                         files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                         style_attr={'width': 640, 'height': 480, 'font size': 10, 'line width': 6, 'color': 'blue', 'circle size': 20, 'y_max': 'auto'})
# # test = TresholdPlotCreatorSingleProcess(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', frame_setting=False, video_setting=True, clf_name='Attack')
# #test.create_plots()
#
#
#








