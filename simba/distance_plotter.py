__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.misc_tools import get_fn_ext
from simba.rw_dfs import read_df
import numpy as np
import matplotlib.pyplot as plt
import PIL
import io
import cv2
import os, glob

class DistancePlotter(object):
    """
     Class for visualizing the distance between two pose-estimated body-parts (e.g., two animals) through line
     charts. Results are saved as individual line charts, and/or a video of line charts.

     Parameters
     ----------
     config_path: str
         path to SimBA project config file in Configparser format
     frame_setting: bool
         If True, creates individual frames
     video_setting: bool
         If True, creates videos

    Notes
    -----
    `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    -----
    >>> distance_plotter = DistancePlotter(config_path=r'MyProjectConfig', frame_setting=False, video_setting=True)
    >>> distance_plotter.create_distance_plot()
    """

    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool):

        if (not frame_setting) and (not video_setting):
            print('SIMBA ERROR: Please choice to create frames and/or video distance plots')
            raise ValueError
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.save_dir = os.path.join(self.project_path, 'frames', 'output', 'line_plot')
        self.video_setting, self.frame_setting = video_setting, frame_setting
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.dir_in = os.path.join(self.project_path, 'csv', 'machine_results')
        self.poi_1 = read_config_entry(self.config, 'Distance plot', 'POI_1', 'str')
        self.poi_2 = read_config_entry(self.config, 'Distance plot', 'POI_2', 'str')
        self.files_found = glob.glob(self.dir_in + "/*." + self.file_type)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print('Processing {} videos...'.format(str(len(self.files_found))))

    def create_distance_plot(self):
        '''
        Creates line charts. Results are stored in the `project_folder/frames/line_plot` directory

        Returns
        ----------
        None
        '''

        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.width, self.height = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])
            self.data_df = read_df(file_path, self.file_type)
            self.data_df["distance"] = np.sqrt((self.data_df[self.poi_1 + '_x'] - self.data_df[self.poi_2 + '_x']) ** 2 + (self.data_df[self.poi_1 + '_y'] - self.data_df[self.poi_2 + '_y']) ** 2) / self.px_per_mm
            self.max_y = int(((self.data_df["distance"].max() / 10) + 10))
            self.y_ticks = list(range(0, self.max_y + 1))
            self.x_ticks, self.x_labels = [], []
            self.y_lbls = self.y_ticks
            self.y_ticks = self.y_ticks[0::10]
            self.y_lbls = self.y_lbls[0::10]
            distance_lst = []
            if self.video_setting:
                self.video_save_path = os.path.join(self.save_dir, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.width, self.height))
            if self.frame_setting:
                self.save_video_folder = os.path.join(self.save_dir, self.video_name)
                if not os.path.exists(self.save_video_folder): os.makedirs(self.save_video_folder)
            for frame_cnt in range(len(self.data_df)):
                distance_lst.append(round(self.data_df.loc[frame_cnt, "distance"], 4))
                plt.plot(distance_lst, color="m", linewidth=6, alpha=0.5)
                self.x_ticks.append(frame_cnt)
                self.x_labels.append(str(round((frame_cnt / self.fps), 2)))

                if len(self.x_labels) > self.fps * 60:
                    self.x_labels_plot, self.x_ticks_plot = self.x_labels[0::250], self.x_ticks[0::250]
                if len(self.x_labels) > ((self.fps * 60) * 10):
                    self.x_labels_plot, self.x_ticks_plot = self.x_labels[0::150], self.x_ticks[0::150]
                if len(self.x_labels) < self.fps * 60:
                    self.x_labels_plot, self.x_ticks_plot = self.x_labels[0::75], self.x_ticks[0::75]

                plt.xlabel('time (s)')
                plt.ylabel('distance (cm)')
                plt.xticks(self.x_ticks_plot, self.x_labels_plot, rotation='vertical', fontsize=8)
                plt.yticks(self.y_ticks, self.y_lbls, fontsize=8)
                plt.suptitle(str(self.poi_1) + ' vs. ' + str(self.poi_2), x=0.5, y=0.92, fontsize=12)

                buffer_ = io.BytesIO()
                plt.savefig(buffer_, format="png")
                buffer_.seek(0)
                image = PIL.Image.open(buffer_)
                ar = np.asarray(image)
                open_cv_image = ar[:, :, ::-1]
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                open_cv_image = cv2.resize(open_cv_image, (self.width, self.height))
                frame = np.uint8(open_cv_image)
                buffer_.close()
                plt.close()

                if self.frame_setting:
                    frame_save_path = os.path.join(self.save_video_folder, str(frame_cnt) + '.png')
                    cv2.imwrite(frame_save_path, frame)
                if self.video_setting:
                    self.writer.write(frame)
                print('Distance frame: {} / {}. Video: {} ({}/{})'.format(str(frame_cnt+1), str(len(self.data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))

            if self.video_setting:
                self.writer.release()
            print('Distance plot for video {} saved...'.format(self.video_name))

        print('SIMBA COMPLETE: All distance visualizations created in project_folder/frames/output/line_plot directory')

# test = DistancePlotter(config_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                        frame_setting=False,
#                        video_setting=True)
# test.create_distance_plot()