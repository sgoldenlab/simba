__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer,
                              get_color_dict)

from simba.misc_visualizations import make_distance_plot

from simba.enums import ReadConfig, Paths, Formats
from simba.rw_dfs import read_df
import numpy as np
import matplotlib.pyplot as plt
import PIL
import io
import cv2
import os

class DistancePlotterSingleCore(object):
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
                 video_setting: bool,
                 final_img: bool,
                 files_found: list,
                 style_attr: dict,
                 line_attr: dict):

        self.video_setting, self.frame_setting, self.files_found, self.style_attr, self.line_attr, self.final_img = video_setting, frame_setting, files_found, style_attr, line_attr, final_img
        if (not frame_setting) and (not video_setting) and (not self.final_img):
            print('SIMBA ERROR: Please choice to create frames and/or video distance plots')
            raise ValueError('SIMBA ERROR: Please choice to create frames and/or video distance plots')
        self.colors_dict = get_color_dict()
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.save_dir = os.path.join(self.project_path, Paths.LINE_PLOT_DIR.value)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.dir_in = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing distances.')
        print('Processing {} videos...'.format(str(len(self.files_found))))
        self.timer = SimbaTimer()
        self.timer.start_timer()

    def create_distance_plot(self):
        '''
        Creates line charts. Results are stored in the `project_folder/frames/line_plot` directory

        Returns
        ----------
        None
        '''

        for file_cnt, file_path in enumerate(self.files_found):

            video_timer = SimbaTimer()
            video_timer.start_timer()

            self.data_df = read_df(file_path, self.file_type)
            distance_arr = np.full((len(self.data_df), len(self.line_attr.keys())), np.nan)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = read_video_info(self.vid_info_df, self.video_name)
            for distance_cnt, data in enumerate(self.line_attr.values()):
                distance_arr[:, distance_cnt] = (np.sqrt((self.data_df[data[0] + '_x'] - self.data_df[data[1] + '_x']) ** 2 + (self.data_df[data[0] + '_y'] - self.data_df[data[1] + '_y']) ** 2) / self.px_per_mm) / 10
            if self.video_setting:
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.video_save_path = os.path.join(self.save_dir, self.video_name + '.mp4')
                writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.style_attr['width'], self.style_attr['height']))
            if self.frame_setting:
                self.save_video_folder = os.path.join(self.save_dir, self.video_name)
                if not os.path.exists(self.save_video_folder): os.makedirs(self.save_video_folder)

            distance_arr = np.nan_to_num(distance_arr, nan=0.0)
            if self.final_img:
                self.final_img_path = os.path.join(self.save_dir, self.video_name + '_final_img.png')
                make_distance_plot(data=distance_arr,
                                   line_attr=self.line_attr,
                                   style_attr=self.style_attr,
                                   fps=self.fps,
                                   save_path=self.final_img_path)

            if self.video_setting or self.frame_setting:
                if self.style_attr['y_max'] == 'auto':
                    max_y = np.amax(distance_arr)
                else:
                    max_y = float(self.style_attr['y_max'])
                y_ticks_locs = y_lbls = np.round(np.linspace(0, max_y, 10), 2)
                for i in range(distance_arr.shape[0]):
                    for j in range(distance_arr.shape[1]):
                        color = (self.colors_dict[self.line_attr[j][-1]][::-1])
                        color = tuple(x / 255 for x in color)
                        plt.plot(distance_arr[0:i, j], color=color, linewidth=self.style_attr['line width'], alpha=self.style_attr['opacity'])

                    x_ticks_locs = x_lbls = np.round(np.linspace(0, i, 5))
                    x_lbls = np.round((x_lbls / self.fps), 1)
                    plt.ylim(0, max_y)

                    plt.xlabel('time (s)')
                    plt.ylabel('distance (cm)')
                    plt.xticks(x_ticks_locs, x_lbls, rotation='horizontal', fontsize=self.style_attr['font size'])
                    plt.yticks(y_ticks_locs, y_lbls, fontsize=self.style_attr['font size'])
                    plt.suptitle('Animal distances', x=0.5, y=0.92, fontsize=self.style_attr['font size']+4)

                    self.buffer_ = io.BytesIO()
                    plt.savefig(self.buffer_, format="png")
                    self.buffer_.seek(0)
                    img = PIL.Image.open(self.buffer_)
                    img = np.uint8(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
                    self.buffer_.close()
                    plt.close()

                    img = cv2.resize(img, (self.style_attr['width'], self.style_attr['height']))

                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_video_folder, str(i) + '.png')
                        cv2.imwrite(frame_save_path, img)
                    if self.video_setting:
                        writer.write(img)
                    print('Distance frame: {} / {}. Video: {} ({}/{})'.format(str(i+1), str(len(self.data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))

                if self.video_setting:
                    writer.release()
                video_timer.stop_timer()
                print('Distance plot for video {} saved (elapsed time: {}s)...'.format(self.video_name, video_timer.elapsed_time_str))
        self.timer.stop_timer()
        print('SIMBA COMPLETE: All distance visualizations created in project_folder/frames/output/line_plot directory (elapsed time: {}s)'.format(self.timer.elapsed_time_str))


# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'y_max': 'auto', 'opacity': 0.9}
# line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
#
# test = DistancePlotterSingleCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                        frame_setting=False,
#                        video_setting=True,
#                        style_attr=style_attr,
#                                  final_img=True,
#                        files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                        line_attr=line_attr)
# test.create_distance_plot()

# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8}
# line_attr = {0: ['Termite_1_Head_1', 'Termite_1_Thorax_1', 'Dark-red']}

# test = DistancePlotterSingleCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                                  frame_setting=False,
#                        video_setting=True,
#                        style_attr=style_attr,
#                        files_found=['/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/csv/outlier_corrected_movement_location/termites_1.csv'],
#                        line_attr=line_attr)
# test.create_distance_plot()