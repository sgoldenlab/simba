__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer,
                              get_color_dict,
                              remove_a_folder,
                              concatenate_videos_in_folder)

from simba.misc_visualizations import make_distance_plot
from simba.enums import Paths, ReadConfig, Formats, Defaults
from simba.rw_dfs import read_df
import numpy as np
import matplotlib.pyplot as plt
import PIL
import io
import multiprocessing
import functools
from numba import jit
import cv2
import os
import platform


def _image_creator(data: np.array,
                   video_setting: bool,
                   frame_setting: bool,
                   video_name: str,
                   video_save_dir: str,
                   frame_folder_dir: str,
                   style_attr: dict,
                   line_attr: dict,
                   fps: int):

    group = int(data[0][0])
    line_data = data[:, 2:]
    color_dict = get_color_dict()
    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (style_attr['width'], style_attr['height']))

    for i in range(line_data.shape[0]):
        frame_id = int(data[i][1])
        for j in range(line_data.shape[1]):
            color = (color_dict[line_attr[j][-1]][::-1])
            color = tuple(x / 255 for x in color)
            plt.plot(line_data[0:i, j], color=color, linewidth=style_attr['line width'], alpha=style_attr['opacity'])

        x_ticks_locs = x_lbls = np.round(np.linspace(0, i, 5))
        x_lbls = np.round((x_lbls / fps), 1)
        plt.ylim(0, style_attr['max_y'])
        plt.xlabel('time (s)')
        plt.ylabel('distance (cm)')
        plt.xticks(x_ticks_locs, x_lbls, rotation='horizontal', fontsize=style_attr['font size'])
        plt.yticks(style_attr['y_ticks_locs'], style_attr['y_ticks_lbls'], fontsize=style_attr['font size'])
        plt.suptitle('Animal distances', x=0.5, y=0.92, fontsize=style_attr['font size'] + 4)

        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        img = PIL.Image.open(buffer_)
        img = np.uint8(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        buffer_.close()
        plt.close()

        img = cv2.resize(img, (style_attr['width'], style_attr['height']))
        if video_setting:
            video_writer.write(np.uint8(img))
        if frame_setting:
            frm_name = os.path.join(frame_folder_dir, str(frame_id) + '.png')
            cv2.imwrite(frm_name, np.uint8(img))

        print('Distance frame created: {}, Video: {}, Processing core: {}'.format(str(frame_id + 1), video_name, str(group + 1)))

    return group

class DistancePlotterMultiCore(object):
    """
     Class for visualizing the distances between pose-estimated body-parts (e.g., two animals) through line
     charts. Results are saved as individual line charts, and/or a video of line charts.

     Parameters
     ----------
     config_path: str
         path to SimBA project config file in Configparser format
     frame_setting: bool
         If True, creates individual frames
     video_setting: bool
         If True, creates videos
     final_img: bool
        If True, creates a single .png representing the entire video.
     style_attr: dict
        Video style attributes (font sizes, line opacity etc.)
     files_found: list
        Files to visualize
     line_attr: dict[list]
        Representing the body-parts to visualize the distance between and their colors.

    Notes
    -----
    `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    -----
    >>> style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5}
    >>> line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
    >>> distance_plotter = DistancePlotterMultiCore(config_path=r'/tests_/project_folder/project_config.ini', frame_setting=False, video_setting=True, final_img=True, style_attr=style_attr, line_attr=line_attr,  files_found=['/test_/project_folder/csv/machine_results/Together_1.csv'], core_cnt=5)

    """

    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool,
                 final_img: bool,
                 files_found: list,
                 style_attr: dict,
                 line_attr: dict,
                 core_cnt: int):

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)


        self.video_setting, self.frame_setting, self.files_found, self.style_attr, self.line_attr, self.final_img = video_setting, frame_setting, files_found, style_attr, line_attr, final_img
        if (not frame_setting) and (not video_setting) and (not self.final_img):
            print('SIMBA ERROR: Please choice to create frames and/or video distance plots')
            raise ValueError('SIMBA ERROR: Please choice to create frames and/or video distance plots')
        self.config = read_config_file(config_path)
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        self.core_cnt = core_cnt
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
            self.save_video_folder = os.path.join(self.save_dir, self.video_name)
            self.temp_folder = os.path.join(self.save_dir, self.video_name, 'temp')
            self.save_frame_folder_dir = os.path.join(self.save_dir, self.video_name)
            for distance_cnt, data in enumerate(self.line_attr.values()):
                distance_arr[:, distance_cnt] = (np.sqrt((self.data_df[data[0] + '_x'] - self.data_df[data[1] + '_x']) ** 2 + (self.data_df[data[0] + '_y'] - self.data_df[data[1] + '_y']) ** 2) / self.px_per_mm) / 10
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir): remove_a_folder(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.save_dir, self.video_name)
                if os.path.exists(self.temp_folder):
                    remove_a_folder(self.temp_folder)
                    remove_a_folder(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.save_dir, self.video_name + '.mp4')

            distance_arr = np.nan_to_num(distance_arr, nan=0.0)

            if self.final_img:
                make_distance_plot(data=distance_arr,
                                   line_attr=self.line_attr,
                                   style_attr=self.style_attr,
                                   fps=self.fps,
                                   save_path=os.path.join(self.save_dir, self.video_name + '_final_img.png'))

            if self.video_setting or self.frame_setting:
                if self.style_attr['y_max'] == 'auto':
                    self.style_attr['max_y'] = np.amax(distance_arr)
                else:
                    self.style_attr['max_y'] = float(self.style_attr['max_y'])
                self.style_attr['y_ticks_locs'] = np.round(np.linspace(0, self.style_attr['max_y'], 10), 2)
                self.style_attr['y_ticks_lbls'] = np.round((self.style_attr['y_ticks_locs'] / self.fps), 1)
                index_column = list(range(0, distance_arr.shape[0]))
                distance_arr = np.column_stack((index_column, distance_arr))

                distance_arr = np.array_split(distance_arr, self.core_cnt)

                data = []
                for cnt, i in enumerate(distance_arr):
                    data.append(self.__insert_group_idx_column(data=i, group=cnt))
                frm_per_core = data[0].shape[0]

                print('Creating gantt, multiprocessing (chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.core_cnt)))
                with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(_image_creator,
                                                  video_setting=self.video_setting,
                                                  video_name=self.video_name,
                                                  frame_setting=self.frame_setting,
                                                  video_save_dir=self.temp_folder,
                                                  frame_folder_dir=self.save_frame_folder_dir,
                                                  style_attr=self.style_attr,
                                                  line_attr=self.line_attr,
                                                  fps=self.fps)
                    for cnt, result in enumerate(pool.imap(constants, data, chunksize=self.chunksize)):
                        print('Image {}/{}, Video {}/{}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), str(file_cnt + 1), str(len(self.files_found))))

                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print('Joining {} multiprocessed video...'.format(self.video_name))
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print('Distance line chart video {} complete (elapsed time: {}s) ...'.format(self.video_name,video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: Distance plot visualizations for {} videos created in project_folder/frames/output/line_plot directory (elapsed time: {}s)'.format(str(len(self.files_found)), self.timer.elapsed_time_str))

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array,
                                  group: int):
        group_col = np.full((data.shape[0], 1), group)
        return np.hstack((group_col, data))

# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5}
# line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
#
# test = DistancePlotterMultiCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                        frame_setting=False,
#                        video_setting=True,
#                        style_attr=style_attr,
#                                  final_img=False,
#                        files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                        line_attr=line_attr,
#                                 core_cnt=5)
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