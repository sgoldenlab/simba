__author__ = "Simon Nilsson"

import numpy as np
import multiprocessing
import functools
from numba import jit
import os
import platform
from typing import Dict, List

from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.printing import stdout_success, SimbaTimer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.read_write import read_df, get_fn_ext, concatenate_videos_in_folder

class DistancePlotterMultiCore(ConfigReader, PlottingMixin):
    """
     Visualize the distances between pose-estimated body-parts (e.g., two animals) through line
     charts. Results are saved as individual line charts, and/or a video of line charts.
     Uses multiprocessing.

     :param str config_path: path to SimBA project config file in Configparser format
     :param bool frame_setting: If True, creates individual frames.
     :param bool video_setting: If True, creates videos.
     :param bool final_img: If True, creates a single .png representing the entire video.
     :param dict style_attr: Video style attributes (font sizes, line opacity etc.)
     :param dict files_found: Files to visualize.
     :param dict line_attr: Representing the body-parts to visualize the distance between and their colors.

    .. note::
       `GitHub tutorial/documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    -----
    >>> style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5}
    >>> line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
    >>> _ = DistancePlotterMultiCore(config_path=r'/tests_/project_folder/project_config.ini', frame_setting=False, video_setting=True, final_img=True, style_attr=style_attr, line_attr=line_attr,  files_found=['/test_/project_folder/csv/machine_results/Together_1.csv'], core_cnt=5)

    """

    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool,
                 final_img: bool,
                 files_found: List[str],
                 style_attr: Dict[str, int],
                 line_attr: Dict[int, list],
                 core_cnt: int):

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)

        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        self.video_setting, self.frame_setting, self.files_found, self.style_attr, self.line_attr, self.final_img = video_setting, frame_setting, files_found, style_attr, line_attr, final_img
        if (not frame_setting) and (not video_setting) and (not self.final_img):
            raise NoSpecifiedOutputError(msg='Please choice to create frames and/or video distance plots')
        self.core_cnt = core_cnt
        if not os.path.exists(self.line_plot_dir): os.makedirs(self.line_plot_dir)
        check_if_filepath_list_is_empty(filepaths=self.outlier_corrected_dir,
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing distances.')
        print(f'Processing {str(len(self.files_found))} videos...')

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array,
                                  group: int):
        group_col = np.full((data.shape[0], 1), group)
        return np.hstack((group_col, data))


    def create_distance_plot(self):
        '''
        Creates line charts. Results are stored in the `project_folder/frames/line_plot` directory

        Returns
        ----------
        None
        '''

        for file_cnt, file_path in enumerate(self.files_found):

            video_timer = SimbaTimer(start=True)
            self.data_df = read_df(file_path, self.file_type)
            distance_arr = np.full((len(self.data_df), len(self.line_attr.keys())), np.nan)
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
            self.save_video_folder = os.path.join(self.line_plot_dir, self.video_name)
            self.temp_folder = os.path.join(self.line_plot_dir, self.video_name, 'temp')
            self.save_frame_folder_dir = os.path.join(self.line_plot_dir, self.video_name)
            for distance_cnt, data in enumerate(self.line_attr.values()):
                distance_arr[:, distance_cnt] = (np.sqrt((self.data_df[data[0] + '_x'] - self.data_df[data[1] + '_x']) ** 2 + (self.data_df[data[0] + '_y'] - self.data_df[data[1] + '_y']) ** 2) / self.px_per_mm) / 10
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    self.remove_a_folder(self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                self.video_folder = os.path.join(self.line_plot_dir, self.video_name)
                if os.path.exists(self.temp_folder):
                    self.remove_a_folder(self.temp_folder)
                    self.remove_a_folder(self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.line_plot_dir, self.video_name + '.mp4')

            distance_arr = np.nan_to_num(distance_arr, nan=0.0)

            if self.final_img:
                self.make_distance_plot(data=distance_arr,
                                        line_attr=self.line_attr,
                                        style_attr=self.style_attr,
                                        fps=self.fps,
                                        save_path=os.path.join(self.line_plot_dir, self.video_name + '_final_img.png'))

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

                print('Creating distance plots, multiprocessing (chunksize: {}, cores: {})...'.format(str(self.multiprocess_chunksize), str(self.core_cnt)))
                with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                    constants = functools.partial(self.distance_plotter_mp,
                                                  video_setting=self.video_setting,
                                                  video_name=self.video_name,
                                                  frame_setting=self.frame_setting,
                                                  video_save_dir=self.temp_folder,
                                                  frame_folder_dir=self.save_frame_folder_dir,
                                                  style_attr=self.style_attr,
                                                  line_attr=self.line_attr,
                                                  fps=self.fps)
                    for cnt, result in enumerate(pool.imap(constants, data, chunksize=self.multiprocess_chunksize)):
                        print('Image {}/{}, Video {}/{}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), str(file_cnt + 1), str(len(self.files_found))))

                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print('Joining {} multiprocessed video...'.format(self.video_name))
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print('Distance line chart video {} complete (elapsed time: {}s) ...'.format(self.video_name,video_timer.elapsed_time_str))

        self.timer.stop_timer()
        stdout_success(f'Distance plot visualizations for {str(len(self.files_found))} video(s) created in project_folder/frames/output/line_plot directory', elapsed_time=self.timer.elapsed_time_str)


# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'y_max': 'auto', 'opacity': 0.9}
# line_attr = {0: ['Center_1', 'Center_2', 'Green'], 1: ['Ear_left_2', 'Ear_left_1', 'Red']}
#
# test = DistancePlotterMultiCore(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                 frame_setting=False,
#                                 video_setting=True,
#                                 style_attr=style_attr,
#                                 final_img=True,
#                                 files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'],
#                                 line_attr=line_attr,
#                                 core_cnt=3)
# test.create_distance_plot()
# #
# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8}
# line_attr = {0: ['Termite_1_Head_1', 'Termite_1_Thorax_1', 'Dark-red']}
#






# style_attr = {'width': 640, 'height': 480, 'line width': 6, 'font size': 8, 'opacity': 0.5, 'y_max': 'auto'}
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