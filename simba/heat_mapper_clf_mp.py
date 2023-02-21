__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer,
                              remove_a_folder,
                              concatenate_videos_in_folder)
from simba.misc_visualizations import make_heatmap_plot
from simba.rw_dfs import read_df
import numpy as np
import os
import cv2
from numba import jit, prange
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from simba.enums import (ReadConfig,
                         Paths,
                         Formats,
                         Defaults)
import multiprocessing
import functools
import platform

def _heatmap_multiprocessor(data: np.array,
                            video_setting: bool,
                            frame_setting: bool,
                            video_temp_dir: str,
                            video_name: str,
                            frame_dir: str,
                            fps: int,
                            style_attr: dict,
                            max_scale: float,
                            clf_name:str,
                            aspect_ratio: float,
                            size: tuple):

    group = int(data[0][0][1])

    if video_setting:
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_save_path = os.path.join(video_temp_dir, '{}.mp4'.format(str(group)))
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, size)

    for i in range(data.shape[0]):
        frame_id = int(data[i, 0, 0])
        frm_data = data[i, :, 2:]
        cum_df = pd.DataFrame(frm_data).reset_index()
        cum_df = cum_df.melt(id_vars='index', value_vars=None, var_name=None, value_name='seconds',
                             col_level=None).rename(columns={'index': 'vertical_idx', 'variable': 'horizontal_idx'})
        cum_df['color'] = (cum_df['seconds'].astype(float) / float(max_scale)).round(2).clip(upper=100)
        color_array = np.zeros((len(cum_df['vertical_idx'].unique()), len(cum_df['horizontal_idx'].unique())))
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                value = cum_df["color"][(cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)].values[0]
                color_array[i, j] = value
        fig = plt.figure()
        im_ratio = color_array.shape[0] / color_array.shape[1]
        plt.pcolormesh(color_array, shading=style_attr['shading'], cmap=style_attr['palette'], rasterized=True, alpha=1, vmin=0.0, vmax=float(max_scale))
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tick_params(axis='both', which='both', length=0)
        cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
        cb.ax.tick_params(size=0)
        cb.outline.set_visible(False)
        cb.set_label('{} (seconds)'.format(clf_name), rotation=270, labelpad=10)
        plt.tight_layout()
        plt.gca().set_aspect(aspect_ratio)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        img = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (size[1], size[0]))
        img = np.uint8(img)
        plt.close()
        print('Heatmap frame created: {}, Video: {}, Processing core: {}'.format(str(frame_id+1), video_name, str(group+1)))

        if video_setting:
            video_writer.write(img)

        if frame_setting:
            file_path = os.path.join(frame_dir, '{}.png'.format(frame_id))
            cv2.imwrite(file_path, img)

    if video_setting:
        video_writer.release()

    return group

class HeatMapperClfMultiprocess(object):

    """
    Class for creating heatmaps representing the locations of the classified behavior

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    final_img_setting: bool
        If True, then  create the a single image representing the last frame of the input video
    video_setting: bool
        If True, then create a video of heatmaps.
    frame_setting: bool
        If True, then create individual heatmap frames
    bin_size: int
        The rectangular size of each heatmap location in millimeters. For example, `50` will divide the video into
        5 centimeter rectangular spatial bins.
    palette: str
        Heatmap pallette. Eg. 'jet', 'magma', 'inferno','plasma', 'viridis', 'gnuplot2'
    bodypart: str
        The name of the body-part used to infer the location of the classified behavior
    clf_name: str
        The name of the classified behavior
    max_scale: int or 'auto'
        The max value in the heatmap in seconds. E.g., with a value of `10`, if the classified behavior has occured
        >= 10 within a rectangular bins, it will be filled with the same color.

    Notes
    -----
    `GitHub visualizations tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    -----

    >>> heat_mapper_clf = HeatMapperClf(config_path='MyConfigPath', final_img_setting=False, video_setting=True, frame_setting=False, bin_size=50, palette='jet', bodypart='Nose_1', clf_name='Attack', max_scale=20)
    >>> heat_mapper_clf.create_heatmaps()
    """

    def __init__(self,
                 config_path: str,
                 final_img_setting: bool,
                 video_setting: bool,
                 frame_setting: bool,
                 bodypart: str,
                 clf_name: str,
                 files_found: list,
                 style_attr: dict,
                 core_cnt: int
                 ):


        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise ValueError('SIMBA ERROR: Please choose to select either heatmap videos, frames, and/or final image.')
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.style_attr = style_attr
        self.bin_size, self.max_scale, self.palette, self.shading, self.core_cnt = style_attr['bin_size'], style_attr['max_scale'], style_attr['palette'], style_attr['shading'], core_cnt
        self.clf_name, self.files_found = clf_name, files_found
        self.config = read_config_file(config_path)
        self.project_path, self.file_type =read_project_path_and_file_type(config=self.config)
        self.out_parent_dir = os.path.join(self.project_path, Paths.HEATMAP_CLF_LOCATION_DIR.value)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        self.dir_in = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.bp_lst = [self.bp + '_x', self.bp + '_y']
        self.maxtasksperchild = Defaults.MAX_TASK_PER_CHILD.value
        self.chunksize = Defaults.CHUNK_SIZE.value
        self.timer = SimbaTimer()
        self.timer.start_timer()
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    @staticmethod
    @jit(nopython=True)
    def __calculate_cum_array(clf_array: np.array,
                              fps: int):
        cum_sum_arr = np.full(clf_array.shape, np.nan)
        for frm_idx in prange(clf_array.shape[0]):
            frame_cum_sum = np.full((clf_array.shape[1], clf_array.shape[2]), 0.0)
            sliced_arr = clf_array[0:frm_idx]
            for i in range(sliced_arr.shape[0]):
                for j in range(sliced_arr.shape[1]):
                    for k in range(sliced_arr.shape[2]):
                        frame_cum_sum[j][k] += sliced_arr[i][j][k]
            cum_sum_arr[frm_idx] = frame_cum_sum


        return cum_sum_arr / fps

    @staticmethod
    @jit(nopython=True)
    def __insert_group_idx_column(data: np.array,
                                  group: int,
                                  last_frm_idx: int):


        results = np.full((data.shape[0], data.shape[1], data.shape[2]+2), np.nan)
        group_col = np.full((data.shape[1], 1), group)
        for frm_idx in prange(data.shape[0]):
            h_stack = np.hstack((group_col, data[frm_idx]))
            frm_col = np.full((h_stack.shape[0], 1), frm_idx+last_frm_idx)
            results[frm_idx] = np.hstack((frm_col, h_stack))

        return results


    def __calculate_bin_attr(self,
                             data_df: pd.DataFrame,
                             clf_name: str,
                             bp_lst: list,
                             px_per_mm: int,
                             img_width: int,
                             img_height: int,
                             bin_size: int,
                             fps: int):

        bin_size_px = int(float(px_per_mm) * float(bin_size))
        horizontal_bin_cnt = int(img_width / bin_size_px)
        vertical_bin_cnt = int(img_height / bin_size_px)
        aspect_ratio = round((vertical_bin_cnt / horizontal_bin_cnt), 3)

        clf_idx = data_df[bp_lst][data_df[clf_name] == 1].reset_index().to_numpy().astype(int)

        bin_dict = {}
        x_location, y_location = 0, 0
        for hbin in range(horizontal_bin_cnt):
            bin_dict[hbin] = {}
            for vbin in range(vertical_bin_cnt):
                bin_dict[hbin][vbin] = {'top_left_x': x_location,
                                        'top_left_y': y_location,
                                        'bottom_right_x': x_location + bin_size_px,
                                        'bottom_right_y': y_location + bin_size_px}
                y_location += bin_size_px
            y_location = 0
            x_location += bin_size_px

        clf_array = np.zeros((len(data_df), vertical_bin_cnt, horizontal_bin_cnt))

        for clf_frame in clf_idx:
            for h_bin_name, v_dict in bin_dict.items():
                for v_bin_name, c in v_dict.items():
                    if (clf_frame[1] <= c['bottom_right_x'] and clf_frame[1] >= c['top_left_x']):
                        if (clf_frame[2] <= c['bottom_right_y'] and clf_frame[2] >= c['top_left_y']):
                            clf_array[int(clf_frame[0])][v_bin_name][h_bin_name] = 1

        clf_array = self.__calculate_cum_array(clf_array=clf_array, fps=fps)

        return clf_array, aspect_ratio

    def __calculate_max_scale(self,
                              clf_array: np.array):
        return np.round(np.max(np.max(clf_array[-1], axis=0)), 3)

    def create_heatmaps(self):
        '''
        Creates heatmap charts. Results are stored in the `project_folder/frames/heatmaps_classifier_locations`
        directory of SimBA project.

        Returns
        ----------
        None
        '''

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=self.video_name)
            self.width, self.height = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])

            self.save_frame_folder_dir = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name)
            self.video_folder = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name)
            self.temp_folder = os.path.join(self.out_parent_dir, self.video_name + '_' + self.clf_name, 'temp')
            if self.frame_setting:
                if os.path.exists(self.save_frame_folder_dir):
                    remove_a_folder(folder_dir=self.save_frame_folder_dir)
                if not os.path.exists(self.save_frame_folder_dir): os.makedirs(self.save_frame_folder_dir)
            if self.video_setting:
                if os.path.exists(self.temp_folder):
                    remove_a_folder(folder_dir=self.temp_folder)
                    remove_a_folder(folder_dir=self.video_folder)
                os.makedirs(self.temp_folder)
                self.save_video_path = os.path.join(self.out_parent_dir, '{}_{}.mp4'.format(self.video_name, self.clf_name))

            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            clf_array, aspect_ratio = self.__calculate_bin_attr(data_df=self.data_df,
                                                                clf_name=self.clf_name,
                                                                bp_lst=self.bp_lst,
                                                                px_per_mm=self.px_per_mm,
                                                                img_width=self.width,
                                                                img_height=self.height,
                                                                bin_size=self.bin_size,
                                                                fps=self.fps)

            if self.max_scale == 'auto':
                self.max_scale = self.__calculate_max_scale(clf_array=clf_array)

            if self.final_img_setting:
                make_heatmap_plot(data_arr=clf_array,
                                  max_scale=self.max_scale,
                                  palette=self.palette,
                                  aspect_ratio=aspect_ratio,
                                  file_name=os.path.join(self.out_parent_dir, self.video_name + '_final_frm.png'),
                                  shading=self.shading,
                                  clf_name=self.clf_name,
                                  img_size=(self.width, self.height))

            if self.video_setting or self.frame_setting:
                frame_arrays = np.array_split(clf_array, self.core_cnt)
                last_frm_idx = 0
                for frm_group in range(len(frame_arrays)):
                    split_arr = frame_arrays[frm_group]
                    frame_arrays[frm_group] = self.__insert_group_idx_column(data=split_arr, group=frm_group, last_frm_idx=last_frm_idx)
                    last_frm_idx = np.max(frame_arrays[frm_group].reshape((frame_arrays[frm_group].shape[0], -1)))
                frm_per_core = frame_arrays[0].shape[0]

                print('Creating heatmaps, multiprocessing (chunksize: {}, cores: {})...'.format(str(self.chunksize), str(self.core_cnt)))
                with multiprocessing.Pool(self.core_cnt, maxtasksperchild=2) as pool:
                    constants = functools.partial(_heatmap_multiprocessor,
                                                  video_setting=self.video_setting,
                                                  frame_setting=self.frame_setting,
                                                  style_attr=self.style_attr,
                                                  fps=self.fps,
                                                  video_temp_dir=self.temp_folder,
                                                  frame_dir=self.save_frame_folder_dir,
                                                  max_scale=self.max_scale,
                                                  aspect_ratio=aspect_ratio,
                                                  clf_name=self.clf_name,
                                                  size=(self.width, self.height),
                                                  video_name=self.video_name)

                    for cnt, result in enumerate(pool.imap(constants, frame_arrays, chunksize=self.chunksize)):
                        print('Image {}/{}, Video {}/{}...'.format(str(int(frm_per_core * (result+1))), str(len(self.data_df)), str(file_cnt + 1), str(len(self.files_found))))
                    pool.terminate()
                    pool.join()

                if self.video_setting:
                    print('Joining {} multiprocessed video...'.format(self.video_name))
                    concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.save_video_path)

                video_timer.stop_timer()
                print('Heatmap video {} complete (elapsed time: {}s) ...'.format(self.video_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: heatmap visualizations for {} videos created in project_folder/frames/output/heatmap_classifier locations directory (elapsed time: {}s)'.format(str(len(self.files_found)), self.timer.elapsed_time_str))



# test = HeatMapperClfMultiprocess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'},
#                      final_img_setting=False,
#                      video_setting=True,
#                      frame_setting=False,
#                      bodypart='Nose_1',
#                      clf_name='Attack',
#                                core_cnt=5,
#                      files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.create_heatmaps()

