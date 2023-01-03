__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from simba.read_config_unit_tests import (read_config_entry, read_config_file)
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import get_fn_ext
from simba.rw_dfs import read_df
import numpy as np
import os, glob
import cv2
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time

class HeatMapperClf(object):

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
                 bin_size: int,
                 palette: str,
                 bodypart: str,
                 clf_name: str,
                 max_scale: int or str):

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise ValueError('SIMBA ERROR: Please choose to select either videos, frames, and/or final image.')
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.bin_size, self.max_scale = bin_size, max_scale
        self.clf_name, self.palette = clf_name, palette
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.save_dir = os.path.join(self.project_path, 'frames', 'output', 'heatmaps_classifier_locations')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.dir_in = os.path.join(self.project_path, 'csv', 'machine_results')
        self.files_found = glob.glob(self.dir_in + "/*." + self.file_type)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.bp_lst = [self.bp + '_x', self.bp + '_y']
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    @staticmethod
    @jit(nopython=True, parallel=True)
    def __calculate_cum_array(cum_array: np.array, current_frm: np.array):
        for row in range(current_frm.shape[0]):
            for col in range(current_frm.shape[1]):
                cum_array[row, col] += current_frm[row, col]
        return cum_array

    @staticmethod
    @jit(nopython=True)
    def __create_color_array(data_arr: np.array):
        print(np.unique(data_arr[:, 0]))
        color_array = np.full((np.unique(data_arr[:, 0]).shape[0], np.unique(data_arr[:, 1]).shape[0]), -1)
        print(color_array.shape)
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                #print(i, j)
                value = np.where((data_arr[:,0]==j) & (data_arr[:,1]==i))
                #print(value)
                #break

    def create_heatmaps(self):
        '''
        Creates heatmap charts. Results are stored in the `project_folder/frames/heatmaps_classifier_locations`
        directory of SimBA project.

        Returns
        ----------
        None
        '''

        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=self.video_name)
            self.width, self.height = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])
            if self.video_setting:
                self.video_save_path = os.path.join(self.save_dir, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.width, self.height))
            if self.frame_setting:
                self.save_video_folder = os.path.join(self.save_dir, self.video_name)
                if not os.path.exists(self.save_video_folder): os.makedirs(self.save_video_folder)
            self.pixels_per_bin = int(float(self.px_per_mm) * float(self.bin_size))
            self.hbins_cnt, self.vbins_cnt = int(self.width / self.pixels_per_bin), int(self.height / self.pixels_per_bin)
            hight_to_width = round((self.vbins_cnt / self.hbins_cnt), 3)
            self.bin_dict = {}
            x_location, y_location = 0, 0
            self.data_df = read_df(file_path, self.file_type)
            self.clf_idx = self.data_df[self.bp_lst][self.data_df[self.clf_name] == 1].reset_index().to_numpy()
            for hbin in range(self.hbins_cnt):
                self.bin_dict[hbin] = {}
                for vbin in range(self.vbins_cnt):
                    self.bin_dict[hbin][vbin] = {'top_left_x': x_location,
                                                 'top_left_y': y_location,
                                                 'bottom_right_x': x_location + self.pixels_per_bin,
                                                 'bottom_right_y': y_location + self.pixels_per_bin}
                    y_location += self.pixels_per_bin
                y_location = 0
                x_location += self.pixels_per_bin
            self.clf_array = np.zeros((len(self.data_df), self.vbins_cnt, self.hbins_cnt))
            for clf_frame in self.clf_idx:
                for h_bin_name, v_dict in self.bin_dict.items():
                    for v_bin_name, c in v_dict.items():
                        if (clf_frame[1] <= c['bottom_right_x'] and clf_frame[1] >= c['top_left_x']):
                            if (clf_frame[2] <= c['bottom_right_y'] and clf_frame[2] >= c['top_left_y']):
                                self.clf_array[int(clf_frame[0])][v_bin_name][h_bin_name] = 1
            cum_array = np.zeros((self.clf_array.shape[1], self.clf_array.shape[2]))
            if self.max_scale == 'auto':
                self.max_scale = int(np.max(np.sum(self.clf_array,axis=0)) / self.fps)
                if self.max_scale == 0: self.max_scale = 0

            if (self.frame_setting) or (self.video_setting):
                for frm_cnt, cumulative_frm in enumerate(range(self.clf_array.shape[0])):
                    start = time.time()
                    current_frm = self.clf_array[cumulative_frm,:,:]
                    cum_array = self.__calculate_cum_array(cum_array=cum_array, current_frm=current_frm)
                    cum_array_s = cum_array / self.fps
                    cum_df = pd.DataFrame(cum_array_s).reset_index()
                    cum_df = cum_df.melt(id_vars='index', value_vars=None, var_name=None, value_name='seconds', col_level=None).rename(columns={'index':'vertical_idx', 'variable': 'horizontal_idx'})
                    cum_df['color'] = (cum_df['seconds'].astype(float) / float(self.max_scale)).round(2).clip(upper=100)
                    cum_array = np.array(cum_df).astype(int)
                    print(cum_array.shape)
                    color_array = self.__create_color_array(data_arr=cum_array)

            #
            #         fig = plt.figure()
            #         im_ratio = color_array.shape[0] / color_array.shape[1]
            #         plt.pcolormesh(color_array, shading='gouraud', cmap=self.palette, rasterized=True, alpha=1, vmin=0.0, vmax=float(self.max_scale))
            #         plt.gca().invert_yaxis()
            #         plt.xticks([])
            #         plt.yticks([])
            #         plt.axis('off')
            #         plt.tick_params(axis='both', which='both', length=0)
            #         cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
            #         cb.ax.tick_params(size=0)
            #         cb.outline.set_visible(False)
            #         cb.set_label('{} (seconds)'.format(self.clf_name), rotation=270, labelpad=10)
            #         plt.tight_layout()
            #         plt.gca().set_aspect(hight_to_width)
            #         canvas = FigureCanvas(fig)
            #         canvas.draw()
            #         mat = np.array(canvas.renderer._renderer)
            #         image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            #         image = cv2.resize(image, (self.width, self.height))
            #         image = np.uint8(image)
            #         plt.close()
            #
            #         if self.video_setting:
            #             self.writer.write(image)
            #         if self.frame_setting:
            #             frame_save_path = os.path.join(self.save_video_folder, str(frm_cnt) + '.png')
            #             cv2.imwrite(frame_save_path, image)
            #         print('Heatmap frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt + 1), str(len(self.data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))
            #
            # if self.video_setting:
            #     self.writer.release()
            # print('Heatmap plot for video {} saved...'.format(self.video_name))

        print('SIMBA COMPLETE: All heatmap visualizations created in project_folder/frames/output/heatmaps_classifier_locations directory')


# test = HeatMapperClf(config_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                  final_img_setting=False,
#                  video_setting=True,
#                  frame_setting=False,
#                  bin_size=100,
#                  palette='jet',
#                  bodypart='Nose_1',
#                  clf_name='Attack',
#                  max_scale=2)
# test.create_heatmaps()

