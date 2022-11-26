__author__ = "Simon Nilsson", "JJ Choong"

from simba. read_config_unit_tests import (read_config_file,
                                           read_config_entry)
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import get_fn_ext
import os, glob
import cv2
from simba.rw_dfs import read_df
import numpy as np
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class HeatmapperLocation(object):
    """
    Class for creating heatmaps representing the the location where animals spend time.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    bodypart: str
        The name of the body-part used to infer the location of the classified behavior
    bin_size: int
        The rectangular size of each heatmap location in millimeters. For example, `50` will divide the video frames
        into 5 centimeter rectangular spatial bins.
    palette: str
        Heatmap pallette. Eg. 'jet', 'magma', 'inferno','plasma', 'viridis', 'gnuplot2'
    max_scale: int or 'auto'
        The max value in the heatmap in seconds. E.g., with a value of `10`, if the classified behavior has occured
        >= 10 within a rectangular bins, it will be filled with the same color.
    final_img_setting: bool
        If True, then  create the a single image representing the last frame of the input video
    video_setting: bool
        If True, then create a video of heatmaps.
    frame_setting: bool
        If True, then create individual heatmap frames
    Notes
    -----
    `GitHub visualizations tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    -----

    >>> heat_mapper_location = HeatmapperLocation(config_path='MyConfigPath', final_img_setting=False, video_setting=True, frame_setting=False, bin_size=50, palette='jet', bodypart='Nose_1', max_scale=20)
    >>> heat_mapper_location.create_heatmaps()


    """


    def __init__(self,
                 config_path: str,
                 bodypart: str,
                 bin_size: int,
                 palette: str,
                 max_scale: int,
                 final_img_setting: bool,
                 video_setting: bool,
                 frame_setting: bool):

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            print('SIMBA ERROR: Please choose to select either videos, frames, and/or final image.')
            raise ValueError()
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.bin_size, self.max_scale = bin_size, max_scale
        self.config, self.palette= read_config_file(config_path), palette
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.save_dir = os.path.join(self.project_path, 'frames', 'output', 'heatmaps_locations')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.dir_in = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.files_found = glob.glob(self.dir_in + "/*." + self.file_type)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.bp_lst = [self.bp + '_x', self.bp + '_y']
        print('Processing heatmaps for {} video(s)...'.format(str(len(self.files_found))))

    @staticmethod
    @jit(nopython=True)
    def __calculate_cum_array(cum_array: np.array, current_frm: np.array):
        for row in range(current_frm.shape[0]):
            for col in range(current_frm.shape[1]):
                cum_array[row, col] += current_frm[row, col]
        return cum_array

    @staticmethod
    @jit(nopython=True)
    def __calculate_cum_array_final_img(loc_array: np.array):
        final_img = np.full((loc_array.shape[1], loc_array.shape[2]), 0)
        for frm in range(loc_array.shape[0]):
            for row in range(loc_array.shape[1]):
                for col in range(loc_array.shape[2]):
                    final_img[row, col] += loc_array[frm, row, col]
        return final_img

    def __create_img(self):
        fig = plt.figure()
        im_ratio = self.color_array.shape[0] / self.color_array.shape[1]
        plt.pcolormesh(self.color_array, shading='gouraud', cmap=self.palette, rasterized=True, alpha=1, vmin=0.0, vmax=float(self.max_scale))
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tick_params(axis='both', which='both', length=0)
        cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
        cb.ax.tick_params(size=0)
        cb.outline.set_visible(False)
        cb.set_label('seconds', rotation=270, labelpad=10)
        plt.tight_layout()
        plt.gca().set_aspect(self.hight_to_width)
        canvas = FigureCanvas(fig)
        canvas.draw()
        mat = np.array(canvas.renderer._renderer)
        image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (self.width, self.height))
        self.image = np.uint8(image)
        plt.close()

    def __save_img(self, frm_cnt=None, final_img=False):
        if self.video_setting:
            self.writer.write(self.image)
        if self.frame_setting:
            frame_save_path = os.path.join(self.save_video_folder, str(frm_cnt) + '.png')
            cv2.imwrite(frame_save_path, self.image)
        if final_img:
            frame_save_path = os.path.join(self.save_dir, self.video_name + '_final_img.png')
            cv2.imwrite(frame_save_path, self.image)

    def create_heatmaps(self):
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=self.video_name)
            self.width, self.height = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])
            if self.video_setting:
                self.video_save_path = os.path.join(self.save_dir, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.width, self.height))
            if self.frame_setting | self.final_img_setting:
                self.save_video_folder = os.path.join(self.save_dir, self.video_name)
                if not os.path.exists(self.save_video_folder): os.makedirs(self.save_video_folder)
            self.pixels_per_bin = int(float(self.px_per_mm) * float(self.bin_size))
            self.hbins_cnt, self.vbins_cnt = int(self.width / self.pixels_per_bin), int(self.height / self.pixels_per_bin)
            self.hight_to_width = round((self.vbins_cnt / self.hbins_cnt), 3)
            self.bin_dict = {}
            x_location, y_location = 0, 0
            self.data_df = read_df(file_path, self.file_type)
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
            self.loc_idx = self.data_df[self.bp_lst].reset_index().to_numpy()
            self.loc_array = np.zeros((len(self.data_df), self.vbins_cnt, self.hbins_cnt))
            for frame in self.loc_idx:
                for h_bin_name, v_dict in self.bin_dict.items():
                    for v_bin_name, c in v_dict.items():
                        if (frame[1] < c['bottom_right_x'] and frame[1] > c['top_left_x']):
                            if (frame[2] < c['bottom_right_y'] and frame[2] > c['top_left_y']):
                                self.loc_array[int(frame[0])][v_bin_name][h_bin_name] = 1
            cum_array = np.zeros((self.loc_array.shape[1], self.loc_array.shape[2]))
            if self.max_scale == 'auto':
                self.max_scale = int(np.max(np.sum(self.loc_array,axis=0)) / self.fps)
                if self.max_scale == 0: self.max_scale = 1

            if self.final_img_setting:
                cum_array = self.__calculate_cum_array_final_img(loc_array=self.loc_array)
                cum_array_s = cum_array / self.fps
                cum_df = pd.DataFrame(cum_array_s).reset_index()
                cum_df = cum_df.melt(id_vars='index', value_vars=None, var_name=None, value_name='seconds',
                                     col_level=None).rename(
                    columns={'index': 'vertical_idx', 'variable': 'horizontal_idx'})
                cum_df['color'] = (cum_df['seconds'].astype(float) / float(self.max_scale)).round(2).clip(upper=100)
                self.color_array = np.zeros((len(cum_df['vertical_idx'].unique()), len(cum_df['horizontal_idx'].unique())))
                for i in range(self.color_array.shape[0]):
                    for j in range(self.color_array.shape[1]):
                        value = cum_df["color"][(cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)].values[0]
                        self.color_array[i, j] = value
                self.__create_img()
                self.__save_img(final_img=True)
                print('Final heatmap image saved at {}'.format(os.path.join(self.save_video_folder, '_final_img.png')))

            if (self.frame_setting) or (self.video_setting):
                for frm_cnt, cumulative_frm in enumerate(range(self.loc_array.shape[0])):
                    current_frm = self.loc_array[cumulative_frm,:,:]
                    cum_array = self.__calculate_cum_array(cum_array=cum_array, current_frm=current_frm)
                    cum_array_s = cum_array / self.fps
                    cum_df = pd.DataFrame(cum_array_s).reset_index()
                    cum_df = cum_df.melt(id_vars='index', value_vars=None, var_name=None, value_name='seconds', col_level=None).rename(columns={'index':'vertical_idx', 'variable': 'horizontal_idx'})
                    cum_df['color'] = (cum_df['seconds'].astype(float) / float(self.max_scale)).round(2).clip(upper=100)
                    self.color_array = np.zeros((len(cum_df['vertical_idx'].unique()), len(cum_df['horizontal_idx'].unique())))
                    for i in range(self.color_array.shape[0]):
                        for j in range(self.color_array.shape[1]):
                            value = cum_df["color"][(cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)].values[0]
                            self.color_array[i,j] = value

                    self.__create_img()
                    self.__save_img()
                    print('Heatmap frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt + 1), str(len(self.data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))

            if self.video_setting:
                self.writer.release()
            print('Heatmap plot for video {} saved...'.format(self.video_name))
        print('SIMBA COMPLETE: Created heatmaps for {} videos'.format(str(len(self.files_found))))

#
#
# test = HeatmapperLocation(config_path='/Users/simon/Desktop/troubleshooting/Open_field_5/project_folder/project_config.ini',
#                           bodypart='Nose',
#                           bin_size=50,
#                           palette='jet',
#                           max_scale=1,
#                           final_img_setting=True,
#                           video_setting=False,
#                           frame_setting=False)
# test.create_heatmaps()


# test = HeatmapperLocation(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                           bodypart='Nose_1',
#                           bin_size=50,
#                           palette='jet',
#                           max_scale='auto',
#                           final_img_setting=True,
#                           video_setting=False,
#                           frame_setting=False)
# test.create_heatmaps()