__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer)
from simba.misc_visualizations import make_heatmap_plot
from simba.rw_dfs import read_df
import numpy as np
import os
import cv2
from numba import jit, prange
import matplotlib.pyplot as plt
from simba.enums import Paths, ReadConfig, Formats
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class HeatMapperClfSingleCore(object):

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
                 style_attr: dict
                 ):

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise ValueError('SIMBA ERROR: Please choose to select either heatmap videos, frames, and/or final image.')
        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.bin_size, self.max_scale, self.palette, self.shading = style_attr['bin_size'], style_attr['max_scale'], style_attr['palette'], style_attr['shading']
        self.clf_name, self.files_found = clf_name, files_found
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.save_dir = os.path.join(self.project_path, Paths.HEATMAP_CLF_LOCATION_DIR.value)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.dir_in = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.bp_lst = [self.bp + '_x', self.bp + '_y']
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
            if self.video_setting:
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.video_save_path = os.path.join(self.save_dir, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.width, self.height))
            if self.frame_setting:
                self.save_video_folder = os.path.join(self.save_dir, self.video_name)
                if not os.path.exists(self.save_video_folder):
                    os.makedirs(self.save_video_folder)
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
                                  file_name=os.path.join(self.save_dir, self.video_name + '_final_frm.png'),
                                  shading=self.shading,
                                  clf_name=self.clf_name,
                                  img_size=(self.width, self.height))

            if self.video_setting or self.frame_setting:
                for frm_cnt, cumulative_frm_idx in enumerate(range(clf_array.shape[0])):
                    frm_data = clf_array[cumulative_frm_idx,:,:]
                    cum_df = pd.DataFrame(frm_data).reset_index()
                    cum_df = cum_df.melt(id_vars='index', value_vars=None, var_name=None, value_name='seconds', col_level=None).rename(columns={'index':'vertical_idx', 'variable': 'horizontal_idx'})
                    cum_df['color'] = (cum_df['seconds'].astype(float) / float(self.max_scale)).round(2).clip(upper=100)
                    color_array = np.zeros((len(cum_df['vertical_idx'].unique()), len(cum_df['horizontal_idx'].unique())))
                    for i in range(color_array.shape[0]):
                        for j in range(color_array.shape[1]):
                            value = cum_df["color"][(cum_df["horizontal_idx"] == j) & (cum_df["vertical_idx"] == i)].values[0]
                            color_array[i,j] = value

                    fig = plt.figure()
                    im_ratio = color_array.shape[0] / color_array.shape[1]
                    plt.pcolormesh(color_array, shading=self.shading, cmap=self.palette, rasterized=True, alpha=1, vmin=0.0, vmax=float(self.max_scale))
                    plt.gca().invert_yaxis()
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                    plt.tick_params(axis='both', which='both', length=0)
                    cb = plt.colorbar(pad=0.0, fraction=0.023 * im_ratio)
                    cb.ax.tick_params(size=0)
                    cb.outline.set_visible(False)
                    cb.set_label('{} (seconds)'.format(self.clf_name), rotation=270, labelpad=10)
                    plt.tight_layout()
                    plt.gca().set_aspect(aspect_ratio)
                    canvas = FigureCanvas(fig)
                    canvas.draw()
                    mat = np.array(canvas.renderer._renderer)
                    image = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                    image = cv2.resize(image, (self.width, self.height))
                    image = np.uint8(image)
                    plt.close()

                    if self.video_setting:
                        self.writer.write(image)
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_video_folder, str(frm_cnt) + '.png')
                        cv2.imwrite(frame_save_path, image)
                    print('Created heatmap frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt + 1), str(len(self.data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))

                if self.video_setting:
                    self.writer.release()

                video_timer.stop_timer()
                print('Heatmap plot for video {} saved (elapsed time: {}s) ... '.format(self.video_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: All heatmap visualizations created in project_folder/frames/output/heatmaps_classifier_locations directory (elapsed time: {}s)'.format(self.timer.elapsed_time_str))





# test = HeatMapperClfSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'},
#                      final_img_setting=False,
#                      video_setting=False,
#                      frame_setting=True,
#                      bodypart='Nose_1',
#                      clf_name='Attack',
#                      files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.create_heatmaps()
#
