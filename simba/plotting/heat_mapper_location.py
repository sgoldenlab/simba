__author__ = "Simon Nilsson", "JJ Choong"

from simba.feature_extractors.unit_tests import read_video_info
from simba.misc_tools import get_fn_ext, SimbaTimer
import os
import cv2
from simba.rw_dfs import read_df
import numpy as np
from numba import jit, prange
import pandas as pd
from simba.enums import Formats
from simba.mixins.config_reader import ConfigReader
from simba.plotting.misc_visualizations import make_location_heatmap_plot
from simba.utils.errors import NoSpecifiedOutputError

class HeatmapperLocationSingleCore(ConfigReader):
    """
    Class for creating heatmaps representing the location where animals spend time.

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
        If True, create a single image representing the last frame of the input video
    video_setting: bool
        If True, then create a video of heatmaps.
    frame_setting: bool
        If True, then create individual heatmap frames
    Notes
    -----
    `GitHub visualizations tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    -----

    >>> heat_mapper_location = HeatmapperLocationSingleCore(config_path='MyConfigPath', final_img_setting=False, video_setting=True, frame_setting=False, bin_size=50, palette='jet', bodypart='Nose_1', max_scale=20)
    >>> heat_mapper_location.create_heatmaps()
    """

    def __init__(self,
                 config_path: str,
                 bodypart: str,
                 style_attr: dict,
                 final_img_setting: bool,
                 video_setting: bool,
                 frame_setting: bool,
                 files_found: list):

        super().__init__(config_path=config_path)

        if (not frame_setting) and (not video_setting) and (not final_img_setting):
            raise NoSpecifiedOutputError(msg='Please choose to select either heatmap videos, frames, and/or final image.')

        self.frame_setting, self.video_setting = frame_setting, video_setting
        self.final_img_setting, self.bp = final_img_setting, bodypart
        self.style_attr = style_attr
        if not os.path.exists(self.heatmap_location_dir): os.makedirs(self.heatmap_location_dir)
        self.files_found = files_found
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.bp_lst = [self.bp + '_x', self.bp + '_y']
        print('Processing heatmaps for {} video(s)...'.format(str(len(self.files_found))))

    @staticmethod
    @jit(nopython=True)
    def __calculate_cum_array_final_img(loc_array: np.array):
        final_img = np.full((loc_array.shape[1], loc_array.shape[2]), 0)
        for frm in range(loc_array.shape[0]):
            for row in range(loc_array.shape[1]):
                for col in range(loc_array.shape[2]):
                    final_img[row, col] += loc_array[frm, row, col]
        return final_img


    def __calculate_bin_attr(self,
                             data_df: pd.DataFrame,
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

        bp_data = data_df[bp_lst].to_numpy().astype(int)

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

        location_array = np.zeros((bp_data.shape[0], vertical_bin_cnt, horizontal_bin_cnt))

        for frm_cnt, frame in enumerate(bp_data):
            for h_bin_name, v_dict in bin_dict.items():
                for v_bin_name, c in v_dict.items():
                    if (frame[0] <= c['bottom_right_x'] and frame[0] >= c['top_left_x']):
                        if (frame[1] <= c['bottom_right_y'] and frame[0] >= c['top_left_y']):
                            location_array[frm_cnt][v_bin_name][h_bin_name] = 1

        location_array = self.__calculate_cum_array(clf_array=location_array, fps=fps)

        return location_array, aspect_ratio

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

    def create_heatmaps(self):
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.video_info_df, video_name=self.video_name)
            self.width, self.height = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])
            if self.video_setting:
                self.video_save_path = os.path.join(self.heatmap_location_dir, self.video_name + '.mp4')
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.width, self.height))
            if self.frame_setting | self.final_img_setting:
                self.save_video_folder = os.path.join(self.heatmap_location_dir, self.video_name)
                if not os.path.exists(self.save_video_folder): os.makedirs(self.save_video_folder)
            self.data_df = read_df(file_path, self.file_type)
            location_array, aspect_ratio = self.__calculate_bin_attr(data_df=self.data_df,
                                                                     bp_lst=self.bp_lst,
                                                                     px_per_mm=self.px_per_mm,
                                                                     img_width=self.width,
                                                                     img_height=self.height,
                                                                     bin_size=self.style_attr['bin_size'],
                                                                     fps=self.fps)

            if self.style_attr['max_scale'] == 'auto':
                self.max_scale = np.round(np.max(np.max(location_array[-1], axis=0)), 3)
                if self.max_scale == 0: self.max_scale = 1

            if self.final_img_setting:
                make_location_heatmap_plot(frm_data=location_array[-1:, :, :],
                                           max_scale=self.max_scale,
                                           palette=self.style_attr['palette'],
                                           aspect_ratio=aspect_ratio,
                                           file_name=os.path.join(self.heatmap_location_dir, self.video_name + '_final_frm.png'),
                                           shading=self.style_attr['shading'],
                                           img_size=(self.width, self.height),
                                           final_img=True)
                print('Final heatmap image saved at {}'.format(os.path.join(self.save_video_folder, '_final_img.png')))

            if (self.frame_setting) or (self.video_setting):
                for frm_cnt, cumulative_frm in enumerate(range(location_array.shape[0])):
                    img = make_location_heatmap_plot(frm_data=location_array[cumulative_frm,:,:],
                                                     max_scale=self.max_scale,
                                                     palette=self.style_attr['palette'],
                                                     aspect_ratio=aspect_ratio,
                                                     file_name=None,
                                                     shading=self.style_attr['shading'],
                                                     img_size=(self.width, self.height),
                                                     final_img=False)
                    if self.video_setting:
                        self.writer.write(img)
                    if self.frame_setting:
                        frame_save_path = os.path.join(self.save_video_folder, str(frm_cnt) + '.png')
                        cv2.imwrite(frame_save_path, img)

                    print('Heatmap frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt + 1), str(len(self.data_df)), self.video_name, str(file_cnt + 1), len(self.files_found)))

            if self.video_setting:
                self.writer.release()
            video_timer.stop_timer()
            print(f'Heatmap plot for video {self.video_name} saved (elapsed time: {video_timer.elapsed_time_str}s')
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: Created heatmaps for {str(len(self.files_found))} videos (elapsed time {self.timer.elapsed_time_str}s)')

#
#
# test = HeatmapperLocationSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 100, 'max_scale': 'auto'},
#                                       final_img_setting=False,
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       bodypart='Nose_1',
#                                       files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.create_heatmaps()



# test = HeatmapperLocationSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini',
#                                       style_attr = {'palette': 'jet', 'shading': 'gouraud', 'bin_size': 50, 'max_scale': 'auto'},
#                                       final_img_setting=False,
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       bodypart='Nose',
#                                       files_found=['/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/csv/outlier_corrected_movement_location/PD1406_2022-05-24_RVDG_GCaMP8s-Gi_Video_Day_22_Baseline.csv'])
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