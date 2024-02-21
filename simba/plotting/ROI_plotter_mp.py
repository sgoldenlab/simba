__author__ = "Simon Nilsson"


import pandas as pd
import os
import itertools
import cv2
import platform
from pathlib import Path
import multiprocessing
import functools
import numpy as np
from copy import deepcopy
from typing import Optional, Union

from simba.utils.enums import Paths, TagNames, TextOptions
from simba.utils.data import detect_bouts
from simba.utils.printing import stdout_success, SimbaTimer, log_event
from simba.utils.warnings import DuplicateNamesWarning
from simba.utils.data import create_color_palettes
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.errors import NoFilesFoundError, DuplicationError
from simba.utils.read_write import get_video_meta_data, concatenate_videos_in_folder, get_fn_ext
pd.options.mode.chained_assignment = None

class ROIPlotMultiprocess(ConfigReader, PlottingMixin):
    """
    Visualize the ROI data (number of entries/exits, time-spent-in etc).

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.
        .. image:: _static/img/roi_visualize.png
           :width: 400
           :align: center

    :param str config_path: Path to SimBA project config file in Configparser format
    :param str video_path: Name of video to create ROI visualizations for
    :param dict style_attr: User-defined visualization settings.
    :param int core_cnt: Number of cores to use.




    Examples
    ----------
    >>> roi_visualizer = ROIPlotMultiprocess(ini_path=r'MyProjectConfig', video_path="MyVideo.mp4")
    >>> roi_visualizer.run()
    """

    def __init__(
            self,
            ini_path: Union[str, os.PathLike],
            video_path: Union[str, os.PathLike],
            core_cnt: int,
            style_attr: dict,
            body_parts: Optional[dict] = None,
            threshold: Optional[float] = None):

        ConfigReader.__init__(self, config_path=ini_path)
        PlottingMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        settings = None
        if body_parts:
            settings = {'body_parts': body_parts}
            if threshold: settings['threshold'] = threshold
            else: settings['threshold'] = 0.0
            self.animal_id_lst = list(body_parts.keys())
        self.roi_analyzer = ROIAnalyzer(ini_path=ini_path, data_path='outlier_corrected_movement_location', settings=settings)
        if not body_parts:
            self.animal_id_lst = self.roi_analyzer.multi_animal_id_list
        if threshold: self.threshold = threshold
        else: self.threshold = 0.0



        self.video_path = os.path.join(self.video_dir, video_path)
        _, self.video_name, _ = get_fn_ext(video_path)
        self.core_cnt = core_cnt
        self.roi_analyzer.files_found = [os.path.join(self.roi_analyzer.input_folder, self.video_name + '.' + self.roi_analyzer.file_type)]
        if not os.path.isfile(self.roi_analyzer.files_found[0]):
            raise NoFilesFoundError(msg=f'SIMBA ERROR: Could not find the file at path {self.roi_analyzer.files_found[0]}. Please make sure you have corrected body-part outliers or indicated that you want to skip outlier correction', source=self.__class__.__name__)
        self.roi_analyzer.run()
        self.roi_entries_df = self.roi_analyzer.detailed_df
        self.data_df, self.style_attr = self.roi_analyzer.data_df, style_attr
        self.out_parent_dir = os.path.join(self.roi_analyzer.project_path, Paths.ROI_ANALYSIS.value)
        if not os.path.exists(self.out_parent_dir): os.makedirs(self.out_parent_dir)
        self.video_save_path = os.path.join(self.out_parent_dir, self.video_name + '.mp4')
        self.video_shapes = list(itertools.chain(self.roi_analyzer.video_recs['Name'].unique(), self.roi_analyzer.video_circs['Name'].unique(),self.roi_analyzer.video_polys['Name'].unique()))
        if len(list(set(self.video_shapes))) != len(self.video_shapes):
            raise DuplicationError(msg='Some SIMBA ROI shapes have identical names. Please use unique names to visualize ROI data.', source=self.__class__.__name__)
        self.roi_analyzer.video_recs = self.add_missing_ROI_cols(self.roi_analyzer.video_recs)
        self.roi_analyzer.video_circs = self.add_missing_ROI_cols(self.roi_analyzer.video_circs)
        self.roi_analyzer.video_polys = self.add_missing_ROI_cols(self.roi_analyzer.video_polys)

        self.shape_dicts = {}
        for df in [self.roi_analyzer.video_recs, self.roi_analyzer.video_circs, self.roi_analyzer.video_polys]:
            if not df['Name'].is_unique:
                df = df.drop_duplicates(subset=['Name'], keep='first')
                DuplicateNamesWarning('Some of your ROIs with the same shape has the same names. E.g., you have two rectangles named "My rectangle". SimBA prefers ROI shapes with unique names. SimBA will keep one of the unique shape names and drop the rest.', source=self.__class__.__name__)
            d = df.set_index('Name').to_dict(orient='index')
            self.shape_dicts = {**self.shape_dicts, **d}

        self.shape_columns = []
        for animal in self.animal_id_lst:
            for shape_name in self.video_shapes:
                self.data_df[animal + '_' + shape_name] = 0
                self.shape_columns.append(animal + '_' + shape_name)
        self.bp_dict = self.roi_analyzer.bp_dict
        self.output_folder = os.path.join(self.roi_analyzer.project_path, Paths.ROI_ANALYSIS.value)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.insert_data()

    def insert_data(self):
        self.roi_entries_dict =self.roi_entries_df[['ANIMAL', 'SHAPE', 'ENTRY FRAMES', 'EXIT FRAMES']].to_dict(orient='records')
        for entry_dict in self.roi_entries_dict:
            entry, exit = int(entry_dict['ENTRY FRAMES']), int(entry_dict['EXIT FRAMES'])
            entry_dict['frame_range'] = list(range(entry, exit + 1))
            col_name = entry_dict['ANIMAL'] + '_' + entry_dict['SHAPE']
            self.data_df[col_name][self.data_df.index.isin(entry_dict['frame_range'])] = 1

    def __calc_text_locs(self):
        self.loc_dict = {}
        line_spacer = deepcopy(TextOptions.FIRST_LINE_SPACING.value)
        for animal_cnt, animal_name in enumerate(self.animal_id_lst):
            self.loc_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape]['timer_text'] = '{} {} {}'.format(shape, animal_name, 'timer:')
                self.loc_dict[animal_name][shape]['entries_text'] = '{} {} {}'.format(shape, animal_name, 'entries:')
                self.loc_dict[animal_name][shape]['timer_text_loc'] = ((self.video_meta_data['width'] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers['space_size'] * line_spacer))
                self.loc_dict[animal_name][shape]['timer_data_loc'] = (int(self.border_img_w-(self.border_img_w/8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers['space_size'] * line_spacer))
                line_spacer += TextOptions.LINE_SPACING.value
                self.loc_dict[animal_name][shape]['entries_text_loc'] = ((self.video_meta_data['width'] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers['space_size'] * line_spacer))
                self.loc_dict[animal_name][shape]['entries_data_loc'] = (int(self.border_img_w - (self.border_img_w / 8)), (self.video_meta_data['height'] - (self.video_meta_data['height'] + TextOptions.BORDER_BUFFER_Y.value) + self.scalers['space_size'] * line_spacer))
                line_spacer += TextOptions.LINE_SPACING.value

    def __create_counters(self):
        self.cnt_dict = {}
        for animal_cnt, animal_name in enumerate(self.animal_id_lst):
            self.cnt_dict[animal_name] = {}
            for shape in self.video_shapes:
                self.cnt_dict[animal_name][shape] = {}
                self.cnt_dict[animal_name][shape]['timer'] = 0
                self.cnt_dict[animal_name][shape]['entries'] = 0
                self.cnt_dict[animal_name][shape]['entry_status'] = False


    def __calculate_cumulative(self):
        for animal in self.animal_id_lst:
            for shape in self.video_shapes:
                self.data_df['{}_{}_cum_sum_time'.format(animal, shape)] = self.data_df['{}_{}'.format(animal, shape)].cumsum() / self.fps
                roi_bouts = list(detect_bouts(data_df=self.data_df, target_lst=['{}_{}'.format(animal, shape)], fps=self.fps)['Start_frame'])
                self.data_df['{}_{}_entry'.format(animal, shape)] = 0
                self.data_df.loc[roi_bouts, '{}_{}_entry'.format(animal, shape)] = 1
                self.data_df['{}_{}_cum_sum_entries'.format(animal, shape)] = self.data_df['{}_{}_entry'.format(animal, shape)].cumsum()


    def __update_video_meta_data(self):
        new_cap = cv2.VideoCapture(self.video_path)
        new_cap.set(1, 1)
        _, img = self.cap.read()
        bordered_img = cv2.copyMakeBorder(img, 0, 0, 0, int(self.video_meta_data['width']), borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.border_img_h, self.border_img_w = bordered_img.shape[0], bordered_img.shape[1]
        new_cap.release()

    def run(self):
        """
        Method to visualize ROI data. Results are stored in the `project_folder/frames/output/ROI_analysis`
        directory of the SimBA project.

        Returns
        -------
        None
        """

        video_timer = SimbaTimer(start=True)
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_meta_data = get_video_meta_data(self.video_path)
        video_settings, pix_per_mm, self.fps = self.read_video_info(video_name=self.video_name)
        self.space_scale, radius_scale, res_scale, font_scale = 25, 10, 1500, 0.8
        max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
        self.scalers = {}
        self.scalers['circle_size'] = int(TextOptions.RADIUS_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        self.scalers['font_size'] = float(TextOptions.FONT_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        self.scalers['space_size'] = int(TextOptions.SPACE_SCALER.value / (TextOptions.RESOLUTION_SCALER.value / max_dim))
        color_lst = create_color_palettes(self.roi_analyzer.animal_cnt, int((len(self.roi_analyzer.bp_names) / 3)))[0]
        self.temp_folder = os.path.join(self.out_parent_dir, self.video_name, 'temp')
        if not os.path.exists(self.temp_folder): os.makedirs(self.temp_folder)
        self.__update_video_meta_data()
        self.__calc_text_locs()
        self.__create_counters()
        self.__calculate_cumulative()

        data_arr = np.array_split(self.data_df.fillna(0), self.core_cnt)
        for df_cnt in range(len(data_arr)):
            data_arr[df_cnt]['group'] = df_cnt
        frm_per_core = len(data_arr[0])

        print(f'Creating ROI images, multiprocessing (determined chunksize: {self.multiprocess_chunksize}, cores: {self.core_cnt})...')
        del self.roi_analyzer.logger
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(self.roi_plotter_mp,
                                          loc_dict=self.loc_dict,
                                          scalers=self.scalers,
                                          video_meta_data=self.video_meta_data,
                                          save_temp_directory= self.temp_folder,
                                          body_part_dict=self.bp_dict,
                                          input_video_path=self.video_path,
                                          roi_analyzer_data=self.roi_analyzer,
                                          video_shape_names=self.video_shapes,
                                          colors=color_lst,
                                          shape_meta_data=self.shape_dicts,
                                          style_attr=self.style_attr,
                                          animal_ids=self.animal_id_lst,
                                          threshold=self.threshold)
            for cnt, result in enumerate(pool.imap(constants, data_arr, chunksize=self.multiprocess_chunksize)):
                print('Image {}/{}, Video {}...'.format(str(int(frm_per_core * (result + 1))), str(len(self.data_df)), self.video_name))
            print('Joining {} multi-processed video...'.format(self.video_name))
            concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=self.video_save_path, video_format='mp4')
            video_timer.stop_timer()
            pool.terminate()
            pool.join()
            stdout_success(msg=f'Video {self.video_name} created. Video saved in project_folder/frames/output/ROI_analysis', elapsed_time=video_timer.elapsed_time_str, source=self.__class__.__name__)

# test = ROIPlotMultiprocess(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini',
#                video_path="termite_test.mp4",
#                core_cnt=5,
#                style_attr={'Show_body_part': True, 'Show_animal_name': True})
# test.run()

# test = ROIPlot(ini_path=r'/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', video_path=r"Together_1.avi")
# test.insert_data()
# test.visualize_ROI_data()

# test = ROIPlot(ini_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\project_config.ini", video_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\videos\Video7.mp4")
# test.insert_data()
# test.visualize_ROI_data()
#
# test = ROIPlotMultiprocess(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                            video_path="Together_1.avi",
#                            style_attr={'Show_body_part': True, 'Show_animal_name': False},
#                            core_cnt=5)
# test.run()


# test = ROIPlotMultiprocess(ini_path=r'/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini',
#                            video_path="Together_1.avi",
#                            style_attr={'Show_body_part': True, 'Show_animal_name': False},
#                            core_cnt=5)
# test.run()