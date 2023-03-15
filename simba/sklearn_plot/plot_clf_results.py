from simba.read_config_unit_tests import (read_config_entry,
                                          check_file_exist_and_readable,
                                          check_float,
                                          check_int)
from simba.misc_tools import (find_video_of_file,
                              get_video_meta_data)
from simba.rw_dfs import read_df
from simba.misc_tools import create_single_color_lst
from simba.feature_extractors.unit_tests import read_video_info
from simba.drop_bp_cords import (createColorListofList,
                                 create_body_part_dictionary,
                                 get_fn_ext)
from simba.enums import ReadConfig, Formats, Dtypes
import os, glob
from copy import deepcopy
from simba.train_model_functions import get_model_info
from simba.mixins.config_reader import ConfigReader
from simba.utils.errors import NoSpecifiedOutputError
import cv2
import numpy as np
from PIL import Image


class PlotSklearnResultsSingleCore(ConfigReader):
    """
    Class for plotting classification results on videos. Results are stored in the
    `project_folder/frames/output/sklearn_results` directory of the SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    rotate: bool
        If True, the output video will be rotated 90 degrees from the input.
    video_setting: bool
        If True, SimBA will create compressed videos.
    frame_setting: bool
        If True, SimBA will create individual frames
    video_file_path: str
       path to video file to create classification visualizations for.

    Notes
    ----------
    `Scikit visualization documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-10-sklearn-visualization__.

    Examples
    ----------
    >>> clf_plotter = PlotSklearnResults(config_path='MyProjectConfig', video_setting=True, frame_setting=False, rotate=False, video_file_path='VideoPath')
    >>> clf_plotter.initialize_visualizations()
    """

    def __init__(self,
                 config_path: str,
                 video_setting: bool,
                 frame_setting: bool,
                 text_settings: dict or bool,
                 rotate: False,
                 video_file_path=None,
                 print_timers=True):

        super().__init__(config_path=config_path)

        if (not video_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(msg='Please choose to create a video and/or frames. SimBA found that you ticked neither video and/or frames')
        self.video_file_path, self.print_timers, self.text_settings = video_file_path, print_timers, text_settings
        self.video_setting, self.frame_setting = video_setting, frame_setting
        if video_file_path is not None:
            check_file_exist_and_readable(os.path.join(self.video_dir, video_file_path))
        if not os.path.exists(self.sklearn_plot_dir): os.makedirs(self.sklearn_plot_dir)
        self.pose_threshold = read_config_entry(self.config, ReadConfig.THRESHOLD_SETTINGS.value, ReadConfig.SKLEARN_BP_PROB_THRESH.value, Dtypes.FLOAT.value, 0.00)
        self.color_lst_of_lst = createColorListofList(self.animal_cnt, int(len(self.x_cols) + 1))
        self.files_found = glob.glob(self.machine_results_dir + '/*.' + self.file_type)
        self.model_dict = get_model_info(self.config, self.clf_cnt)
        self.clf_colors = create_single_color_lst(pallete_name ='Set1', increments=self.clf_cnt+3)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_list, self.animal_cnt, self.x_cols, self.y_cols, [], self.color_lst_of_lst)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.rotate = rotate
        self.a = np.deg2rad(90)
        print(f'Processing {str(len(self.files_found))} videos...')

    def __get_print_settings(self):
        if self.text_settings is False:
            self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 60, 12, 1500, 1.1
            self.max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
            self.circle_scale = int(self.radius_scale / (self.res_scale / self.max_dim))
            self.font_size = float(self.font_scale / (self.res_scale / self.max_dim))
            self.spacing_scale = int(self.space_scale / (self.res_scale / self.max_dim))
            self.text_thickness = 2
        else:
            check_float(name='ERROR: TEXT SIZE', value=self.text_settings['font_size'])
            check_int(name='ERROR: SPACE SIZE', value=self.text_settings['space_size'])
            check_int(name='ERROR: TEXT THICKNESS', value=self.text_settings['text_thickness'])
            check_int(name='ERROR: CIRCLE SIZE', value=self.text_settings['circle_size'])
            self.font_size = float(self.text_settings['font_size'])
            self.spacing_scale = int(self.text_settings['space_size'])
            self.text_thickness = int(self.text_settings['text_thickness'])
            self.circle_scale = int(self.text_settings['circle_size'])

    def create_visualizations(self):
        _, self.video_name, _ = get_fn_ext(self.file_path)
        self.data_df = read_df(self.file_path, self.file_type).reset_index(drop=True)
        self.video_settings, _, self.fps = read_video_info(self.video_info_df, self.video_name)
        self.video_path = find_video_of_file(self.video_dir, self.video_name)
        self.cap = cv2.VideoCapture(self.video_path)
        self.save_path = os.path.join(self.sklearn_plot_dir, self.video_name + '.mp4')
        self.video_meta_data = get_video_meta_data(self.video_path)
        height, width = deepcopy(self.video_meta_data['height']), deepcopy(self.video_meta_data['width'])
        if self.frame_setting:
            self.video_frame_dir = os.path.join(self.sklearn_plot_dir, self.video_name)
            if not os.path.exists(self.video_frame_dir): os.makedirs(self.video_frame_dir)
        if self.rotate:
            self.video_meta_data['height'], self.video_meta_data['width'] = width, height
        self.writer = cv2.VideoWriter(self.save_path, self.fourcc, self.fps, (self.video_meta_data['width'], self.video_meta_data['height']))
        self.__get_print_settings()
        self.video_model_dict = deepcopy(self.model_dict)
        for model in self.video_model_dict:
            self.video_model_dict[model]['time'] = 0

        row_n = 0
        while (self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            try:
                if ret:
                    self.id_flag_cords = {}
                    for animal_name, animal_data in self.animal_bp_dict.items():
                        animal_clr = animal_data['colors']
                        ID_flag = False
                        for bp_no in range(len(animal_data['X_bps'])):
                            bp_clr = animal_clr[bp_no]
                            x_bp, y_bp = animal_data['X_bps'][bp_no], animal_data['Y_bps'][bp_no],
                            p_bp = x_bp[:-2] + '_p'
                            bp_cords = self.data_df.loc[row_n, [x_bp, y_bp, p_bp]]
                            if bp_cords[p_bp] > self.pose_threshold:
                                cv2.circle(self.frame, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), 0, bp_clr, self.circle_scale)
                                if ('centroid' in x_bp.lower()) or ('center' in x_bp.lower()):
                                    self.id_flag_cords[animal_name] = (int(bp_cords[x_bp]), int(bp_cords[y_bp]))
                                    ID_flag = True

                        if not ID_flag:
                            self.id_flag_cords[animal_name] = (int(bp_cords[x_bp]), int(bp_cords[y_bp]))

                    for animal_name, animal_cords in self.id_flag_cords.items():
                        cv2.putText(self.frame, animal_name, animal_cords, self.font, self.font_size,
                                    self.animal_bp_dict[animal_name]['colors'][0], 2)

                    if self.rotate:
                        self.frame = np.array(Image.fromarray(self.frame).rotate(90, Image.BICUBIC, expand=True))
                    if self.print_timers:
                        cv2.putText(self.frame, str('Timers'), (10, ((self.video_meta_data['height'] - self.video_meta_data['height']) + self.spacing_scale)), self.font, self.font_size, (0, 255, 0), self.text_thickness)
                    self.add_spacer = 2
                    for model_no, model_info in self.video_model_dict.items():
                        frame_results = self.data_df.loc[row_n, model_info['model_name']]
                        self.video_model_dict[model_no]['frame_results'] = frame_results
                        self.video_model_dict[model_no]['time'] += frame_results / self.fps
                        if self.print_timers:
                            cv2.putText(self.frame, model_info['model_name'] + ' ' + str(round(self.video_model_dict[model_no]['time'], 2)) + str('s'), (10, (self.video_meta_data['height'] - self.video_meta_data['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, (255, 0, 0), self.text_thickness)
                            self.add_spacer += 1
                    cv2.putText(self.frame, str('Ensemble prediction'), (10, (self.video_meta_data['height'] - self.video_meta_data['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, (0, 255, 0), self.text_thickness)
                    self.add_spacer += 1

                    for model_cnt, model_info in self.video_model_dict.items():
                        if self.video_model_dict[model_cnt]['frame_results'] == 1:
                            cv2.putText(self.frame, model_info['model_name'], (10, ( self.video_meta_data['height'] - self.video_meta_data['height']) + self.spacing_scale * self.add_spacer), self.font, self.font_size, self.clf_colors[model_cnt], self.text_thickness)
                            self.add_spacer += 1
                    if self.video_setting:
                        self.writer.write(self.frame)
                    if self.frame_setting:
                        frame_save_name = os.path.join(self.video_frame_dir, str(row_n) + '.png')
                        cv2.imwrite(frame_save_name, self.frame)
                    # cv2.imshow('window', self.frame)
                    # cv2.waitKey(50000)
                    print('Frame: {} / {}. Video: {} ({}/{})'.format(str(row_n), str(self.video_meta_data['frame_count']),
                                                                     self.video_name, str(self.file_cnt + 1),
                                                                     len(self.files_found)))
                    row_n += 1

                else:
                    print('Video {} saved...'.format(self.video_name))
                    self.cap.release()
                    self.writer.release()

            except KeyError as e:
                print(e.args, e)
                print('SIMBA INDEX WARNING: Some frames appears to be missing in the dataframe and could not be created')
                print('Video {} saved...'.format(self.video_name))
                self.cap.release()
                self.writer.release()

    def initialize_visualizations(self):
        if self.video_file_path is None:
            for file_cnt, file_path in enumerate(self.files_found):
                self.file_cnt, self.file_path = file_cnt, file_path
                self.create_visualizations()
        else:
            self.file_cnt, file_path = 0, self.video_file_path
            _, file_name, _ = get_fn_ext(file_path)
            self.file_path = os.path.join(self.machine_results_dir, file_name + '.' + self.file_type)
            self.files_found = [self.file_path]
            check_file_exist_and_readable(self.file_path)
            self.create_visualizations()

        self.timer.stop_timer()
        print('SIMBA COMPLETE: All visualizations created in project_folder/frames/output/sklearn_results directory (elapsed time: {}s)'.format(self.timer.elapsed_time_str))

# test = PlotSklearnResults(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini',
#                           video_setting=True,
#                           frame_setting=False,
#                           video_file_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/videos/Together_1.avi',
#                           print_timers=False)
# test.initialize_visualizations()

# test = PlotSklearnResultsSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/project_config.ini',
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       video_file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/videos/Together_1.avi',
#                                       print_timers=True,
#                                       text_settings=False,
#                                       rotate=False)
# test.initialize_visualizations()








