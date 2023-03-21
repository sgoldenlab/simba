from simba.read_config_unit_tests import (read_config_entry,
                                          check_file_exist_and_readable,
                                          check_float,
                                          check_int)
from simba.misc_tools import (find_video_of_file,
                              get_video_meta_data,
                              concatenate_videos_in_folder)
from simba.rw_dfs import read_df
from simba.misc_tools import (create_single_color_lst,
                              SimbaTimer)
from simba.feature_extractors.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.drop_bp_cords import (getBpNames,
                                 createColorListofList,
                                 create_body_part_dictionary,
                                 get_fn_ext)

from simba.enums import ReadConfig, Formats, Paths, Dtypes
from simba.mixins.config_reader import ConfigReader
from simba.train_model_functions import get_model_info
import os, glob
from copy import deepcopy
import multiprocessing
import cv2
import numpy as np
import functools
import platform
from simba.utils.errors import NoSpecifiedOutputError


def _multiprocess_sklearn_video(data: np.array,
                                video_path: str,
                                video_save_dir: str,
                                frame_save_dir: str,
                                clf_colors: list,
                                models_info: dict,
                                bp_dict: dict,
                                text_attr: dict,
                                rotate: bool,
                                print_timers: bool,
                                video_setting: bool,
                                frame_setting: bool,
                                pose_threshold: float):

    fourcc, font = cv2.VideoWriter_fourcc(*'mp4v'), cv2.FONT_HERSHEY_COMPLEX
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    group = data['group'].iloc[0]
    start_frm, current_frm, end_frm = data['index'].iloc[0], data['index'].iloc[0], data['index'].iloc[-1]

    if video_setting:
        video_save_path = os.path.join(video_save_dir, '{}.mp4'.format(str(group)))
        video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data['fps'], (video_meta_data['width'], video_meta_data['height']))

    cap.set(1, start_frm)
    while current_frm < end_frm:
        ret, img = cap.read()
        add_spacer = 2
        for animal_name, animal_data in bp_dict.items():
            animal_clr = animal_data['colors']
            id_flag_cords = None
            for bp_no in range(len(animal_data['X_bps'])):
                bp_clr = animal_clr[bp_no]
                x_bp, y_bp, p_bp = animal_data['X_bps'][bp_no], animal_data['Y_bps'][bp_no], animal_data['P_bps'][bp_no]
                bp_cords = data.loc[current_frm, [x_bp, y_bp, p_bp]]
                if bp_cords[p_bp] > pose_threshold:
                    cv2.circle(img, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), 0, bp_clr, text_attr['circle_scale'])
                    if ('centroid' in x_bp.lower()) or ('center' in x_bp.lower()):
                        id_flag_cords = (int(bp_cords[x_bp]), int(bp_cords[y_bp]))

            if not id_flag_cords:
                id_flag_cords = (int(bp_cords[x_bp]), int(bp_cords[y_bp]))
            cv2.putText(img, animal_name, id_flag_cords, font, text_attr['font_size'], animal_clr[0], text_attr['text_thickness'])
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if print_timers:
            cv2.putText(img, 'Timers', (10, text_attr['spacing_scale']), font, text_attr['font_size'], (0, 255, 0), text_attr['text_thickness'])
        frame_results = {}
        for model in models_info.values():
            frame_results[model['model_name']] = data.loc[current_frm, model['model_name']]
            if print_timers:
                cumulative_time = round(data.loc[current_frm, model['model_name'] + '_cumsum'] / video_meta_data['fps'], 3)
                cv2.putText(img, '{} {}s'.format(model['model_name'], str(cumulative_time)), (10, text_attr['spacing_scale'] * add_spacer), font, text_attr['font_size'], (255, 0, 0), text_attr['text_thickness'])
                add_spacer += 1

        cv2.putText(img, 'Ensemble prediction', (10, int(text_attr['spacing_scale'] * add_spacer)), font, text_attr['font_size'], (0, 255, 0), text_attr['text_thickness'])
        add_spacer += 1
        for clf_cnt, (clf_name, clf_results) in enumerate(frame_results.items()):
            if clf_results == 1:
                cv2.putText(img, clf_name, (10, int(text_attr['spacing_scale'] * add_spacer)), font, text_attr['font_size'], clf_colors[clf_cnt], text_attr['text_thickness'])
                add_spacer += 1
        if video_setting:
            video_writer.write(img)
        if frame_setting:
            frame_save_name = os.path.join(frame_save_dir, '{}.png'.format(str(current_frm)))
            cv2.imwrite(frame_save_name, img)
        current_frm += 1
        print('Multi-processing video frame {} on core {}...'.format(str(current_frm), str(group)))

    cap.release()
    if video_setting:
        video_writer.release()

    return group


class PlotSklearnResultsMultiProcess(ConfigReader):
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
       path to video file to create classification visualizations for
    cores: int
        Number of cores to use

    Notes
    ----------
    `Scikit visualization documentation <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-10-sklearn-visualization__.

    Examples
    ----------
    >>> clf_plotter = PlotSklearnResults(config_path='MyProjectConfig', video_setting=True, frame_setting=False, rotate=False, video_file_path='VideoPath', cores=5)
    >>> clf_plotter.initialize_visualizations()
    """

    def __init__(self,
                 config_path: str,
                 video_setting: bool,
                 frame_setting: bool,
                 text_settings: dict or bool,
                 cores: int,
                 rotate: False,
                 video_file_path=None,
                 print_timers=True):

        super().__init__(config_path=config_path)
        if platform.system() == "Darwin":
            multiprocessing.set_start_method('spawn', force=True)

        if (not video_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(msg='SIMBA ERROR: Please choose to create a video and/or frames. SimBA found that you ticked neither video and/or frames')
        self.video_file_path, self.print_timers, self.text_settings = video_file_path, print_timers, text_settings
        self.video_setting, self.frame_setting, self.cores = video_setting, frame_setting, cores
        if video_file_path is not None:
            check_file_exist_and_readable(os.path.join(self.video_dir, video_file_path))
        if not os.path.exists(self.sklearn_plot_dir): os.makedirs(self.sklearn_plot_dir)
        self.pose_threshold = read_config_entry(self.config, ReadConfig.THRESHOLD_SETTINGS.value, ReadConfig.SKLEARN_BP_PROB_THRESH.value, Dtypes.FLOAT.value, 0.00)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.color_lst_of_lst = createColorListofList(self.animal_cnt, int(len(self.x_cols) + 1))
        self.files_found = glob.glob(self.machine_results_dir + '/*.' + self.file_type)
        self.model_dict = get_model_info(self.config, self.model_cnt)
        self.clf_colors = create_single_color_lst(pallete_name ='Set1', increments=self.model_cnt+3)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_list, self.animal_cnt, self.x_cols, self.y_cols, self.pcols, self.color_lst_of_lst)
        self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        self.rotate = rotate

    def __get_print_settings(self):
        self.text_attr = {}
        if self.text_settings is False:
            self.space_scale, self.radius_scale, self.res_scale, self.font_scale = 60, 12, 1500, 1.1
            self.max_dim = max(self.video_meta_data['width'], self.video_meta_data['height'])
            self.text_attr['circle_scale'] = int(self.radius_scale / (self.res_scale / self.max_dim))
            self.text_attr['font_size'] = float(self.font_scale / (self.res_scale / self.max_dim))
            self.text_attr['spacing_scale'] = int(self.space_scale / (self.res_scale / self.max_dim))
            self.text_attr['text_thickness'] = 2
        else:
            check_float(name='ERROR: TEXT SIZE', value=self.text_settings['font_size'])
            check_int(name='ERROR: SPACE SIZE', value=self.text_settings['space_size'])
            check_int(name='ERROR: TEXT THICKNESS', value=self.text_settings['text_thickness'])
            check_int(name='ERROR: CIRCLE SIZE', value=self.text_settings['circle_size'])
            self.text_attr['font_size'] = float(self.text_settings['font_size'])
            self.text_attr['spacing_scale'] = int(self.text_settings['space_size'])
            self.text_attr['text_thickness'] = int(self.text_settings['text_thickness'])
            self.text_attr['circle_scale'] = int(self.text_settings['circle_size'])

    def __index_df_for_multiprocessing(self, data: list) -> list:
        for cnt, df in enumerate(data):
            df['group'] = cnt
        return data

    def create_visualizations(self):
        video_timer = SimbaTimer()
        video_timer.start_timer()
        _, self.video_name, _ = get_fn_ext(self.file_path)
        self.data_df = read_df(self.file_path, self.file_type).reset_index(drop=True)
        self.video_settings, _, self.fps = read_video_info(self.vid_info_df, self.video_name)
        self.video_path = find_video_of_file(self.video_dir, self.video_name)
        self.video_meta_data = get_video_meta_data(self.video_path)
        height, width = deepcopy(self.video_meta_data['height']), deepcopy(self.video_meta_data['width'])
        self.video_frame_dir, self.video_temp_dir = None, None
        if self.frame_setting:
            self.video_frame_dir = os.path.join(self.sklearn_plot_dir, self.video_name)
            if not os.path.exists(self.video_frame_dir): os.makedirs(self.video_frame_dir)
        if self.video_setting:
            self.video_save_path = os.path.join(self.sklearn_plot_dir, self.video_name + '.mp4')
            self.video_temp_dir = os.path.join(self.sklearn_plot_dir, self.video_name,  'temp')
            if not os.path.exists(self.video_temp_dir): os.makedirs(self.video_temp_dir)
        if self.rotate:
            self.video_meta_data['height'], self.video_meta_data['width'] = width, height
        self.__get_print_settings()

        for model in self.model_dict.values():
            self.data_df[model['model_name'] + '_cumsum'] = self.data_df[model['model_name']].cumsum()
        self.data_df['index'] = self.data_df.index
        data = np.array_split(self.data_df, self.cores)
        frm_per_core = data[0].shape[0]

        data = self.__index_df_for_multiprocessing(data=data)
        with multiprocessing.Pool(self.cores, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_multiprocess_sklearn_video,
                                          clf_colors=self.clf_colors,
                                          bp_dict=self.animal_bp_dict,
                                          video_save_dir=self.video_temp_dir,
                                          frame_save_dir=self.video_frame_dir,
                                          models_info=self.model_dict,
                                          text_attr=self.text_attr,
                                          rotate=self.rotate,
                                          video_path=self.video_path,
                                          print_timers=self.print_timers,
                                          video_setting=self.video_setting,
                                          frame_setting=self.frame_setting,
                                          pose_threshold=self.pose_threshold)

            for cnt, result in enumerate(pool.imap(constants, data, chunksize=self.multiprocess_chunksize)):
                print('Image {}/{}, Video {}/{}...'.format(str(int(frm_per_core * (result+1))), str(len(self.data_df)), str(self.file_cnt + 1), str(len(self.files_found))))
            if self.video_setting:
                print('Joining {} multiprocessed video...'.format(self.video_name))
                concatenate_videos_in_folder(in_folder=self.video_temp_dir, save_path=self.video_save_path)
            video_timer.stop_timer()
            pool.terminate()
            pool.join()
            print('Video {} complete (elapsed time: {}s)...'.format(self.video_name, video_timer.elapsed_time_str))

    def initialize_visualizations(self):
        if self.video_file_path is None:
            print('Processing {} videos...'.format(str(len(self.files_found))))
            for file_cnt, file_path in enumerate(self.files_found):
                self.file_cnt, self.file_path = file_cnt, file_path
                self.create_visualizations()
        else:
            print('Processing 1 video...')
            self.file_cnt, file_path = 0, self.video_file_path
            _, file_name, _ = get_fn_ext(file_path)
            self.file_path = os.path.join(self.machine_results_dir, file_name + '.' + self.file_type)
            self.files_found = [self.file_path]
            check_file_exist_and_readable(self.file_path)
            self.create_visualizations()

        self.timer.stop_timer()
        if self.video_setting:
            print('SIMBA COMPLETE: {} videos saved in project_folder/frames/output/sklearn_results directory (elapsed time: {}s)'.format(len(self.files_found), self.timer.elapsed_time_str))
        if self.frame_setting:
            print('SIMBA COMPLETE: Frames for {} videos saved in sub-folders within project_folder/frames/output/sklearn_results directory. (elapsed time: {}s)'.format(len(self.files_found), self.timer.elapsed_time_str))




# test = PlotSklearnResultsMultiProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/project_config.ini',
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       video_file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/videos/Together_1.avi',
#                                       print_timers=True,
#                                       text_settings=False,
# cores=6,
#                                       rotate=False)
# test.initialize_visualizations()


# test = PlotSklearnResultsMultiProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       video_file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.avi',
#                                       print_timers=True,
#                                       text_settings=False,
#                                       cores=6,
#                                       rotate=False)
# test.initialize_visualizations()


# test = PlotSklearnResultsMultiProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/Rat_NOR_BAD/project_folder/project_config.ini',
#                                       video_setting=True,
#                                       frame_setting=False,
#                                       cores=6,
#                                       video_file_path='/Users/simon/Desktop/envs/troubleshooting/Rat_NOR_BAD/project_folder/videos/03142021_NOB_DOT_5_upsidedown.mp4',
#                                       print_timers=True,
#                                       text_settings=False,
#                                       rotate=False)
# test.initialize_visualizations()