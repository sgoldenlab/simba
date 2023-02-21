__author__ = "Simon Nilsson", "JJ Choong"

import cv2

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from collections import deque
from simba.misc_tools import (check_multi_animal_status,
                              get_fn_ext,
                              SimbaTimer,
                              get_color_dict)
from simba.enums import ReadConfig, Paths, Formats
from simba.misc_visualizations import make_path_plot
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.drop_bp_cords import getBpNames
from simba.rw_dfs import read_df
import numpy as np
import os



class PathPlotterSingleCore(object):
    """
    Class for creating "path plots" videos and/or images detailing the movement paths of
    individual animals in SimBA.

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format
    frame_setting: bool
        If True, individual frames will be created.
    video_setting: bool
        If True, compressed videos will be created.
    files_found: list
        Data paths to create from which to create plots
    animal_attr: dict
        Animal body-parts and colors
    style_attr: dict
        Plot sttributes

    Notes
    ----------
    `Visualization tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-11-visualizations>`__.

    Examples
    ----------
    >>> style_attr = {'width': 'As input', 'height': 'As input', 'line width': 5, 'font size': 5, 'font thickness': 2, 'circle size': 5, 'bg color': 'White', 'max lines': 100}
    >>> animal_attr = {0: ['Ear_right_1', 'Red']}
    >>> path_plotter = PathPlotter(config_path=r'MyConfigPath', frame_setting=False, video_setting=True, style_attr=style_attr, animal_attr=animal_attr, files_found=['project_folder/csv/machine_results/MyVideo.csv'])
    >>> path_plotter.create_path_plots()
    """


    def __init__(self,
                 config_path: str,
                 frame_setting: bool,
                 video_setting: bool,
                 last_frame: bool,
                 files_found: list,
                 style_attr: dict,
                 animal_attr: dict,
                 clf_attr: dict):

        self.video_setting, self.frame_setting, self.style_attr, self.files_found, self.animal_attr, self.clf_attr, self.last_frame = video_setting, frame_setting, style_attr, files_found, animal_attr, clf_attr, last_frame
        if (not frame_setting) and (not video_setting) and (not last_frame):
            print('SIMBA ERROR: Please choice to create path frames and/or video path plots')
            raise ValueError('SIMBA ERROR: Please choice to create path frames and/or video path plots')
        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.no_animals_path_plot = len(animal_attr.keys())
        self.save_folder = os.path.join(self.project_path, Paths.PATH_PLOT_DIR.value)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals_path_plot)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.font = Formats.FONT.value
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. To plot paths without performing machine classifications, use path plotter functions in [ROI] tab.')
        print('Processing {} videos...'.format(str(len(self.files_found))))

    def create_path_plots(self):
        """
        Method to create path plot videos and/or frames.Results are store in the
        'project_folder/frames/path_plots' directory of the SimBA project.
        """

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, self.video_name, _ = get_fn_ext(file_path)
            self.video_info, _, self.fps = read_video_info(self.vid_info_df, self.video_name)
            self.__get_styles()
            self.__get_deque_lookups()
            self.data_df = read_df(file_path, self.file_type)
            if self.video_setting:
                self.video_save_path = os.path.join(self.save_folder, self.video_name + '.mp4')
                self.fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
                self.writer = cv2.VideoWriter(self.video_save_path, self.fourcc, self.fps, (self.style_attr['width'], self.style_attr['height']))

            if self.frame_setting:
                self.save_video_folder = os.path.join(self.save_folder, self.video_name)
                if not os.path.exists(self.save_video_folder):
                    os.makedirs(self.save_video_folder)

            if self.clf_attr:
                clf_names = []
                for v in self.clf_attr.values():
                    clf_names.append(v[0])
                clf_df = self.data_df[clf_names]

            if self.last_frame:
                _ = make_path_plot(data_df=self.data_df,
                                   video_info=self.video_info,
                                   style_attr=self.style_attr,
                                   deque_dict=self.deque_dict,
                                   clf_attr=self.clf_attr,
                                   save_path=os.path.join(self.save_folder, self.video_name + '_final_frame.png'))

            if self.video_setting or self.frame_setting:
                for frm_cnt in range(len(self.data_df)):
                    img = np.zeros((int(self.video_info['Resolution_height'].values[0]), int(self.video_info['Resolution_width'].values[0]), 3))
                    img[:] = self.style_attr['bg color']
                    for animal_cnt, (animal_name, animal_data) in enumerate(self.deque_dict.items()):
                        bp_x = int(self.data_df.loc[frm_cnt, '{}_{}'.format(animal_data['bp'], 'x')])
                        bp_y = int(self.data_df.loc[frm_cnt, '{}_{}'.format(animal_data['bp'], 'y')])
                        self.deque_dict[animal_name]['deque'].appendleft((bp_x, bp_y))
                    for animal_name, animal_data in self.deque_dict.items():
                        cv2.circle(img, (self.deque_dict[animal_name]['deque'][0]), 0, self.deque_dict[animal_name]['clr'], self.style_attr['circle size'])
                        cv2.putText(img, animal_name, (self.deque_dict[animal_name]['deque'][0]), self.font, self.style_attr['font size'], self.deque_dict[animal_name]['clr'], self.style_attr['font thickness'])

                    for animal_name, animal_data in self.deque_dict.items():
                        for i in range(len(self.deque_dict[animal_name]['deque'])-1):
                            line_clr = self.deque_dict[animal_name]['clr']
                            position_1 = self.deque_dict[animal_name]['deque'][i]
                            position_2 = self.deque_dict[animal_name]['deque'][i+1]
                            cv2.line(img, position_1, position_2, line_clr, self.style_attr['line width'])

                    if self.clf_attr:
                        animal_1_name = list(self.deque_dict.keys())[0]
                        animal_bp_x, animal_bp_y = self.deque_dict[animal_1_name]['bp'] + '_x', self.deque_dict[animal_1_name]['bp'] + '_y'
                        for clf_cnt, clf_data in self.clf_attr.items():
                            clf_size = int(clf_data[-1].split(': ')[-1])
                            clf_clr = self.color_dict[clf_data[1]]
                            sliced_df = clf_df.loc[0: frm_cnt]
                            sliced_df_idx = list(sliced_df[sliced_df[clf_data[0]] == 1].index)
                            locations = self.data_df.loc[sliced_df_idx, [animal_bp_x, animal_bp_y]].astype(int).values
                            for i in range(locations.shape[0]):
                                cv2.circle(img, (locations[i][0], locations[i][1]), 0, clf_clr, clf_size)

                    img = cv2.resize(img, (self.style_attr['width'], self.style_attr['height']))
                    if self.video_setting:
                        self.writer.write(np.uint8(img))
                    if self.frame_setting:
                        frm_name = os.path.join(self.save_video_folder, str(frm_cnt) + '.png')
                        cv2.imwrite(frm_name, np.uint8(img))
                    print('Frame: {} / {}. Video: {} ({}/{})'.format(str(frm_cnt+1), str(len(self.data_df)),self.video_name, str(file_cnt + 1), len(self.files_found)))

                if self.video_setting:
                    self.writer.release()
                video_timer.stop_timer()
                print('Path visualization for video {} saved (elapsed time {}s)...'.format(self.video_name, video_timer.elapsed_time_str))

        self.timer.stop_timer()
        print('SIMBA COMPLETE: Path visualizations for {} videos saved in project_folder/frames/output/path_plots directory (elapsed time {}s)'.format(str(len(self.files_found)), self.timer.elapsed_time_str))

    def __get_styles(self):
        self.color_dict = get_color_dict()
        if self.style_attr is not None:
            self.style_attr['bg color'] = self.color_dict[self.style_attr['bg color']]
            self.style_attr['max lines'] = int(self.style_attr['max lines'] * (int(self.video_info['fps'].values[0]) / 1000))
            if self.style_attr['width'] == 'As input':
                self.style_attr['width'], self.style_attr['height'] = int(self.video_info['Resolution_width'].values[0]), int(self.video_info['Resolution_height'].values[0])
            else:
                pass
        else:
            self.style_attr = {}
            space_scaler, radius_scaler, res_scaler, font_scaler = 25, 10, 1500, 0.8
            self.style_attr['width'] = int(self.video_info['Resolution_width'].values[0])
            self.style_attr['height'] = int(self.video_info['Resolution_height'].values[0])
            max_res = max(self.style_attr['width'], self.style_attr['height'])
            self.style_attr['circle size'] = int(radius_scaler / (res_scaler / max_res))
            self.style_attr['font size'] = int(font_scaler / (res_scaler / max_res))
            self.style_attr['bg color'] = self.color_dict['White']
            self.style_attr['max lines'] = int(self.video_info['fps'].values[0] * 2)
            self.style_attr['font thickness'] = 2
            self.style_attr['line width'] = 2

    def __get_deque_lookups(self):
        self.deque_dict = {}
        for animal_cnt, animal in enumerate(self.multi_animal_id_lst):
            self.deque_dict[animal] = {}
            self.deque_dict[animal]['deque'] = deque(maxlen=self.style_attr['max lines'])
            self.deque_dict[animal]['bp'] = self.animal_attr[animal_cnt][0]
            self.deque_dict[animal]['clr'] = self.color_dict[self.animal_attr[animal_cnt][1]]


# style_attr = {'width': 'As input',
#               'height': 'As input',
#               'line width': 5,
#               'font size': 5,
#               'font thickness': 2,
#               'circle size': 5,
#               'bg color': 'White',
#               'max lines': 2000}
#
# animal_attr = {0: ['Ear_right_1', 'Red']}
# clf_attr = {0: ['Attack', 'Black', 'Size: 30'], 1: ['Sniffing', 'Red', 'Size: 30']}
#
#
# #style_attr = None
# test = PathPlotterSingleCore(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                    frame_setting=False,
#                    video_setting=False,
#                              last_frame=True,
#                    style_attr=style_attr,
#                    animal_attr=animal_attr,
#                              clf_attr=clf_attr,
#                     files_found=['/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/machine_results/Together_1.csv'])
# test.create_path_plots()