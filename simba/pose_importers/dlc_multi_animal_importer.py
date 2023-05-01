__author__ = "Simon Nilsson"

import os, glob
from datetime import datetime
import itertools
import pandas as pd
import numpy as np
import cv2
from copy import deepcopy

from simba.data_processors.interpolation_smoothing import Smooth, Interpolate
from simba.utils.errors import InvalidFilepathError, NoFilesFoundError, BodypartColumnNotFoundError, InvalidInputError
from simba.utils.warnings import InValidUserInputWarning, InvalidValueWarning
from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.read_write import read_df, write_df, read_config_entry, get_video_meta_data, get_fn_ext, find_all_videos_in_project
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.printing import stdout_success
from simba.utils.enums import Paths, ConfigKey, Dtypes, Formats

class MADLC_Importer(ConfigReader, PlottingMixin):
    """
    Class for importing multi-animal deeplabcut (maDLC) pose-estimation data (in H5 format)
    into a SimBA project in parquet or CSV format.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    data_folder: str
        Path to folder containing maDLC data in `.h5` format.
    file_type: str
        Method used to perform pose-estimation in maDLC. OPTIONS: `skeleton`, `box`, `ellipse`.
    id_lst: list
        Animal names.
    interpolation_settings: str
        String defining the pose-estimation interpolation method. OPTIONS: 'None', 'Animal(s): Nearest',
        'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear',
        'Body-parts: Quadratic'.
    smoothing_settings: dict
        Dictionary defining the pose estimation smoothing method. EXAMPLE: {'Method': 'Savitzky Golay',
        'Parameters': {'Time_window': '200'}})

    Notes
    -----
    `Multi-animal import tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md>`__.

    Examples
    -----
    >>> madlc_importer =MADLC_Importer(config_path=r'MyConfigPath', data_folder=r'maDLCDataFolder', file_type='ellipse', id_lst=['Animal_1', 'Animal_2'], interpolation_settings='None', smoothing_settings={'Method': 'None', 'Parameters': {'Time_window': '200'}})
    >>> madlc_importer.run()

    References
    ----------
    .. [1] Lauer et al., Multi-animal pose estimation, identification and tracking with DeepLabCut, `Nature Methods`,
           2022.
    """


    def __init__(self,
                 config_path: str,
                 data_folder: str,
                 file_type: str,
                 id_lst: list,
                 interpolation_settings: str,
                 smoothing_settings: dict):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PlottingMixin.__init__(self)

        self.interpolation_settings, self.smoothing_settings = interpolation_settings, smoothing_settings
        self.input_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(self.logs_path, f'data_import_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        self.videos_in_project = find_all_videos_in_project(videos_dir=self.video_dir)
        self.videos_in_project_lower_case = [os.path.basename(x).lower() for x in self.videos_in_project]
        self.save_folder = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        self.pose_setting = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, ConfigKey.POSE_SETTING.value, data_type=Dtypes.STR.value)
        if file_type == 'skeleton':
            dlc_file_ending, dlc_filtered_file_ending = 'sk.h5', 'sk_filtered.h5'
        elif file_type == 'box':
            dlc_file_ending, dlc_filtered_file_ending = 'bx.h5', 'bx_filtered.h5'
        elif file_type == 'ellipse':
            dlc_file_ending, dlc_filtered_file_ending = 'el.h5', 'el_filtered.h5'
        else:
            raise InvalidInputError(f'DLC FILETYPE {file_type} NOT SUPPORTED')
        self.files_found = glob.glob(self.input_folder + '/*' + dlc_file_ending) + glob.glob(self.input_folder + '/*' + dlc_filtered_file_ending)
        self.files_in_folder = glob.glob(self.input_folder + '/*')
        if not self.multi_animal_status:
            self.config.set('Multi animal IDs', 'id_list', '')
            self.id_lst = ['Animal_1']
            with open(self.config_path, 'w') as configfile:
                self.config.write(configfile)
        self.split_file_exts = list(itertools.product(*[Formats.DLC_NETWORK_FILE_NAMES.value, ['.mp4', '.avi']]))
        self.space_scaler, self.radius_scaler, self.res_scaler, self.font_scaler = 40, 10, 1500, 1.2
        self.bp_lst = []
        for animal in self.animal_bp_dict.keys():
            for currXcol, currYcol, currPcol in zip(self.animal_bp_dict[animal]['X_bps'], self.animal_bp_dict[animal]['Y_bps'], self.animal_bp_dict[animal]['P_bps']):
                self.bp_lst.extend((animal + '_' + currXcol, animal + '_' + currYcol, animal + '_' + currPcol))

        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Found 0 files in {} path that satisfy the criterion for maDLC {} filetype. SimBA detected {} other files within in directory'.format(self.input_folder, file_type, str(len(self.files_in_folder))))
        print('Importing {} file(s)...'.format(str(len(self.files_found))))

    def __find_video_file(self):
        assessed_file_paths, self.video_path = [], None
        for combination in self.split_file_exts:
            possible_vid_name = self.file_name.lower().split(combination[0])[0] + combination[1]
            for video_cnt, video_name in enumerate(self.videos_in_project_lower_case):
                if possible_vid_name == video_name:
                    self.video_path = self.videos_in_project[video_cnt]
                else:
                    assessed_file_paths.append(possible_vid_name)
        if self.video_path is None:
            print(assessed_file_paths)
            raise NoFilesFoundError(msg=f'SimBA searched your project_folder/videos directory for a video file representing {self.file_name}, and could not find a match. Above is a list of possible video filenames that SimBA searched for within your projects video directory without success.')
        else:
             _, self.video_basename, _ = get_fn_ext(self.video_path)

    def __insert_all_bps(self, frame=None):
        for animal, bp_data in self.img_bp_cords_dict.items():
            for bp_cnt, bp_tuple in enumerate(bp_data):
                try:
                    cv2.circle(frame, bp_tuple, self.vid_circle_scale, self.animal_bp_dict[animal]['colors'][bp_cnt], -1, lineType=cv2.LINE_AA)
                except Exception as err:
                    if type(err) == OverflowError:
                        InvalidValueWarning(f'SimBA encountered a pose-estimated body-part located at pixel position {str(bp_tuple)}. '
                              'This value is too large to be converted to an integer. '
                              'Please check your pose-estimation data to make sure that it is accurate.')
                    print(err.args)

    def __create_first_side_img(self):
        side_img = np.ones((int(self.video_info['height'] / 2), self.video_info['width'], 3))
        cv2.putText(side_img, 'Current video: ' + self.video_basename, (10, self.vid_space_scale), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 3)
        cv2.putText(side_img, 'Can you assign identities based on the displayed frame ?', (10, int(self.vid_space_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 2)
        cv2.putText(side_img, 'Press "x" to display new, random, frame', (10, int(self.vid_space_scale * (self.add_spacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 0), 3)
        cv2.putText(side_img, 'Press "c" to continue to start assigning identities using this frame', (10, int(self.vid_space_scale * (self.add_spacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (0, 255, 0), 2)
        self.img_concat = np.uint8(np.concatenate((self.img_overlay, side_img), axis=0))

    def create_choose_animals_side_img(self, animal_id):
        self.side_img = np.ones((int(self.video_info['height'] / 2), self.video_info['width'], 3))
        cv2.putText(self.side_img, 'Double left mouse click on:', (10, self.vid_space_scale), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, animal_id, (10, int(self.vid_space_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 0), 2)
        self.img_concat = np.uint8(np.concatenate((self.img_overlay, self.side_img), axis=0))

    def __initiate_choose_frame(self):
        cv2.destroyAllWindows()
        self.cap.set(1, self.frame_no)
        self.all_frame_data = self.data_df.loc[self.frame_no, :]
        cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
        self.img_bp_cords_dict = {}
        ret, self.img = self.cap.read()
        self.img_overlay = deepcopy(self.img)
        for animal_cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
            self.img_bp_cords_dict[animal_name] = []
            for bp_cnt in range(len(animal_bps['X_bps'])):
                x_cord = int(self.data_df.loc[self.frame_no, animal_name + '_' + animal_bps['X_bps'][bp_cnt]])
                y_cord = int(self.data_df.loc[self.frame_no, animal_name + '_' + animal_bps['Y_bps'][bp_cnt]])
                self.img_bp_cords_dict[animal_name].append((x_cord, y_cord))
        self.__insert_all_bps(frame=self.img_overlay)
        self.__create_first_side_img()
        cv2.imshow('Define animal IDs', self.img_concat)
        cv2.resizeWindow('Define animal IDs', self.video_info['height'], self.video_info['width'])

        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(20)
            if k == ord('x'):
                cv2.destroyWindow('Define animal IDs')
                cv2.waitKey(0)
                self.frame_no += 50
                self.__initiate_choose_frame()
                break
            elif k == ord('c'):
                cv2.destroyWindow('Define animal IDs')
                cv2.waitKey(0)
                self.__initiate_choose_animals()
                break

    def __get_x_y_loc(self, event, x, y, flags, param):
        if event == 7:
            self.click_loc = (x,y)
            self.ID_cords[self.animal_cnt] = {}
            self.ID_cords[self.animal_cnt]['cord'] = self.click_loc
            self.ID_cords[self.animal_cnt]['name'] = self.animal_name

    def __insert_all_animal_names(self):
        for animal_cnt, animal_data in self.ID_cords.items():
            cv2.putText(self.new_frame, animal_data['name'], animal_data['cord'], cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 3)

    def __initiate_choose_animals(self):
        self.ID_cords = {}
        for animal_cnt, animal in enumerate(self.animal_bp_dict.keys()):
            self.new_overlay = deepcopy(self.img_overlay)
            cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
            self.animal_name = animal
            self.animal_cnt = animal_cnt
            self.side_img = np.ones((int(self.video_info['height'] / 2), self.video_info['width'], 3))
            cv2.putText(self.side_img, 'Double left mouse click on:', (10, self.vid_space_scale), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 3)
            cv2.putText(self.side_img, animal, (10, int(self.vid_space_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 0), 3)
            for id in self.ID_cords.keys():
                cv2.putText(self.new_overlay, self.ID_cords[id]['name'], self.ID_cords[id]['cord'], cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 3)
            self.new_overlay = np.uint8(np.concatenate((self.new_overlay, self.side_img), axis=0))
            cv2.imshow('Define animal IDs', self.new_overlay)
            cv2.resizeWindow('Define animal IDs', self.video_info['height'], self.video_info['width'])
            while animal_cnt not in self.ID_cords.keys():
                cv2.setMouseCallback('Define animal IDs', self.__get_x_y_loc)
                cv2.waitKey(200)
        self.__initiate_confirm()

    def __initiate_confirm(self):
        cv2.destroyAllWindows()
        cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Define animal IDs', self.video_info['height'], self.video_info['width'])
        self.new_frame = deepcopy(self.img)
        self.side_img = np.ones((int(self.video_info['height'] / 2), self.video_info['width'], 3))
        cv2.putText(self.side_img, 'Current video: {}'.format(self.video_basename), (10, int(self.vid_space_scale)), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 3)
        cv2.putText(self.side_img, 'Are you happy with your assigned identities ?', (10, int(self.vid_space_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, 'Press "c" to continue (to finish, or proceed to the next video)', (10, int(self.vid_space_scale * (self.add_spacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 0), 2)
        cv2.putText(self.side_img, 'Press "x" to re-start assigning identities', (10, int(self.vid_space_scale * (self.add_spacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (0, 255, 255), 2)
        self.__insert_all_bps(frame=self.new_frame)
        self.__insert_all_animal_names()
        self.img_concat = np.uint8(np.concatenate((self.new_frame, self.side_img), axis=0))
        cv2.imshow('Define animal IDs', self.img_concat)
        cv2.resizeWindow('Define animal IDs', self.video_info['height'], self.video_info['width'])
        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(20)
            if k == ord('x'):
                cv2.destroyWindow('Define animal IDs')
                cv2.waitKey(0)
                self.frame_no += 50
                self.__initiate_choose_frame()
                break
            elif k == ord('c'):
                cv2.destroyAllWindows()
                cv2.waitKey(0)
                break

    def __check_intergity_of_order(self):
        for click_key_combination in itertools.combinations(list(self.animal_order.keys()), 2):
            click_n, click_n1 = click_key_combination[0], click_key_combination[1]
            animal_1, animal_2 = self.animal_order[click_n]['animal_name'], self.animal_order[click_n1]['animal_name']
            if animal_1 == animal_2:
                InValidUserInputWarning(msg=f'The animal most proximal to click number {str(click_n)} is animal named {animal_1}. The animal most proximal to click number {str(click_n1)} is also animal {animal_2}.'
                      'Please indicate which animal is which using a video frame where the animals are clearly separated')
            else:
                pass

    def __find_closest_animals(self):
        self.animal_order = {}
        for animal_number, animal_click_data in self.ID_cords.items():
            animal_name, animal_cord = animal_click_data['name'], animal_click_data['cord']
            closest_animal = {}
            closest_animal['animal_name'] = None
            closest_animal['body_part_name'] = None
            closest_animal['distance'] = np.inf
            for other_animal_name, animal_bps in self.animal_bp_dict.items():
                animal_bp_names_x = self.animal_bp_dict[other_animal_name]['X_bps']
                animal_bp_names_y = self.animal_bp_dict[other_animal_name]['Y_bps']
                for x_col, y_col in zip(animal_bp_names_x, animal_bp_names_y):
                    bp_location = (int(self.all_frame_data['{}_{}'.format(other_animal_name, x_col)]), int(self.all_frame_data['{}_{}'.format(other_animal_name, y_col)]))
                    distance = np.sqrt((animal_cord[0] - bp_location[0]) ** 2 + (animal_cord[1] - bp_location[1]) ** 2)
                    if distance < closest_animal['distance']:
                        closest_animal['animal_name'] = other_animal_name
                        closest_animal['body_part_name'] = (x_col, y_col)
                        closest_animal['distance'] = distance
            self.animal_order[animal_number] = closest_animal
        self.__check_intergity_of_order()

    def __organize_df(self):
        self.out_df = pd.DataFrame()
        for animal_cnt, animal_data in self.animal_order.items():
            closest_animal_dict = self.animal_bp_dict[animal_data['animal_name']]
            x_cols, y_cols, p_cols = closest_animal_dict['X_bps'], closest_animal_dict['Y_bps'], closest_animal_dict['P_bps']
            x_cols = [animal_data['animal_name'] + '_' + x for x in x_cols]
            y_cols = [animal_data['animal_name'] + '_' + x for x in y_cols]
            p_cols = [animal_data['animal_name'] + '_' + x for x in p_cols]
            for x_col, y_col, p_cols in zip(x_cols, y_cols, p_cols):
                df = self.data_df[[x_col, y_col, p_cols]]
                self.out_df = pd.concat([self.out_df, df], axis=1)

        self.out_df.columns = self.bp_lst
        print(self.animal_bp_dict)
        for animal_name in self.id_lst:
            for x_col, y_col in zip(self.animal_bp_dict[animal_name]['X_bps'], self.animal_bp_dict[animal_name]['Y_bps']):
                self.out_df['{}_{}'.format(animal_name, x_col)] = self.out_df['{}_{}'.format(animal_name, x_col)].astype(int)
                self.out_df['{}_{}'.format(animal_name, y_col)] = self.out_df['{}_{}'.format(animal_name, y_col)].astype(int)

    def __insert_multi_idx_header(self):
        multi_idx_cols = []
        for col_idx in range(len(self.out_df.columns)):
            multi_idx_cols.append(tuple(('DLC_multi', 'DLC_multi', self.out_df.columns[col_idx])))
        self.out_df.columns = pd.MultiIndex.from_tuples(multi_idx_cols, names=('scorer', 'bodypart', 'coords'))

    def __save_df(self):
        self.save_path = os.path.join(os.path.join(self.save_folder, f'{self.video_basename }.{self.file_type}'))
        write_df(df=self.out_df, file_type=self.file_type, save_path=self.save_path, multi_idx_header=True)

    def __run_interpolation(self):
        print('Interpolating missing values in video {} (Method: {}) ...'.format(self.video_basename, self.interpolation_settings))
        _ = Interpolate(input_path=self.save_path,config_path=self.config_path, method=self.interpolation_settings, initial_import_multi_index=True)


    def __run_smoothing(self):
        print(f'Performing {self.smoothing_settings["Method"]} smoothing on video {self.video_basename}...')
        Smooth(config_path=self.config_path,
               input_path=self.save_path,
               time_window=int(self.smoothing_settings['Parameters']['Time_window']),
               smoothing_method=self.smoothing_settings['Method'],
               initial_import_multi_index=True)

    def run(self):
        """
        Method for initializing maDLC importing GUI.

        Returns
        ----------
        None

        """

        import_log = pd.DataFrame(columns=['VIDEO', 'IMPORT_TIME', 'IMPORT_SOURCE', 'INTERPOLATION_SETTING', 'SMOOTHING_SETTING'])
        for file_cnt, file_path in enumerate(self.files_found):
            self.add_spacer = 2
            _, self.file_name, _ = get_fn_ext(file_path)
            print('Processing file {} ...'.format(self.file_name))
            self.__find_video_file()
            self.data_df = pd.read_hdf(file_path).replace([np.inf, -np.inf], np.nan).fillna(0)
            try:
                self.data_df.columns = self.bp_lst
            except ValueError as err:
                raise BodypartColumnNotFoundError(msg=f'SIMBA ERROR: The number of body-parts in the input file {file_path} do not match the number of body-parts in your SimBA project. '
                      f'The number of of body-parts expected by your SimBA project is {str(len(self.x_cols))}. '
                      f'The number of of body-parts contained in file {file_path} is {str(int(len(self.data_df.columns) / 3))}. '
                      f'Make sure you have specified the correct number of animals and body-parts in your project.')
            self.video_info = get_video_meta_data(self.video_path)
            self.max_video_dimension = max(self.video_info['width'], self.video_info['height'])
            self.vid_circle_scale = int(self.radius_scaler / (self.res_scaler / self.max_video_dimension))
            self.vid_font_scale = float(self.font_scaler / (self.res_scaler / self.max_video_dimension))
            self.vid_space_scale = int(self.space_scaler / (self.res_scaler / self.max_video_dimension))
            self.frame_no = 1
            self.cap = cv2.VideoCapture(self.video_path)
            self.__initiate_choose_frame()
            self.cap.release()
            self.__find_closest_animals()
            self.__organize_df()
            self.__insert_multi_idx_header()
            self.__save_df()
            if self.interpolation_settings != 'None':
                self.__run_interpolation()
            if self.smoothing_settings['Method'] != 'None':
                self.__run_smoothing()
            import_log.loc[len(import_log)] = [self.file_name,
                                               datetime.now().strftime('%Y%m%d%H%M%S'),
                                               'MADLC',
                                               str(self.interpolation_settings),
                                               str(self.smoothing_settings)]
            print('SimBA import of file {} complete!'.format(self.file_name))

        import_log.to_csv(self.import_log_path)
        stdout_success(msg=f'{str(len(self.files_found))} files imported to your SimBA project. Imported files are located in the project_folder/csv/input_csv directory.')

# test = MADLC_Importer(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/h5',
#                    file_type='ellipse',
#                    id_lst=['Simon', 'JJ'],
#                    interpolation_settings='Body-parts: Nearest',
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()



# test = MADLC_Importer(config_path=r'/Users/simon/Desktop/troubleshooting/B1-MS_US/project_folder/project_config.ini',
#                    data_folder=r'/Users/simon/Desktop/troubleshooting/B1-MS_US/el_import/other_2',
#                    file_type='ellipse',
#                    id_lst=['MS', 'US'],
#                    interpolation_settings='None',
#                    smoothing_settings={'Method': 'None', 'Parameters': {'Time_window': '200'}})
# test.import_data()

# test = MADLC_Importer(config_path=r'/Users/simon/Desktop/troubleshooting/Soong/project_folder/project_config.ini',
#                    data_folder=r'/Users/simon/Desktop/troubleshooting/Soong/import',
#                    file_type='ellipse',
#                    id_lst=['Animal1'],
#                    interpolation_settings='None',
#                    smoothing_settings={'Method': 'None', 'Parameters': {'Time_window': '200'}})
# test.import_data()
