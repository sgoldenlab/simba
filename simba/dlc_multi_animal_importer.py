__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
import os, glob
from simba.misc_tools import (check_multi_animal_status,
                              get_video_meta_data,
                              smooth_data_gaussian,
                              smooth_data_savitzky_golay)
from simba.drop_bp_cords import (getBpNames,
                                 get_fn_ext,
                                 createColorListofList,
                                 create_body_part_dictionary)
from simba.enums import Paths, ReadConfig, Dtypes
from datetime import datetime
import itertools
import pandas as pd
import numpy as np
import cv2
from simba.interpolate_pose import Interpolate
from copy import deepcopy
import pyarrow.parquet as pq
import pyarrow

class MADLC_Importer(object):
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
        List of animal names.
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
    >>> madlc_importer.import_data()

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

        self.config_path = config_path
        self.config = read_config_file(config_path)
        self.interpolation_settings = interpolation_settings
        self.smoothing_settings = smoothing_settings
        self.input_folder = data_folder
        self.id_lst = id_lst
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.animal_cnt = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.video_folder = os.path.join(self.project_path, 'videos')
        self.import_log_path = os.path.join(self.project_path, 'logs', f'data_import_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        self.videos_in_project = self.__find_all_videos_in_project(self.video_folder)
        self.videos_in_project_lower_case = [os.path.basename(x).lower() for x in self.videos_in_project]
        self.save_folder = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        self.pose_setting = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.POSE_SETTING.value, data_type=Dtypes.STR.value)
        if file_type == 'skeleton': dlc_file_ending, dlc_filtered_file_ending = 'sk.h5', 'sk_filtered.h5'
        elif file_type == 'box': dlc_file_ending, dlc_filtered_file_ending = 'bx.h5', 'bx_filtered.h5'
        elif file_type == 'ellipse': dlc_file_ending, dlc_filtered_file_ending = 'el.h5', 'el_filtered.h5'
        else: raise ValueError('SIMBA ERROR: DLC FILETYPE {} NOT SUPPORTED'.format(file_type))
        self.files_found = glob.glob(self.input_folder + '/*' + dlc_file_ending) + glob.glob(self.input_folder + '/*' + dlc_filtered_file_ending)
        self.files_in_folder = glob.glob(self.input_folder + '/*')
        self.multi_animal_status, self.multi_animal_ids = check_multi_animal_status(self.config, self.animal_cnt)
        if not self.multi_animal_status:
            self.config.set('Multi animal IDs', 'id_list', '')
            self.id_lst = ['Animal_1']
            with open(self.config_path, 'w') as configfile:
                self.config.write(configfile)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.clr_lst_of_lst = createColorListofList(self.animal_cnt, int(len(self.x_cols) / self.animal_cnt) + 1)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_ids, self.animal_cnt, self.x_cols, self.y_cols, self.pcols, self.clr_lst_of_lst)
        self.split_file_exts = list(itertools.product(*[['dlc_resnet50', 'dlc_resnet_50', 'dlc_dlcrnetms5', 'dlc_effnet_b0', 'dlc_resnet101'], ['.mp4', '.avi']]))
        self.space_scaler, self.radius_scaler, self.res_scaler, self.font_scaler = 40, 10, 1500, 1.2
        self.bp_lst = []
        for animal in self.animal_bp_dict.keys():
            for currXcol, currYcol, currPcol in zip(self.animal_bp_dict[animal]['X_bps'], self.animal_bp_dict[animal]['Y_bps'], self.animal_bp_dict[animal]['P_bps']):
                self.bp_lst.extend((animal + '_' + currXcol, animal + '_' + currYcol, animal + '_' + currPcol))

        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Found 0 files in {} path that satisfy the criterion for maDLC {} filetype. SimBA detected {} other files within in directory'.format(self.input_folder, file_type, str(len(self.files_in_folder))))
        print('Importing {} file(s)...'.format(str(len(self.files_found))))

    def __find_all_videos_in_project(self, folder_path=None):
        video_paths = []
        file_paths_in_folder = [f for f in next(os.walk(folder_path))[2] if not f[0] == '.']
        file_paths_in_folder = [os.path.join(self.video_folder, f) for f in file_paths_in_folder]
        for file_cnt, file_path in enumerate(file_paths_in_folder):
            try:
                _, file_name, file_ext = get_fn_ext(file_path)
            except ValueError:
                print('SIMBA ERROR: {} is not a valid filepath'.format(file_path))
                raise ValueError('SIMBA ERROR: {} is not a valid filepath'.format(file_path))
            if (file_ext.lower() == '.mp4') or (file_ext.lower() == '.avi'):
                video_paths.append(file_path)
        if len(video_paths) == 0:
            print('SIMBA ERROR: No videos in mp4 or avi format imported to SimBA project')
            raise FileNotFoundError
        else:
            return video_paths

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
            print('SimBA ERROR: SimBA searched your project_folder/videos directory for a video file representing {}, and could not find a match. Above is a list of possible video filenames that SimBA searched for within your projects video directory without success.'.format(self.file_name))
            raise AttributeError()
        else:
             _, self.video_basename, _ = get_fn_ext(self.video_path)

    def __insert_all_bps(self, frame=None):
        for animal, bp_data in self.img_bp_cords_dict.items():
            for bp_cnt, bp_tuple in enumerate(bp_data):
                try:
                    cv2.circle(frame, bp_tuple, self.vid_circle_scale, self.animal_bp_dict[animal]['colors'][bp_cnt], -1, lineType=cv2.LINE_AA)
                except Exception as err:
                    if type(err) == OverflowError:
                        print('SIMBA ERROR: SimBA encountered a pose-estimated body-part located at pixel position {}. '
                              'This value is too large to be converted to an integer. '
                              'Please check your pose-estimation data to make sure that it is accurate.'.format(str(bp_tuple)))
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
                print('SIMBA ERROR: The animal most proximal to click number {} is animal named {}. The animal most proximal to click number {} is also animal {}.'
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
        save_name = os.path.join(self.video_basename + '.' + self.file_type)
        self.save_path = os.path.join(self.save_folder, save_name)
        if self.file_type == 'parquet':
            table = pyarrow.Table.from_pandas(self.out_df)
            pyarrow.parquet.write_table(table, self.save_path)
        if self.file_type == 'csv':
            self.out_df.to_csv(self.save_path)

    def __run_interpolation(self):
        print('Interpolating missing values in video {} (Method: {} ...'.format(self.video_basename, self.interpolation_settings))
        if self.file_type == 'parquet':
            data_df = pd.read_parquet(self.save_path)
        if self.file_type == 'csv':
            data_df = pd.read_csv(self.save_path, index_col=0)
        interpolate_body_parts = Interpolate(self.config_path, data_df)
        interpolate_body_parts.detect_headers()
        interpolate_body_parts.fix_missing_values(self.interpolation_settings)
        interpolate_body_parts.reorganize_headers()
        if self.file_type == 'parquet':
            table = pyarrow.Table.from_pandas(interpolate_body_parts.new_df)
            pyarrow.parquet.write_table(table, self.save_path)
        if self.file_type == 'csv':
            interpolate_body_parts.new_df.to_csv(self.save_path)

    def __run_smoothing(self):
        if self.smoothing_settings['Method'] == 'Gaussian':
            print('Performing Gaussian smoothing on video {}...'.format(self.video_basename))
            time_window = self.smoothing_settings['Parameters']['Time_window']
            smooth_data_gaussian(config=self.config, file_path=self.save_path, time_window_parameter=time_window)

        if self.smoothing_settings['Method'] == 'Savitzky Golay':
            print('Performing Savitzky Golay smoothing on video {}...'.format(self.video_basename))
            time_window = self.smoothing_settings['Parameters']['Time_window']
            smooth_data_savitzky_golay(config=self.config, file_path=self.save_path, time_window_parameter=time_window)

    def import_data(self):
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
                print(err)
                print('The number of body-parts in the input file {} do not match the number of body-parts in your SimBA project. '
                      'The number of of body-parts expected by your SimBA project is {}. '
                      'The number of of body-parts contained in file {} is {}. '
                      'Make sure you have specified the correct number of animals and body-parts in your project.'.format(file_path, str(len(self.x_cols)), file_path, str(len(self.data_df.columns))))
                raise ValueError
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
        print('SIMBA COMPLETE: {} files imported to your SimBA project. Imported files are located in the project_folder/csv/input_csv directory.'.format(str(len(self.files_found))))

# test = MADLC_Importer(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals/project_folder/project_config.ini',
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals/h5',
#                    file_type='ellipse',
#                    id_lst=['Simon', 'JJ'],
#                    interpolation_settings='Body-parts: Nearest',
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.import_data()



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
