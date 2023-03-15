import pandas as pd
from simba.read_config_unit_tests import read_config_entry, read_config_file
import os, glob
from numba import jit, prange
import numpy as np
from copy import deepcopy
from simba.misc_tools import (find_video_of_file,
                              check_multi_animal_status,
                              smooth_data_gaussian,
                              smooth_data_savitzky_golay,
                              get_video_meta_data)
from simba.drop_bp_cords import (get_fn_ext,
                                 createColorListofList,
                                 create_body_part_dictionary,
                                 getBpNames,
                                 getBpHeaders)
from datetime import datetime
import itertools
import cv2
import pyarrow.parquet as pq
import pyarrow
from simba.interpolate_pose import Interpolate
from simba.utils.errors import NoFilesFoundError


class SleapCsvImporter(object):

    """
    Class for importing SLEAP pose-estimation data into SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    data_folder: str
        Path to folder containing SLEAP data in `.slp` format.
    actor_IDs: list
         Animal names.
    interpolation_settings: str
        String defining the pose-estimation interpolation method. OPTIONS: 'None', 'Animal(s): Nearest',
        'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear',
        'Body-parts: Quadratic'.
    smoothing_settings: dict
        Dictionary defining the pose estimation smoothing method. EXAMPLE: {'Method': 'Savitzky Golay',
        'Parameters': {'Time_window': '200'}})

    Notes
    ----------
    `Google Colab notebook for converting SLEAP .slp to CSV  <https://colab.research.google.com/drive/1EpyTKFHVMCqcb9Lj9vjMrriyaG9SvrPO?usp=sharing>`__.
    `Example expected SLEAP csv data file for 5 animals / 4 pose-estimated body-parts  <https://github.com/sgoldenlab/simba/blob/master/misc/sleap_csv_example.csv>`__.


    Example
    ----------

    >>> sleap_csv_importer = SleapCsvImporter(config_path=r'/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/project_folder/project_config.ini', data_folder=r'/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/import/temp', actor_IDs=['Termite_1', 'Termite_2', 'Termite_3', 'Termite_4', 'Termite_5'], interpolation_settings="Body-parts: Nearest", smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
    >>> sleap_csv_importer.initate_import_slp()
    """

    def __init__(self,
                 config_path: str,
                 data_folder: str,
                 actor_IDs: list,
                 interpolation_settings: str,
                 smoothing_settings: dict):

        self.config_path, self.config = config_path, read_config_file(ini_path=config_path)
        self.interpolation_settings = interpolation_settings
        self.smoothing_settings, self.actor_ids = smoothing_settings, actor_IDs
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.animal_cnt = len(self.actor_ids)
        multi_animal_status, multi_animal_id_lst = check_multi_animal_status(self.config, self.animal_cnt)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(self.config_path)
        color_lst = createColorListofList(self.animal_cnt, len(self.x_cols))
        self.video_dir = os.path.join(self.project_path, 'videos')
        self.files_found = glob.glob(data_folder + '/*.csv')
        self.import_log_path = os.path.join(self.project_path, 'logs', f'data_import_log_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        if len(self.files_found) == 0:
            raise NoFilesFoundError(msg='SIMBA ERROR: Zero CSV files found in the data directory ({})'.format(data_folder))

        self.save_folder = os.path.join(self.project_path, 'csv', 'input_csv')
        self.space_scaler, self.radius_scaler, self.res_scaler, self.font_scaler, self.add_spacer, self.frame_no = 40, 10, 1500, 1.2, 2, 1
        self.bp_names_csv_path = os.path.join(self.project_path, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
        self.pose_settings = self.config.get('create ensemble settings', 'pose_estimation_body_parts')
        if (self.pose_settings is 'user_defined'):
            self.__update_config_animal_cnt()
        self.animal_bp_dict = create_body_part_dictionary(multi_animal_status, self.actor_ids, self.animal_cnt, self.x_cols, self.y_cols, self.p_cols, color_lst)
        if self.animal_cnt > 1:
            self.__update_bp_headers_file()
            multi_animal_status, multi_animal_id_lst = check_multi_animal_status(self.config, self.animal_cnt)
            self.x_cols, self.y_cols, self.p_cols = getBpNames(self.config_path)
            self.animal_bp_dict = create_body_part_dictionary(multi_animal_status, self.actor_ids, self.animal_cnt, self.x_cols, self.y_cols, self.p_cols, color_lst)
        self.df_headers = getBpHeaders(inifile=self.config_path)

    def __update_config_animal_cnt(self):
        self.config.set("General settings", "animal_no", str(self.animal_cnt))
        with open(self.project_path, "w+") as f:
            self.config.write(f)
        f.close()

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
        cv2.putText(side_img, 'Current video: ' + self.video_basename, (10, self.vid_space_scale), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 2)
        cv2.putText(side_img, 'Can you assign identities based on the displayed frame ?', (10, int(self.vid_space_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 255), 2)
        cv2.putText(side_img, 'Press "x" to display new, random, frame', (10, int(self.vid_space_scale * (self.add_spacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (255, 255, 0), 2)
        cv2.putText(side_img, 'Press "c" to continue to start assigning identities using this frame', (10, int(self.vid_space_scale * (self.add_spacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, (0, 255, 0), 2)
        self.img_concat = np.uint8(np.concatenate((self.img_overlay, side_img), axis=0))

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
                x_cord = int(self.data_df.loc[self.frame_no, animal_bps['X_bps'][bp_cnt]])
                y_cord = int(self.data_df.loc[self.frame_no, animal_bps['Y_bps'][bp_cnt]])
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
                self.frame_no = np.random.randint(0, self.video_info['frame_count']-1, size=1)[0]
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
            cv2.putText(self.new_frame, animal_data['name'], animal_data['cord'], cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, self.animal_bp_dict[animal_data['name']]['colors'][0], 2)

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
                cv2.putText(self.new_overlay, self.ID_cords[id]['name'], self.ID_cords[id]['cord'], cv2.FONT_HERSHEY_SIMPLEX, self.vid_font_scale, self.animal_bp_dict[self.ID_cords[id]['name']]['colors'][0], 3)
            self.new_overlay = np.uint8(np.concatenate((self.new_overlay, self.side_img), axis=0))
            cv2.imshow('Define animal IDs', self.new_overlay)
            cv2.resizeWindow('Define animal IDs', self.video_info['height'], self.video_info['width'])
            while animal_cnt not in self.ID_cords.keys():
                cv2.setMouseCallback('Define animal IDs', self.__get_x_y_loc)
                cv2.waitKey(200)
        self.__initiate_confirm()

    def __check_intergity_of_order(self):
        for click_key_combination in itertools.combinations(list(self.animal_order.keys()), 2):
            click_n, click_n1 = click_key_combination[0], click_key_combination[1]
            animal_1, animal_2 = self.animal_order[click_n]['animal_name'], self.animal_order[click_n1]['animal_name']
            if animal_1 == animal_2:
                print('SIMBA ERROR: The animal most proximal to click number {} is animal named {}. The animal most proximal to click number {} is also animal {}.'
                      'Please indicate which animal is which using a video frame where the animals are clearly separated'.format(click_n, animal_1, click_n1, animal_2))
            else:
                pass

    def __update_bp_headers_file(self):
        new_headers = []
        for animal_name in self.animal_bp_dict.keys():
            for bp in self.animal_bp_dict[animal_name]['X_bps']:
                if animal_name not in bp:
                    new_headers.append('{}_{}'.format(animal_name, bp[:-2]))
                else:
                    new_headers.append(bp[:-2])
        new_bp_df = pd.DataFrame(new_headers)
        new_bp_df.to_csv(self.bp_names_csv_path, index=False, header=False)

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
                    bp_location = (int(self.all_frame_data[x_col]), int(self.all_frame_data[y_col]))
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
            for x_col, y_col, p_cols in zip(x_cols, y_cols, p_cols):
                df = self.data_df[[x_col, y_col, p_cols]]
                self.out_df = pd.concat([self.out_df, df], axis=1)

    def __insert_multi_idx_header(self):
        multi_index_columns = []
        for column in range(len(self.data_df.columns)):
            multi_index_columns.append(tuple(('SLEAP_multi', 'SLEAP_multi', self.data_df.columns[column])))
        self.out_df.columns = pd.MultiIndex.from_tuples(multi_index_columns, names=['scorer', 'bodypart', 'coords'])

    def __save_df(self):
        save_name = os.path.join(self.video_basename + '.' + self.file_type)
        self.save_path = os.path.join(self.save_folder, save_name)
        if self.file_type == 'parquet':
            table = pyarrow.Table.from_pandas(self.out_df)
            pyarrow.parquet.write_table(table, self.save_path)
        if self.file_type == 'csv':
            self.out_df.to_csv(self.save_path)

    def __run_interpolation(self):
        print('Interpolating missing values in video {} (Method: {}) ...'.format(self.video_basename, self.interpolation_settings))
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
            print('Performing Gaussian smoothing on video "{}" ...'.format(self.video_basename))
            time_window = self.smoothing_settings['Parameters']['Time_window']
            smooth_data_gaussian(config=self.config, file_path=self.save_path, time_window_parameter=time_window)

        if self.smoothing_settings['Method'] == 'Savitzky Golay':
            print('Performing Savitzky Golay smoothing on video {}...'.format(self.video_basename))
            time_window = self.smoothing_settings['Parameters']['Time_window']
            smooth_data_savitzky_golay(config=self.config, file_path=self.save_path, time_window_parameter=time_window)

    @staticmethod
    @jit(nopython=True)
    def __transpose_multi_animal_data_table(data: np.array, idx: np.array, animal_cnt: int) -> np.array:
        results = np.full((np.max(idx[:, 1]), data.shape[1]*animal_cnt), 0.0)
        for i in prange(np.max(idx[:, 1])):
            for j in prange(animal_cnt):
                data_idx = np.argwhere((idx[:, 0] == j) & (idx[:, 1] == i)).flatten()
                if len(data_idx) == 1:
                    animal_frm_data = data[data_idx[0]]
                else:
                    animal_frm_data = np.full((data.shape[1]), 0.0)
                results[i][j*animal_frm_data.shape[0]:j*animal_frm_data.shape[0]+animal_frm_data.shape[0]] = animal_frm_data
        return results

    def initate_import_slp(self):
        import_log = pd.DataFrame(columns=['VIDEO', 'IMPORT_TIME', 'IMPORT_SOURCE', 'INTERPOLATION_SETTING', 'SMOOTHING_SETTING'])
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_basename, _ = get_fn_ext(filepath=file_path)
            print('Analysing {}...'.format(os.path.basename(file_path)))
            data_df = pd.read_csv(file_path)
            idx = data_df.iloc[:, :2]
            idx['track'] = idx['track'].str.replace(r'[^\d.]+', '').astype(int)
            data_df = data_df.iloc[:, 2:]
            if self.animal_cnt > 1:
                self.data_df = pd.DataFrame(self.__transpose_multi_animal_data_table(data=data_df.values, idx=idx.values, animal_cnt=self.animal_cnt))
                p_df = pd.DataFrame(1.0, index=self.data_df.index, columns=self.data_df.columns[1::2] + .5)
                self.data_df = pd.concat([self.data_df, p_df], axis=1).sort_index(axis=1)
                self.data_df.columns = self.df_headers
            else:
                idx = list(idx.drop('track', axis=1)['frame_idx'])
                self.data_df = data_df.set_index([idx]).sort_index()
                self.data_df.columns = np.arange(len(self.data_df.columns))
                self.data_df = self.data_df.reindex(range(self.data_df.index[0], self.data_df.index[-1] + 1), fill_value=0)
                p_df = pd.DataFrame(1.0, index=self.data_df.index, columns=self.data_df.columns[1::2] + .5)
                self.data_df = pd.concat([self.data_df, p_df], axis=1).sort_index(axis=1)
                self.data_df.columns = self.df_headers
                self.out_df = deepcopy(self.data_df)

            if self.animal_cnt > 1:
                self.video_path = find_video_of_file(video_dir=self.video_dir, filename=self.video_basename)
                self.video_info = get_video_meta_data(self.video_path)
                self.max_video_dimension = max(self.video_info['width'], self.video_info['height'])
                self.vid_circle_scale = int(self.radius_scaler / (self.res_scaler / self.max_video_dimension))
                self.vid_font_scale = float(self.font_scaler / (self.res_scaler / self.max_video_dimension))
                self.vid_space_scale = int(self.space_scaler / (self.res_scaler / self.max_video_dimension))
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
            import_log.loc[len(import_log)] = [self.video_basename,
                                               datetime.now().strftime('%Y%m%d%H%M%S'),
                                               'SLEAP_CSV',
                                               str(self.interpolation_settings),
                                               str(self.smoothing_settings)]
            print('Video "{}" imported...'.format(self.video_basename))

        import_log.to_csv(self.import_log_path)
        print('SIMBA COMPLETE: {} file(s) imported to the SimBA project (project_folder/csv/input_csv directory)'.format(str(len(self.files_found))))


# test = SleapCsvImporter(config_path=r'/Users/simon/Desktop/envs/troubleshooting/Hornet/project_folder/project_config.ini',
#                  data_folder=r'/Users/simon/Desktop/envs/troubleshooting/Hornet_single_slp/import',
#                  actor_IDs=['Hornet'],
#                  interpolation_settings="Body-parts: Nearest",
#                  smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.initate_import_slp()


# test = SleapCsvImporter(config_path=r'/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/project_folder/project_config.ini',
#                  data_folder=r'/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/import',
#                  actor_IDs=['Termite_1', 'Termite_2', 'Termite_3', 'Termite_4', 'Termite_5'],
#                  interpolation_settings="Body-parts: Nearest",
#                  smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.initate_import_slp()