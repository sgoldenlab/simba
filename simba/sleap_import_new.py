__author__ = "Simon Nilsson, JJ Choong"

import itertools
from configparser import ConfigParser, MissingSectionHeaderError
import os, glob
import numpy as np
import h5py
from simba.drop_bp_cords import (get_workflow_file_format,
                                 get_fn_ext,
                                 createColorListofList,
                                 create_body_part_dictionary,
                                 getBpNames,
                                 getBpHeaders)
from simba.misc_tools import (find_video_of_file,
                              check_multi_animal_status,
                              smooth_data_gaussian,
                              smooth_data_savitzky_golay,
                              get_video_meta_data)
from simba.read_config_unit_tests import (check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type,
                                          read_config_file)
from simba.enums import Paths, ReadConfig, Methods
from simba.rw_dfs import read_df
import json
from collections import defaultdict
import pandas as pd
import cv2
import random
from simba.interpolate_pose import Interpolate
import pyarrow.parquet as pq
import pyarrow as pa


class ImportSLEAP(object):
    """
    Class for importing SLEAP pose-estimation data into a SimBA project.

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

    Example
    ----------
    >>> slp_importer = ImportSLEAP(project_path="MyConfigPath", data_folder=r'MySLPDataFolder', actor_IDs=['Mouse_1', 'Mouse_2'], interpolation_settings="Body-parts: Nearest", smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
    >>> slp_importer.initate_import_slp()
    >>> slp_importer.visualize_sleap()
    >>> slp_importer.perform_interpolation()
    >>> slp_importer.perform_smothing()
    """


    def __init__(self,
                 project_path: str,
                 data_folder: str,
                 actor_IDs: list,
                 interpolation_settings: str,
                 smoothing_settings: dict):

        self.ini_path = project_path
        self.interpolation_settings = interpolation_settings
        self.smoothing_settings = smoothing_settings
        self.config = ConfigParser()
        self.actors_IDs = actor_IDs
        self.config = read_config_file(ini_path=project_path)
        self.project_path, self.file_format = read_project_path_and_file_type(config=self.config)
        self.video_folder = os.path.join(self.project_path, 'videos')
        self.files_found = glob.glob(data_folder + '/*.slp')
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero .slp files found in {} directory'.format(data_folder))
        self.save_folder = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        self.animals_no = len(self.actors_IDs)
        self.add_spacer = 2
        self.bp_names_csv_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
        self.pose_settings = self.config.get(ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.POSE_SETTING.value)
        if self.pose_settings is Methods.USER_DEFINED.value:
            self.__update_config()

        print('Converting .SLP file(s) into SimBA dataframes...')

    def visualize_sleap(self):
        self.frame_number = 0
        multi_animal_status, multi_animal_id_lst = check_multi_animal_status(self.config, self.animals_no)
        Xcols, Ycols, Pcols = getBpNames(self.ini_path)
        color_lst = createColorListofList(self.animals_no, len(self.analysis_dict['ordered_bps']))
        self.animal_bp_dict = create_body_part_dictionary(multi_animal_status, multi_animal_id_lst, self.animals_no, Xcols, Ycols, [], color_lst)

        for file_path in self.save_paths_lst:
            self.data_df = read_df(file_path, self.file_format)
            self.data_df.columns = getBpHeaders(self.ini_path)
            _, video_name, _ = get_fn_ext(file_path)
            video_path = find_video_of_file(self.video_folder, video_name)
            self.cap = cv2.VideoCapture(video_path)
            self.cap.set(1, self.frame_number)
            if not self.cap.isOpened():
                raise Exception('Can\'t open video file ' + video_path)
            self.video_meta_data = get_video_meta_data(video_path)
            mySpaceScale, myRadius, myResolution, myFontScale = 40, 10, 1500, 1.2
            maxResDimension = max(self.video_meta_data['width'], self.video_meta_data['height'])
            self.circle_scale = int(myRadius / (myResolution / maxResDimension))
            self.font_scale = float(myFontScale / (myResolution / maxResDimension))
            self.spacing_scale = int(mySpaceScale / (myResolution / maxResDimension))
            self.__show_clean_window()

    def __update_config(self):
        self.config.set("General settings", "animal_no", str(self.animals_no))
        with open(self.project_path, "w+") as f:
            self.config.write(f)
        f.close()

    def __h5_to_dict(self, name, obj):
        attr = list(obj.attrs.items())
        if name == 'metadata':
            jsonList = (attr[1][1])
            jsonList = jsonList.decode('utf-8')
            final_dictionary = json.loads(jsonList)
            final_dictionary = dict(final_dictionary)
            return final_dictionary

    def __get_provenance(self):
        try:
            video_path = os.path.basename(self.sleap_dict['provenance']['video.path'])
            _, video_name, _ = get_fn_ext(video_path)
        except KeyError:
            _, video_name, _ = get_fn_ext(self.file_path)

        return find_video_of_file(self.video_folder, video_name)

    def __get_video_frame_cnt(self):
        cap = cv2.VideoCapture(self.video_path)
        self.video_frame_cnt_opencv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __fill_missing_indexes(self):
        missing_indexes = list(set(list(range(0, self.video_frame_cnt_opencv))) - set(list(self.data_df.index)))
        missing_df = pd.DataFrame(0, index=missing_indexes, columns=self.analysis_dict['xyp_headers'])
        self.data_df = pd.concat([self.data_df, missing_df], axis=0)

    def __save_multi_index_header_df(self, df=None, filetype=None, savepath=None):
        if filetype == 'csv':
            df.to_csv(savepath)
        if filetype == 'parquet':
            table = pa.Table.from_pandas(df)
            pq.write_table(table, savepath)
        self.save_paths_lst.append(savepath)

    def __update_bp_headers_file(self):
        self.new_headers = []
        if len(list(self.animal_bp_dict.keys())) > 1:
            for cnt, animal in enumerate(self.animal_bp_dict.keys()):
                for bp in self.analysis_dict['ordered_bps']:
                    self.new_headers.append('{}_{}_{}'.format(animal, bp, str(cnt + 1)))
        else:
            for cnt, animal in enumerate(self.animal_bp_dict.keys()):
                for bp in self.analysis_dict['ordered_bps']:
                    self.new_headers.append('{}_{}'.format(bp, str(cnt + 1)))
        new_bp_df = pd.DataFrame(self.new_headers)
        new_bp_df.to_csv(self.bp_names_csv_path, index=False, header=False)

    def initate_import_slp(self):
        """
        Method to initiate SLEAP import GUI.

        Returns
        -------
        Attribute: dict
            analysis_dict

        """
        self.analysis_dict = defaultdict(list)
        self.save_paths_lst = []
        for vdn_cnt, file_path in enumerate(self.files_found):
            print('Analysing {}{}'.format(os.path.basename(file_path), '...'))
            self.file_path = file_path
            in_h5 = h5py.File(file_path, 'r')
            self.video_counter = vdn_cnt
            self.sleap_dict = in_h5.visititems(self.__h5_to_dict)
            self.video_path = self.__get_provenance()
            self.video_dir, self.video_name, self.video_ext = get_fn_ext(self.video_path)
            self.save_path = os.path.join(self.save_folder, self.video_name + '.{}'.format(self.file_format))
            self.__get_video_frame_cnt()
            self.analysis_dict['bp_names'] = []
            self.analysis_dict['ordered_ids'] = []
            self.analysis_dict['ordered_bps'] = []
            self.analysis_dict['xy_headers'] = []
            self.analysis_dict['xyp_headers'] = []
            self.analysis_dict['animals_in_each_frame'] = []
            for bp in self.sleap_dict['nodes']:
                self.analysis_dict['bp_names'].append(bp['name'])
            for orderVar in self.sleap_dict['skeletons'][0]['nodes']:
                self.analysis_dict['ordered_ids'].append((orderVar['id']))
            for index in self.analysis_dict['ordered_ids']:
                self.analysis_dict['ordered_bps'].append(self.analysis_dict['bp_names'][index])

            with h5py.File(file_path, 'r') as file:
                self.analysis_dict['frames'] = file['frames'][:]
                self.analysis_dict['instances'] = file['instances'][:]
                self.analysis_dict['predicted_points'] = np.reshape(file['pred_points'][:], (file['pred_points'][:].size, 1))

            self.analysis_dict['no_frames'] = len(self.analysis_dict['frames'])
            for c in itertools.product(self.actors_IDs, self.analysis_dict['ordered_bps']):
                x, y, p = str('{}_{}_x'.format(c[0], c[1])), str('{}_{}_y'.format(c[0], c[1])), (str('{}_{}_p'.format(c[0], c[1])))
                self.analysis_dict['xy_headers'].extend((x, y))
                self.analysis_dict['xyp_headers'].extend((x, y, p))

            self.data_df = pd.DataFrame(columns=self.analysis_dict['xyp_headers'])
            frames_lst = [l.tolist() for l in self.analysis_dict['frames']]
            self.analysis_dict['animals_in_each_frame'] = [x[4] - x[3] for x in frames_lst]
            self.__create_tracks()


    def __check_that_all_animals_exist_in_frame(self):
        existing_animals = list(self.frame_dict.keys())
        missing_animals = [x for x in range(self.animals_no) if x not in existing_animals]
        for missing_animal in missing_animals:
            self.frame_dict[missing_animal] = [0] * ((len(self.analysis_dict['ordered_bps']))) * 3

    def __create_tracks(self):
        start_frame = 0
        for frame_cnt, frame in enumerate(range(self.analysis_dict['no_frames'])):
            frame_idx = self.analysis_dict['frames'][frame_cnt][2]
            self.frame_dict = {}
            print('Restructuring SLEAP frame: {}/{}, Video: {} ({}/{})'.format(str(frame_cnt), str(self.analysis_dict['no_frames']), str(self.video_name), str(self.video_counter + 1), str(len(self.files_found))))
            self.cnt_animals_frm = self.analysis_dict['animals_in_each_frame'][frame]
            if self.cnt_animals_frm == 0:
                self.frame_dict[0] = [0] * len(self.analysis_dict['xyp_headers'])
                end_frame = start_frame + (len(self.analysis_dict['ordered_bps']) * self.cnt_animals_frm)

            else:
                end_frame = start_frame + (len(self.analysis_dict['ordered_bps']) * self.cnt_animals_frm)
                start_animal, end_animal = 0, len(self.analysis_dict['ordered_bps'])
                frame_arr = self.analysis_dict['predicted_points'][start_frame:end_frame]
                for instance_counter, animal in enumerate(range(self.cnt_animals_frm)):
                    currRow = []
                    animal_arr = frame_arr[start_animal:end_animal]
                    track_id = self.analysis_dict['instances'][instance_counter][4]
                    for bp in animal_arr:
                        currRow.extend((bp[0][0], bp[0][1], bp[0][4]))
                    self.frame_dict[track_id] = currRow
                    start_animal += len(self.analysis_dict['ordered_bps'])
                    end_animal += len(self.analysis_dict['ordered_bps'])

            if self.animals_no > 1:
                self.__check_that_all_animals_exist_in_frame()
            frame_lst = [item for sublist in list(self.frame_dict.values()) for item in sublist]
            start_frame = end_frame
            try:
                self.data_df.loc[frame_idx] = frame_lst
            except ValueError:
                break

        self.data_df.fillna(0, inplace=True)
        self.__fill_missing_indexes()
        self.data_df.sort_index(inplace=True)
        multi_animal_status, multi_animal_id_lst = check_multi_animal_status(self.config, self.animals_no)
        if self.animals_no < 2:
            multi_animal_status = False
        Xcols, Ycols, Pcols = getBpNames(self.ini_path)
        color_lst = createColorListofList(self.animals_no, len(self.analysis_dict['ordered_bps']))
        self.animal_bp_dict = create_body_part_dictionary(multi_animal_status, multi_animal_id_lst, self.animals_no, Xcols, Ycols, [], color_lst)
        self.__update_bp_headers_file()
        self.__save_multi_index_header_df(df=self.data_df,filetype=self.file_format,savepath=self.save_path)
        print('Re-organized {} for SimBA analysis...'.format(os.path.basename(self.save_path)))

    def __create_first_side_img(self):
        self.side_img = np.ones((int(self.video_meta_data['height'] / 1.5), self.video_meta_data['width'], 3))
        cv2.putText(self.side_img, 'Current video: ' + self.video_name, (10, int(self.spacing_scale)), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, 'Can you assign identities based on the displayed frame ?', (10, int(self.spacing_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,(255, 255, 255), 2)
        cv2.putText(self.side_img, 'Press "x" to display new - random - frame', (10, int(self.spacing_scale * (self.add_spacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 0), 2)
        cv2.putText(self.side_img, 'Press "c" to continue to start assigning identities using this frame', (10, int(self.spacing_scale * (self.add_spacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 255), 2)

    def __update_frame(self):
        new_frame_option_lst = list(range(0, self.video_meta_data['frame_count']))
        new_frame_option_lst.remove(self.frame_number)
        self.frame_number = random.choice(new_frame_option_lst)
        self.cap.set(1, self.frame_number)
        self.__show_clean_window()

    def __get_x_y_loc(self, event, x, y, flags, param):
        if event == 1:
            self.click_loc = (x,y)
            self.ID_cords[self.animal_cnt] = {}
            self.ID_cords[self.animal_cnt]['cord'] = self.click_loc
            self.ID_cords[self.animal_cnt]['name'] = self.current_animal
            self.clicked = True
            for id in self.ID_cords.keys():
                cv2.putText(self.frame, self.ID_cords[id]['name'], self.ID_cords[id]['cord'], cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 0), 2)
                self.concat_img = np.uint8(np.concatenate((self.frame, self.side_img), axis=0))
                cv2.imshow('Define animal IDs', self.concat_img)

    def __create_third_side_img(self):
        self.side_img = np.ones((int(self.video_meta_data['height'] / 1.5), self.video_meta_data['width'], 3))
        cv2.putText(self.side_img, 'Current video: ' + self.video_name, (10, int(self.spacing_scale)), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, 'Are you happy with your assigned identities ?', (10, int(self.spacing_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
        cv2.putText(self.side_img, 'Press "c" to continue (to finish, or proceed to the next video)', (10, int(self.spacing_scale * (self.add_spacer * 3))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 0), 2)
        cv2.putText(self.side_img, 'Press "x" to re-start assigning identities', (10, int(self.spacing_scale * (self.add_spacer * 4))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 255), 2)

    def __assign_individuals(self):
        self.ID_cords = {}
        for animal_cnt, animal in enumerate(self.animal_bp_dict.keys()):
            self.current_animal = animal
            self.animal_cnt = animal_cnt
            self.side_img = np.ones((int(self.video_meta_data['height'] / 1.5), self.video_meta_data['width'], 3))
            cv2.putText(self.side_img, 'Double left mouse click on:', (10, int(self.spacing_scale)), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 2)
            cv2.putText(self.side_img, animal, (10, int(self.spacing_scale * (self.add_spacer * 2))), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 0), 2)
            self.concat_img = np.uint8(np.concatenate((self.frame, self.side_img), axis=0))
            cv2.imshow('Define animal IDs', self.concat_img)
            while animal_cnt not in self.ID_cords.keys():
                cv2.setMouseCallback('Define animal IDs', self.__get_x_y_loc)
                cv2.waitKey(20)

    def __show_clean_window(self):
        cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
        ret, self.frame = self.cap.read()
        for animal in self.animal_bp_dict.keys():
            for cnt, bp in enumerate(zip(self.animal_bp_dict[animal]['X_bps'], self.animal_bp_dict[animal]['Y_bps'])):
                bp_cord = (int(self.data_df.at[self.frame_number, bp[0]]), int(self.data_df.at[self.frame_number, bp[1]]))
                cv2.circle(self.frame, bp_cord, self.circle_scale, self.animal_bp_dict[animal]['colors'][cnt], -1, lineType=cv2.LINE_AA)
        self.__create_first_side_img()
        self.concat_img = np.uint8(np.concatenate((self.frame, self.side_img), axis=0))
        cv2.imshow('Define animal IDs', self.concat_img)
        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(10)
            if k == ord('x'):
                self.__update_frame()
            elif k == ord('c'):
                self.__assign_individuals()
                keyboard_choice = True

        self.__create_third_side_img()
        self.concat_img = np.uint8(np.concatenate((self.frame, self.side_img), axis=0))
        cv2.imshow('Define animal IDs', self.concat_img)
        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(50)
            if k == ord('x'):
                self.__update_frame()
            elif k == ord('c'):
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                self.__sort_df()
                keyboard_choice = True

    def __insert_multiindex_header(self):
        multi_index_columns = []
        for column in range(len(self.data_df.columns)):
            multi_index_columns.append(tuple(('SLEAP_multi', 'SLEAP_multi', self.data_df.columns[column])))
        self.data_df.columns = pd.MultiIndex.from_tuples(multi_index_columns, names=['scorer', 'bodypart', 'coords'])

    def __sort_df(self):
        header_order = []
        for animal_id in self.ID_cords:
           d = self.animal_bp_dict[self.ID_cords[animal_id]['name']]
           p_cols = [x.replace(x[-1],'p') for x in d['X_bps']]
           header_order.extend((list(itertools.chain.from_iterable(zip(d['X_bps'],d['Y_bps'], p_cols)))))
        self.data_df = self.data_df[header_order]

    def save_df(self):
        """
        Method to save data created in SLEAP import GUI. Data is saved in the ``project_folder/csv/input_csv``
        directory in the SimBA project.

        Returns
        -------
        None

        """

        self.__insert_multiindex_header()
        self.__save_multi_index_header_df(self.data_df,self.file_format,self.save_path)

    def perform_interpolation(self):
        """
        Method to save perform interpolation of imported SLEAP data.

        Returns
        -------
        None

        """



        if self.interpolation_settings != 'None':
            print('Interpolating missing values (Method: {} {} {}'.format(self.interpolation_settings, '...', ')'))
            self.data_df = read_df(self.save_path, self.file_format)
            interpolate_body_parts = Interpolate(self.ini_path, self.data_df)
            interpolate_body_parts.detect_headers()
            interpolate_body_parts.fix_missing_values(self.interpolation_settings)
            interpolate_body_parts.reorganize_headers()
            self.__save_multi_index_header_df(df=interpolate_body_parts.new_df, filetype=self.file_format, savepath=self.save_path)
            print('Interpolation complete for video {}{}'.format(self.video_name, '...'))

    def perform_smothing(self):
        """
        Method to save perform smoothing of imported SLEAP data.

        Returns
        -------
        None
        """

        if self.smoothing_settings['Method'] == 'Gaussian':
            print('Performing Gaussian smoothing on video {}{}'.format(self.video_name, '...'))
            time_window = self.smoothing_settings['Parameters']['Time_window']
            smooth_data_gaussian(config=self.config, file_path=self.save_path, time_window_parameter=time_window)

        if self.smoothing_settings['Method'] == 'Savitzky Golay':
            print('Performing Savitzky Golay smoothing on video {}{}'.format(self.video_name, '...'))
            time_window = self.smoothing_settings['Parameters']['Time_window']
            smooth_data_savitzky_golay(config=self.config, file_path=self.save_path, time_window_parameter=time_window)
#
# test = ImportSLEAP(project_path=r"Z:\DeepLabCut\DLC_extract\Troubleshooting\test_slp_import\project_folder\project_config.ini",
#             data_folder=r'Z:\DeepLabCut\DLC_extract\Troubleshooting\test_slp_import\data\slp',
#             actor_IDs=['Mouse_1'],
#             interpolation_settings="Body-parts: Nearest", #
#             smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.initate_import_slp()
# if test.animals_no > 1:
#     test.visualize_sleap()
# test.save_df()
# test.perform_interpolation()
# test.perform_smothing()
# print('All SLEAP imports complete.')
#

# test = ImportSLEAP(project_path="Z:\DeepLabCut\DLC_extract\Troubleshooting\SLEAP_9Test_2\project_folder\project_config.ini",
#             data_folder='Z:\DeepLabCut\DLC_extract\Troubleshooting\SLEAP_9Test_2\data',
#             actor_IDs=['Animal_1', 'Animal_2', 'Animal_3', 'Animal_4', 'Animal_5'],
#             interpolation_settings="None", #
#             smoothing_settings = {'Method': 'None', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.initate_import_slp()
# if test.animals_no > 1:
#     test.visualize_sleap()
# test.save_df()
# test.perform_interpolation()
# test.perform_smothing()
# print('All SLEAP imports complete.')

#
# test = ImportSLEAP(project_path='/Volumes/GoogleDrive/My Drive/GitHub/SLEAP_import_2/project_folder/project_config.ini',
#             data_folder='/Volumes/GoogleDrive/My Drive/GitHub/SLEAP_import_2/data',
#             actor_IDs=['Animal_1'],
#             interpolation_settings="Body-parts: Nearest", #
#             smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.initate_import_slp()
# if test.animals_no > 1:
#     test.visualize_sleap()
# test.save_df()
# test.perform_interpolation()
# test.perform_smothing()
# print('All SLEAP imports complete.')
#

# test = ImportSLEAP(project_path="/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/data',
#                    actor_IDs=['Simon', 'Nastacia', 'JJ', 'Sam', 'Liana'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.initate_import_slp()
# if test.animals_no > 1:
#     test.visualize_sleap()
# test.save_df()
# test.perform_interpolation()
# test.perform_smothing()
# print('All SLEAP imports complete.')


# test = ImportSLEAP(project_path="/Users/simon/Desktop/envs/troubleshooting/sleap_cody/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_cody/import_data',
#                    actor_IDs=['Animal_1', 'Animal_2'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.initate_import_slp()
# if test.animals_no > 1:
#     test.visualize_sleap()
# test.save_df()
# test.perform_interpolation()
# test.perform_smothing()
# print('All SLEAP imports complete.')


