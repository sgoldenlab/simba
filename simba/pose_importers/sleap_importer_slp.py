__author__ = "Simon Nilsson"

import itertools
import os, glob
import numpy as np
import h5py
import json
from collections import defaultdict
import pandas as pd
import cv2
import random


from simba.mixins.config_reader import ConfigReader
from simba.data_processors.interpolation_smoothing import Smooth, Interpolate
from simba.utils.read_write import get_fn_ext, find_video_of_file, get_video_meta_data, read_df, write_df
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.data import create_color_palettes
from simba.utils.enums import Paths, ConfigKey, Methods



class SLEAPImporterSLP(ConfigReader):
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

        ConfigReader.__init__(self, config_path=project_path)
        self.interpolation_settings = interpolation_settings
        self.smoothing_settings = smoothing_settings
        self.actors_IDs = actor_IDs
        self.video_folder = os.path.join(self.project_path, 'videos')
        self.files_found = glob.glob(data_folder + '/*.slp')
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Zero .slp files found in {} directory'.format(data_folder))
        self.save_folder = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        self.animals_no = len(self.actors_IDs)
        self.add_spacer = 2
        self.bp_names_csv_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
        self.pose_settings = self.config.get(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, ConfigKey.POSE_SETTING.value)
        if self.pose_settings is Methods.USER_DEFINED.value:
            self.__update_config()

        print('Converting .SLP file(s) into SimBA dataframes...')

    def visualize_sleap(self):
        self.frame_number = 0
        for file_path in self.save_paths_lst:
            self.data_df = read_df(file_path, self.file_type)
            self.data_df.columns = self.bp_headers
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
            self.save_path = os.path.join(self.save_folder, self.video_name + '.{}'.format(self.file_type))
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
        self.check_multi_animal_status()
        if self.animals_no < 2:
            self.multi_animal_status = False
        color_lst = create_color_palettes(self.animals_no, len(self.analysis_dict['ordered_bps']))
        self.animal_bp_dict = self.create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_list, self.animals_no, self.x_cols, self.x_cols, [], color_lst)
        self.__update_bp_headers_file()
        self.__save_multi_index_header_df(df=self.data_df, filetype=self.file_type, savepath=self.save_path)
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
            print('Interpolating missing values in video {} (Method: {})...'.format(self.video_name, self.interpolation_settings))
            _ = Interpolate(input_path=self.save_path, config_path=self.config_path, method=self.interpolation_settings, initial_import_multi_index=True)

    def perform_smothing(self):
        """
        Method to save perform smoothing of imported SLEAP data.

        Returns
        -------
        None
        """

        if (self.smoothing_settings['Method'] == Methods.GAUSSIAN.value) or (self.smoothing_settings['Method'] == Methods.SAVITZKY_GOLAY.value):
            print(f'Performing {self.smoothing_settings["Method"]} smoothing on video {self.video_name}...')
            Smooth(config_path=self.config_path,
                   input_path=self.save_path,
                   time_window=int(self.smoothing_settings['Parameters']['Time_window']),
                   smoothing_method=self.smoothing_settings['Method'],
                   initial_import_multi_index=True)


# test = SLEAPImporterSLP(project_path="/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini",
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


