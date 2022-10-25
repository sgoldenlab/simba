from simba.read_config_unit_tests import read_config_file, read_config_entry
import glob, os
import scipy.io as sio
import numpy as np
import h5py
import pandas as pd
from copy import deepcopy
from simba.misc_tools import check_multi_animal_status, get_fn_ext, find_video_of_file, get_video_meta_data
from simba.drop_bp_cords import getBpNames, createColorListofList, create_body_part_dictionary, getBpHeaders
import cv2

class TRKImporter(object):
    def __init__(self,
                 config_path: str,
                 trk_folder: str,
                 animal_id_lst: list,
                 interpolation_method: str,
                 smooth_settings: dict):

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.trk_folder, self.id_lst = trk_folder, animal_id_lst
        self.interpolation_method, self.smooth_settings = interpolation_method, smooth_settings
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        if self.no_animals == 1:
            self.animal_ids = ['Animal_1']
        else:
            self.animal_ids = read_config_entry(self.config, 'Multi animal IDs', 'id_list', 'str')
            self.animal_ids = self.animal_ids.split(",")
        self.files_found = glob.glob(self.trk_folder + '/*.trk')
        if len(self.files_found) == 0:
            print('SIMBA ERROR: No TRK files (with .trk file-ending) found in {}'.format(self.trk_folder))
            raise ValueError('SIMBA ERROR: No TRK files (with .trk file-ending) found in {}'.format(self.trk_folder))
        self.video_folder = os.path.join(self.project_path, 'videos')
        self.save_folder = os.path.join(self.project_path, 'csv', 'input_csv')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.color_lst = createColorListofList(self.no_animals, int(len(self.x_cols) / self.no_animals) + 1)
        self.animal_pose_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, self.p_cols, self.color_lst)
        self.bp_headers = getBpHeaders(config_path)
        self.space_scaler, self.radius_scaler, self.resolution_scaler, self.font_scaler = 40, 10, 1500, 1.2
        self.frm_number = 0

    def trk_read(self, file_path: str):
        print('Reading data using scipy.io...')
        try:
            trk_dict = sio.loadmat(file_path)
            trk_coordinates = trk_dict['pTrk']
            track_cnt = trk_coordinates.shape[3]
            animals_tracked_list = [trk_coordinates[..., i] for i in range(track_cnt)]

        except NotImplementedError:
            print('Failed to read data using scipy.io. Reading data using h5py...')
            with h5py.File(file_path, 'r') as trk_dict:
                trk_list = list(trk_dict['pTrk'])
                t_second = np.array(trk_list)
                if len(t_second.shape) > 3:
                    t_third = np.swapaxes(t_second, 0, 3)
                    trk_coordinates = np.swapaxes(t_third, 1, 2)
                    track_cnt = trk_coordinates.shape[3]
                    animals_tracked_list = [trk_coordinates[..., i] for i in range(track_cnt)]
                else:
                    animals_tracked_list = np.swapaxes(t_second, 0, 2)
                    track_cnt = 1

        print('Number of animals detected in TRK {}: {}'.format(str(file_path), str(track_cnt)))
        if track_cnt != self.no_animals:
            print('SIMBA ERROR: There are {} tracks in the .trk file {}. But your SimBA project expects {} tracks.'.format(str(track_cnt), file_path, str(self.no_animals)))
            raise ValueError('SIMBA ERROR: There are {} tracks in the .trk file {}. But your SimBA project expects {} tracks.'.format(str(track_cnt), file_path, str(self.no_animals)))
        return animals_tracked_list

    def import_trk(self):
        for file_cnt, file_path in enumerate(self.files_found):
            _, file_name, file_ext = get_fn_ext(file_path)
            video_path = find_video_of_file(self.video_folder, file_name)
            video_meta_data = get_video_meta_data(video_path=video_path)
            animal_tracks = self.trk_read(file_path=file_path)

            if self.no_animals != 1:
                animal_df_lst = []
                for animal in animal_tracks:
                    m, n, r = animal.shape
                    out_arr = np.column_stack((np.repeat(np.arange(m), n), animal.reshape(m * n, -1)))
                    animal_df_lst.append(pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True))
                self.animal_df = pd.concat(animal_df_lst, axis=1).fillna(0)
            else:
                m, n, r = animal_tracks.shape
                out_arr = np.column_stack((np.repeat(np.arange(m), n), animal_tracks.reshape(m * n, -1)))
                self.animal_df = pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True)
            p_cols = pd.DataFrame(1, index=self.animal_df.index, columns=self.animal_df.columns[1::2] + .5)
            self.animal_df = pd.concat([self.animal_df, p_cols], axis=1).sort_index(axis=1)
            if len(self.bp_headers) != len(self.animal_df.columns):
                print('SIMBA ERROR: SimBA detected {} body-parts in the .TRK file {}. Your SimBA project however expects {} body-parts'.format(str(len(animal_df.columns)), file_path, len(self.bp_headers)))
                raise ValueError('SIMBA ERROR: SimBA detected {} body-parts in the .TRK file {}. Your SimBA project however expects {} body-parts'.format(str(len(animal_df.columns)), file_path, len(self.bp_headers)))
            self.animal_df.columns = self.bp_headers

            max_dim = max(video_meta_data['width'], video_meta_data['height'])
            self.circle_scale = int(self.radius_scaler / (self.resolution_scaler / max_dim))
            self.font_scale = float(self.font_scaler / (self.resolution_scaler / max_dim))
            self.spacingScale = int(self.space_scaler / (self.resolution_scaler / max_dim))
            cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
            self.cap = cv2.VideoCapture(video_path)
            self.create_first_interface()

    def __insert_all_animal_bps(self, frame=None):
        for animal, bp_data in self.img_bp_cords_dict.items():
            for bp_cnt, bp_tuple in enumerate(bp_data):
                try:
                    cv2.circle(frame, bp_tuple, self.circle_scale, self.animal_pose_dict[animal]['colors'][bp_cnt], -1, lineType=cv2.LINE_AA)
                except Exception as err:
                    if type(err) == OverflowError:
                        print('SIMBA ERROR: SimBA encountered a pose-estimated body-part located at pixel position {}. '
                              'This value is too large to be converted to an integer. '
                              'Please check your pose-estimation data to make sure that it is accurate.'.format(str(bp_tuple)))
                        raise OverflowError()
                    print(err.args)

    def create_first_interface(self):
        while True:
            self.cap.set(1, self.frm_number)
            _, self.frame = self.cap.read()
            self.overlay = self.frame.copy()
            self.img_bp_cords_dict = {}
            for animal_cnt, (animal_name, animal_bps) in enumerate(self.animal_pose_dict.items()):
                self.img_bp_cords_dict[animal_name] = []
                for bp_cnt in range(len(animal_bps['X_bps'])):
                    x_cord = int(self.animal_df.loc[self.frm_number, animal_name + '_' + animal_bps['X_bps'][bp_cnt]])
                    y_cord = int(self.animal_df.loc[self.frm_number, animal_name + '_' + animal_bps['Y_bps'][bp_cnt]])
                    self.img_bp_cords_dict[animal_name].append((x_cord, y_cord))
            self.__insert_all_animal_bps(frame=self.overlay)




















