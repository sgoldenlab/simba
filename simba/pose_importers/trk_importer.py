import glob
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.errors import CountError, NoFilesFoundError
from simba.utils.read_write import (find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_config_file)
from simba.utils.warnings import InvalidValueWarning


class TRKImporter(ConfigReader):
    def __init__(
        self,
        config_path: str,
        data_path: str,
        animal_id_lst: list,
        interpolation_method: str,
        smoothing_settings: dict,
    ):
        ConfigReader.__init__(self, config_path=config_path)
        check_if_dir_exists(in_dir=data_path)
        self.data_path, self.id_lst = data_path, animal_id_lst
        self.interpolation_method, self.smooth_settings = (
            interpolation_method,
            smoothing_settings,
        )
        if self.animal_cnt == 1:
            self.animal_ids = ["Animal_1"]
        else:
            self.animal_ids = read_config_entry(
                self.config, "Multi animal IDs", "id_list", "str"
            )
            self.animal_ids = self.animal_ids.split(",")
        self.data_paths = glob.glob(self.data_path + "/*.trk")
        check_if_filepath_list_is_empty(
            filepaths=self.data_paths,
            error_msg=f"No TRK files (with .trk file-ending) found in {self.data_path}",
        )
        (
            self.space_scaler,
            self.radius_scaler,
            self.resolution_scaler,
            self.font_scaler,
        ) = (40, 10, 1500, 1.2)
        self.frm_number = 0

    def trk_read(self, file_path: str):
        print("Reading data using scipy.io...")
        try:
            trk_dict = sio.loadmat(file_path)
            trk_coordinates = trk_dict["pTrk"]
            track_cnt = trk_coordinates.shape[3]
            animals_tracked_list = [trk_coordinates[..., i] for i in range(track_cnt)]

        except NotImplementedError:
            print("Failed to read data using scipy.io. Reading data using h5py...")
            with h5py.File(file_path, "r") as trk_dict:
                trk_list = list(trk_dict["pTrk"])
                t_second = np.array(trk_list)
                if len(t_second.shape) > 3:
                    t_third = np.swapaxes(t_second, 0, 3)
                    trk_coordinates = np.swapaxes(t_third, 1, 2)
                    track_cnt = trk_coordinates.shape[3]
                    animals_tracked_list = [
                        trk_coordinates[..., i] for i in range(track_cnt)
                    ]
                else:
                    animals_tracked_list = np.swapaxes(t_second, 0, 2)
                    track_cnt = 1

        print(
            "Number of animals detected in TRK {}: {}".format(
                str(file_path), str(track_cnt)
            )
        )
        if track_cnt != self.animal_cnt:
            raise CountError(
                msg=f"There are {str(track_cnt)} tracks in the .trk file {file_path}. But your SimBA project expects {str(self.animal_cnt)} tracks."
            )
        return animals_tracked_list

    def import_trk(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            _, file_name, file_ext = get_fn_ext(file_path)
            if self.animal_cnt > 0:
                pass
            video_path = find_video_of_file(self.video_dir, file_name)
            if not video_path:
                raise NoFilesFoundError(msg="Could not find a video jj")

    #         video_meta_data = get_video_meta_data(video_path=video_path)
    #         animal_tracks = self.trk_read(file_path=file_path)
    #
    #         if self.animal_cnt != 1:
    #             animal_df_lst = []
    #             for animal in animal_tracks:
    #                 m, n, r = animal.shape
    #                 out_arr = np.column_stack((np.repeat(np.arange(m), n), animal.reshape(m * n, -1)))
    #                 animal_df_lst.append(pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True))
    #             self.animal_df = pd.concat(animal_df_lst, axis=1).fillna(0)
    #         else:
    #             m, n, r = animal_tracks.shape
    #             out_arr = np.column_stack((np.repeat(np.arange(m), n), animal_tracks.reshape(m * n, -1)))
    #             self.animal_df = pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True)
    #         p_cols = pd.DataFrame(1, index=self.animal_df.index, columns=self.animal_df.columns[1::2] + .5)
    #         self.animal_df = pd.concat([self.animal_df, p_cols], axis=1).sort_index(axis=1)
    #         if len(self.bp_headers) != len(self.animal_df.columns):
    #             raise CountError(msg=f'SimBA detected {str(len(self.animal_df.columns))} body-parts in the .TRK file {file_path}. Your SimBA project however expects {len(self.bp_headers)} body-parts')
    #         self.animal_df.columns = self.bp_headers
    #
    #         max_dim = max(video_meta_data['width'], video_meta_data['height'])
    #         self.circle_scale = int(self.radius_scaler / (self.resolution_scaler / max_dim))
    #         self.font_scale = float(self.font_scaler / (self.resolution_scaler / max_dim))
    #         self.spacingScale = int(self.space_scaler / (self.resolution_scaler / max_dim))
    #         cv2.namedWindow('Define animal IDs', cv2.WINDOW_NORMAL)
    #         self.cap = cv2.VideoCapture(video_path)
    #         self.create_first_interface()
    #
    # def __insert_all_animal_bps(self, frame=None):
    #     for animal, bp_data in self.img_bp_cords_dict.items():
    #         for bp_cnt, bp_tuple in enumerate(bp_data):
    #             try:
    #                 cv2.circle(frame, bp_tuple, self.circle_scale, self.animal_bp_dict[animal]['colors'][bp_cnt], -1, lineType=cv2.LINE_AA)
    #             except Exception as err:
    #                 if type(err) == OverflowError:
    #                     InvalidValueWarning(f'SimBA encountered a pose-estimated body-part located at pixel position {str(bp_tuple)}. This value is too large to be converted to an integer. Please check your pose-estimation data to make sure that it is accurate.')
    #                 print(err.args)
    #
    # def create_first_interface(self):
    #     while True:
    #         self.cap.set(1, self.frm_number)
    #         _, self.frame = self.cap.read()
    #         self.overlay = self.frame.copy()
    #         self.img_bp_cords_dict = {}
    #         for animal_cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
    #             self.img_bp_cords_dict[animal_name] = []
    #             for bp_cnt in range(len(animal_bps['X_bps'])):
    #                 x_cord = int(self.animal_df.loc[self.frm_number, animal_name + '_' + animal_bps['X_bps'][bp_cnt]])
    #                 y_cord = int(self.animal_df.loc[self.frm_number, animal_name + '_' + animal_bps['Y_bps'][bp_cnt]])
    #                 self.img_bp_cords_dict[animal_name].append((x_cord, y_cord))
    #         self.__insert_all_animal_bps(frame=self.overlay)


# test = TRKImporter(config_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini',
#                    data_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/trk_data',
#                    animal_id_lst=['Animal_1', 'Animal_2'],
#                    interpolation_method="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
#


# test = SLEAPImporterCSV(config_path=r'/Users/simon/Desktop/envs/troubleshooting/Hornet/project_folder/project_config.ini',
#                  data_folder=r'/Users/simon/Desktop/envs/troubleshooting/Hornet_single_slp/import',
#                  actor_IDs=['Hornet'],
#                  interpolation_settings="Body-parts: Nearest",
#                  smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()


# def __init__(self,
#              config_path: str,
#              data_folder: str,
#              animal_id_lst: list,
#              interpolation_method: str,
#              smooth_settings: dict):
