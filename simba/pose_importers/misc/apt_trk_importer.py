import glob
import os

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.utils.checks import (check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.errors import CountError, NoFilesFoundError
from simba.utils.read_write import (find_all_videos_in_project,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_config_file)
from simba.utils.warnings import InvalidValueWarning


class APTImporterTRK(ConfigReader, PoseImporterMixin):
    def __init__(
        self,
        config_path: str,
        data_folder: str,
        id_lst: list,
        interpolation_settings: str,
        smoothing_settings: dict,
    ):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PoseImporterMixin.__init__(self)
        check_if_dir_exists(in_dir=data_folder)
        self.interpolation_settings, self.smoothing_settings = (
            interpolation_settings,
            smoothing_settings,
        )
        self.data_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(
            self.logs_path, f"data_import_log_{self.datetime}.csv"
        )
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
        self.input_data_paths = self.find_data_files(
            dir=self.data_folder, extensions=[".trk"]
        )
        self.data_and_videos_lk = self.link_video_paths_to_data_paths(
            data_paths=self.input_data_paths, video_paths=self.video_paths
        )
        print(f"Importing {len(list(self.data_and_videos_lk.keys()))} file(s)...")

    def run(self):
        for file_cnt, (video_name, video_data) in enumerate(
            self.data_and_videos_lk.items()
        ):
            data = self.read_apt_trk_file(file_path=video_data["DATA"])
            data.columns = self.bp_headers
            data = self.intertwine_probability_cols(data=data)
            if self.animal_cnt > 1:
                self.initialize_multi_animal_ui(
                    animal_bp_dict=self.animal_bp_dict,
                    video_info=get_video_meta_data(video_data["VIDEO"]),
                    video_path=video_data["VIDEO"],
                    data_df=data,
                )
                self.multianimal_identification()
                # TODO WHEN SOMEONE SHARE APT DATA.

    #
    # def import_trk(self):
    #     for file_cnt, file_path in enumerate(self.data_paths):
    #         _, file_name, file_ext = get_fn_ext(file_path)
    #         if self.animal_cnt > 0:
    #             pass
    #         video_path = find_video_of_file(self.video_dir, file_name)
    #         if not video_path:
    #             raise NoFilesFoundError(msg='Could not find a video jj')
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
    #
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


test = APTImporterTRK(
    config_path="/Users/simon/Desktop/envs/troubleshooting/trk_test/project_folder/project_config.ini",
    data_folder="/Users/simon/Desktop/envs/troubleshooting/trk_test/data/",
    id_lst=["Animal_1", "Animal_2", "Animal_3"],
    interpolation_settings="Body-parts: Nearest",
    smoothing_settings={
        "Method": "Savitzky Golay",
        "Parameters": {"Time_window": "200"},
    },
)
test.run()


# test = TRKImporter(config_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini',
#                    data_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/trk_data',
#                    animal_id_lst=['Animal_1', 'Animal_2'],
#                    interpolation_method="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})


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
