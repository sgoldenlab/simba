#### MODIFIED FROM @Toshea111 - https://github.com/Toshea111/sleap/blob/develop/docs/notebooks/Convert_HDF5_to_CSV_updated.ipynb
import numpy as np
import pandas as pd
import h5py
import os
from copy import deepcopy

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.data_processors.interpolation_smoothing import Smooth, Interpolate
from simba.utils.errors import BodypartColumnNotFoundError
from simba.utils.read_write import get_video_meta_data, write_df, find_all_videos_in_project
from simba.utils.printing import stdout_warning, stdout_success, SimbaTimer
from simba.utils.enums import Methods

class SLEAPImporterH5(ConfigReader, PoseImporterMixin):
    """
    Importing SLEAP pose-estimation data into SimBA project in ``H5`` format

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str data_folder: Path to folder containing SLEAP data in `csv` format.
    :parameter List[str] id_lst: Animal names. This will be ignored in one animal projects and default to ``Animal_1``.
    :parameter str interpolation_settings: String defining the pose-estimation interpolation method. OPTIONS: 'None', 'Animal(s): Nearest',
        'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear',
        'Body-parts: Quadratic'.
    :parameter str smoothing_settings: Dictionary defining the pose estimation smoothing method. EXAMPLE: {'Method': 'Savitzky Golay',
        'Parameters': {'Time_window': '200'}}

    .. note::
       `Multi-animal import tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md>`__.

    References
    ----------
    .. [1] Pereira et al., SLEAP: A deep learning system for multi-animal pose tracking, `Nature Methods`,
           2022.

    >>> sleap_h5_importer = SLEAPImporterH5(config_path=r'project_folder/project_config.ini', data_folder=r'data_folder', actor_IDs=['Termite_1', 'Termite_2', 'Termite_3', 'Termite_4', 'Termite_5'], interpolation_settings="Body-parts: Nearest", smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
    >>> sleap_h5_importer.run()
    """

    def __init__(self,
                 config_path: str,
                 data_folder: str,
                 id_lst: list,
                 interpolation_settings: str,
                 smoothing_settings: dict):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PoseImporterMixin.__init__(self)
        self.interpolation_settings, self.smoothing_settings = interpolation_settings, smoothing_settings
        self.data_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(self.logs_path, f'data_import_log_{self.datetime}.csv')
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
        self.input_data_paths = self.find_data_files(dir=self.data_folder, extensions=['.h5'])
        self.data_and_videos_lk = self.link_video_paths_to_data_paths(data_paths=self.input_data_paths, video_paths=self.video_paths)
        if (self.pose_setting is Methods.USER_DEFINED.value):
            self.__update_config_animal_cnt()
        if self.animal_cnt > 1:
            self.check_multi_animal_status()
            self.animal_bp_dict = self.create_body_part_dictionary(self.multi_animal_status, self.id_lst, self.animal_cnt, self.x_cols, self.y_cols, self.p_cols, self.clr_lst)
            self.update_bp_headers_file()
        print(f'Importing {len(list(self.data_and_videos_lk.keys()))} file(s)...')

    def run(self):
        import_log = pd.DataFrame(columns=['VIDEO', 'IMPORT_TIME', 'IMPORT_SOURCE', 'INTERPOLATION_SETTING', 'SMOOTHING_SETTING'])
        for file_cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            print(f'Importing {video_name}...')
            video_timer = SimbaTimer(start=True)
            self.video_name = video_name
            try:
                with h5py.File(video_data['DATA'], "r") as sleap_dict:
                    data = {k: v[()] for k, v in sleap_dict.items()}
                    data["point_scores"] = np.transpose(data["point_scores"][0])
                    data["node_names"] = [s.decode() for s in data["node_names"].tolist()]
                    data["track_names"] = [s.decode() for s in data["track_names"].tolist()]
                    data["tracks"] = np.transpose(data["tracks"])
                    data["track_occupancy"] = data["track_occupancy"].astype(bool)
            except OSError:
                stdout_warning(msg=f'{video_data["DATA"]} is not a valid SLEAP H5 file. Skipping {video_name}...')
                continue
            valid_frame_idxs = np.argwhere(data["track_occupancy"].any(axis=1)).flatten()
            tracks = []
            for frame_idx in valid_frame_idxs:
                frame_tracks = data["tracks"][frame_idx]
                for i in range(frame_tracks.shape[-1]):
                    pts = frame_tracks[..., i]
                    if np.isnan(pts).all():
                        continue
                    detection = {"track": data["track_names"][i], "frame_idx": frame_idx}
                    for node_name, (x, y) in zip(data["node_names"], pts):
                        detection[f"{node_name}.x"] = x
                        detection[f"{node_name}.y"] = y
                    tracks.append(detection)
            self.data_df = pd.DataFrame(tracks).fillna(0)
            idx = self.data_df.iloc[:, :2]
            idx['track'] = pd.Categorical(idx['track'])
            idx['track'] = idx['track'].cat.codes.astype(int)
            self.data_df = self.data_df.iloc[:, 2:]
            if self.animal_cnt > 1:
                self.data_df = pd.DataFrame(self.transpose_multi_animal_table(data=self.data_df.values, idx=idx.values, animal_cnt=self.animal_cnt))
                self.data_df = self.data_df.reindex(range(0, self.data_df.index[-1] + 1), fill_value=0)
                p_df = pd.DataFrame(data['point_scores'], index=self.data_df.index, columns=self.data_df.columns[1::2] + .5).fillna(0).clip(0.0, 1.0)
                self.data_df = pd.concat([self.data_df, p_df], axis=1).sort_index(axis=1)
                if len(self.data_df.columns) != len(self.bp_headers):
                    raise BodypartColumnNotFoundError(
                        msg=f'The number of body-parts in data file {video_data["DATA"]} do not match the number of body-parts in your SimBA project. '
                            f'The number of of body-parts expected by your SimBA project is {int(len(self.bp_headers) / 3)}. '
                            f'The number of of body-parts contained in data file {video_data["DATA"]} is {int(len(self.data_df.columns) / 3)}. '
                            f'Make sure you have specified the correct number of animals and body-parts in your project.')
                self.data_df.columns = self.bp_headers
            else:
                idx = list(idx.drop('track', axis=1)['frame_idx'])
                self.data_df = self.data_df.set_index([idx]).sort_index()
                self.data_df.columns = np.arange(len(self.data_df.columns))
                self.data_df = self.data_df.reindex(range(0, self.data_df.index[-1] + 1), fill_value=0)
                p_df = pd.DataFrame(data['point_scores'], index=self.data_df.index, columns=self.data_df.columns[1::2] + .5).fillna(0).clip(0.0, 1.0)
                self.data_df = pd.concat([self.data_df, p_df], axis=1).sort_index(axis=1)
                if len(self.data_df.columns) != len(self.bp_headers):
                    raise BodypartColumnNotFoundError(
                        msg=f'The number of body-parts in data file {video_data["DATA"]} do not match the number of body-parts in your SimBA project. '
                            f'The number of of body-parts expected by your SimBA project is {int(len(self.bp_headers) / 3)}. '
                            f'The number of of body-parts contained in data file {video_data["DATA"]} is {int(len(self.data_df.columns) / 3)}. '
                            f'Make sure you have specified the correct number of animals and body-parts in your project.')
                self.data_df.columns = self.bp_headers
                multi_idx_header = []
                for i in range(len(self.data_df.columns)): multi_idx_header.append(('IMPORTED_POSE', 'IMPORTED_POSE', list(self.data_df.columns)[i]))
                self.data_df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
                self.out_df = deepcopy(self.data_df)

            if self.animal_cnt > 1:
                self.initialize_multi_animal_ui(animal_bp_dict=self.animal_bp_dict,
                                                video_info=get_video_meta_data(video_data['VIDEO']),
                                                data_df=self.data_df,
                                                video_path=video_data['VIDEO'])
                self.multianimal_identification()
            self.save_path = os.path.join(os.path.join(self.input_csv_dir, f'{self.video_name}.{self.file_type}'))
            write_df(df=self.out_df, file_type=self.file_type, save_path=self.save_path, multi_idx_header=True)
            if self.interpolation_settings != 'None':
                self.__run_interpolation()
            if self.smoothing_settings['Method'] != 'None':
                self.__run_smoothing()
            video_timer.stop_timer()
            stdout_success(msg=f'Video {video_name} data imported...', elapsed_time=video_timer.elapsed_time_str)
        self.timer.stop_timer()
        stdout_success(msg='All SLEAP H5 data files imported', elapsed_time=self.timer.elapsed_time_str)


    def __run_interpolation(self):
        print(f'Interpolating missing values in video {self.video_name} (Method: {self.interpolation_settings})...')
        _ = Interpolate(input_path=self.save_path,config_path=self.config_path, method=self.interpolation_settings, initial_import_multi_index=True)

    def __run_smoothing(self):
        print(f'Performing {self.smoothing_settings["Method"]} smoothing on video {self.video_name}...')
        Smooth(config_path=self.config_path,
               input_path=self.save_path,
               time_window=int(self.smoothing_settings['Parameters']['Time_window']),
               smoothing_method=self.smoothing_settings['Method'],
               initial_import_multi_index=True)

#
# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_dropped_frms/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_dropped_frms/data',
#                    id_lst=['Simon'],
#                    interpolation_settings= 'None', #'"Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'None', 'Parameters': {'Time_window': '200'}})
# test.run()






# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/import_h5',
#                    id_lst=['Simon', 'Nastacia', 'Ben', 'John', 'JJ'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()
# print('All SLEAP imports complete.')

# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/SLEAP_2_Animals_16_body_parts/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/SLEAP_2_Animals_16_body_parts/data/h5',
#                    id_lst=['Simon', 'Nastacia'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()
# print('All SLEAP imports complete.')







# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/Termites_5/import_h5',
#                    id_lst=['Simon', 'Nastacia', 'Ben', 'John', 'JJ'],
#                    interpolation_settings='None',
#                    smoothing_settings = {'Method': 'None'})
# test.run()
# print('All SLEAP imports complete.')

# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_cody/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_cody/import_h5',
#                    actor_IDs=['Nastacia', 'Sam'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.import_sleap()
# print('All SLEAP imports complete.')


# test = SLEAPImporterH5(config_path="/Users/simon/Desktop/envs/troubleshooting/sleap_cody/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_cody/import_h5',
#                    actor_IDs=['Nastacia'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.import_sleap()
# # print('All SLEAP imports complete.')
