__author__ = "Simon Nilsson"

import pandas as pd
import os
import numpy as np
from copy import deepcopy

from simba.data_processors.interpolation_smoothing import Interpolate, Smooth
from simba.utils.read_write import find_all_videos_in_project
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.utils.read_write import write_df, find_video_of_file, get_video_meta_data, get_fn_ext
from simba.utils.printing import stdout_success

class SLEAPImporterCSV(ConfigReader, PoseImporterMixin):

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

    >>> sleap_csv_importer = SLEAPImporterCSV(config_path=r'/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/project_folder/project_config.ini', data_folder=r'/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/import/temp', actor_IDs=['Termite_1', 'Termite_2', 'Termite_3', 'Termite_4', 'Termite_5'], interpolation_settings="Body-parts: Nearest", smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
    >>> sleap_csv_importer.run()
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
        self.input_data_paths = self.find_data_files(dir=self.data_folder, extensions=['.csv'])
        self.data_and_videos_lk = self.link_video_paths_to_data_paths(data_paths=self.input_data_paths, video_paths=self.video_paths)
        print(f'Importing {len(list(self.data_and_videos_lk.keys()))} file(s)...')


    def run(self):
        for file_cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            print(f'Analysing {video_name}...')
            self.video_name = video_name
            self.save_path = os.path.join(os.path.join(self.input_csv_dir, f'{self.video_name}.{self.file_type}'))
            data_df = pd.read_csv(video_data['DATA'])
            idx = data_df.iloc[:, :2]
            idx['track'] = idx['track'].str.replace(r'[^\d.]+', '').astype(int)
            data_df = data_df.iloc[:, 2:]
            if self.animal_cnt > 1:
                self.data_df = pd.DataFrame(self.transpose_multi_animal_table(data=data_df.values, idx=idx.values, animal_cnt=self.animal_cnt))
                p_df = pd.DataFrame(1.0, index=self.data_df.index, columns=self.data_df.columns[1::2] + .5)
                self.data_df = pd.concat([self.data_df, p_df], axis=1).sort_index(axis=1)
                self.data_df.columns = self.bp_headers
            else:
                idx = list(idx.drop('track', axis=1)['frame_idx'])
                self.data_df = data_df.set_index([idx]).sort_index()
                self.data_df.columns = np.arange(len(self.data_df.columns))
                self.data_df = self.data_df.reindex(range(self.data_df.index[0], self.data_df.index[-1] + 1), fill_value=0)
                p_df = pd.DataFrame(1.0, index=self.data_df.index, columns=self.data_df.columns[1::2] + .5)
                self.data_df = pd.concat([self.data_df, p_df], axis=1).sort_index(axis=1)
                self.data_df.columns = self.bp_headers
                self.out_df = deepcopy(self.data_df)

            if self.animal_cnt > 1:
                self.initialize_multi_animal_ui(animal_bp_dict=self.animal_bp_dict,
                                                video_info=get_video_meta_data(video_data['VIDEO']),
                                                data_df=self.data_df,
                                                video_path=video_data['VIDEO'])
                self.multianimal_identification()
            write_df(df=self.out_df, file_type=self.file_type, save_path=self.save_path, multi_idx_header=True)
            if self.interpolation_settings != 'None':
                self.__run_interpolation()
            if self.smoothing_settings['Method'] != 'None':
                self.__run_smoothing()
        stdout_success(msg=f'{len(list(self.data_and_videos_lk.keys()))} file(s) imported to the SimBA project (project_folder/csv/input_csv directory)')

    def __run_interpolation(self):
        print(f'Interpolating missing values in video {self.video_name} (Method: {self.interpolation_settings})...')
        _ = Interpolate(input_path=self.save_path, config_path=self.config_path, method=self.interpolation_settings, initial_import_multi_index=True)

    def __run_smoothing(self):
        print(f'Performing {self.smoothing_settings["Method"]} smoothing on video {self.video_name}...')
        Smooth(config_path=self.config_path,
               input_path=self.save_path,
               time_window=int(self.smoothing_settings['Parameters']['Time_window']),
               smoothing_method=self.smoothing_settings['Method'],
               initial_import_multi_index=True)

# test = SLEAPImporterCSV(config_path=r'/Users/simon/Desktop/envs/troubleshooting/Hornet/project_folder/project_config.ini',
#                  data_folder=r'/Users/simon/Desktop/envs/troubleshooting/Hornet_single_slp/import',
#                  id_lst=['Hornet'],
#                  interpolation_settings="Body-parts: Nearest",
#                  smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()


# test = SLEAPImporterCSV(config_path=r'/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/project_folder',
#                  data_folder='/Users/simon/Desktop/envs/troubleshooting/slp_1_animal_1_bp/import',
#                  actor_IDs=['Termite_1'],
#                  interpolation_settings="Body-parts: Nearest",
#                  smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
# test.run()