__author__ = "Simon Nilsson"

import os
from datetime import datetime
import pandas as pd
import numpy as np

from simba.data_processors.interpolation_smoothing import Smooth, Interpolate
from simba.utils.errors import BodypartColumnNotFoundError
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import write_df, get_video_meta_data, find_all_videos_in_project

from simba.utils.enums import Formats
from simba.mixins.pose_importer_mixin import PoseImporterMixin

class MADLC_Importer(ConfigReader, PoseImporterMixin):
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
        PoseImporterMixin.__init__(self)

        self.interpolation_settings, self.smoothing_settings = interpolation_settings, smoothing_settings
        self.data_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(self.logs_path, f'data_import_log_{self.datetime}.csv')
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
        self.input_data_paths = self.find_data_files(dir=self.data_folder, extensions=Formats.DLC_FILETYPES.value[file_type])
        self.data_and_videos_lk = self.link_video_paths_to_data_paths(data_paths=self.input_data_paths, video_paths=self.video_paths, str_splits=Formats.DLC_NETWORK_FILE_NAMES.value)
        print(f'Importing {len(list(self.data_and_videos_lk.keys()))} file(s)...')

    def run(self):
        import_log = pd.DataFrame(columns=['VIDEO', 'IMPORT_TIME', 'IMPORT_SOURCE', 'INTERPOLATION_SETTING', 'SMOOTHING_SETTING'])
        for cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            self.add_spacer, self.frame_no, self.video_data, self.video_name = 2, 1, video_data, video_name
            print(f'Processing {video_name} ...')
            self.data_df = pd.read_hdf(video_data['DATA']).replace([np.inf, -np.inf], np.nan).fillna(0)
            if len(self.data_df.columns) != len(self.bp_headers):
                raise BodypartColumnNotFoundError(msg=f'The number of body-parts in data file {video_data["DATA"]} do not match the number of body-parts in your SimBA project. '
                      f'The number of of body-parts expected by your SimBA project is {int(len(self.bp_headers) / 3)}. '
                      f'The number of of body-parts contained in data file {video_data["DATA"]} is {int(len(self.data_df.columns) / 3)}. '
                      f'Make sure you have specified the correct number of animals and body-parts in your project.')
            self.data_df.columns = self.bp_headers
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

    def __run_interpolation(self):
        print('Interpolating missing values in video {} (Method: {}) ...'.format(self.video_name, self.interpolation_settings))
        _ = Interpolate(input_path=self.save_path,config_path=self.config_path, method=self.interpolation_settings, initial_import_multi_index=True)

    def __run_smoothing(self):
        print(f'Performing {self.smoothing_settings["Method"]} smoothing on video {self.video_name}...')
        Smooth(config_path=self.config_path,
               input_path=self.save_path,
               time_window=int(self.smoothing_settings['Parameters']['Time_window']),
               smoothing_method=self.smoothing_settings['Method'],
               initial_import_multi_index=True)


test = MADLC_Importer(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
                   data_folder=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/h5',
                   file_type='ellipse',
                   id_lst=['Simon', 'JJ'],
                   interpolation_settings='Body-parts: Nearest',
                   smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
test.run()
