__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from simba.data_processors.interpolate import Interpolate
from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_instance,
                                check_int, check_str, check_valid_lst)
from simba.utils.enums import Formats, Methods, Options
from simba.utils.errors import BodypartColumnNotFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_all_videos_in_project,
                                    get_video_meta_data, write_df)


class MADLCImporterH5(ConfigReader, PoseImporterMixin):
    """
    Importing multi-animal deeplabcut (maDLC) pose-estimation data (in H5 format)
    into a SimBA project in parquet or CSV format.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str data_folder: Path to folder containing maDLC data in ``.h5`` format.
    :parameter str file_type: Method used to perform pose-estimation in maDLC. OPTIONS: `skeleton`, `box`, `ellipse`.
    :param List[str] id_lst: Names of animals.
    :parameter Optional[Dict[str, str]] interpolation_setting: Dict defining the type and method to use to perform interpolation {'type': 'animals', 'method': 'linear'}.
    :parameter Optional[Dict[str, Union[str, int]]] smoothing_settings: Dictionary defining the pose estimation smoothing method {'time_window': 500, 'method': 'gaussian'}.

    .. note::
       `Multi-animal import tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md>`__.

    :examples:
    >>> _ = MADLCImporterH5(config_path=r'MyConfigPath', data_folder=r'maDLCDataFolder', file_type='ellipse', id_lst=['Animal_1', 'Animal_2'], interpolation_settings={'type': 'animals', 'method': 'linear'}, smoothing_settings={'time_window': 500, 'method': 'gaussian'}).run()

    References
    ----------
    .. [1] Lauer et al., Multi-animal pose estimation, identification and tracking with DeepLabCut, `Nature Methods`,
           2022.
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_folder: Union[str, os.PathLike],
                 file_type: Literal['skeleton', 'box', 'ellipse'],
                 id_lst: List[str],
                 interpolation_settings: Optional[Dict[str, str]] = None,
                 smoothing_settings: Optional[Dict[str, Any]] = None):

        check_file_exist_and_readable(file_path=config_path)
        check_if_dir_exists(in_dir=data_folder)
        check_str(name=f'{self.__class__.__name__} file_type', value=file_type, options=Options.MULTI_DLC_TYPE_IMPORT_OPTION.value)
        check_valid_lst(data=id_lst, source=f'{self.__class__.__name__} id_lst', valid_dtypes=(str,))
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(data=interpolation_settings, key=['method', 'type'], name=f'{self.__class__.__name__} interpolation_settings')
            check_str(name=f'{self.__class__.__name__} interpolation_settings type', value=interpolation_settings['type'], options=('body-parts', 'animals'))
            check_str(name=f'{self.__class__.__name__} interpolation_settings method', value=interpolation_settings['method'], options=('linear', 'quadratic', 'nearest'))
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(data=smoothing_settings, key=['method', 'time_window'], name=f'{self.__class__.__name__} smoothing_settings')
            check_str(name=f'{self.__class__.__name__} smoothing_settings method', value=smoothing_settings['method'], options=('savitzky-golay', 'gaussian'))
            check_int(name=f'{self.__class__.__name__} smoothing_settings time_window', value=smoothing_settings['time_window'], min_value=1)

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PoseImporterMixin.__init__(self)
        self.interpolation_settings, self.smoothing_settings = interpolation_settings, smoothing_settings
        self.data_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(self.logs_path, f"data_import_log_{self.datetime}.csv")
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
        self.input_data_paths = self.find_data_files(dir=self.data_folder, extensions=Formats.DLC_FILETYPES.value[file_type])
        self.data_and_videos_lk = self.link_video_paths_to_data_paths(data_paths=self.input_data_paths, video_paths=self.video_paths, str_splits=Formats.DLC_NETWORK_FILE_NAMES.value)
        if self.pose_setting is Methods.USER_DEFINED.value:
            self.__update_config_animal_cnt()
        if self.animal_cnt > 1:
            self.check_multi_animal_status()
            self.animal_bp_dict = self.create_body_part_dictionary(self.multi_animal_status, self.id_lst, self.animal_cnt, self.x_cols, self.y_cols, self.p_cols, self.clr_lst)
            if self.pose_setting is Methods.USER_DEFINED.value:
                self.update_bp_headers_file(update_bp_headers=True)
        print(f"Importing {len(list(self.data_and_videos_lk.keys()))} multi-animal DLC file(s)...")

    def run(self):
        import_log = pd.DataFrame(columns=["VIDEO", "IMPORT_TIME", "IMPORT_SOURCE", "INTERPOLATION_SETTING", "SMOOTHING_SETTING"])
        for cnt, (video_name, video_data) in enumerate(self.data_and_videos_lk.items()):
            video_timer = SimbaTimer(start=True)
            self.add_spacer, self.frame_no, self.video_data, self.video_name = (2, 1, video_data, video_name)
            print(f"Processing {video_name} ({cnt+1}/{len(self.input_data_paths)})...")
            self.data_df = pd.read_hdf(video_data["DATA"]).replace([np.inf, -np.inf], np.nan).fillna(0)
            if len(self.data_df.columns) != len(self.bp_headers):
                raise BodypartColumnNotFoundError(
                    msg=f'The number of body-parts in data file {video_data["DATA"]} do not match the number of body-parts in your SimBA project. '
                    f"The number of of body-parts expected by your SimBA project is {int(len(self.bp_headers) / 3)}. "
                    f'The number of of body-parts contained in data file {video_data["DATA"]} is {int(len(self.data_df.columns) / 3)}. '
                    f"Make sure you have specified the correct number of animals and body-parts in your project. NOTE: The project body-parts is stored at {self.body_parts_path}."
                )
            self.data_df.columns = self.bp_headers
            self.out_df = deepcopy(self.data_df)
            if self.animal_cnt > 1:
                self.initialize_multi_animal_ui(animal_bp_dict=self.animal_bp_dict, video_info=get_video_meta_data(video_data["VIDEO"]), data_df=self.data_df, video_path=video_data["VIDEO"])
                self.multianimal_identification()
            self.save_path = os.path.join(os.path.join(self.input_csv_dir, f"{self.video_name}.{self.file_type}"))
            write_df(df=self.out_df, file_type=self.file_type, save_path=self.save_path, multi_idx_header=True)
            if self.interpolation_settings is not None:
                interpolator = Interpolate(config_path=self.config_path, data_path=self.save_path, type=self.interpolation_settings['type'], method=self.interpolation_settings['method'], multi_index_df_headers=True, copy_originals=False)
                interpolator.run()
            if self.smoothing_settings is not None:
                smoother = Smoothing(config_path=self.config_path, data_path=self.save_path, time_window=self.smoothing_settings['time_window'], method=self.smoothing_settings['method'], multi_index_df_headers=True, copy_originals=False)
                smoother.run()
            video_timer.stop_timer()
            stdout_success(msg=f"Video {video_name} data imported...", elapsed_time=video_timer.elapsed_time_str)
        self.timer.stop_timer()
        stdout_success(msg="All maDLC H5 data files imported", elapsed_time=self.timer.elapsed_time_str)


# test = MADLCImporterH5(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                    data_folder=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/h5',
#                    file_type='ellipse',
#                    id_lst=['Simon', 'JJ'],
#                    interpolation_settings= {'type': 'animals', 'method': 'linear'},
#                    smoothing_settings = {'time_window': 500, 'method': 'gaussian'})
# test.run()
