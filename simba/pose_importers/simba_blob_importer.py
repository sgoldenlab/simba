import os
from copy import copy
from typing import Optional, Union

import numpy as np
import pandas as pd

from simba.data_processors.interpolate import Interpolate
from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_int,
                                check_str, check_valid_boolean,
                                check_valid_dataframe)
from simba.utils.enums import ConfigKey, Dtypes, Formats, Methods
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)

REQUIRED_FIELDS = ['nose_x', 'nose_y', 'left_x', 'left_y', 'center_x', 'center_y', 'right_x', 'right_y', 'tail_x',
                   'tail_y']
BP_NAMES = ['nose', 'left', 'center', 'right', 'tail']

class SimBABlobImporter(ConfigReader):
    """
    :example:
    >>> r = SimBABlobImporter(config_path=r"C:\troubleshooting\simba_blob_project\project_folder\project_config.ini", data_path=r'C:\troubleshooting\simba_blob_project\data')
    >>> r.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike],
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 smoothing_settings: Optional[dict] = None,
                 interpolation_settings: Optional[dict] = None,
                 verbose: Optional[bool] = True):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        pose_config_name = self.read_config_entry(config=self.config, section=ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                  option=ConfigKey.POSE_SETTING.value, default_value=None,
                                                  data_type=Dtypes.STR.value).strip()
        if pose_config_name != Methods.SIMBA_BLOB.value:
            raise InvalidInputError(
                msg=f'The project {config_path} is not a SimBA blob project. Cannot import SimBA blob data to a non SimBA blob project ({ConfigKey.POSE_SETTING.value}: {pose_config_name}, expected: {Methods.SIMBA_BLOB.value})',
                source=self.__class__.__name__)
        if os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.csv'],
                                                                   raise_error=True)
        elif os.path.isfile(data_path):
            self.data_paths = [data_path]
        else:
            raise NoFilesFoundError(msg=f'{data_path} is not a valid file path or valid directory path',
                                    source=self.__class__.__name__)
        if interpolation_settings is not None:
            check_if_keys_exist_in_dict(data=interpolation_settings, key=['method', 'type'], name=f'{self.__class__.__name__} interpolation_settings')
            check_str(name=f'{self.__class__.__name__} interpolation_settings type', value=interpolation_settings['type'], options=('body-parts', 'animals'))
            check_str(name=f'{self.__class__.__name__} interpolation_settings method', value=interpolation_settings['method'], options=('linear', 'quadratic', 'nearest'))
            self.interpolation_type, self.interpolation_method = interpolation_settings['type'], interpolation_settings['method']
        else:
            self.interpolation_type, self.interpolation_method = None, None
        if smoothing_settings is not None:
            check_if_keys_exist_in_dict(data=smoothing_settings, key=['method', 'time_window'], name=f'{self.__class__.__name__} smoothing_settings')
            check_str(name=f'{self.__class__.__name__} smoothing_settings method', value=smoothing_settings['method'], options=('savitzky-golay', 'gaussian'))
            check_int(name=f'{self.__class__.__name__} smoothing_settings time_window', value=smoothing_settings['time_window'], min_value=1)
            self.smoothing_time, self.smoothing_method = smoothing_settings['time_window'], smoothing_settings['method']
        else:
            self.smoothing_time, self.smoothing_method = None, None
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', raise_error=True)
        else:
            save_dir = copy(self.outlier_corrected_dir)
        self.interpolation_settings, self.smoothing_settings,  = (interpolation_settings, smoothing_settings)
        self.save_dir, self.verbose = save_dir, verbose

    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            file_timer = SimbaTimer(start=True)
            df = read_df(file_path=file_path, file_type='csv')
            df.columns = [x.strip().lower() for x in df.columns]
            video_name = get_fn_ext(filepath=file_path)[1]
            if self.verbose:
                print(f'Importing SimBA blob data for video {video_name}...')
            save_path = os.path.join(self.save_dir, f'{video_name}.csv')
            check_valid_dataframe(df=df, source=f'{self.__class__.__name__} {file_path}',
                                  valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=REQUIRED_FIELDS)
            df = df[REQUIRED_FIELDS].astype(np.int32)
            df_out = pd.DataFrame()
            for i in range(0, df.shape[1], 2):
                df_out = pd.concat(
                    [df_out, df.iloc[:, i:i + 2], pd.DataFrame(1, index=df.index, columns=[f'{BP_NAMES[i // 2]}_p'])],
                    axis=1)
            del df
            write_df(df=df_out, file_type=self.file_type, save_path=save_path, multi_idx_header=False)
            if self.interpolation_settings is not None:
                interpolator = Interpolate(config_path=self.config_path, data_path=save_path, type=self.interpolation_type, method=self.interpolation_method, multi_index_df_headers=False, copy_originals=False)
                interpolator.run()
            if self.smoothing_settings is not None:
                smoother = Smoothing(config_path=self.config_path, data_path=save_path, time_window=self.smoothing_time, method=self.smoothing_method, multi_index_df_headers=False, copy_originals=False)
                smoother.run()
            file_timer.stop_timer()
            print(f'Imported data for video {video_name} (elapsed time: {file_timer.elapsed_time}s)')
        self.timer.stop_timer()
        stdout_success(
            msg=f"{len(self.data_paths)} SimBA blob tracking files file(s) imported to the SimBA project {self.save_dir}",
            source=self.__class__.__name__, elapsed_time=self.timer.elapsed_time)


#
# r = SimBABlobImporter(config_path=r"C:\troubleshooting\simba_blob_project\project_folder\project_config.ini",
#                       data_path=r'C:\troubleshooting\simba_blob_project\data',
#                       smoothing_settings={'method': 'savitzky-golay', 'time_window': 100},
#                       interpolation_settings='Body-parts: Nearest')
# r.run()