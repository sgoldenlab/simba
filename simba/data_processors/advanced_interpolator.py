__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from datetime import datetime

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_instance, check_int,
                                check_str, check_that_column_exist,
                                check_valid_boolean)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats, Options
from simba.utils.errors import DataHeaderError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (copy_files_to_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)

BODY_PART_TYPE = 'body-part'
ANIMAL_TYPE = 'animal'
NEAREST = 'nearest'
LINEAR = 'linear'
QUADRATIC = 'quadratic'

class AdvancedInterpolator(ConfigReader):
    """
    Interpolation method that allows different interpolation parameters for different animals or body-parts.
    For example, interpolate some body-parts of animals using linear interpolation, and other body-parts of animals using nearest interpolation.

    .. image:: _static/img/AdvancedInterpolator.webp
       :width: 600
       :align: center

    .. video:: _static/img/smoothing_example_2.webm
       :width: 800
       :autoplay:
       :loop:

    :param Union[str, os.PathLike] data_path: Path to folder containing pose-estimation data or a file with pose-estimation data.
    :param Union[str, os.PathLike] config_path: Optional path to SimBA project config file in Configparser format.
    :param Literal["animal", "body-part"] type: Type of interpolation: animal or body-part. Default: 'body-part'.
    :param Dict settings: Interpolation rules for each animal or each animal body-part. See examples.
    :param bool verbose: If True, prints progress messages. Default: True.
    :param Union[str, os.PathLike] save_dir: Optional directory to save results. If None, saves in input directory.
    :param bool multi_index_data: If True, the incoming data has multi-index columns. Default: False.
    :param bool save_copy: If True, saves original data in datetime-stamped sub-directory. Default: True.
    :param Optional[int] max_interpolation_length: Maximum length of gaps to interpolate. If None, interpolates all gaps. Default: None.

    :examples:
    >>> # Animal-level interpolation
    >>> interpolator = AdvancedInterpolator(
    ...     data_path='/path/to/project_folder/csv/input_csv',
    ...     config_path='/path/to/project_folder/project_config.ini',
    ...     type='animal',
    ...     settings={'Animal_1': 'linear', 'Animal_2': 'quadratic'},
    ...     multi_index_data=True
    ... )
    >>> interpolator.run()
    >>> 
    >>> # Body-part level interpolation
    >>> interpolator = AdvancedInterpolator(
    ...     data_path='/path/to/project_folder/csv/input_csv',
    ...     config_path='/path/to/project_folder/project_config.ini',
    ...     type='body-part',
    ...     settings={
    ...         'Simon': {
    ...             'Ear_left_1': 'linear',
    ...             'Ear_right_1': 'linear',
    ...             'Nose_1': 'quadratic',
    ...             'Lat_left_1': 'quadratic',
    ...             'Lat_right_1': 'quadratic',
    ...             'Center_1': 'nearest',
    ...             'Tail_base_1': 'nearest'
    ...         },
    ...         'JJ': {
    ...             'Ear_left_2': 'nearest',
    ...             'Ear_right_2': 'nearest',
    ...             'Nose_2': 'quadratic',
    ...             'Lat_left_2': 'quadratic',
    ...             'Lat_right_2': 'quadratic',
    ...             'Center_2': 'linear',
    ...             'Tail_base_2': 'linear'
    ...         }
    ...     },
    ...     multi_index_data=True
    ... )
    >>> interpolator.run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 settings: Dict[str, Any],
                 type: Optional[Literal["animal", "body-part"]] = 'body-part',
                 verbose: Optional[bool] = True,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 multi_index_data: Optional[bool] = False,
                 save_copy: Optional[bool] = True,
                 max_interpolation_length: Optional[int] = None):



        if config_path is not None:
            ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        else:
            self.file_type, self.datetime = Formats.CSV.value, datetime.now().strftime("%Y%m%d%H%M%S")
            self.timer = SimbaTimer(start=True)
        check_str(name=f'{self.__class__.__name__} type', value=type, options=["animal", "body-part"], raise_error=True)
        if os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.file_paths = [data_path]
            self.input_dir = os.path.dirname(data_path)
            self.cpy_dir = os.path.join(os.path.dirname(data_path), f"Pre_Advanced_Interpolation_{self.datetime}")
        elif os.path.isdir(data_path):
            self.file_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=[f".{self.file_type}"], raise_warning=False,raise_error=True)
            self.cpy_dir = os.path.join(data_path, f"Pre_Advanced_Interpolation_{self.datetime}")
            self.input_dir = data_path
        else:
            raise InvalidInputError(msg=f'{data_path} is not a valid file path or file directory', source=self.__class__.__name__)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir)
        else:
            save_dir = self.input_dir

        check_valid_boolean(value=[multi_index_data, save_copy], source=self.__class__.__name__, raise_error=True)
        check_instance(source=self.__class__.__name__, instance=settings, accepted_types=(dict,))
        for animal, animal_data in settings.items():
            if type == BODY_PART_TYPE:
                check_instance(source=self.__class__.__name__, instance=animal_data, accepted_types=(dict,))
                for bp_name, bp_data in animal_data.items():
                    if config_path is not None:
                        check_str(name='method', value=bp_name, options=self.project_bps)
                    check_str(name='method', value=bp_data, options=[LINEAR, NEAREST, QUADRATIC])
            else:
               check_str(name='method', value=animal_data, options=[LINEAR, NEAREST, QUADRATIC])
        self.settings, self.type, self.multi_index_data, self.verbose = settings, type, multi_index_data, verbose
        if type == ANIMAL_TYPE and config_path is not None: self.__transpose_settings()
        self.save_copy, self.save_dir, self.config_path = save_copy, save_dir, config_path
        if max_interpolation_length is not None:
            check_int(name=f'{self.__class__.__name__} max_interpolation_length', min_value=1, raise_error=True, value=max_interpolation_length)
        self.max_interpolation_length = max_interpolation_length
        if save_copy and not os.path.isdir(self.cpy_dir): os.makedirs(self.cpy_dir)

    def __transpose_settings(self):
        """Helper to transpose settings dict if interpolating per animal, so the same method can be used for both animal and body-part interpolation"""
        transposed_settings = {}
        for animal_name, body_part_data in self.animal_bp_dict.items():
            transposed_settings[animal_name] = {}
            for animal_body_part in body_part_data["X_bps"]:
                transposed_settings[animal_name][animal_body_part[:-2]] = self.settings[animal_name]
        self.settings = transposed_settings

    def __insert_multi_index(self, df: pd.DataFrame) -> pd.DataFrame:
        multi_idx_header = []
        for i in range(len(df.columns)):
            multi_idx_header.append(("IMPORTED_POSE", "IMPORTED_POSE", list(df.columns)[i]))
        df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
        return df

    def run(self):
        for file_cnt, file_path in enumerate(self.file_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            if self.config_path is not None:
                df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=self.multi_index_data).fillna(0).reset_index(drop=True)
                if len(df.columns) != len(self.bp_col_names):
                    raise DataHeaderError(msg=f"The SimBA project suggest the data should have {len(self.bp_col_names)} columns, but the input data has {len(df.columns)} columns", source=self.__class__.__name__)
                df.columns = self.bp_headers
                df[df < 0] = 0
            else:
                df = pd.read_csv(filepath_or_buffer=file_path, index_col=0)

            df.columns = [x.lower() for x in df.columns]
            df_cpy = deepcopy(df)
            for animal_name, animal_body_parts in self.settings.items():
                for bp, interpolation_setting in animal_body_parts.items():
                    bp = bp.lower()
                    check_that_column_exist(df=df, column_name=[f"{bp}_x", f"{bp}_y"], file_name=file_path)
                    df[[f"{bp}_x", f"{bp}_y"]] = df[[f"{bp}_x", f"{bp}_y"]].astype(np.int32)
                    if self.max_interpolation_length is None:
                        df[df <= 0] = 0
                        idx = df.loc[(df[f"{bp}_x"] <= 0.0) & (df[f"{bp}_y"] <= 0.0)].index.tolist()
                        if self.verbose: print(f"Interpolating {len(idx)} {bp} body-parts in video {video_name}...")
                        df[[f"{bp}_x", f"{bp}_y"]] = (df[[f"{bp}_x", f"{bp}_y"]].interpolate(method=interpolation_setting, axis=0).ffill().bfill().astype(np.int32))
                        df[[f"{bp}_x", f"{bp}_y"]][df[[f"{bp}_x", f"{bp}_y"]] < 0] = 0
                    else:
                        df_cpy.loc[df_cpy[f'{bp}_x'] <= 0, f'{bp}_x'] = 0
                        df_cpy.loc[df_cpy[f'{bp}_y'] <= 0, f'{bp}_y'] = 0
                        idx = df.loc[(df[f"{bp}_x"] <= 0.0) & (df[f"{bp}_y"] <= 0.0)].index.tolist()
                        df_cpy[f'{bp}_temp'] = 0
                        df_cpy.loc[idx, [f'{bp}_temp']] = 1
                        bouts = detect_bouts(data_df=df_cpy, target_lst=[f'{bp}_temp'], fps=1)
                        bouts = bouts[bouts['Bout_time'] <= self.max_interpolation_length]
                        if len(bouts) > 0:
                            idx = bouts.apply(lambda row: list(range(row['Start_frame'], row['End_frame'] + 1)), axis=1).explode().tolist()
                            if self.verbose: print(f"Interpolating {len(idx)} {bp} body-parts in video {video_name}...")
                            df.loc[idx, [f"{bp}_x", f"{bp}_y"]] = np.nan
                            df[[f"{bp}_x", f"{bp}_y"]] = (df[[f"{bp}_x", f"{bp}_y"]].interpolate(method=interpolation_setting, axis=0).astype(np.int32))
            if self.multi_index_data:
                df = self.__insert_multi_index(df=df)
            if self.save_copy:
                copy_files_to_directory(file_paths=[file_path], dir=self.cpy_dir, verbose=False)
            save_path = os.path.join(self.save_dir, f'{video_name}{self.file_type}')
            write_df(df=df, file_type=self.file_type, save_path=save_path, multi_idx_header=self.multi_index_data)
            video_timer.stop_timer()
            print(f'Video {video_name} complete. Elapsed time {video_timer.elapsed_time_str}s')
        self.timer.stop_timer()
        if self.save_copy:
            msg = f"Advanced interpolation complete. Data saved in {self.save_dir}. Original copies saved in {self.cpy_dir}."
        else:
            msg = f"Advanced interpolation complete. Data saved in {self.save_dir}."
        stdout_success(msg=msg, elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

# SMOOTHING_SETTINGS = {'Simon': {'Ear_left_1': {'method': 'Savitzky Golay', 'time_window': 3500},
#                                'Ear_right_1': {'method': 'Gaussian', 'time_window': 500},
#                                'Nose_1': {'method': 'Savitzky Golay', 'time_window': 2000},
#                                'Lat_left_1': {'method': 'Savitzky Golay', 'time_window': 2000},
#                                'Lat_right_1': {'method': 'Gaussian', 'time_window': 2000},
#                                'Center_1': {'method': 'Savitzky Golay', 'time_window': 2000},
#                                'Tail_base_1': {'method': 'Gaussian', 'time_window': 500}},
#                         'JJ': {'Ear_left_2': {'method': 'Savitzky Golay', 'time_window': 2000},
#                                'Ear_right_2': {'method': 'Savitzky Golay', 'time_window': 500},
#                                'Nose_2': {'method': 'Gaussian', 'time_window': 3500},
#                                'Lat_left_2': {'method': 'Savitzky Golay', 'time_window': 500},
#                                'Lat_right_2': {'method': 'Gaussian', 'time_window': 3500},
#                                'Center_2': {'method': 'Gaussian', 'time_window': 2000},
#                                'Tail_base_2': {'method': 'Savitzky Golay', 'time_window': 3500}}}
#
#
# INTERPOLATION_SETTINGS = {'Animal_1': {'NOSE': 'linear',
#                           'LEFT_EAR': 'linear',
#                           'RIGHT_EAR': 'quadratic',
#                           'LEFT_SIDE': 'quadratic',
#                           'CENTER': 'quadratic',
#                           'RIGHT_SIDE': 'nearest',
#                           'TAIL_BASE': 'nearest'}}

# advanced_interpolator = AdvancedInterpolator(data_path=r'D:\netholabs\data', settings=INTERPOLATION_SETTINGS, type='body-part', multi_index_data=True, save_copy=False, max_interpolation_length=100)

# advanced_interpolator.run()

# for animal, animal_data in settings.items():
#     check_instance(source=self.__class__.__name__, instance=animal_data, accepted_types=(dict,))
#     if type == BODY_PART_TYPE:
#         for bp_name, bp_data in animal_data.items():
#             check_if_keys_exist_in_dict(data=bp_data, key=['method', 'time_window'])
#             check_str(name='method', value=bp_data['method'], options=[GAUSSIAN, SAVITZKY_GOLAY])
#             check_int(name='time_window', value=bp_data['time_window'], min_value=1)
#     else:
#         check_if_keys_exist_in_dict(data=animal_data, key=['method', 'time_window'])
#         check_str(name='method', value=animal_data['method'], options=[GAUSSIAN, SAVITZKY_GOLAY])
#         check_int(name='time_window', value=animal_data['time_window'], min_value=1)