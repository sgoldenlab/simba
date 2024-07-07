__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable, check_instance,
                                check_int, check_str, check_that_column_exist,
                                check_valid_boolean)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.enums import TagNames
from simba.utils.errors import (DataHeaderError, InvalidInputError,
                                NoFilesFoundError)
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (copy_files_to_directory,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df, write_df)

BODY_PART_TYPE = 'body-part'
ANIMAL_TYPE = 'animal'
GAUSSIAN = 'gaussian'
SAVITZKY_GOLAY = 'savitzky_golay'
TIME_WINDOW = 'time_window'

class AdvancedSmoother(ConfigReader):
    """
    Smoothing method that allows different smoothing parameters for different animals or body-parts.
    For example, smooth some body-parts of animals using Savitzky-Golay smoothing, and other body-parts of animals using Gaussian smoothing.

    .. image:: _static/img/AdvancedSmoother.webp
       :width: 600
       :align: center

    :parameter str data_dir: path to pose-estimation data in CSV or parquet format
    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter Literal type: Level of smoothing: animal or body-part.
    :parameter Dict settings: Smoothing rules for each animal or each animal body-part.
    :parameter bool initial_import_multi_index: If True, the incoming data is multi-index columns dataframes. Use of input data is the ``project_folder/csv/input_csv`` directory. Default: False.
    :parameter bool overwrite: If True, overwrites the input data. If False, then saves a copy input data in datetime-stamped sub-directory.
    :parameter Optional[verbose] bool: If True, prints the progress. Default: True.

    :examples:
    >>> smoother = AdvancedSmoother(data_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv',
    >>>                             config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
    >>>                             type='animal',
    >>>                             settings={'Simon': {'method': 'Savitzky Golay', 'time_window': 200},
    >>>                                       'JJ': {'method': 'Savitzky Golay', 'time_window': 200}},
    >>>                             initial_import_multi_index=True,
    >>>                             overwrite=False)
    >>> smoother.run()
    >>> SMOOTHING_SETTINGS = {'Simon': {'Ear_left_1': {'method': 'savitzky_golay', 'time_window': 3500},
    >>>                            'Ear_right_1': {'method': 'gaussian', 'time_window': 500},
    >>>                            'Nose_1': {'method': 'savitzky_golay', 'time_window': 2000},
    >>>                            'Lat_left_1': {'method': 'savitzky_golay', 'time_window': 2000},
    >>>                            'Lat_right_1': {'method': 'gaussian', 'time_window': 2000},
    >>>                            'Center_1': {'method': 'savitzky_golay', 'time_window': 2000},
    >>>                            'Tail_base_1': {'method': 'gaussian', 'time_window': 500}},
    >>>                     'JJ': {'Ear_left_2': {'method': 'savitzky_golay', 'time_window': 2000},
    >>>                            'Ear_right_2': {'method': 'savitzky_golay', 'time_window': 500},
    >>>                            'Nose_2': {'method': 'gaussian', 'time_window': 3500},
    >>>                            'Lat_left_2': {'method': 'savitzky_golay', 'time_window': 500},
    >>>                            'Lat_right_2': {'method': 'gaussian', 'time_window': 3500},
    >>>                            'Center_2': {'method': 'gaussian', 'time_window': 2000},
    >>>                            'Tail_base_2': {'method': 'savitzky_golay', 'time_window': 3500}}}
    >>> advanced_smoother = AdvancedSmoother(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
    >>>                  data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/new_data',
    >>>                  settings=SMOOTHING_SETTINGS, type='body-part', multi_index_data=True, overwrite=False)
    >>> advanced_smoother.run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 config_path: Union[str, os.PathLike],
                 settings: Dict[str, Any],
                 type: Optional[Literal["animal", "body-part"]] = 'body-part',
                 verbose: Optional[bool] = True,
                 multi_index_data: Optional[bool] = False,
                 overwrite: Optional[bool] = True):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=f"data_dir: {data_path}, type: {type}, settings: {settings}, initial_import_multi_index: {multi_index_data}, overwrite: {overwrite}",)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        check_str(name=f'{self.__class__.__name__} type', value=type, options=[ANIMAL_TYPE, BODY_PART_TYPE], raise_error=True)
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
        check_valid_boolean(value=[multi_index_data, overwrite], source=self.__class__.__name__, raise_error=True)
        check_instance(source=self.__class__.__name__, instance=settings, accepted_types=(dict,))
        for animal, animal_data in settings.items():
            if type == BODY_PART_TYPE:
                check_instance(source=self.__class__.__name__, instance=animal_data, accepted_types=(dict,))
                for bp_name, bp_data in animal_data.items():
                    check_str(name='body_part', value=bp_name, options=self.project_bps)
                    check_str(name='method', value=bp_data['method'], options=[SAVITZKY_GOLAY, GAUSSIAN])
                    check_int(name='method', value=bp_data[TIME_WINDOW], min_value=1)
            else:
                check_str(name='method', value=animal_data['method'], options=[SAVITZKY_GOLAY, GAUSSIAN])
                check_int(name='method', value=animal_data[TIME_WINDOW], min_value=1)
        self.settings, self.type, self.multi_index_data, self.verbose = settings, type, multi_index_data, verbose
        if type == ANIMAL_TYPE:
            self.__transpose_settings()

        self.overwrite = overwrite
        if not overwrite and not os.path.isdir(self.cpy_dir): os.makedirs(self.cpy_dir)

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
            df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=self.multi_index_data).fillna(0).reset_index(drop=True)
            if self.verbose: print(f"Smoothing data in video {video_name} ({file_cnt+1}/{len(self.file_paths)})...")
            if len(df.columns) != len(self.bp_col_names):
                raise DataHeaderError(msg=f"The SimBA project suggest the data should have {len(self.bp_col_names)} columns, but the input data has {len(df.columns)} columns", source=self.__class__.__name__)
            df.columns = self.bp_headers
            df[df < 0] = 0
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, warning=False, raise_error=False)
            if video_path is None:
                try:
                    video_meta_data = {}
                    self.video_info_df = self.read_video_info_csv(file_path=self.video_info_path)
                    _, _, fps = self.read_video_info(video_name=video_name)
                    video_meta_data["fps"] = fps
                except:
                    raise NoFilesFoundError(msg=f"No video for file {video_name} found in SimBA project. Import the video before doing smoothing. To perform smoothing, SimBA needs the video fps from the video itself OR the logs/video_info.csv file in order to read the video FPS.", source=self.__class__.__name__)
            else:
                video_meta_data = get_video_meta_data(video_path=video_path)
            out_df = deepcopy(df)
            for animal_name, animal_body_parts in self.settings.items():
                for bp, smoothing_setting in animal_body_parts.items():
                    if self.verbose: print(f"Smoothing body-part {bp} in video {video_name} using method {smoothing_setting['method']} (time window: {smoothing_setting[TIME_WINDOW]}ms)...")
                    check_that_column_exist(df=df, column_name=[f"{bp}_x", f"{bp}_y"], file_name=file_path)
                    bp_df = df[[f"{bp}_x", f"{bp}_y"]]
                    if smoothing_setting['method'] == SAVITZKY_GOLAY:
                        bp_df = savgol_smoother(data=bp_df, fps=video_meta_data['fps'], time_window=smoothing_setting[TIME_WINDOW], source=f'{file_path} {bp}')
                    else:
                        bp_df = df_smoother(data=bp_df, fps=video_meta_data['fps'], time_window=smoothing_setting[TIME_WINDOW], source=f'{file_path} {bp}')
                    out_df[[f"{bp}_x", f"{bp}_y"]] = bp_df
            if self.multi_index_data:
                out_df = self.__insert_multi_index(df=out_df)
            if not self.overwrite:
                copy_files_to_directory(file_paths=[file_path], dir=self.cpy_dir, verbose=False)
            write_df(df=out_df, file_type=self.file_type, save_path=file_path, multi_idx_header=self.multi_index_data)
            video_timer.stop_timer()
            print(f'Smoothing video {video_name} complete ({file_cnt+1}/{len(self.file_paths)}). Elapsed time {video_timer.elapsed_time_str}s')
        self.timer.stop_timer()
        if self.overwrite:
            msg = f"Advanced smoothing complete. Data saved in {self.input_dir}."
        else:
            msg = f"Advanced smoothing complete. Data saved in {self.input_dir}. Original data saved in {self.cpy_dir}."
        stdout_success(msg=msg, elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

# SMOOTHING_SETTINGS = {'Simon': {'Ear_left_1': {'method': 'savitzky_golay', 'time_window': 3500},
#                                'Ear_right_1': {'method': 'gaussian', 'time_window': 500},
#                                'Nose_1': {'method': 'savitzky_golay', 'time_window': 2000},
#                                'Lat_left_1': {'method': 'savitzky_golay', 'time_window': 2000},
#                                'Lat_right_1': {'method': 'gaussian', 'time_window': 2000},
#                                'Center_1': {'method': 'savitzky_golay', 'time_window': 2000},
#                                'Tail_base_1': {'method': 'gaussian', 'time_window': 500}},
#                         'JJ': {'Ear_left_2': {'method': 'savitzky_golay', 'time_window': 2000},
#                                'Ear_right_2': {'method': 'savitzky_golay', 'time_window': 500},
#                                'Nose_2': {'method': 'gaussian', 'time_window': 3500},
#                                'Lat_left_2': {'method': 'savitzky_golay', 'time_window': 500},
#                                'Lat_right_2': {'method': 'gaussian', 'time_window': 3500},
#                                'Center_2': {'method': 'gaussian', 'time_window': 2000},
#                                'Tail_base_2': {'method': 'savitzky_golay', 'time_window': 3500}}}


#SMOOTHING_SETTINGS = {'Animal_1': {'method': 'savitzky_golay', 'time_window': 3500}, 'Animal_2': {'method': 'savitzky_golay', 'time_window': 3500}}
# advanced_smoother = AdvancedSmoother(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/new_data',
#                      settings=SMOOTHING_SETTINGS, type='body-part', multi_index_data=True, overwrite=False)
#
# advanced_smoother.run()
