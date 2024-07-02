__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from typing import List, Optional, Union

import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable, check_int,
                                check_str, check_valid_lst)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.enums import TagNames
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (copy_files_to_directory,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    read_video_info, write_df)


class Smoothing(ConfigReader):
    """
    Smooth pose-estimation data according to user-defined method.

    .. image:: _static/img/smoothing.gif
       :width: 600
       :align: center

    .. note::
       `Smoothing tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files>`__.

    .. importants::
        The wmoothened data overwrites the original data on disk. If the original data is required, pass ``copy_originals = True`` to save a copy of the original data.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format.
    :param Union[str, os.PathLike, List[Union[str, os.PathLike]]] data_path: Path to directory containing pose-estimation data, to a file containing pose-estimation data, or a list of paths containing pose-estimation data.
    :param int time_window: Rolling time window in millisecond to use when smoothing. Larger time-windows and greater smoothing.
    :param Optional[Literal["gaussian", "savitzky-golay"]] method: Type of smoothing_method. OPTIONS: ``gaussian``, ``savitzky-golay``. Default `gaussian`.
    :param bool multi_index_df_headers: If True, the incoming data is multi-index columns dataframes. Default: False.
    :param bool copy_originals: If truth-like, then the pre-smoothened, original data, will be bo stored in a subdirectory of the original data. The subdirectory is named according to the type of smoothing method and datetime of the operation.

    :references:
        .. [1] `Video expected putput <https://www.youtube.com/watch?v=d9-Bi4_HyfQ>`__.

    :examples:
    >>> smoother = Smoothing(data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv/Together_1.csv', config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', method='Savitzky-Golay', time_window=500, multi_index_df_headers=True, copy_originals=True)
    >>> smoother.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                 time_window: int,
                 method: Optional[Literal["gaussian", "savitzky-golay"]] = 'Savitzky-Golay',
                 multi_index_df_headers: Optional[bool] = False,
                 copy_originals: Optional[bool] = False) -> None:

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if isinstance(data_path, list):
            check_valid_lst(data=data_path, source=self.__class__.__name__, valid_dtypes=(str,))
            for i in data_path: check_file_exist_and_readable(file_path=i)
            self.file_paths = deepcopy(data_path)
        elif os.path.isdir(data_path):
            self.file_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=[f'.{self.file_type}'], raise_error=True)
        elif os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.file_paths = [data_path]
        else:
            raise InvalidInputError(msg=f'{data_path} is not a valid data directory, or a valid file path, or a valid list of file paths', source=self.__class__.__name__)
        check_int(value=time_window, min_value=1, name=f'{self.__class__.__name__} time_window')
        check_str(name=f'{self.__class__.__name__} method', value=method.lower(), options=("gaussian", "savitzky-golay"))
        if copy_originals:
            self.originals_dir = os.path.join(os.path.dirname(self.file_paths[0]), f"Pre_{method}_{time_window}_smoothing_{self.datetime}")
            os.makedirs(self.originals_dir)
        self.multi_index_df_headers, self.method, self.time_window, self.copy_originals = multi_index_df_headers, method.lower(), time_window, copy_originals

    def __insert_multiindex_header(self, df: pd.DataFrame):
        multi_idx_header = []
        for i in range(len(df.columns)):
            multi_idx_header.append(("IMPORTED_POSE", "IMPORTED_POSE", list(df.columns)[i]))
        df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
        return df

    def run(self):
        print(f'Running smoothing on {len(self.file_paths)} data files...')
        for file_cnt, file_path in enumerate(self.file_paths):
            df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=True)
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=False, warning=False)
            if video_path is None:
                if not os.path.isfile(self.video_info_path):
                    raise NoFilesFoundError(msg=f"To perform smoothing, SimBA needs to read the video FPS. SimBA could not find the video {video_name} in represented in the {self.video_dir} directory or in {self.video_info_path} file. Please import the video and/or include it in the video_logs.csv file so SimBA can know the video FPS", source=self.__class__.__name__)
                else:
                    self.video_info_df = self.read_video_info_csv(file_path=self.video_info_path)
                    video_info = read_video_info(vid_info_df=self.video_info_df,video_name=video_name, raise_error=False)
                    if video_info is None:
                        raise NoFilesFoundError(msg=f"To perform smoothing, SimBA needs to read the video FPS. SimBA could not find the video {video_name} in represented in the {self.video_dir} directory or in {self.video_info_path} file. Please import the video and/or include it in the video_logs.csv file so SimBA can know the video FPS", source=self.__class__.__name__)
                    fps = video_info[2]
            else:
                fps = get_video_meta_data(video_path=video_path)['fps']
            if self.method == 'savitzky-golay':
                df = savgol_smoother(data=df, fps=fps, time_window=self.time_window, source=video_name)
            else:
                df = df_smoother(data=df, fps=fps, time_window=self.time_window, source=video_name, method='gaussian')
            if self.multi_index_df_headers:
                df = self.__insert_multiindex_header(df=df)
            if self.copy_originals:
                copy_files_to_directory(file_paths=[file_path], dir=self.originals_dir)
            write_df(df=df, file_type=self.file_type, save_path=file_path, multi_idx_header=self.multi_index_df_headers)
            video_timer.stop_timer()
            print(f"Video {video_name} smoothed ({self.method}: {str(self.time_window)}ms) (elapsed time {video_timer.elapsed_time_str})...")
        self.timer.stop_timer()
        if self.copy_originals:
            msg = f"{len(self.file_paths)} data file(s) smoothened using {self.method} method and {self.time_window} time-window. Originals saved in {self.originals_dir} directory."
        else:
            msg = f"{len(self.file_paths)} data file(s) smoothened using {self.method} method and {self.time_window} time-window."
        stdout_success(msg=msg, elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)