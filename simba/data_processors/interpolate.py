import pandas as pd

pd.options.mode.chained_assignment = None
import os
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable, check_str,
                                check_valid_lst)
from simba.utils.data import animal_interpolator, body_part_interpolator
from simba.utils.enums import TagNames
from simba.utils.errors import DataHeaderError, InvalidInputError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (copy_files_to_directory,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)


class Interpolate(ConfigReader):
    """
    Interpolate missing body-parts in pose-estimation data. "Missing" is defined as either (i) when a single body-parts is None, or
    when all body-parts belonging to an animal are identical (i.e., the same 2D coordinate or all None).

    .. image:: _static/img/interpolation_comparison.png
       :width: 500
       :align: center

    .. note::
       `Interpolation tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#to-import-multiple-dlc-csv-files>`__.

    .. importants::
        The interpolated data overwrites the original data on disk. If the original data is required, pass ``copy_originals = True`` to save a copy of the original data.


    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format.
    :param Union[str, os.PathLike] data_path: Path to a directory, path to a file, or a list of file paths to files with pose-estimation data in CSV or parquet format.
    :param Optional[Literal['body-parts', 'animals']] type: If 'animals', then interpolation is performed when all body-parts belonging to an animal are identical (i.e., the same 2D coordinate or all None). If 'body-parts` then all body-parts that are None will be interpolated. Default: body-parts.
    :param Optional[Literal['nearest', 'linear', 'quadratic']] method: If 'animals', then interpolation is performed when all body-parts belonging to an animal are identical (i.e., the same 2D coordinate or all None). If 'body-parts` then all body-parts that are None will be interpolated. Default: body-parts.
    :param Optional[bool] multi_index_df_headers: If truth-like, then the input data is anticipated to have multiple header columns, and output columns will have multiple header columns. Default: False.
    :param Optional[bool] copy_originals: If truth-like, then the pre-interpolated, original data, will be bo stored in a subdirectory of the original data. The subdirectory is named according to the type of interpolation and datetime of the operation.

    :example:
    >>> interpolator = Interpolate(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv/test', type='body-parts', multi_index_df_headers=True, copy_originals=True)
    >>> interpolator.run()

    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                 type: Optional[Literal['body-parts', 'animals']] = 'body-parts',
                 method: Optional[Literal['nearest', 'linear', 'quadratic']] = 'nearest',
                 multi_index_df_headers: Optional[bool] = False,
                 copy_originals: Optional[bool] = False) -> None:

        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        check_str(name=f'{self.__class__.__name__} type', value=type.lower(), options=('body-parts', 'animals'))
        check_str(name=f'{self.__class__.__name__} method', value=method.lower(), options=('nearest', 'linear', 'quadratic'))
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
        if copy_originals:
            self.originals_dir = os.path.join(os.path.dirname(self.file_paths[0]), f"Pre_{method}_{type}_interpolation_{self.datetime}")
            os.makedirs(self.originals_dir)
        self.type, self.method, self.multi_index_df_headers, self.copy_originals = type.lower(), method.lower(), multi_index_df_headers, copy_originals

    def __insert_multiindex_header(self, df: pd.DataFrame):
        multi_idx_header = []
        for i in range(len(df.columns)):
            multi_idx_header.append(("IMPORTED_POSE", "IMPORTED_POSE", list(df.columns)[i]))
        df.columns = pd.MultiIndex.from_tuples(multi_idx_header)
        return df

    def run(self):
        print(f'Running interpolation on {len(self.file_paths)} data files...')
        for file_cnt, file_path in enumerate(self.file_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            df = read_df(file_path=file_path, file_type=self.file_type, check_multiindex=self.multi_index_df_headers)
            if self.multi_index_df_headers:
                if len(df.columns) != len(self.bp_headers):
                    raise DataHeaderError( msg=f"The file {file_path} contains {len(df.columns)} columns, but your SimBA project expects {len(self.bp_headers)} columns representing {int(len(self.bp_headers) / 3)} body-parts (x, y, p). Check that the {self.body_parts_path} lists the correct body-parts associated with the project", source=self.__class__.__name__)
                df.columns = self.bp_headers
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
            df[df < 0] = 0
            if self.type == 'animals':
                df = animal_interpolator(df=df, animal_bp_dict=self.animal_bp_dict, source=file_path, method=self.method)
            else:
                df = body_part_interpolator(df=df, animal_bp_dict=self.animal_bp_dict, source=file_path, method=self.method)
            if self.multi_index_df_headers:
                df = self.__insert_multiindex_header(df=df)
            if self.copy_originals:
                copy_files_to_directory(file_paths=[file_path], dir=self.originals_dir)
            write_df(df=df.astype(np.int64), file_type=self.file_type, save_path=file_path, multi_idx_header=self.multi_index_df_headers)
            video_timer.stop_timer()
            print(f"Video {self.video_name} interpolated (elapsed time {video_timer.elapsed_time_str}) ...")
        self.timer.stop_timer()
        if self.copy_originals:
            msg = f"{len(self.file_paths)} data file(s) interpolated using {self.type} {self.method} methods. Originals saved in {self.originals_dir} directory."
        else:
            msg = f"{len(self.file_paths)} data file(s) interpolated using {self.type} {self.method} methods."
        stdout_success(msg=msg, elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

