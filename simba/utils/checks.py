__author__ = "Simon Nilsson"
"""
@authors: Xiaoyu Tong, Jia Jie Choong, Simon Nilsson
"""


import os
import trafaret as t
import pandas as pd
from typing import Any, Optional, Tuple, List, Union
import subprocess
import re

from simba.utils.errors import (NoFilesFoundError,
                                CorruptedFileError,
                                IntegerError,
                                FloatError,
                                StringError,
                                NotDirectoryError,
                                ColumnNotFoundError,
                                InvalidInputError,
                                CountError)


def check_file_exist_and_readable(file_path: Union[str, os.PathLike]) -> None:
    """
    Checks if a path points to a readable file.

    :param str file_path: Path to file on disk.
    :raise NoFilesFoundError: The file does not exist.
    :raise CorruptedFileError: The file can not be read or is zero byte size.
    """
    if not os.path.isfile(file_path):
        raise NoFilesFoundError(msg=f'{file_path} is not a valid file path', source=check_file_exist_and_readable.__name__)
    elif not os.access(file_path, os.R_OK):
        raise CorruptedFileError(msg=f'{file_path} is not readable', source=check_file_exist_and_readable.__name__)
    elif os.stat(file_path).st_size == 0:
        raise CorruptedFileError(msg=f'{file_path} is 0 bytes and contains no data.', source=check_file_exist_and_readable.__name__)
    else:
        pass

def check_int(name: str,
              value: Any,
              max_value: Optional[int] = None,
              min_value: Optional[int] = None,
              raise_error: Optional[bool] = True) -> (bool, str):
    """
    Check if variable is a valid integer.

    :param str name: Name of variable
    :param Any value: Value of variable
    :param Optional[int] max_value: Maximum allowed value of the variable. If None, then no maximum. Default: None.
    :param Optional[int]: Minimum allowed value of the variable. If None, then no minimum. Default: None.
    :param Optional[bool] raise_error: If True, then raise error if invalid integer. Default: True.

    :return bool: False if invalid. True if valid.
    :return str: If invalid, then error msg. Else empty str.

    :examples:
    >>> check_int(name='My_fps', input=25, min_value=1)
    """
    msg = ''
    try:
        t.Int().check(value)
    except t.DataError as e:
        msg=f'{name} should be an integer number in SimBA, but is set to {str(value)}'
        if raise_error:
            raise IntegerError(msg=msg, source=check_int.__name__)
        else:
            return False, msg
    if (min_value != None):
        if int(value) < min_value:
            msg = f'{name} should be MORE THAN OR EQUAL to {str(min_value)}. It is set to {str(value)}'
            if raise_error:
                raise IntegerError(msg=msg, source=check_int.__name__)
            else:
                return False, msg
    if (max_value != None):
        if int(value) > max_value:
            msg = f'{name} should be LESS THAN OR EQUAL to {str(max_value)}. It is set to {str(value)}'
            if raise_error:
                raise IntegerError(msg=msg, source=check_int.__name__)
            else:
                return False, msg
    return True, msg


def check_str(name: str,
              value: Any,
              options: Optional[Tuple[Any]] = (),
              allow_blank: bool = False,
              raise_error: bool = True) -> (bool, str):
    """
    Check if variable is a valid string.

    :param str name: Name of variable
    :param Any value: Value of variable
    :param Optional[Tuple[Any]] options: Tuple of allowed strings. If empty tuple, then any string allowed. Default: ().
    :param Optional[bool] allow_blank: If True, allow empty string. Default: False.
    :param Optional[bool] raise_error: If True, then raise error if invalid string. Default: True.

    :return bool: False if invalid. True if valid.
    :return str: If invalid, then error msg. Else empty str.

    :examples:
    >>> check_str(name='split_eval', input='gini', options=['entropy', 'gini'])
    """

    msg = ''
    try:
        t.String(allow_blank=allow_blank).check(value)
    except t.DataError as e:
        msg = f'{name} should be an string in SimBA, but is set to {str(value)}'
        if raise_error:
            raise StringError(msg=msg, source=check_str.__name__)
        else:
            return False, msg
    if len(options) > 0:
        if value not in options:
            msg = f'{name} is set to {str(value)} in SimBA, but this is not a valid option: {options}'
            if raise_error:
                raise StringError(msg=msg, source=check_str.__name__)
            else:
                return False, msg
        else:
            return True, msg
    else:
        return True, msg


def check_float(name: str,
                value: Any,
                max_value: Optional[float] = None,
                min_value: Optional[float] = None,
                raise_error: bool = True) -> (bool, str):
    """
    Check if variable is a valid float.

    :param str name: Name of variable
    :param Any value: Value of variable
    :param Optional[int] max_value: Maximum allowed value of the float. If None, then no maximum. Default: None.
    :param Optional[int]: Minimum allowed value of the float. If None, then no minimum. Default: None.
    :param Optional[bool] raise_error: If True, then raise error if invalid float. Default: True.

    :return bool: False if invalid. True if valid.
    :return str: If invalid, then error msg. Else empty str.

    :examples:
    >>> check_float(name='My_float', value=0.5, max_value=1.0, min_value=0.0)
    """


    msg = ''
    try:
        t.Float().check(value)
    except t.DataError as e:
        msg = f'{name} should be a float number in SimBA, but is set to {str(value)}'
        if raise_error:
            raise FloatError(msg=msg, source=check_float.__name__)
        else:
            return False, msg
    if (min_value != None):
        if float(value) < min_value:
            msg = f'{name} should be MORE THAN OR EQUAL to {str(min_value)}. It is set to {str(value)}'
            if raise_error:
                raise FloatError(msg=msg, source=check_float.__name__)
            else:
                return False, msg
    if (max_value != None):
        if float(value) > max_value:
            msg = f'{name} should be LESS THAN OR EQUAL to {str(max_value)}. It is set to {str(value)}'
            if raise_error:
                raise FloatError(msg=msg, source=check_float.__name__)
            else:
                return False, msg
    return True, msg


def check_if_filepath_list_is_empty(filepaths: List[str],
                                    error_msg: str) -> None:
    """
    Check if a list is empty

    :param List[str]: List of file-paths.
    :raise NoFilesFoundError: The list is empty.
    """


    if len(filepaths) == 0:
        raise NoFilesFoundError(msg=error_msg, source=check_if_filepath_list_is_empty.__name__)
    else:
        pass


def check_if_dir_exists(in_dir: Union[str, os.PathLike]) -> None:
    """
    Check if a directory path exists.

    :param str in_dir: Putative directory path.
    :raise NotDirectoryError: The directory does not exist.
    """
    if not os.path.isdir(in_dir):
        raise NotDirectoryError(msg=f'{in_dir} is not a valid directory', source=check_if_dir_exists.__name__)

def check_that_column_exist(df: pd.DataFrame,
                            column_name: str,
                            file_name: str) -> None:
    """
    Check if single named field exist within a dataframe.

    :param pd.DataFrame df:
    :param str column_name: Name of putative field.
    :param str file_name: Path of ``df`` on disk.
    :raise ColumnNotFoundError: The ``column_name`` does not exist within ``df``.
    """

    if column_name not in df.columns:
        raise ColumnNotFoundError(column_name=column_name, file_name=file_name, source=check_that_column_exist.__name__)

def check_if_valid_input(name: str,
                         input: str,
                         options: List[str],
                         raise_error: bool = True) -> (bool, str):
    """
    Check if string variable is valid option

    :param str name: Atrbitrary name of variable.
    :param Any input: Value of variable.
    :param List[str] options: Allowed options of ``input``
    :param Optional[bool] raise_error: If True, then raise error if invalid value. Default: True.

    :return bool: False if invalid. True if valid.
    :return str: If invalid, then error msg. Else, empty str.

    :example:
    >>> check_if_valid_input(name='split_eval', input='gini', options=['entropy', 'gini'])
    >>> (True, '')
    """

    msg = ''
    if input not in options:
        msg = f'{name} is set to {str(input)}, which is an invalid setting. OPTIONS {options}'
        if raise_error:
            raise InvalidInputError(msg=msg, source=check_if_valid_input.__name__)
        else:
            return False, msg
    else:
        return True, msg



def check_minimum_roll_windows(roll_windows_values: List[int],
                               minimum_fps: float) -> List[int]:

    """
    Remove any rolling temporal window that are shorter than a single frame in
    any of the videos within the project.

    :param List[int] roll_windows_values: Rolling temporal windows represented as frame counts. E.g., [10, 15, 30, 60]
    :param float minimum_fps: The lowest fps of the videos that are to be analyzed. E.g., 10.

    :return List[int]: roll_windows_values without impassable windows.
    """

    for win in range(len(roll_windows_values)):
        if minimum_fps < roll_windows_values[win]:
            roll_windows_values[win] = minimum_fps
        else:
            pass
    roll_windows_values = list(set(roll_windows_values))
    return roll_windows_values


def check_same_number_of_rows_in_dfs(dfs: List[pd.DataFrame]) -> bool:
    """
    Helper to check that each dataframe in list contains an equal number of rows

    :param List[pd.DataFrame] dfs: List of dataframes.
    :return bool: True if dataframes has an equal number of rows. Else False.

    >>> df_1, df_2 = pd.DataFrame([[1, 2], [1, 2]]), pd.DataFrame([[4, 2], [9, 3], [1, 5]])
    >>> check_same_number_of_rows_in_dfs(dfs=[df_1, df_2])
    >>> False
    >>> df_1, df_2 = pd.DataFrame([[1, 2], [1, 2]]), pd.DataFrame([[4, 2], [9, 3]])
    >>> True
    """

    row_cnt = None
    for df_cnt, df in enumerate(dfs):
        if df_cnt == 0:
            row_cnt = len(df)
        else:
            if len(df) != row_cnt:
                return False
    return True

def check_if_headers_in_dfs_are_unique(dfs: List[pd.DataFrame]) -> List[str]:
    """
    Helper to check heaaders in multiple dataframes are unique.

    :param List[pd.DataFrame] dfs: List of dataframes.
    :return List[str]: List of columns headers seen in multiple dataframes. Empty if None.

    :examples:
    >>> df_1, df_2 = pd.DataFrame([[1, 2]], columns=['My_column_1', 'My_column_2']), pd.DataFrame([[4, 2]], columns=['My_column_3', 'My_column_1'])
    >>> check_if_headers_in_dfs_are_unique(dfs=[df_1, df_2])
    >>> ['My_column_1']
    """
    seen_headers = []
    for df_cnt, df in enumerate(dfs):
        seen_headers.extend(list(df.columns))
    duplicates = list(set([x for x in seen_headers if seen_headers.count(x) > 1]))
    return duplicates

def check_if_string_value_is_valid_video_timestamp(value: str, name: str) -> None:
    """
    Helper to check if a string is in a valid HH:MM:SS format

    :param str value: Timestamp in HH:MM:SS format.
    :param str name: An arbitrary string name of the timestamp.
    :raises InvalidInputError: If the timestamp is in invalid format

    :example:
    >>> check_if_string_value_is_valid_video_timestamp(value='00:0b:10', name='My time stamp')
    >>> "InvalidInputError: My time stamp is should be in the format XX:XX:XX where X is an integer between 0-9"
    """
    r = re.compile('.{2}:.{2}:.{2}')
    if (len(value) != 8) or (not r.match(value)) or (re.search('[a-zA-Z]', value)):
        raise InvalidInputError(msg=f'{name} is should be in the format XX:XX:XX where X is an integer between 0-9', source=check_if_string_value_is_valid_video_timestamp.__name__)

def check_that_hhmmss_start_is_before_end(start_time: str,
                                          end_time: str,
                                          name: str) -> None:
    """
    Helper to check that a start time in HH:MM:SS format is before an end time in HH:MM:SS format

    :param str start_time: Period start time in HH:MM:SS format.
    :param str end_time: Period end time in HH:MM:SS format.
    :param int name: Name of the variable
    :raises InvalidInputError: If end time is before the start time.

    :example:
    >>> check_that_hhmmss_start_is_before_end(start_time='00:00:05', end_time='00:00:01', name='My time period')
    >>> "InvalidInputError: My time period has an end-time which is before the start-time"
    >>> check_that_hhmmss_start_is_before_end(start_time='00:00:01', end_time='00:00:05')
    """
    start_h, start_m, start_s = start_time.split(':')
    end_h, end_m, end_s = end_time.split(':')
    start_in_s = int(start_h) * 3600 + int(start_m) * 60 + int(start_s)
    end_in_s = int(end_h) * 3600 + int(end_m) * 60 + int(end_s)
    if end_in_s < start_in_s:
        raise InvalidInputError(f'{name} has an end-time which is before the start-time.', source=check_that_hhmmss_start_is_before_end.__name__)

def check_nvidea_gpu_available() -> bool:
    """
    Helper to check of NVIDEA GPU is available via ``nvidia-smi``.
    returns bool: True if nvidia-smi returns not None. Else False.
    """
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except Exception:
        return False

def check_ffmpeg_available() -> bool:
    """
    Helper to check of FFMpeg is available via subprocess ``ffmpeg``.

    returns bool: True if ``ffmpeg`` returns not None. Else False.
    """
    try:
        subprocess.call('ffmpeg', stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def check_if_valid_rgb_str(input: str,
                           delimiter: str = ',',
                           return_cleaned_rgb_tuple: bool = True,
                           reverse_returned: bool = True):
    """
    Helper to check if a string is a valid representation of an RGB color.

    :param str input: Value to check as string. E.g., '(166, 29, 12)' or '22,32,999'
    :param str delimiter: The delimiter between subsequent values in the rgb input string.
    :param bool return_cleaned_rgb_tuple: If True, and input is a valid rgb, then returns a "clean" rgb tuple: Eg. '166, 29, 12' -> (166, 29, 12). Else, returns None.
    :param bool reverse_returned: If True and return_cleaned_rgb_tuple is True, reverses to returned cleaned rgb tuple (e.g., RGB becomes BGR) before returning it.

    :example:
    >>> check_if_valid_rgb_str(input='(50, 25, 100)', return_cleaned_rgb_tuple=True, reverse_returned=True)
    >>> (100, 25, 50)
    """

    input = input.replace(" ", "")
    if input.count(delimiter) != 2:
        raise InvalidInputError(msg=f'{input} in not a valid RGB color')
    values = input.split(',')
    rgb = []
    for value in values:
        val = ''.join(c for c in value if c.isdigit())
        check_int(name='RGB value', value=val, max_value=255, min_value=0, raise_error=True)
        rgb.append(val)
    rgb = tuple([int(x) for x in rgb])

    if return_cleaned_rgb_tuple:
        if reverse_returned:
            rgb = rgb[::-1]
        return rgb




#check_if_valid_rgb_str(input='(255, 0, 255)')







