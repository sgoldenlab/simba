__author__ = "Simon Nilsson"
"""
@authors: Xiaoyu Tong, Jia Jie Choong, Simon Nilsson
"""
import os
import trafaret as t
import pandas as pd
from typing import Any, Optional, Tuple, List, Union

from simba.utils.errors import (NoFilesFoundError,
                                CorruptedFileError,
                                IntegerError,
                                FloatError,
                                StringError,
                                NotDirectoryError,
                                ColumnNotFoundError,
                                InvalidInputError)


def check_file_exist_and_readable(file_path: Union[str, os.PathLike]) -> None:
    """
    Checks if a path points to a readable file.

    :param str file_path:
    :raise NoFilesFoundError: The file does not exist.
    :raise CorruptedFileError: The file can not be read.
    """
    if not os.path.isfile(file_path):
        raise NoFilesFoundError(msg=f'{file_path} is not a valid file path')
    elif not os.access(file_path, os.R_OK):
        raise CorruptedFileError(msg=f'{file_path} is not readable')
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
    """
    msg = ''
    try:
        t.Int().check(value)
    except t.DataError as e:
        msg=f'{name} should be an integer number in SimBA, but is set to {str(value)}'
        if raise_error:
            raise IntegerError(msg=msg)
        else:
            return False, msg
    if (min_value != None):
        if int(value) < min_value:
            msg = f'{name} should be MORE THAN OR EQUAL to {str(min_value)}. It is set to {str(value)}'
            if raise_error:
                raise IntegerError(msg=msg)
            else:
                return False, msg
    if (max_value != None):
        if int(value) > max_value:
            msg = f'{name} should be LESS THAN OR EQUAL to {str(max_value)}. It is set to {str(value)}'
            if raise_error:
                raise IntegerError(msg=msg)
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
    """

    msg = ''
    try:
        t.String(allow_blank=allow_blank).check(value)
    except t.DataError as e:
        msg = f'{name} should be an string in SimBA, but is set to {str(value)}'
        if raise_error:
            raise StringError(msg=msg)
        else:
            return False, msg
    if len(options) > 0:
        if value not in options:
            msg = f'{name} is set to {str(value)} in SimBA, but this is not a valid option: {options}'
            if raise_error:
                raise StringError(msg=msg)
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
    """


    msg = ''
    try:
        t.Float().check(value)
    except t.DataError as e:
        msg = f'{name} should be a float number in SimBA, but is set to {str(value)}'
        if raise_error:
            raise FloatError(msg=msg)
        else:
            return False, msg
    if (min_value != None):
        if float(value) < min_value:
            msg = f'{name} should be MORE THAN OR EQUAL to {str(min_value)}. It is set to {str(value)}'
            if raise_error:
                raise FloatError(msg=msg)
            else:
                return False, msg
    if (max_value != None):
        if float(value) > max_value:
            msg = f'{name} should be LESS THAN OR EQUAL to {str(max_value)}. It is set to {str(value)}'
            if raise_error:
                raise FloatError(msg=msg)
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
        raise NoFilesFoundError(msg=error_msg)
    else:
        pass


def check_if_dir_exists(in_dir: Union[str, os.PathLike]) -> None:
    """
    Check if a directory path exists.

    :param str in_dir: Putative directory path.
    :raise NotDirectoryError: The directory does not exist.
    """
    if not os.path.isdir(in_dir):
        raise NotDirectoryError(msg=f'{in_dir} is not a valid directory')


def check_that_column_exist(df: pd.DataFrame,
                            column_name: str,
                            file_name: str) -> None:
    """
    Check if named field exist within a dataframe.

    :param pd.DataFrame df:
    :param str column_name: Name of putative field.
    :param str file_name: Path of ``df`` on disk.
    :raise ColumnNotFoundError: The  ``column_name`` does not exist within ``df``.
    """

    if column_name not in df.columns:
        raise ColumnNotFoundError(column_name=column_name, file_name=file_name)

def check_if_valid_input(name: str,
                         input: str,
                         options: List[str],
                         raise_error: bool = True) -> (bool, str):
    """
    Check if string variable is valid option

    :param str name: Name of variable
    :param Any input: Value of variable
    :param List[str] options: Allowed options of ``input``
    :param Optional[bool] raise_error: If True, then raise error if invalid value. Default: True.

    :return bool: False if invalid. True if valid.
    :return str: If invalid, then error msg. Else empty str.
    """

    msg = ''
    if input not in options:
        msg = f'{name} is set to {str(input)}, which is an invalid setting. OPTIONS {options}'
        if raise_error:
            raise InvalidInputError(msg=msg)
        else:
            return False, msg
    else:
        return True, msg



def check_minimum_roll_windows(roll_windows_values: List[int],
                               minimum_fps: float) -> List[int]:

    """
    Remove any rolling temporal window that are shorter than a single frame in
    any of the videos in the project.

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