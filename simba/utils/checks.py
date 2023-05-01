__author__ = "Simon Nilsson"

import os
import trafaret as t
import pandas as pd
from simba.utils.errors import (NoFilesFoundError,
                                CorruptedFileError,
                                IntegerError,
                                FloatError,
                                StringError,
                                NotDirectoryError,
                                ColumnNotFoundError,
                                InvalidInputError)

def check_file_exist_and_readable(file_path: str):
    """ Checks if a path points to a readable file"""
    if not os.path.isfile(file_path):
        raise NoFilesFoundError(msg=f'{file_path} is not a valid file path')
    elif not os.access(file_path, os.R_OK):
        raise CorruptedFileError(msg=f'{file_path} is not readable')
    else:
        pass


def check_int(name: str,
              value: None,
              max_value=None,
              min_value=None,
              raise_error: bool=True):
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
              value: None,
              options=(),
              allow_blank=False,
              raise_error: bool = True):
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
                value=None,
                max_value=None,
                min_value=None,
                raise_error: bool=True):
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


def check_if_filepath_list_is_empty(filepaths: list,
                                    error_msg: str):
    if len(filepaths) == 0:
        raise NoFilesFoundError(msg=error_msg)
    else:
        pass


def check_if_dir_exists(in_dir: str):
    if not os.path.isdir(in_dir):
        raise NotDirectoryError(msg=f'{in_dir} is not a valid directory')


def check_that_column_exist(df: pd.DataFrame,
                            column_name: str,
                            file_name: str):
    if column_name not in df.columns:
        raise ColumnNotFoundError(column_name=column_name, file_name=file_name)

def check_if_valid_input(name: str,
                         input: str,
                         options: list,
                         raise_error: bool=True):
    msg = ''
    if input not in options:
        msg = f'{name} is set to {str(input)}, which is an invalid setting. OPTIONS {options}'
        if raise_error:
            raise InvalidInputError(msg=msg)
        else:
            return False, msg
    else:
        return True, msg



def check_minimum_roll_windows(roll_windows_values: list,
                               minimum_fps: float):

    """
    Helper to remove any rolling temporal window that are shorter than a single frame in
    any of the videos in the project.

    Parameters
    ----------
    roll_windows_values: list
        Rolling temporal windows represented as frame counts. E.g., [10, 15, 30, 60]
    minimum_fps: float
        The lowest fps of the videos that are to be analyzed. E.g., 10

    Returns
    -------
    roll_windows_values: list
    """

    for win in range(len(roll_windows_values)):
        if minimum_fps < roll_windows_values[win]:
            roll_windows_values[win] = minimum_fps
        else:
            pass
    roll_windows_values = list(set(roll_windows_values))
    return roll_windows_values