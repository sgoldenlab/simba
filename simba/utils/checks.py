__author__ = "Simon Nilsson"

import ast
import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

import cv2
import numpy as np
import pandas as pd
import trafaret as t
from shapely.geometry import Polygon

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.utils.enums import Formats, Keys, Options, UMAPParam
from simba.utils.errors import (ArrayError, ColumnNotFoundError,
                                CorruptedFileError, CountError,
                                DirectoryNotEmptyError, FFMPEGNotFoundError,
                                FloatError, FrameRangeError, IntegerError,
                                InvalidFilepathError, InvalidInputError,
                                MissingColumnsError, NoDataError,
                                NoFilesFoundError, NoROIDataError,
                                NotDirectoryError, ParametersFileError,
                                SimBAGPUError, StringError)
from simba.utils.warnings import (CorruptedFileWarning, FrameRangeWarning,
                                  InvalidValueWarning, NoDataFoundWarning)


def check_file_exist_and_readable(file_path: Union[str, os.PathLike], raise_error: bool = True) -> bool:
    """
    Checks if a path points to a readable file.

    :param str file_path: Path to file on disk.
    :raise NoFilesFoundError: The file does not exist.
    :raise CorruptedFileError: The file can not be read or is zero byte size.
    """
    check_instance(source="FILE PATH", instance=file_path, accepted_types=(str, os.PathLike))
    if not os.path.isfile(file_path):
        if raise_error:
            raise NoFilesFoundError(msg=f"{file_path} is not a valid file path", source=check_file_exist_and_readable.__name__)
        else:
            return False
    elif not os.access(file_path, os.R_OK):
        if raise_error:
            raise CorruptedFileError(msg=f"{file_path} is not readable", source=check_file_exist_and_readable.__name__)
        else:
            return False
    elif os.stat(file_path).st_size == 0:
        if raise_error:
            raise CorruptedFileError(msg=f"{file_path} is 0 bytes and contains no data.", source=check_file_exist_and_readable.__name__)
        else:
            return False
    else:
        return True


def check_int(name: str,
              value: Any,
              max_value: Optional[int] = None,
              min_value: Optional[int] = None,
              unaccepted_vals: Optional[List[int]] = None,
              accepted_vals: Optional[List[int]] = None,
              raise_error: Optional[bool] = True) -> Tuple[bool, str]:
    """
    Check if variable is a valid integer.

    :param str name: Name of variable
    :param Any value: Value of variable
    :param Optional[int] max_value: Maximum allowed value of the variable. If None, then no maximum. Default: None.
    :param Optional[int]: Minimum allowed value of the variable. If None, then no minimum. Default: None.
    :param Optional[List[int]] unaccepted_vals: Optional list of values that are not accepted. Default: None.
    :param Optional[List[int]] accepted_vals: Optional list of values that are accepted. Default: None.
    :param Optional[bool] raise_error: If True, then raise error if invalid integer. Default: True.
    :return: If `raise_error` is False, then returns size-2 tuple, with first value being a bool representing if valid integer, and second value a string representing error (if valid is False, else empty string)
    :rtype: Tuple[bool, str]

    :examples:
    >>> check_int(name='My_fps', input=25, min_value=1)
    """
    msg = ""
    try:
        t.Int().check(value)
    except t.DataError as e:
        msg = f"{name} should be an integer number in SimBA, but is set to {str(value)}"
        if raise_error:
            raise IntegerError(msg=msg, source=check_int.__name__)
        else:
            return False, msg
    if min_value != None:
        if int(value) < min_value:
            msg = f"{name} should be MORE THAN OR EQUAL to {str(min_value)}. It is set to {str(value)}"
            if raise_error:
                raise IntegerError(msg=msg, source=check_int.__name__)
            else:
                return False, msg
    if max_value != None:
        if int(value) > max_value:
            msg = f"{name} should be LESS THAN OR EQUAL to {str(max_value)}. It is set to {str(value)}"
            if raise_error:
                raise IntegerError(msg=msg, source=check_int.__name__)
            else:
                return False, msg
    if unaccepted_vals != None:
        check_valid_lst(data=unaccepted_vals, source=name, valid_dtypes=(int,), min_len=1)
        if int(value) in unaccepted_vals:
            msg = f"{name} is an not an accepted value. Unaccepted values {unaccepted_vals}."
            if raise_error:
                raise IntegerError(msg=msg, source=check_int.__name__)
            else:
                return False, msg
    if accepted_vals != None:
        check_valid_lst(data=accepted_vals, source=name, valid_dtypes=(int,), min_len=1)
        if int(value) not in accepted_vals:
            msg = f"{name} is an not an accepted value. Got: {value}. Accepted values {accepted_vals}."
            if raise_error:
                raise IntegerError(msg=msg, source=check_int.__name__)
            else:
                return False, msg


    return True, msg


def check_str(name: str,
              value: Any,
              options: Optional[Union[Tuple[Any], List[Any], Iterable[Any]]] = (),
              allow_blank: bool = False,
              invalid_options: Optional[Union[List[str], Tuple[str]]] = None,
              raise_error: bool = True) -> Tuple[bool, str]:

    """
    Check if variable is a valid string.

    :param str name: Name of variable
    :param Any value: Value of variable
    :param Optional[Tuple[Any]] options: Tuple of allowed strings. If empty tuple, then any string allowed. Default: ().
    :param Optional[bool] allow_blank: If True, allow empty string. Default: False.
    :param Optional[bool] raise_error: If True, then raise error if invalid string. Default: True.
    :param Optional[List[str]] invalid_options: If not None, then a list of strings that are invalid.
    :return: If `raise_error` is False, then returns size-2 Tuple, with first value being a bool representing if valid string, and second value a string representing error reason (if valid is False, else empty string).
    :rtype: Tuple[bool, str]

    :examples:
    >>> check_str(name='split_eval', input='gini', options=['entropy', 'gini'])
    """

    msg = ""
    try:
        t.String(allow_blank=allow_blank).check(value)
    except t.DataError as e:
        msg = f"{name} should be an string in SimBA, but is set to {str(value)}"
        if raise_error:
            raise StringError(msg=msg, source=check_str.__name__)
        else:
            return False, msg
    if len(options) > 0:
        if value not in options:
            msg = f"{name} is set to {value} in SimBA, but this is not a valid option: {options}"
            if raise_error:
                raise StringError(msg=msg, source=check_str.__name__)
            else:
                return False, msg
        else:
            return True, msg

    if invalid_options is not None:
        check_instance(source=f'{name} invalid_options', accepted_types=(tuple, list,), instance=invalid_options)
        if isinstance(invalid_options, tuple):
            invalid_options = list(invalid_options)
        check_valid_lst(data=invalid_options, valid_dtypes=(str,), min_len=1)
        if value in invalid_options:
            msg = f"{name} is set to {value} in SimBA, but this is among invalid options: {invalid_options}"
            if raise_error:
                raise StringError(msg=msg, source=check_str.__name__)
            else:
                return False, msg
        else:
            return True, msg
    else:
        return True, msg

def check_float(
    name: str,
    value: Any,
    max_value: Optional[float] = None,
    min_value: Optional[float] = None,
    raise_error: bool = True,
) -> Tuple[bool, str]:
    """
    Check if variable is a valid float.

    :param str name: Name of variable
    :param Any value: Value of variable
    :param Optional[int] max_value: Maximum allowed value of the float. If None, then no maximum. Default: None.
    :param Optional[int]: Minimum allowed value of the float. If None, then no minimum. Default: None.
    :param Optional[bool] raise_error: If True, then raise error if invalid float. Default: True.
    :return: If `raise_error` is False, then returns size-2 tuple, with first value being a bool representing if valid float, and second value a string representing error (if valid is False, else empty string)
    :rtype: Tuple[bool, str]


    :examples:
    >>> check_float(name='My_float', value=0.5, max_value=1.0, min_value=0.0)
    """

    msg = ""
    try:
        t.Float().check(value)
    except t.DataError as e:
        msg = f"{name} should be a float number in SimBA, but is set to {str(value)}"
        if raise_error:
            raise FloatError(msg=msg, source=check_float.__name__)
        else:
            return False, msg
    if min_value != None:
        if float(value) < min_value:
            msg = f"{name} should be MORE THAN OR EQUAL to {str(min_value)}. It is set to {str(value)}"
            if raise_error:
                raise FloatError(msg=msg, source=check_float.__name__)
            else:
                return False, msg
    if max_value != None:
        if float(value) > max_value:
            msg = f"{name} should be LESS THAN OR EQUAL to {str(max_value)}. It is set to {str(value)}"
            if raise_error:
                raise FloatError(msg=msg, source=check_float.__name__)
            else:
                return False, msg
    return True, msg


def check_iterable_length(source: str, val: int, exact_accepted_length: Optional[int] = None, max: Optional[int] = np.inf, min: int = 1, raise_error: bool = True) -> bool:
    if (not exact_accepted_length) and (not max) and (not min):
        if raise_error:
            raise InvalidInputError(msg=f"Provide exact_accepted_length or max and min values for {source}", source=check_iterable_length.__name__)
        else:
            return False
    if exact_accepted_length:
        if val != exact_accepted_length:
            if raise_error:
                raise InvalidInputError(msg=f"{source} length is {val}, expected {exact_accepted_length}", source=check_iterable_length.__name__)
            else:
                return False
    elif (val > max) or (val < min):
        if raise_error:
            raise InvalidInputError(msg=f"{source} value {val} does not full-fill criterion: min {min}, max{max} ", source=check_iterable_length.__name__)
        else:
            return False
    return True


def check_instance(source: str, instance: object, accepted_types: Union[Tuple[Any], Any], raise_error: bool = True, warning: bool = True) -> bool:
    """
    Check if an instance is an acceptable type.

    :param str name: Arbitrary name of instance used for interpretable error msg. Can also be the name of the method.
    :param object instance: A data object.
    :param Union[Tuple[object], object] accepted_types: Accepted instance types. E.g., (Polygon, pd.DataFrame) or Polygon.
    :param Optional[bool] raise_error: If True, raises error. Else, prints warning.
    """

    if not isinstance(instance, accepted_types):
        msg = f"{source} requires {accepted_types}, got {type(instance)}"
        if raise_error:
            raise InvalidInputError(msg=msg, source=source)
        else:
            if warning:
                InvalidValueWarning(msg=msg, source=source)
            return False
    return True


def get_fn_ext(filepath: Union[os.PathLike, str]) -> (str, str, str):
    """
    Split file path into three components: (i) directory, (ii) file name, and (iii) file extension.

    :parameter str filepath: Path to file.
    :return str: File directory name
    :return str: File name
    :return str: File extension

    :example:
    >>> get_fn_ext(filepath='C:/My_videos/MyVideo.mp4')
    >>> ('My_videos', 'MyVideo', '.mp4')
    """
    file_extension = Path(filepath).suffix
    try:
        file_name = os.path.basename(filepath.rsplit(file_extension, 1)[0])
    except ValueError:
        raise InvalidFilepathError(
            msg=f"{filepath} is not a valid filepath", source=get_fn_ext.__name__
        )
    dir_name = os.path.dirname(filepath)
    return dir_name, file_name, file_extension


def check_if_filepath_list_is_empty(filepaths: List[str], error_msg: str) -> None:
    """
    Check if a list is empty

    :param List[str]: List of file-paths.
    :raise NoFilesFoundError: The list is empty.
    """

    if len(filepaths) == 0:
        raise NoFilesFoundError(
            msg=error_msg, source=check_if_filepath_list_is_empty.__name__
        )
    else:
        pass


def check_all_file_names_are_represented_in_video_log(
    video_info_df: pd.DataFrame, data_paths: List[Union[str, os.PathLike]]
) -> None:
    """
    Helper to check that all files are represented in a dataframe of the SimBA `project_folder/logs/video_info.csv`
    file.

    :param pd.DataFrame video_info_df: List of file-paths.
    :param List[Union[str, os.PathLike]] data_paths: List of file-paths.
    :raise ParametersFileError: The list is empty.
    """

    missing_videos = []
    for file_path in data_paths:
        video_name = get_fn_ext(file_path)[1]
        if video_name not in list(video_info_df["Video"]):
            missing_videos.append(video_name)
    if len(missing_videos) > 0:
        raise ParametersFileError(
            msg=f"SimBA could not find {len(missing_videos)} video(s) in the video_info.csv file. Make sure all videos analyzed are represented in the project_folder/logs/video_info.csv file. MISSING VIDEOS: {missing_videos}"
        )


def check_if_dir_exists(in_dir: Union[str, os.PathLike],
                        source: Optional[str] = None,
                        create_if_not_exist: Optional[bool] = False,
                        raise_error: bool = True) -> Union[None, bool]:
    """
    Check if a directory path exists.

    :param Union[str, os.PathLike] in_dir: Putative directory path.
    :param Optional[str] source: String source for interpretable error messaging.
    :param Optional[bool] create_if_not_exist: If directory does not exist, then create it. Default False,
    :raise NotDirectoryError: The directory does not exist.
    """

    if not isinstance(in_dir, (str, Path, os.PathLike)):
        if raise_error:
            raise NotDirectoryError(msg=f"{in_dir} is not a valid directory", source=check_if_dir_exists.__name__)
        else:
            return False

    elif not os.path.isdir(in_dir):
        if create_if_not_exist:
            try:
                os.makedirs(in_dir)
            except:
                pass
        else:
            if source is None:
                if raise_error:
                    raise NotDirectoryError(msg=f"{in_dir} is not a valid directory", source=check_if_dir_exists.__name__)
                else:
                    return False
            else:
                if raise_error:
                    raise NotDirectoryError(msg=f"{in_dir} is not a valid directory", source=source)
                else:
                    return False
    else:
        return True


def check_that_column_exist(df: pd.DataFrame,
                            column_name: Union[str, os.PathLike, List[str]],
                            file_name: str,
                            raise_error: bool = True) -> Union[None, bool]:
    """
    Check if single named field or a list of fields exist within a dataframe.

    :param pd.DataFrame df:
    :param str column_name: Name or names of field(s).
    :param str file_name: Path of ``df`` on disk.
    :param bool raise_error: If True, raise error if column doesnt exist. If False, returns bool.
    :raise ColumnNotFoundError: The ``column_name`` does not exist within ``df``.
    """

    if type(column_name) == str:
        column_name = [column_name]
    for column in column_name:
        if column not in df.columns:
            if raise_error:
                raise ColumnNotFoundError(column_name=column, file_name=file_name, source=check_that_column_exist.__name__)
            else:
                return False
    return True


def check_if_valid_input(
    name: str, input: str, options: List[str], raise_error: bool = True
) -> (bool, str):
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

    msg = ""
    if input not in options:
        msg = f"{name} is set to {str(input)}, which is an invalid setting. OPTIONS {options}"
        if raise_error:
            raise InvalidInputError(msg=msg, source=check_if_valid_input.__name__)
        else:
            return False, msg
    else:
        return True, msg


def check_minimum_roll_windows(
    roll_windows_values: List[int], minimum_fps: float
) -> List[int]:
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
    >>> check_if_string_value_is_valid_video_timestamp(value='00:00:10', name='My time stamp'
    """
    r = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d+)?$")
    if not r.match(value):
        raise InvalidInputError(
            msg=f"{name} should be in the format XX:XX:XX:XXXX or XX:XX:XX where X is an integer between 0-9. Got: {value}",
            source=check_if_string_value_is_valid_video_timestamp.__name__,
        )
    else:
        pass


def check_that_hhmmss_start_is_before_end(
    start_time: str, end_time: str, name: str
) -> None:
    """
    Helper to check that a start time in HH:MM:SS or HH:MM:SS:MS format is before an end time in HH:MM:SS or HH:MM:SS:MS format

    :param str start_time: Period start time in HH:MM:SS format.
    :param str end_time: Period end time in HH:MM:SS format.
    :param int name: Name of the variable
    :raises InvalidInputError: If end time is before the start time.

    :example:
    >>> check_that_hhmmss_start_is_before_end(start_time='00:00:05', end_time='00:00:01', name='My time period')
    >>> "InvalidInputError: My time period has an end-time which is before the start-time"
    >>> check_that_hhmmss_start_is_before_end(start_time='00:00:01', end_time='00:00:05')
    """

    if len(start_time.split(":")) != 3:
        raise InvalidInputError(
            f"Invalid time-stamp: ({start_time}). HH:MM:SS or HH:MM:SS.MS format required"
        )
    elif len(end_time.split(":")) != 3:
        raise InvalidInputError(
            f"Invalid time-stamp: ({end_time}). HH:MM:SS or HH:MM:SS.MS format required"
        )
    start_h, start_m, start_s = start_time.split(":")
    end_h, end_m, end_s = end_time.split(":")
    start_val = int(start_h) * 3600 + int(start_m) * 60 + float(start_s)
    end_val = int(end_h) * 3600 + int(end_m) * 60 + float(end_s)
    if end_val < start_val:
        raise InvalidInputError(
            f"{name} has an end-time which is before the start-time.",
            source=check_that_hhmmss_start_is_before_end.__name__,
        )


def check_nvidea_gpu_available() -> bool:
    """
    Helper to check of NVIDEA GPU is available via ``nvidia-smi``.
    returns bool: True if nvidia-smi returns not None. Else False.
    """
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False


def check_ffmpeg_available(raise_error: Optional[bool] = False) -> Union[bool, None]:
    """
    Helper to check of FFMpeg is available via subprocess ``ffmpeg``.

    :param Optional[bool] raise_error: If True, raises ``FFMPEGNotFoundError`` if FFmpeg can't be found. Else return False. Default False.
    :returns bool: True if ``ffmpeg`` returns not None and raise_error is False. Else False.
    """

    try:
        subprocess.call("ffmpeg", stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        if raise_error:
            raise FFMPEGNotFoundError(
                msg="FFMpeg could not be found on the instance (as evaluated via subprocess ffmpeg). Please make sure FFMpeg is installed."
            )
        else:
            return False


def check_if_valid_rgb_str(
    input: str,
    delimiter: str = ",",
    return_cleaned_rgb_tuple: bool = True,
    reverse_returned: bool = True,
):
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
        raise InvalidInputError(msg=f"{input} in not a valid RGB color")
    values = input.split(",")
    rgb = []
    for value in values:
        val = "".join(c for c in value if c.isdigit())
        check_int(
            name="RGB value", value=val, max_value=255, min_value=0, raise_error=True
        )
        rgb.append(val)
    rgb = tuple([int(x) for x in rgb])

    if return_cleaned_rgb_tuple:
        if reverse_returned:
            rgb = rgb[::-1]
        return rgb


def check_if_valid_rgb_tuple(data: Tuple[int, int, int],
                             raise_error: bool = True,
                             source: Optional[str] = None) -> bool:
    check_instance(source=check_if_valid_rgb_tuple.__name__, instance=data, accepted_types=tuple, raise_error=raise_error)
    check_iterable_length(source=check_if_valid_rgb_tuple.__name__, val=len(data), exact_accepted_length=3, raise_error=raise_error)
    for i in range(len(data)):
        if source is None:
            check_int(name="RGB value", value=data[i], max_value=255, min_value=0, raise_error=raise_error)
        else:
            check_int(name=f"RGB value {source}", value=data[i], max_value=255, min_value=0, raise_error=raise_error)
    return True


def check_if_list_contains_values(
    data: List[Union[float, int, str]],
    values: List[Union[float, int, str]],
    name: str,
    raise_error: bool = True,
) -> None:
    """
    Helper to check if values are represeted in a list. E.g., make sure annotatations of behvaior absent and present are represented in annitation column

    :param List[Union[float, int, str]] data: List of values. E.g., annotation column represented as list.
    :param List[Union[float, int, str]] values: Values to conform present. E.g., [0, 1].
    :param str name: Arbitrary name of the data for more useful error msg.
    :param bool raise_error: If True, raise error of not all values can be found in data. Else, print warning.

    :example:
    >>> check_if_list_contains_values(data=[1,2, 3, 4, 0], values=[0, 1, 6], name='My_data')
    """

    data, missing_values = list(set(data)), []
    for value in values:
        if value not in data:
            missing_values.append(value)

    if len(missing_values) > 0 and raise_error:
        raise NoDataError(
            msg=f"{name} does not contain the following expected values: {missing_values}",
            source=check_if_list_contains_values.__name__,
        )

    elif len(missing_values) > 0 and not raise_error:
        NoDataFoundWarning(
            msg=f"{name} does not contain the following expected values: {missing_values}",
            source=check_if_list_contains_values.__name__,
        )


def check_valid_hex_color(color_hex: str, raise_error: Optional[bool] = True) -> bool:
    """
    Check if given string represents a valid hexadecimal color code.

    :param str color_hex: A string representing a hexadecimal color code, either in the format '#RRGGBB' or '#RGB'.
    :param bool raise_error: If True, raise an exception when the color_hex is invalid; if False, return False instead. Default is True.
    :return bool: True if the color_hex is a valid hexadecimal color code; False otherwise (if raise_error is False).
    :raises IntegerError: If the color_hex is an invalid hexadecimal color code and raise_error is True.
    """

    hex_regex = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$")
    match = hex_regex.match(color_hex)
    if match is None and raise_error:
        raise IntegerError(
            msg=f"{color_hex} is an invalid hex color",
            source=check_valid_hex_color.__name__,
        )
    elif match is None and not raise_error:
        return False
    else:
        return True

def check_valid_url(url: str) -> bool:
    """ Helper to check if a string is a valid url"""
    regex = re.compile(
        r'^(https?|ftp)://'  # protocol
        r'(\S+(:\S*)?@)?'  # user:password (optional)
        r'((\d{1,3}\.){3}\d{1,3}|'  # IP address
        r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}))'  # domain name
        r'(:\d+)?'  # port (optional)
        r'(/[\S]*)?$',  # path (optional)
        re.IGNORECASE)
    return re.match(regex, url) is not None


def check_if_2d_array_has_min_unique_values(data: np.ndarray, min: int) -> bool:
    """
    Check if a 2D NumPy array has at least a minimum number of unique rows.

    For example, use when creating shapely Polygons or Linestrings, which typically requires at least 2 or three unique
    body-part coordinates.

    :param np.ndarray data: Input 2D array to be checked.
    :param np.ndarray min: Minimum number of unique rows required.
    :return bool: True if the input array has at least the specified minimum number of unique rows, False otherwise.

    :example:
    >>> data = np.array([[0, 0], [0, 0], [0, 0], [0, 1]])
    >>> check_if_2d_array_has_min_unique_values(data=data, min=2)
    >>> True
    """

    if len(data.shape) != 2:
        raise CountError(
            msg=f"Requires input array of two dimensions, found {data.size}",
            source=check_if_2d_array_has_min_unique_values.__name__,
        )
    sliced_data = np.unique(data, axis=0)
    if sliced_data.shape[0] < min:
        return False
    else:
        return True


def check_if_module_has_import(parsed_file: ast.Module, import_name: str) -> bool:
    """
    Check if a Python module has a specific import statement.

    Used for e.g., user custom feature extraction classes in ``simba.utils.custom_feature_extractor.CustomFeatureExtractor``.

    :parameter ast.Module file_path: The abstract syntax tree (AST) of the Python module.
    :parameter str import_name: The name of the module or package to check for in the import statements.
    :parameter bool: True if the specified import is found in the module, False otherwise.

    :example:
    >>> parsed_file = ast.parse(Path('/simba/misc/piotr.py').read_text())
    >>> check_if_module_has_import(parsed_file=parsed_file, import_name='argparse')
    >>> True
    """
    imports = [
        n for n in parsed_file.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    for i in imports:
        for name in i.names:
            if name.name == import_name:
                return True
    return False


def check_valid_extension(
    path: Union[str, os.PathLike], accepted_extensions: Union[List[str], str]
):
    """
    Checks if the file extension of the provided path is in the list of accepted extensions.

    :param Union[str, os.PathLike] file_path: The path to the file whose extension needs to be checked.
    :param List[str] accepted_extensions: A list of accepted file extensions. E.g., ['pickle', 'csv'].
    """
    if isinstance(accepted_extensions, (list, tuple)):
        check_valid_lst(data=accepted_extensions, source=f"{check_valid_extension.__name__} accepted_extensions", valid_dtypes=(str,), min_len=1)
    elif isinstance(accepted_extensions, str):
        check_str(name=f"{check_valid_extension.__name__} accepted_extensions", value=accepted_extensions)
        accepted_extensions = [accepted_extensions]
    accepted_extensions = [x.lower() for x in accepted_extensions]
    check_file_exist_and_readable(file_path=path)
    extension = get_fn_ext(filepath=path)[2][1:]
    if extension.lower() not in accepted_extensions:
        raise InvalidFilepathError(msg=f"File extension for file {path} has an invalid extension. Found {extension}, accepted: {accepted_extensions}", source=check_valid_extension.__name__)


def check_if_valid_img(data: np.ndarray,
                       source: str = "",
                       raise_error: bool = True,
                       greyscale: bool = False,
                       color: bool = False) -> Union[bool, None]:
    """
    Check if a variable is a valid image.

    :param str source: Name of the variable and/or class origin for informative error messaging and logging.
    :param np.ndarray data: Data variable to check if a valid image representation.
    :param bool greyscale: Checks that the image is greyscale. Default False.
    :param bool color: Checks that the image is color. Default False.
    :parameter bool raise_error: If True, raise InvalidInputError if invalid image representation. Else, return bool.
    """

    check_instance(source=check_if_valid_img.__name__, instance=data, accepted_types=(np.ndarray, cp.ndarray))
    if (data.ndim != 2) and (data.ndim != 3):
        if raise_error:
            raise InvalidInputError(msg=f"The {source} data is not a valid image. It has {data.ndim} dimensions", source=check_if_valid_img.__name__)
        else:
            return False
    if data.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        if raise_error:
            raise InvalidInputError(msg=f"The {source} data is not a valid image. It is dtype {data.dtype}", source=check_if_valid_img.__name__)
        else:
            return False
    if np.max(data) > 255:
        if raise_error:
            raise InvalidInputError(msg=f"The {source} data is not a valid image. Values found that are above 255: {np.max(data)}", source=check_if_valid_img.__name__)
    if greyscale:
        if (data.ndim != 2):
            if raise_error:
                raise InvalidInputError(msg=f"The {source} image is not a greyscale image. Got {data.ndim} dimensions", source=check_if_valid_img.__name__)
            else:
                return False
    if color:
        if (data.ndim != 3):
            if raise_error:
                raise InvalidInputError(msg=f"The {source} image is not a color image. Got {data.ndim} dimensions", source=check_if_valid_img.__name__)
            else:
                return False



    return True


def check_that_dir_has_list_of_filenames(
    dir: Union[str, os.PathLike],
    file_name_lst: List[str],
    file_type: Optional[str] = "csv",
):
    """
    Check that all file names in a list has an equivalent file in a specified directory. E.g., check if all files in the outlier corrected folder has an equivalent file in the featurues_extracted directory.

    :example:
    >>> file_name_lst = glob.glob('/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement' + '/*.csv')
    >>> check_that_dir_has_list_of_filenames(dir = '/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/features_extracted', file_name_lst=file_name_lst)
    """

    files_in_dir = glob.glob(dir + f"/*.{file_type}")
    files_in_dir = [os.path.basename(x) for x in files_in_dir]
    for file_name in file_name_lst:
        if os.path.basename(file_name) not in files_in_dir:
            raise NoFilesFoundError(msg=f"File name {os.path.basename(file_name)} could not be found in the directory {dir}", source=check_that_dir_has_list_of_filenames.__name__)


def check_valid_array(data: np.ndarray,
                      source: Optional[str] = "",
                      accepted_ndims: Optional[Union[Tuple[int], Any]] = None,
                      accepted_sizes: Optional[List[int]] = None,
                      accepted_axis_0_shape: Optional[List[int]] = None,
                      accepted_axis_1_shape: Optional[List[int]] = None,
                      accepted_dtypes: Optional[Union[List[Union[str, Type]], Tuple[Union[str, Type]], Iterable[Any]]] = None,
                      accepted_values: Optional[List[Any]] = None,
                      accepted_shapes: Optional[List[Tuple[int]]] = None,
                      min_axis_0: Optional[int] = None,
                      max_axis_1: Optional[int] = None,
                      min_axis_1: Optional[int] = None,
                      min_value: Optional[Union[float, int]] = None,
                      max_value: Optional[Union[float, int]] = None,
                      raise_error: bool = True) -> Union[None, bool]:
    """
    Check if the given  array satisfies specified criteria regarding its dimensions, shape, and data type.

    :parameter np.ndarray data: The numpy array to be checked.
    :parameter Optional[str] source: A string identifying the source, name, or purpose of the array for interpretable error messaging.
    :parameter Optional[Tuple[int]] accepted_ndims: List of tuples representing acceptable dimensions. If provided, checks whether the array's number of dimensions matches any tuple in the list.
    :parameter Optional[List[str]] accepted_axis_0_shape: List of accepted number of rows of 2-dimensional array. Will also raise error if value passed and input is not a 2-dimensional array.
    :parameter Optional[List[str]] accepted_axis_1_shape: List of accepted number of columns or fields of 2-dimensional array. Will also raise error if value passed and input is not a 2-dimensional array.
    :parameter Optional[List[int]] accepted_sizes: List of acceptable sizes for the array's shape. If provided, checks whether the length of the array's shape matches any value in the list.
    :parameter Optional[List[str]] accepted_dtypes: List of acceptable data types for the array. If provided, checks whether the array's data type matches any string in the list.

    :example:
    >>> data = np.array([[1, 2], [3, 4]])
    >>> check_valid_array(data, source="Example", accepted_ndims=(4, 3), accepted_sizes=[2], accepted_dtypes=['int'])
    """

    check_instance(source=source, instance=data, accepted_types=np.ndarray)
    if accepted_ndims is not None:
        if data.ndim not in accepted_ndims:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable dimensions. Found {data.ndim}, accepted: {accepted_ndims}: {source}", source=check_valid_array.__name__)
            else:
                return False
    if accepted_sizes is not None:
        if len(data.shape) not in accepted_sizes:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable size. Found {len(data.shape)}, accepted: {accepted_sizes}: {source}", source=check_valid_array.__name__)
            else:
                return False
    if accepted_dtypes is not None:
        if data.dtype not in accepted_dtypes:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable type. Found {data.dtype}, accepted: {accepted_dtypes}: {source}", source=check_valid_array.__name__)
            else:
                return False
    if accepted_shapes is not None:
        if data.shape not in accepted_shapes:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable shape. Found {data.shape}, accepted: {accepted_shapes}: {source}", source=check_valid_array.__name__)
            else:
                return False
    if accepted_axis_0_shape is not None:
        # if data.ndim != 2:
        #     raise ArrayError(
        #         msg=f"Array not of acceptable dimension. Found {data.ndim}, accepted: 2, {source}",
        #         source=check_valid_array.__name__,
        #     )
        if data.shape[0] not in accepted_axis_0_shape:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable shape. Found {data.shape[0]} rows, accepted: {accepted_axis_0_shape}, {source}", source=check_valid_array.__name__)
            else:
                return False
    if accepted_axis_1_shape is not None:
        if data.ndim != 2:
            raise ArrayError(msg=f"Array not of acceptable dimension. Found {data.ndim}, accepted: 2, {source}",  source=check_valid_array.__name__,)
        elif data.shape[1] not in accepted_axis_1_shape:
            if raise_error:
                raise ArrayError( msg=f"Array not of acceptable shape. Found {data.shape[0]} columns (axis=1), accepted: {accepted_axis_1_shape}, {source}", source=check_valid_array.__name__)
            else:
                return False
    if min_axis_0 is not None:
        check_int(name=f"{source} min_axis_0", value=min_axis_0)
        if data.shape[0] < min_axis_0:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable shape. Found  {data.shape[0]} rows, minimum accepted: {min_axis_0}, {source}", source=check_valid_array.__name__)
            else:
                return False

    if max_axis_1 is not None and data.ndim > 1:
        check_int(name=f"{source} max_axis_1", value=max_axis_1)
        if data.shape[1] > max_axis_1:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable shape. Found  {data.shape[1]} columns, maximum columns accepted: {max_axis_1}, {source}", source=check_valid_array.__name__)
            else:
                return False
    if min_axis_1 is not None and data.ndim > 1:
        check_int(name=f"{source} min_axis_1", value=min_axis_1)
        if data.shape[1] < min_axis_1:
            if raise_error:
                raise ArrayError(msg=f"Array not of acceptable shape. Found  {data.shape[1]} columns, minimum columns accepted: {min_axis_1}, {source}", source=check_valid_array.__name__)
            else:
                return False
    if accepted_values is not None:
        check_valid_lst(data=accepted_values, source=f"{source} accepted_values")
        additional_vals = list(set(np.unique(data)) - set(accepted_values))
        if len(additional_vals) > 0:
            if raise_error:
                raise ArrayError(msg=f"Array contains unacceptable values. Found  {additional_vals}, accepted: {accepted_values}, {source}", source=check_valid_array.__name__,)
            return False

    if min_value is not None:
        check_float(name=f'{source} min_value', value=min_value)
        if np.min(data) < min_value:
            if raise_error:
                raise ArrayError(msg=f"Array contains value below accepted value. Found  {np.min(data)}, accepted minimum: {min_value}, {source}", source=check_valid_array.__name__, )
            else:
                return False

    if max_value is not None:
        check_float(name=f'{source} max_value', value=max_value)
        if np.max(data) > max_value:
            if raise_error:
                raise ArrayError(msg=f"Array contains value above accepted maximum value. Found  {np.max(data)}, accepted minimum: {max_value}, {source}", source=check_valid_array.__name__, )
            else:
                return False
    else:
        return True

def check_valid_lst(data: list,
                    source: Optional[str] = "",
                    valid_dtypes: Optional[Union[Tuple[Any], List[Any], Any]] = None,
                    valid_values: Optional[List[Any]] = None,
                    min_len: Optional[int] = 1,
                    max_len: Optional[int] = None,
                    min_value: Optional[float] = None,
                    exact_len: Optional[int] = None,
                    raise_error: Optional[bool] = True) -> bool:
    """
    Check the validity of a list based on passed  criteria.

    :param list data: The input list to be validated.
    :param Optional[str] source: A string indicating the source or context of the data for informative error messaging.
    :param Optional[Tuple[Any]] valid_dtypes: A tuple of accepted data types. If provided, check if all elements in the list have data types in this tuple.
    :param Optional[List[Any]] valid_values: A list of accepted list values. If provided, check if all elements in the list have matching values in this list.
    :param Optional[int] min_len: The minimum allowed length of the list.
    :param Optional[int] max_len: The maximum allowed length of the list.
    :param Optional[bool] raise_error: If True, raise an InvalidInputError if any validation fails. If False, return False instead of raising an error.
    :return bool: True if all validation criteria are met, False otherwise.

    :example:
    >>> check_valid_lst(data=[1, 2, 'three'], valid_dtypes=(int, str), min_len=2, max_len=5)
    >>> check_valid_lst(data=[1, 2, 3], valid_dtypes=(int,), min_len=3)
    """
    check_instance(source=source, instance=data, accepted_types=list)
    if min_len is not None:
        check_int(
            name=f"{source} {min_len}",
            value=min_len,
            min_value=0,
            raise_error=raise_error,
        )
        if len(data) < min_len:
            if raise_error:
                raise InvalidInputError(
                    msg=f"Invalid length of list. Found {len(data)}, minimum accepted: {min_len}",
                    source=source,
                )
            else:
                return False

    check_instance(source=source, instance=data, accepted_types=list)
    if valid_dtypes is not None:
        for dtype in set([type(x) for x in data]):
            if dtype not in valid_dtypes:
                if raise_error:
                    raise InvalidInputError(msg=f"Invalid data type found in list. Found {dtype}, accepted: {valid_dtypes}", source=source)
                else:
                    return False

    if max_len is not None:
        check_int(
            name=f"{source} {max_len}",
            value=max_len,
            min_value=0,
            raise_error=raise_error,
        )
        if len(data) > max_len:
            if raise_error:
                raise InvalidInputError(
                    msg=f"Invalid length of list. Found {len(data)}, maximum accepted: {min_len}",
                    source=source,
                )
            else:
                return False
    if exact_len is not None:
        check_int(
            name=f"{source} {exact_len}",
            value=exact_len,
            min_value=0,
            raise_error=raise_error,
        )
        if len(data) != exact_len:
            if raise_error:
                raise InvalidInputError(
                    msg=f"Invalid length of list. Found {len(data)}, accepted: {exact_len}",
                    source=source,
                )
            else:
                return False

        if valid_values != None:
            check_valid_lst(
                data=valid_values, source=check_valid_lst.__name__, min_len=1
            )
            invalids = list(set(data) - set(valid_values))
            if len(invalids):
                if raise_error:
                    raise InvalidInputError(
                        msg=f"Invalid list entries. Found {invalids}, accepted: {valid_values}",
                        source=source,
                    )
                else:
                    return False

        if min_value != None:
            check_float(name=check_valid_lst.__name__, value=min_value)
            invalids = [x for x in data if x < min_value]
            if len(invalids) > 0:
                if raise_error:
                    raise InvalidInputError(msg=f"Invalid list entries. Found {invalids}, minimum accepted value: {min_value}", source=source)
                else:
                    return False

    return True


def check_if_keys_exist_in_dict(
    data: dict,
    key: Union[str, int, tuple, List],
    name: Optional[str] = "",
    raise_error: Optional[bool] = True,
) -> bool:

    check_instance(source=name, instance=data, accepted_types=(dict,))
    check_instance(
        source=name,
        instance=key,
        accepted_types=(
            str,
            int,
            tuple,
            List,
        ),
    )
    if not isinstance(key, (list, tuple)):
        key = [key]

    for k in key:
        if k not in list(data.keys()):
            if raise_error:
                raise InvalidInputError(
                    msg=f"{k} does not exist in object {name}",
                    source=check_if_keys_exist_in_dict.__class__.__name__,
                )
        else:
            pass
    return True


def check_that_directory_is_empty(directory: Union[str, os.PathLike], raise_error: Optional[bool] = True) -> None:
    """
    Checks if a directory is empty. If the directory has content, then returns False or raises ``DirectoryNotEmptyError``.

    :param str directory: Directory to check.
    :raises DirectoryNotEmptyError: If ``directory`` contains files.
    """

    check_if_dir_exists(in_dir=directory)
    try:
        all_files_in_folder = [
            f for f in next(os.walk(directory))[2] if not f[0] == "."
        ]
    except StopIteration:
        return 0
    else:
        if len(all_files_in_folder) > 0:
            if raise_error:
                raise DirectoryNotEmptyError(
                    msg=f"The {directory} is not empty and contains {str(len(all_files_in_folder))} files. Use a directory that is empty.",
                    source=check_that_directory_is_empty.__name__,
                )
            else:
                return False
        else:
            return True


def check_umap_hyperparameters(hyper_parameters: Dict[str, Any]) -> None:
    """
    Checks if dictionary of paramameters (umap, scaling, etc) are valid for grid-search umap dimensionality reduction .

    :param dict hyper_parameters: Dictionary holding umap hyerparameters.
    :raises InvalidInputError: If any input is invalid

    :example:
    >>> check_umap_hyperparameters(hyper_parameters={'n_neighbors': [2], 'min_distance': [0.1], 'spread': [1], 'scaler': 'MIN-MAX', 'variance': 0.2})
    """
    for key in UMAPParam.HYPERPARAMETERS.value:
        if key not in hyper_parameters.keys():
            raise InvalidInputError(
                msg=f"Hyperparameter dictionary is missing {key} entry.",
                source=check_umap_hyperparameters.__name__,
            )
    for key in [
        UMAPParam.N_NEIGHBORS.value,
        UMAPParam.MIN_DISTANCE.value,
        UMAPParam.SPREAD.value,
    ]:
        if not isinstance(hyper_parameters[key], list):
            raise InvalidInputError(
                msg=f"Hyperparameter dictionary key {key} has to be a list but got {type(hyper_parameters[key])}.",
                source=check_umap_hyperparameters.__name__,
            )
        if len(hyper_parameters[key]) == 0:
            raise InvalidInputError(
                msg=f"Hyperparameter dictionary key {key} has 0 entries.",
                source=check_umap_hyperparameters.__name__,
            )
        for value in hyper_parameters[key]:
            if not isinstance(value, (int, float)):
                raise InvalidInputError(
                    msg=f"Hyperparameter dictionary key {key} have to have numeric entries but got {type(value)}.",
                    source=check_umap_hyperparameters.__name__,
                )
    if hyper_parameters[UMAPParam.SCALER.value] not in Options.SCALER_OPTIONS.value:
        raise InvalidInputError(
            msg=f"Scaler {hyper_parameters[UMAPParam.SCALER.value]} not supported. Opitions: {Options.SCALER_OPTIONS.value}",
            source=check_umap_hyperparameters.__name__,
        )
    check_float(
        "VARIANCE THRESHOLD",
        value=hyper_parameters[UMAPParam.VARIANCE.value],
        min_value=0.0,
        max_value=100.0,
    )
def check_video_has_rois(roi_dict: Dict[str, pd.DataFrame],
                         roi_names: List[str] = None,
                         video_names: List[str] = None,
                         source: str = 'roi dict',
                         raise_error: bool = True):
    """
    Check that specified videos all have user-defined ROIs with specified names.
    """

    check_valid_dict(x=roi_dict, valid_key_dtypes=(str,), valid_values_dtypes=(pd.DataFrame,), required_keys=(Keys.ROI_RECTANGLES.value, Keys.ROI_CIRCLES.value, Keys.ROI_POLYGONS.value,),)
    check_valid_dataframe(df=roi_dict[Keys.ROI_RECTANGLES.value], source=f'{check_video_has_rois.__name__} {source} roi_dict {Keys.ROI_RECTANGLES.value}', required_fields=['Video', 'Name'])
    check_valid_dataframe(df=roi_dict[Keys.ROI_CIRCLES.value], source=f'{check_video_has_rois.__name__} {source} roi_dict {Keys.ROI_CIRCLES.value}', required_fields=['Video', 'Name'])
    check_valid_dataframe(df=roi_dict[Keys.ROI_POLYGONS.value], source=f'{check_video_has_rois.__name__} {source} roi_dict {Keys.ROI_POLYGONS.value}', required_fields=['Video', 'Name'])
    if roi_names is not None:
        check_valid_lst(data=roi_names, source=f'{check_video_has_rois.__name__} {source} roi_names', valid_dtypes=(str,), min_len=1)
    else:
        roi_names = list(set(list(roi_dict[Keys.ROI_RECTANGLES.value]['Name'].unique()) + list(roi_dict[Keys.ROI_CIRCLES.value]['Name'].unique()) + list(roi_dict[Keys.ROI_POLYGONS.value]['Name'].unique())))
    if video_names is not None:
        check_valid_lst(data=video_names, source=f'{check_video_has_rois.__name__} {source} video_names', min_len=1,)
    else:
        video_names = list(set(list(roi_dict[Keys.ROI_RECTANGLES.value]['Video'].unique()) + list(roi_dict[Keys.ROI_CIRCLES.value]['Video'].unique()) + list(roi_dict[Keys.ROI_POLYGONS.value]['Video'].unique())))
    missing_rois = {}
    rois_missing = False
    for video_name in video_names:
        missing_rois[video_name] = []
        for roi_name in roi_names:
            rect_filt = roi_dict[Keys.ROI_RECTANGLES.value][(roi_dict[Keys.ROI_RECTANGLES.value]['Video'] == video_name) & (roi_dict[Keys.ROI_RECTANGLES.value]['Name'] == roi_name)]
            circ_filt = roi_dict[Keys.ROI_CIRCLES.value][(roi_dict[Keys.ROI_CIRCLES.value]['Video'] == video_name) & (roi_dict[Keys.ROI_CIRCLES.value]['Name'] == roi_name)]
            poly_filt = roi_dict[Keys.ROI_POLYGONS.value][(roi_dict[Keys.ROI_POLYGONS.value]['Video'] == video_name) & (roi_dict[Keys.ROI_POLYGONS.value]['Name'] == roi_name)]
            if (len(rect_filt) + len(circ_filt) + len(poly_filt)) == 0:
                missing_rois[video_name].append(roi_name); rois_missing = True
    if rois_missing and raise_error:
        raise NoROIDataError(msg=f'Some videos are missing some ROIs: {missing_rois}', source=f'{check_video_has_rois.__name__} {source}')
    elif rois_missing:
        return False, missing_rois
    else:
        return True


def check_if_df_field_is_boolean(df: pd.DataFrame,
                                 field: str,
                                 raise_error: bool = True,
                                 bool_values: Optional[Tuple[Any]] = (0, 1),
                                 df_name: Optional[str] = ''):

    """Helper to check if a dataframe field is a boolean value"""
    check_instance(source=f'{check_if_df_field_is_boolean.__name__} df', instance=df, accepted_types=(pd.DataFrame,))
    check_str(name=f"{check_if_df_field_is_boolean.__name__} field", value=field)
    check_that_column_exist(df=df, column_name=field, file_name=check_if_df_field_is_boolean.__name__)
    additional = list((set(list(df[field])) - set(bool_values)))
    if len(additional) > 0:
        if raise_error:
            raise CountError(msg=f"Field {field} not a boolean in {df_name}. Found values {additional}. Accepted: {bool_values}", source=check_if_df_field_is_boolean.__name__)
        else:
            return False
    return True


def check_valid_dataframe(
    df: pd.DataFrame,
    source: Optional[str] = "",
    valid_dtypes: Optional[Tuple[Any]] = None,
    required_fields: Optional[List[str]] = None,
    min_axis_0: Optional[int] = None,
    min_axis_1: Optional[int] = None,
    max_axis_0: Optional[int] = None,
    max_axis_1: Optional[int] = None,
):
    """Helper to check if a dataframe is valid"""
    check_instance(source=source, instance=df, accepted_types=(pd.DataFrame,))
    if valid_dtypes is not None:
        dtypes = list(set(df.dtypes))
        additional = [x for x in dtypes if x not in valid_dtypes]
        if len(additional) > 0:
            raise InvalidInputError(
                msg=f"The dataframe {source} has invalid data format(s) {additional}. Valid: {valid_dtypes}",
                source=source,
            )
    if min_axis_1 is not None:
        check_int(name=f"{source} min_axis_1", value=min_axis_1, min_value=1)
        if len(df.columns) < min_axis_1:
            raise InvalidInputError(
                msg=f"The dataframe {source} has less than ({df.columns}) the required minimum number of columns ({min_axis_1}).",
                source=source,
            )
    if min_axis_0 is not None:
        check_int(name=f"{source} min_axis_0", value=min_axis_0, min_value=1)
        if len(df) < min_axis_0:
            raise InvalidInputError(
                msg=f"The dataframe {source} has less than ({len(df)}) the required minimum number of rows ({min_axis_0}).",
                source=source,
            )
    if max_axis_0 is not None:
        check_int(name=f"{source} max_axis_0", value=min_axis_0, min_value=1)
        if len(df) > max_axis_0:
            raise InvalidInputError(
                msg=f"The dataframe {source} has more than ({len(df)}) the required maximum number of rows ({max_axis_0}).",
                source=source,
            )
    if max_axis_1 is not None:
        check_int(name=f"{source} max_axis_1", value=min_axis_1, min_value=1)
        if len(df.columns) > max_axis_1:
            raise InvalidInputError(
                msg=f"The dataframe {source} has more than ({df.columns}) the required maximum number of columns ({max_axis_1}).",
                source=source,
            )
    if required_fields is not None:
        check_valid_lst(
            data=required_fields,
            source=check_valid_dataframe.__name__,
            valid_dtypes=(str,),
        )
        missing = list(set(required_fields) - set(df.columns))
        if len(missing) > 0:
            raise InvalidInputError(
                msg=f"The dataframe {source} are missing required columns {missing}.",
                source=source,
            )

def check_valid_boolean(value: Union[Any, List[Any]], source: Optional[str] = '', raise_error: Optional[bool] = True):
    if not isinstance(value, list):
        value = [value]
    for val in value:
        if val in (True, False):
            return True
        else:
            if raise_error:
                raise InvalidInputError(msg=f'{val} is not a valid boolean', source=source)
            else:
                return False

def check_valid_tuple(x: tuple,
                      source: Optional[str] = "",
                      accepted_lengths: Optional[Tuple[int]] = None,
                      valid_dtypes: Optional[Tuple[Any]] = None,
                      minimum_length: Optional[int] = None,
                      accepted_values: Optional[Iterable[Any]] = None,
                      min_integer: Optional[int] = None):

    if not isinstance(x, (tuple)):
        raise InvalidInputError(msg=f"{check_valid_tuple.__name__} {source} is not a valid tuple, got: {type(x)}", source=source,)
    if accepted_lengths is not None:
        if len(x) not in accepted_lengths:
            raise InvalidInputError(
                msg=f"Tuple is not of valid lengths. Found {len(x)}. Accepted: {accepted_lengths}",
                source=source,
            )
    if valid_dtypes is not None:
        dtypes = list(set([type(v) for v in x]))
        additional = [x for x in dtypes if x not in valid_dtypes]
        if len(additional) > 0:
            raise InvalidInputError(msg=f"The tuple {source} has invalid data format(s) {additional}. Valid: {valid_dtypes}", source=source)

    if minimum_length is not None:
        check_int(name=f'{check_valid_tuple.__name__} minimum_length', value=minimum_length, min_value=1)
        tuple_len = len(x)
        if tuple_len < minimum_length:
            raise InvalidInputError(msg=f"The tuple {source} is shorter ({tuple_len}) than the minimum required length ({minimum_length}).", source=source)

    if accepted_values is not None:
        check_instance(source=f'{check_valid_tuple.__name__} accepted_values', accepted_types=(list, tuple,), instance=accepted_values)
        for i in x:
            if i not in accepted_values:
                raise InvalidInputError(msg=f"The tuple {source} has a value that is NOT accepted: {i}, (accepted: {accepted_values}).", source=source)

    if min_integer is not None:
        check_int(name=f'{check_valid_tuple.__name__} min_integer', value=min_integer)
        for i in x:
            if isinstance(i, int):
                if i < min_integer:
                    raise InvalidInputError(msg=f"The tuple {source} has an integer value below the minimum allowed integer value: {i}, (minimum: {min_integer}).", source=source)


def check_video_and_data_frm_count_align(video: Union[str, os.PathLike, cv2.VideoCapture],
                                         data: Union[str, os.PathLike, pd.DataFrame],
                                         name: Optional[str] = "",
                                         raise_error: Optional[bool] = True) -> Union[None, bool]:
    """
    Check if the frame count of a video matches the row count of a data file.

    :param Union[str, os.PathLike, cv2.VideoCapture] video: Path to the video file or cv2.VideoCapture object.
    :param Union[str, os.PathLike, pd.DataFrame] data: Path to the data file or DataFrame containing the data.
    :param Optional[str] name: Name of the video (optional for interpretable error msgs).
    :param Optional[bool] raise_error: Whether to raise an error if the counts don't align (default is True). If False, prints warning.
    :return None:

    :example:
    >>> data_1 = '/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/outlier_corrected_movement_location/SI_DAY3_308_CD1_PRESENT.csv'
    >>> video_1 = '/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/frames/output/ROI_analysis/SI_DAY3_308_CD1_PRESENT.mp4'
    >>> check_video_and_data_frm_count_align(video=video_1, data=data_1, raise_error=True)
    """

    def _count_generator(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)

    check_instance(
        source=f"{check_video_and_data_frm_count_align.__name__} video",
        instance=video,
        accepted_types=(str, cv2.VideoCapture),
    )
    check_instance(
        source=f"{check_video_and_data_frm_count_align.__name__} data",
        instance=data,
        accepted_types=(str, pd.DataFrame),
    )
    check_str(
        name=f"{check_video_and_data_frm_count_align.__name__} name",
        value=name,
        allow_blank=True,
    )
    if isinstance(video, str):
        check_file_exist_and_readable(file_path=video)
        video = cv2.VideoCapture(video)
    video_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if isinstance(data, str):
        check_file_exist_and_readable(file_path=data)
        with open(data, "rb") as fp:
            c_generator = _count_generator(fp.raw.read)
            data_count = (sum(buffer.count(b"\n") for buffer in c_generator)) - 1
    else:
        data_count = len(data)
    if data_count != video_count:
        if not raise_error:
            FrameRangeWarning(msg=f"The video {name} has {video_count} frames, but the associated data file for this video has {data_count} rows", source=check_video_and_data_frm_count_align.__name__)
            return False
        else:
            raise FrameRangeError(msg=f"The video {name} has {video_count} frames, but the associated data file for this video has {data_count} rows", source=check_video_and_data_frm_count_align.__name__,)
    return True

def check_if_video_corrupted(video: Union[str, os.PathLike, cv2.VideoCapture],
                             frame_interval: Optional[int] = None,
                             frame_n: Optional[int] = 20,
                             raise_error: Optional[bool] = True) -> None:

    """
    Check if a video file is corrupted by inspecting a set of its frames.

    .. note::
       For decent run-time regardless of video length, pass a smaller ``frame_n`` (<100).

    :param Union[str, os.PathLike] video_path: Path to the video file or cv2.VideoCapture OpenCV object.
    :param Optional[int] frame_interval: Interval between frames to be checked. If None, ``frame_n`` will be used.
    :param Optional[int] frame_n: Number of frames to be checked, will be sampled at large allowed interval. If None, ``frame_interval`` will be used.
    :param Optional[bool] raise_error: Whether to raise an error if corruption is found. If False, prints warning.
    :return None:

    :example:
    >>> check_if_video_corrupted(video_path='/Users/simon/Downloads/NOR ENCODING FExMP8.mp4')
    """
    check_instance(source=f'{check_if_video_corrupted.__name__} video', instance=video, accepted_types=(str, cv2.VideoCapture))
    if isinstance(video, str):
        check_file_exist_and_readable(file_path=video)
        cap = cv2.VideoCapture(video)
    else:
        cap = video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if (frame_interval is not None and frame_n is not None) or (frame_interval is None and frame_n is None):
        raise InvalidInputError(msg='Pass frame_interval OR frame_n', source=check_if_video_corrupted.__name__)
    if frame_interval is not None:
        frms_to_check = list(range(0, frame_count, frame_interval))
    else:
        frms_to_check = np.array_split(np.arange(0, frame_count), frame_n)
        frms_to_check = [x[-1] for x in frms_to_check]
    errors = []
    for frm_id in frms_to_check:
        cap.set(1, frm_id)
        ret, _ = cap.read()
        if not ret: errors.append(frm_id)
    if len(errors) > 0:
        if raise_error:
            raise CorruptedFileError(msg=f'Found {len(errors)} corrupted frame(s) at indexes {errors} in video {video}', source=check_if_video_corrupted.__name__)
        else:
            CorruptedFileWarning(msg=f'Found {len(errors)} corrupted frame(s) at indexes {errors} in video {video}', source=check_if_video_corrupted.__name__)
    else:
        pass


def check_valid_dict(x: dict,
                     valid_key_dtypes: Optional[Tuple[Any]] = None,
                     valid_values_dtypes: Optional[Tuple[Any, ...]] = None,
                     valid_keys: Optional[Union[Tuple[Any], List[Any]]] = None,
                     max_len_keys: Optional[int] = None,
                     min_len_keys: Optional[int] = None,
                     required_keys: Optional[Tuple[Any, ...]] = None,
                     max_value: Optional[Union[float, int]] = None,
                     min_value: Optional[Union[float, int]] = None,
                     source: Optional[str] = None):


    source = check_valid_dict.__name__ if source is None else source
    check_instance(source=check_valid_dict.__name__, instance=x, accepted_types=(dict,))
    if valid_key_dtypes is not None:
        for i in list(x.keys()):
            if not isinstance(i, valid_key_dtypes):
                raise InvalidInputError(msg=f'{type(i)} is not a valid key DTYPE. Valid: {valid_key_dtypes}', source=source)
    if valid_values_dtypes is not None:
        for i in list(x.values()):
            if not isinstance(i, valid_values_dtypes):
                raise InvalidInputError(msg=f'{type(i)} is not a valid value DTYPE. Valid: {valid_values_dtypes}', source=source)
    if max_len_keys is not None:
        check_int(name=f'{check_valid_dict.__name__} max_len_keys', min_value=1, value=max_len_keys)
        key_cnt = len(list(x.keys()))
        if key_cnt > max_len_keys:
            raise InvalidInputError(msg=f'Dictionary have {key_cnt} keys. Maximum allowed: {max_len_keys}', source=source)
    if min_len_keys is not None:
        check_int(name=f'{check_valid_dict.__name__} min_len_keys', min_value=1, value=min_len_keys)
        key_cnt = len(list(x.keys()))
        if key_cnt < min_len_keys:
            raise InvalidInputError(msg=f'Dictionary have {key_cnt} keys. Minimum allowed: {min_len_keys}', source=source)
    if required_keys is not None:
        for i in list(required_keys):
            if i not in list(x.keys()):
                raise InvalidInputError(msg=f'The required key {i} does not exist in the dictionary. Existing keys: {list(x.keys())}', source=source)
    if max_value is not None:
        if not isinstance(max_value, (float, int)):
            raise InvalidInputError(msg=f'{check_valid_dict.__name__} max_value has to be a float or integer, got {type(max_value)}.')
        for k, v in x.items():
            if isinstance(v, (float, int)):
                if v > max_value:
                    raise InvalidInputError(msg=f'The required key {k} has value {v} which is above the max allowed: {max_value}.', source=source)
    if min_value is not None:
        if not isinstance(min_value, (float, int)):
            raise InvalidInputError(msg=f'{check_valid_dict.__name__} max_value has to be a float or integer, got {type(min_value)}.')
        for k, v in x.items():
            if isinstance(v, (float, int)):
                if v < min_value:
                    raise InvalidInputError(msg=f'The required key {k} has value {v} which is less than the minimum allowed: {min_value}.', source=source)
    if valid_keys is not None:
        if not isinstance(valid_keys, (tuple, list)):
            raise InvalidInputError(msg=f'{check_valid_dict.__name__} valid_keys has to tuple, got {type(valid_keys)}.')
        invalid_keys = [i for i in x.keys() if i not in valid_keys]
        if len(invalid_keys) > 0:
            raise InvalidInputError(msg=f'The dictionary has keys that are invalid ({invalid_keys}). Accepted, valid keys are: {valid_keys}.', source=source)




def is_video_color(video: Union[str, os.PathLike, cv2.VideoCapture]) -> bool:
    """
    Determines whether a video is in color or greyscale.

    .. seealso::
       :func:`simba.mixins.image_mixin.ImageMixin.is_video_color`

    :param Union[str, os.PathLike, cv2.VideoCapture] video: The video source, either a cv2.VideoCapture object or a path to a file on disk.
    :return: Returns `True` if the video is in color (has more than one channel), and `False` if the video is greyscale (single channel).
    :rtype: bool
    """

    check_instance(source=is_video_color.__name__, instance=video, accepted_types=(str, cv2.VideoCapture))
    
    # Handle string path vs VideoCapture object
    should_release = False
    if isinstance(video, str):
        check_file_exist_and_readable(file_path=video)
        video = cv2.VideoCapture(video)
        should_release = True

    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frm = video.read()
        
        # If frame has only 2 dimensions, it's definitely greyscale
        if frm.ndim == 2:
            return False
        
        # If frame has 3 dimensions, check if it's actually greyscale
        # (some greyscale videos are stored as 3-channel with identical values)
        if frm.ndim == 3:
            # Check if all channels are identical (indicating greyscale)
            if frm.shape[2] == 3:  # BGR format
                # Compare B, G, and R channels
                if np.array_equal(frm[:, :, 0], frm[:, :, 1]) and np.array_equal(frm[:, :, 1], frm[:, :, 2]):
                    return False  # All channels identical = greyscale
                else:
                    return True   # Channels different = color
            elif frm.shape[2] == 1:
                return False  # Single channel = greyscale
            else:
                return True   # Other multi-channel formats = color
        
        # Default case: assume greyscale
        return False
    
    finally:
        # Clean up VideoCapture if we created it
        if should_release and video.isOpened():
            video.release()


def check_filepaths_in_iterable_exist(file_paths: Iterable[str],
                                      name: Optional[str] = None):

    check_instance(source=f'{check_filepaths_in_iterable_exist.__name__} file_paths {name}', instance=file_paths, accepted_types=(list, tuple,))
    if len(file_paths) == 0:
        raise NoFilesFoundError(msg=f'{name} {file_paths} is empty')
    for file_path in file_paths:
        check_str(name=f'{check_filepaths_in_iterable_exist.__name__} {file_path} {name}', value=file_path)
        if not os.path.isfile(file_path):
            raise NoFilesFoundError(msg=f'{name} {file_path} is not a valid file path')

def check_all_dfs_in_list_has_same_cols(dfs: List[pd.DataFrame], raise_error: bool = True, source: str = '') -> bool:
    """ Checks that all dataframes in list has the same column names"""
    check_valid_lst(data=dfs, source=check_all_dfs_in_list_has_same_cols.__name__, valid_dtypes=(pd.DataFrame,), min_len=1)
    col_headers = [list(x.columns) for x in dfs]
    common_headers = set(col_headers[0]).intersection(*col_headers[1:])
    all_headers = set(item for sublist in col_headers for item in sublist)
    missing_headers = list(all_headers - common_headers)
    if len(missing_headers) > 0:
        if raise_error:
            raise MissingColumnsError(msg=f"The data in {source} directory do not contain the same headers. Some files are missing the headers: {missing_headers}", source=check_all_dfs_in_list_has_same_cols.__name__)
        else:
            return False
    return True


def is_valid_video_file(file_path: Union[str, os.PathLike], raise_error: bool = True):
    """
    Check if a file path is a valid video file.
    """
    check_file_exist_and_readable(file_path=file_path)
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            if not raise_error:
                return False
            else:
                raise InvalidFilepathError(msg=f'The path {file_path} is not a valid video file', source=is_valid_video_file.__name__)
        return True
    except Exception:
        if not raise_error:
            return False
        else:
            raise InvalidFilepathError(msg=f'The path {file_path} is not a valid video file', source=is_valid_video_file.__name__)
    finally:
        if 'cap' in locals():
            if cap.isOpened():
                cap.release()


def check_valid_polygon(polygon: Union[np.ndarray, Polygon], raise_error: bool = True, name: Optional[str] = None) -> Union[bool, None]:
    """
    Validates whether the given polygon is a valid geometric shape.

    :param Union[np.ndarray, Polygon] polygon: The polygon to validate, either as a NumPy array of shape (N, 2) or a shapely Polygon object.
    :param bool raise_error: If True, raises an InvalidInputError if the polygon is invalid; otherwise, returns False.
    :param Optional[str] name: An optional name for the polygon to include in error messages.
    :return: True if the polygon is valid, False if invalid (and raise_error is False), or None if an error is raised.
    """


    name = '' if name is None else name
    check_instance(source=f'{check_valid_polygon.__name__} polygon', accepted_types=(np.ndarray, Polygon,), instance=polygon)
    if isinstance(polygon, np.ndarray):
        check_valid_array(data=polygon, source=f'{check_valid_polygon.__name__} polygon', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_0=3, accepted_axis_1_shape=[2,])
        polygon = Polygon(polygon.astype(np.int32))
    if not polygon.is_valid:
        if raise_error:
            raise InvalidInputError(msg=f'The polygon {name} is invalid', source=check_valid_polygon.__name__)
        else:
            return False
    else:
        return True


def is_img_bw(img: np.ndarray,
              raise_error: bool = True,
              source: Optional[str] = '') -> bool:
    """
    Helper to check if an image is binarey black and white.
    """
    check_if_valid_img(data=img, source=is_img_bw.__name__, raise_error=True)
    px_vals = set(list(np.sort(np.unique(img)).astype(np.int32)))
    additional = list(px_vals - {0, 255})
    if len(additional) > 0:
        if raise_error:
            raise InvalidInputError(msg=f'The image {source} is not a black-and-white image. Expected: [0, 255], got {additional}', source=is_img_bw.__name__)
        else:
            return False
    else:
        return True

def is_img_greyscale(img: np.ndarray,
                      raise_error: bool = True,
                      source: Optional[str] = '') -> bool:

    check_if_valid_img(data=img, source=is_img_greyscale.__name__, raise_error=False, greyscale=True)
    if not img.ndim == 2:
        if raise_error:
            raise InvalidInputError(msg=f'The image {source} is not a greyscale image. Expected 2 dimensions got: {img.ndim}', source=is_img_greyscale.__name__)
        else:
            return False
    else:
        return True


def is_wsl() -> bool:
    """ Helper to check if SimBA is running in Microsoft WSL """
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False

def is_windows_path(value):
    """Check if the value is a valid Windows path."""
    return isinstance(value, str) and (len(value) > 1 and value[1] == ':' and value[0].isalpha())


def check_same_files_exist_in_all_directories(dirs: List[Union[str, os.PathLike]], raise_error: bool = False, file_type: str = "csv") -> bool:
    """
    Check if the same files of a given type exist in all specified directories.

    :param List[Union[str, os.PathLike]] dirs: List of directory paths to check.
    :param bool raise_error: If True, raises an error when file names do not match across directories. Defaults to False.
    :param bool raise_error: File extension (without the dot) to check for (e.g., 'csv', 'txt'). Defaults to 'csv'.
    """

    check_valid_lst( data=dirs, source=f"{check_same_files_exist_in_all_directories.__name__} dirs", valid_dtypes=(str, os.PathLike), min_len=2)
    file_sets = [{os.path.basename(f) for f in glob.glob(os.path.join(dir, f"*.{file_type}"))} for dir in dirs]
    common_files = set.intersection(*file_sets) if file_sets else set()
    if not all(files == common_files for files in file_sets):
        if raise_error:
            raise NoFilesFoundError( msg=f"Files of type '{file_type}' do not match across directories: {dirs}.",  source=check_same_files_exist_in_all_directories.__name__)
        return False
    return True



def check_valid_img_path(path: Union[str, os.PathLike], raise_error: bool = True):
    check_file_exist_and_readable(path)
    try:
        _ = cv2.imread(path)
    except Exception as e:
        if raise_error:
            print(e.args)
            raise InvalidInputError(msg=f'{path} could not be read as a valid image file', source=check_valid_img_path.__name__)
        else:
            return False
    return True




def check_valid_device(device: Union[Literal['cpu'], int], raise_error: bool = True) -> bool:
    """
    Validate a compute device specification, ensuring it is either 'cpu' or a valid GPU index.

    :param device: The device to validate. Should be the string 'cpu' for CPU usage, or an integer representing a CUDA device index (e.g., 0 for 'cuda:0').
    :param raise_error: If True, raises `InvalidInputError` or `SimBAGPUError` when the device is invalid. If False, returns `False` instead of raising errors. Default True.
    """
    source = check_valid_device.__name__
    if isinstance(device, str):
        valid, msg = check_str(name=f'{source} format', value=device.lower(), options=['cpu'], raise_error=False)
        if not valid:
            if raise_error:
                raise InvalidInputError(msg=msg, source=source)
            return False
        return True

    valid, msg = check_int(name=f'{source} device', value=device, min_value=0, raise_error=False)
    if not valid:
        if raise_error:
            raise InvalidInputError(msg=msg, source=source)
        return False

    gpu_available, gpus = _is_cuda_available()
    if not gpu_available:
        if raise_error:
            raise SimBAGPUError(msg=f'No GPU detected but device {device} passed', source=source)
        return False

    if device not in gpus:
        if raise_error:
            raise SimBAGPUError(msg=f'Unaccepted GPU device {device} passed. Accepted: {list(gpus.keys())}', source=source)
        return False

def is_lxc_container() -> bool:
    """
    Helper to check if the current environment is inside a LXC Linux container.

    .. note::
       See GitHub issue 457 for origin - https://github.com/sgoldenlab/simba/issues/457#issuecomment-3052631284
       Thanks Heinrich2818 - https://github.com/Heinrich2818

    :return: True if current environment is a LXC linux container, False if not.
    :rtype: bool
    """

    try:
        with open('/proc/1/cgroup') as f:
            for line in f:
                if 'lxc' in line:
                    return True
    except IOError:
        pass
    try:
        with open('/proc/self/mountinfo') as f:
            for line in f:
                if ' - cgroup2 ' in line:
                    mount_point = line.strip().split()[-1]
                    if 'lxc' in mount_point:
                        return True
    except IOError:
        pass
    try:
        with open('/proc/1/environ', 'rb') as f:
            env = f.read().split(b'\0')
            for e in env:
                if e == b'container=lxc':
                    return True
    except IOError:
        pass
    try:
        import subprocess
        r = subprocess.run(['systemd-detect-virt', '--container'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, text=True)
        if r.stdout.strip() == 'lxc':
            return True
    except Exception:
        pass
    return False



