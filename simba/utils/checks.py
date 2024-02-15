__author__ = "Simon Nilsson"

import ast
import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import trafaret as t

from simba.utils.errors import (ArrayError, ColumnNotFoundError,
                                CorruptedFileError, CountError, FloatError,
                                IntegerError, InvalidFilepathError,
                                InvalidInputError, NoDataError,
                                NoFilesFoundError, NotDirectoryError,
                                ParametersFileError, StringError)
from simba.utils.warnings import NoDataFoundWarning


def check_file_exist_and_readable(file_path: Union[str, os.PathLike]) -> None:
    """
    Checks if a path points to a readable file.

    :param str file_path: Path to file on disk.
    :raise NoFilesFoundError: The file does not exist.
    :raise CorruptedFileError: The file can not be read or is zero byte size.
    """
    check_instance(
        source="FILE PATH", instance=file_path, accepted_types=(str, os.PathLike)
    )
    if not os.path.isfile(file_path):
        raise NoFilesFoundError(
            msg=f"{file_path} is not a valid file path",
            source=check_file_exist_and_readable.__name__,
        )
    elif not os.access(file_path, os.R_OK):
        raise CorruptedFileError(
            msg=f"{file_path} is not readable",
            source=check_file_exist_and_readable.__name__,
        )
    elif os.stat(file_path).st_size == 0:
        raise CorruptedFileError(
            msg=f"{file_path} is 0 bytes and contains no data.",
            source=check_file_exist_and_readable.__name__,
        )
    else:
        pass


def check_int(
    name: str,
    value: Any,
    max_value: Optional[int] = None,
    min_value: Optional[int] = None,
    raise_error: Optional[bool] = True,
) -> (bool, str):
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
    return True, msg


def check_str(
    name: str,
    value: Any,
    options: Optional[Tuple[Any]] = (),
    allow_blank: bool = False,
    raise_error: bool = True,
) -> (bool, str):
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
            msg = f"{name} is set to {str(value)} in SimBA, but this is not a valid option: {options}"
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
) -> (bool, str):
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


def check_iterable_length(
    source: str,
    val: int,
    exact_accepted_length: Optional[int] = None,
    max: Optional[int] = np.inf,
    min: int = 1,
) -> None:

    if (not exact_accepted_length) and (not max) and (not min):
        raise InvalidInputError(
            msg=f"Provide exact_accepted_length or max and min values for {source}",
            source=check_iterable_length.__name__,
        )
    if exact_accepted_length:
        if val != exact_accepted_length:
            raise InvalidInputError(
                msg=f"{source} length is {val}, expected {exact_accepted_length}",
                source=check_iterable_length.__name__,
            )

    elif (val > max) or (val < min):
        raise InvalidInputError(
            msg=f"{source} value {val} does not full-fill criterion: min {min}, max{max} ",
            source=check_iterable_length.__name__,
        )


def check_instance(
    source: str, instance: object, accepted_types: Union[Tuple[object], object]
) -> None:
    """
    Check if an instance is an acceptable type.

    :param str name: Arbitrary name of instance used for interpretable error msg. Can also be the name of the method.
    :param object instance: A data object.
    :param Union[Tuple[object], object] accepted_types: Accepted instance types. E.g., (Polygon, pd.DataFrame) or Polygon.
    """

    if not isinstance(instance, accepted_types):
        raise InvalidInputError(
            msg=f"{source} requires {accepted_types}, got {type(instance)}",
            source=source,
        )


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


def check_if_dir_exists(in_dir: Union[str, os.PathLike]) -> None:
    """
    Check if a directory path exists.

    :param str in_dir: Putative directory path.
    :raise NotDirectoryError: The directory does not exist.
    """
    if not os.path.isdir(in_dir):
        raise NotDirectoryError(
            msg=f"{in_dir} is not a valid directory",
            source=check_if_dir_exists.__name__,
        )


def check_that_column_exist(
    df: pd.DataFrame, column_name: Union[str, os.PathLike, List[str]], file_name: str
) -> None:
    """
    Check if single named field or a list of fields exist within a dataframe.

    :param pd.DataFrame df:
    :param str column_name: Name or names of field(s).
    :param str file_name: Path of ``df`` on disk.
    :raise ColumnNotFoundError: The ``column_name`` does not exist within ``df``.
    """

    if type(column_name) == str:
        column_name = [column_name]
    for column in column_name:
        if column not in df.columns:
            raise ColumnNotFoundError(
                column_name=column,
                file_name=file_name,
                source=check_that_column_exist.__name__,
            )


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


def check_ffmpeg_available() -> bool:
    """
    Helper to check of FFMpeg is available via subprocess ``ffmpeg``.

    returns bool: True if ``ffmpeg`` returns not None. Else False.
    """
    try:
        subprocess.call("ffmpeg", stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except Exception:
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


def check_if_valid_rgb_tuple(data: Tuple[int, int, int]) -> bool:
    check_instance(
        source=check_if_valid_rgb_tuple.__name__, instance=data, accepted_types=tuple
    )
    check_iterable_length(
        source=check_if_valid_rgb_tuple.__name__, val=len(data), exact_accepted_length=3
    )
    for i in range(len(data)):
        check_int(
            name="RGB value",
            value=data[i],
            max_value=255,
            min_value=0,
            raise_error=True,
        )


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


def check_if_valid_img(
    data: np.ndarray, source: Optional[str] = "", raise_error: Optional[bool] = True
) -> Union[bool, None]:
    """
    Check if a variable is a valid image.

    :parameter str source: Name of the variable and/or class origin for informative error messaging and logging.
    :parameter np.ndarray data: Data variable to check if a valid image representation.
    :parameter Optional[bool] raise_error: If True, raise InvalidInputError if invalid image representation. Else, return bool.
    """

    check_instance(
        source=check_if_valid_img.__name__, instance=data, accepted_types=np.ndarray
    )
    if (data.ndim != 2) and (data.ndim != 3):
        if raise_error:
            raise InvalidInputError(
                msg=f"The {source} data is not a valid image. It has {data.ndim} dimensions",
                source=check_if_valid_img.__name__,
            )
        else:
            return False
    if data.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        if raise_error:
            raise InvalidInputError(
                msg=f"The {source} data is not a valid image. It is dtype {data.dtype}",
                source=check_if_valid_img.__name__,
            )
        else:
            return False


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
            raise NoFilesFoundError(
                msg=f"File name {os.path.basename(file_name)} could not be found in the directory {dir}",
                source=check_that_dir_has_list_of_filenames.__name__,
            )


def check_valid_array(
    data: np.ndarray,
    source: Optional[str] = "",
    accepted_ndims: Optional[List[Tuple[int]]] = None,
    accepted_sizes: Optional[List[int]] = None,
    accepted_dtypes: Optional[List[str]] = None,
    accepted_shapes: Optional[List[Tuple[int]]] = None,
) -> None:
    """
    Check if the given  array satisfies specified criteria regarding its dimensions, shape, and data type.

    :parameter np.ndarray data: The numpy array to be checked.
    :parameter Optional[str] source: A string identifying the source, name, or purpose of the array for interpretable error messaging.
    :parameter Optional[Tuple[int]] accepted_ndims: List of tuples representing acceptable dimensions. If provided, checks whether the array's number of dimensions matches any tuple in the list.
    :parameter Optional[List[int]] accepted_sizes: List of acceptable sizes for the array's shape. If provided, checks whether the length of the array's shape matches any value in the list.
    :parameter Optional[List[str]] accepted_dtypes: List of acceptable data types for the array. If provided, checks whether the array's data type matches any string in the list.

    :example:
    >>> data = np.array([[1, 2], [3, 4]])
    >>> check_valid_array(data, source="Example", accepted_ndims=(4, 3), accepted_sizes=[2], accepted_dtypes=['int'])
    """

    check_instance(source=source, instance=data, accepted_types=np.ndarray)
    if accepted_ndims is not None:
        if data.ndim not in accepted_ndims:
            raise ArrayError(
                msg=f"Array not of acceptable dimensions. Found {data.ndim}, accepted: {accepted_ndims}: {source}",
                source=check_valid_array.__name__,
            )
    if accepted_sizes is not None:
        if len(data.shape) not in accepted_sizes:
            raise ArrayError(
                msg=f"Array not of acceptable size. Found {len(data.shape)}, accepted: {accepted_sizes}: {source}",
                source=check_valid_array.__name__,
            )
    if accepted_dtypes is not None:
        if data.dtype not in accepted_dtypes:
            raise ArrayError(
                msg=f"Array not of acceptable type. Found {data.dtype}, accepted: {accepted_dtypes}: {source}",
                source=check_valid_array.__name__,
            )
    if accepted_shapes is not None:
        if data.shape not in accepted_shapes:
            raise ArrayError(
                msg=f"Array not of acceptable shape. Found {data.shape}, accepted: {accepted_shapes}: {source}",
                source=check_valid_array.__name__,
            )


def check_valid_lst(
    data: list,
    source: Optional[str] = "",
    valid_dtypes: Optional[Tuple[Any]] = None,
    min_len: Optional[int] = 1,
    max_len: Optional[int] = None,
    raise_error: Optional[bool] = True,
) -> bool:
    """
    Check the validity of a list based on passed  criteria.

    :parameter list data: The input list to be validated.
    :parameter Optional[str] source: A string indicating the source or context of the data for informative error messaging.
    :parameter Optional[Tuple[Any]] valid_dtypes: A tuple of accepted data types. If provided, check if all elements in the list have data types in this tuple.
    :parameter Optional[int] min_len: The minimum allowed length of the list.
    :parameter Optional[int] max_len: The maximum allowed length of the list.
    :parameter Optional[bool] raise_error: If True, raise an InvalidInputError if any validation fails. If False, return False instead of raising an error.
    :return bool: True if all validation criteria are met, False otherwise.

    :example:
    >>> check_valid_lst(data=[1, 2, 'three'], valid_dtypes=(int, str), min_len=2, max_len=5)
    >>> check_valid_lst(data=[1, 2, 3], valid_dtypes=(int,), min_len=3)
    """

    check_instance(source=source, instance=data, accepted_types=list)
    if valid_dtypes is not None:
        for dtype in set([type(x) for x in data]):
            if dtype not in valid_dtypes:
                if raise_error:
                    raise InvalidInputError(
                        msg=f"Invalid data type found in list. Found {dtype}, accepted: {valid_dtypes}",
                        source=source,
                    )
                else:
                    return False
    if min_len is not None:
        if len(data) < min_len:
            if raise_error:
                raise InvalidInputError(
                    msg=f"Invalid length of list. Found {len(data)}, minimum accepted: {min_len}",
                    source=source,
                )
            else:
                return False
    if max_len is not None:
        if len(data) > max_len:
            if raise_error:
                raise InvalidInputError(
                    msg=f"Invalid length of list. Found {len(data)}, maximum accepted: {min_len}",
                    source=source,
                )
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
    if not isinstance(key, list):
        key = [key]
    for k in key:
        if k not in data.keys():
            if raise_error:
                raise InvalidInputError(
                    msg=f"{k} does not exist in object {name}",
                    source=check_if_keys_exist_in_dict.__class__.__name__,
                )
        else:
            return False

    return True
