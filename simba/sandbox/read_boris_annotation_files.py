from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import os

import numpy as np
import pandas as pd

from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_dir_exists, check_int, check_str,
    check_valid_boolean, check_valid_dataframe, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import Methods
from simba.utils.errors import (ColumnNotFoundError, FrameRangeError,
                                InvalidFileTypeError, InvalidInputError)
from simba.utils.read_write import (bento_file_reader,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_video_info,
                                    read_video_info_csv, write_pickle)
from simba.utils.warnings import ThirdPartyAnnotationsInvalidFileFormatWarning


def is_new_boris_version(pd_df: pd.DataFrame):
    """
    Check the format of a boris annotation file.

    In the new version, additional column names are present, while
    others have slightly different name. Here, we check for the presence
    of a column name present only in the newer version.

    :return: True if newer version
    """
    return "Media file name" in list(pd_df.columns)

def read_boris_file(file_path: Union[str, os.PathLike],
                    fps: Optional[Union[int, float]] = None,
                    orient: Optional[Literal['index', 'columns']] = 'index',
                    save_path: Optional[Union[str, os.PathLike]] = None,
                    raise_error: Optional[bool] = False,
                    log_setting: Optional[bool] = False) -> Union[None, Dict[str, pd.DataFrame]]:

    """
    Reads a BORIS behavioral annotation file, processes the data, and optionally saves the results to a file.

    :param Union[str, os.PathLike] file_path: The path to the BORIS file to be read. The file should be a CSV containing behavioral annotations.
    :param Optional[Union[int, float]] fps: Frames per second (FPS) to convert time annotations into frame numbers. If not provided, it will be extracted from the BORIS file if available.
    :param Optional[Literal['index', 'columns']] orient: Determines the orientation of the results. 'index' will organize data with start and stop times as indices, while 'columns' will store data in columns.
    :param Optional[Union[str, os.PathLike] save_path: The path where the processed results should be saved as a pickle file. If not provided, the results will be returned instead.
    :param Optional[bool] raise_error: Whether to raise errors if the file format or content is invalid. If False, warnings will be logged instead of raising exceptions.
    :param Optional[bool] log_setting: Whether to log warnings and errors.  This is relevant when `raise_error` is set to False.
    :return: If `save_path` is None, returns a dictionary where keys are behaviors and values are dataframes  containing start and stop frames for each behavior. If `save_path` is provided, the results are saved and nothing is returned.
    """

    MEDIA_FILE_NAME = "Media file name"
    BEHAVIOR_TYPE = 'Behavior type'
    OBSERVATION_ID = "Observation id"
    TIME = "Time"
    FPS = 'FPS'
    EVENT = 'EVENT'
    BEHAVIOR = "Behavior"
    START = 'START'
    FRAME = 'FRAME'
    STOP = 'STOP'
    STATUS = "Status"
    MEDIA_FILE_PATH = "Media file path"

    results = {}
    check_file_exist_and_readable(file_path=file_path)
    if fps is not None:
        check_int(name=f'{read_boris_file.__name__} fps', min_value=1, value=fps)
    check_str(name=f'{read_boris_file.__name__} orient', value=orient, options=('index', 'columns'))
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
    boris_df = pd.read_csv(file_path)
    if not is_new_boris_version(boris_df):
        expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
        if not OBSERVATION_ID in boris_df.columns:
            if raise_error:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file', source=read_boris_file.__name__)
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return results
        start_idx = boris_df[boris_df[OBSERVATION_ID] == TIME].index.values
        if len(start_idx) != 1:
            if raise_error:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file', source=read_boris_file.__name__)
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return results
        df = pd.read_csv(file_path, skiprows=range(0, int(start_idx + 1)))
    else:
        MEDIA_FILE_PATH, STATUS = MEDIA_FILE_NAME, BEHAVIOR_TYPE
        expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
        df = pd.read_csv(file_path)
    check_valid_dataframe(df=df, source=f'{read_boris_file.__name__} {file_path}', required_fields=expected_headers)
    _, video_base_name, _ = get_fn_ext(df.loc[0, MEDIA_FILE_PATH])
    numeric_check = pd.to_numeric(df[TIME], errors='coerce').notnull().all()
    if not numeric_check:
        if raise_error:
            raise InvalidInputError(msg=f'SimBA found TIME DATA annotation in file {file_path} that could not be interpreted as numeric values (seconds or frame numbers)')
        else:
            ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
            return results
    df[TIME] = df[TIME].astype(np.float32)
    fps = None
    if fps is None:
        if not FPS in df.columns:
            if raise_error:
                raise FrameRangeError(f'The annotations are in seconds and FPS was not passed. FPS could also not be read from the BORIS file', source=bento_file_reader.__name__)
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return results
        fps = df[FPS].iloc[0]
        if not isinstance(fps, (float, int)):
            if raise_error:
                raise FrameRangeError(f'The annotations are in seconds and FPS was not passed. FPS could also not be read from the BORIS file', source=bento_file_reader.__name__)
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return results
    df = df[expected_headers]
    df['FRAME'] = (df[TIME] * fps).astype(int)
    df = df.drop([TIME, MEDIA_FILE_PATH], axis=1)
    df = df.rename(columns={BEHAVIOR: 'BEHAVIOR', STATUS: EVENT})

    for clf in df['BEHAVIOR'].unique():
        clf_df = df[df['BEHAVIOR'] == clf].reset_index(drop=True)
        if orient == 'column':
            results[clf] = clf_df
        else:
            start_clf, stop_clf = clf_df[clf_df[EVENT] == START].reset_index(drop=True), clf_df[clf_df[EVENT] == STOP].reset_index(drop=True)
            start_clf = start_clf.rename(columns={FRAME: START}).drop([EVENT, 'BEHAVIOR'], axis=1)
            stop_clf = stop_clf.rename(columns={FRAME: STOP}).drop([EVENT], axis=1)
            if len(start_clf) != len(stop_clf):
                if raise_error:
                    raise FrameRangeError(f'In file {file_path}, the number of start events ({len(start_clf)}) and stop events ({len(stop_clf)}) for behavior {clf} is not equal', source=bento_file_reader.__name__)
                else:
                    ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                    return results
            clf_df = pd.concat([start_clf, stop_clf], axis=1)[['BEHAVIOR', START, STOP]]
            results[clf] = clf_df
    if save_path is None:
        return results
    else:
        write_pickle(data=results, save_path=save_path)


def read_boris_annotation_files(data_paths: Union[List[str], str, os.PathLike],
                                video_info_df: Union[str, os.PathLike, pd.DataFrame],
                                error_setting: Literal[Union[None, Methods.ERROR.value, Methods.WARNING.value]] = None,
                                log_setting: Optional[bool] = False) -> Dict[str, pd.DataFrame]:
    """
    Reads multiple BORIS behavioral annotation files and compiles the data into a dictionary of dataframes.

    :param Union[List[str], str, os.PathLike] data_paths: Paths to the BORIS annotation files. This can be a list of file paths, a single directory containing the files, or a single file path.
    :param Union[str, os.PathLike, pd.DataFrame] video_info_df: The path to a CSV file, an existing dataframe, or a file-like object containing video information  (e.g., FPS, video name). This data is used to align the annotation files with their respective videos.
    :param Literal[Union[None, Methods.ERROR.value, Methods.WARNING.value]] error_setting: Defines the behavior when encountering issues in the files. Options are `Methods.ERROR.value` to raise errors, `Methods.WARNING.value` to log warnings, or `None` for no action.
    :param Optional[bool] log_setting: Whether to log warnings and errors when `error_setting` is set to `Methods.WARNING.value`.  Defaults to `False`.
    :return: A dictionary where each key is a video name, and each value is a dataframe containing the compiled behavioral annotations from the corresponding BORIS file.
    """


    if error_setting is not None:
        check_str(name=f'{read_boris_annotation_files.__name__} error_setting', value=error_setting, options=(Methods.ERROR.value, Methods.WARNING.value))
    check_valid_boolean(value=log_setting, source=f'{read_boris_annotation_files.__name__} log_setting')
    raise_error = False
    if error_setting == Methods.ERROR.value:
        raise_error = True
    if isinstance(video_info_df, str):
        check_file_exist_and_readable(file_path=video_info_df)
        video_info_df = read_video_info_csv(file_path=video_info_df)
    if isinstance(data_paths, list):
        check_valid_lst(data=data_paths, source=f'{read_boris_annotation_files.__name__} data_paths', min_len=1, valid_dtypes=(str,))
    elif isinstance(data_paths, str):
        check_if_dir_exists(in_dir=data_paths, source=f'{read_boris_annotation_files.__name__} data_paths')
        data_paths = find_files_of_filetypes_in_directory(directory=data_paths, extensions=['.csv'], raise_error=True)
    check_all_file_names_are_represented_in_video_log(video_info_df=video_info_df, data_paths=data_paths)
    check_valid_dataframe(df=video_info_df, source=read_boris_annotation_files.__name__)
    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(file_path)
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        boris_dict = read_boris_file(file_path=file_path, fps=fps, orient='columns', raise_error=raise_error, log_setting=log_setting)
        dfs[video_name] = pd.concat(boris_dict.values(), ignore_index=True)
    return dfs




    #     boris_df = pd.read_csv(file_path)
    #     try:
    #         if not is_new_boris_version(boris_df):
    #             expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
    #             start_idx = boris_df[boris_df[OBSERVATION_ID] == TIME].index.values
    #             df = pd.read_csv(file_path, skiprows=range(0, int(start_idx + 1)))[
    #                 expected_headers
    #             ]
    #         else:
    #             # Adjust column names to newer BORIS annotation format
    #             MEDIA_FILE_PATH = "Media file name"
    #             STATUS = "Behavior type"
    #             expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
    #             df = pd.read_csv(file_path)[expected_headers]
    #         _, video_base_name, _ = get_fn_ext(df.loc[0, MEDIA_FILE_PATH])
    #         df.drop(MEDIA_FILE_PATH, axis=1, inplace=True)
    #         df.columns = ["TIME", "BEHAVIOR", "EVENT"]
    #         df["TIME"] = df["TIME"].astype(float)
    #         dfs[video_base_name] = df.sort_values(by="TIME")
    #     except Exception as e:
    #         print(e)
    #         if error_setting == Methods.WARNING.value:
    #             ThirdPartyAnnotationsInvalidFileFormatWarning(
    #                 annotation_app="BORIS", file_path=file_path, log_status=log_setting
    #             )
    #         elif error_setting == Methods.ERROR.value:
    #             raise InvalidFileTypeError(
    #                 msg=f"{file_path} is not a valid BORIS file. See the docs for expected file format."
    #             )
    #         else:
    #             pass
    # for video_name, video_df in dfs.items():
    #     _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
    #     video_df["FRAME"] = (video_df["TIME"] * fps).astype(int)
    #     video_df.drop("TIME", axis=1, inplace=True)
    # return dfs


# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
df = read_boris_annotation_files(data_paths=[r"C:\troubleshooting\boris_test\project_folder\boris_files\c_oxt23_190816_132617_s_trimmcropped.csv"],
                                 error_setting='WARNING',
                                 log_setting=False,
                                 video_info_df=r"C:\troubleshooting\boris_test\project_folder\logs\video_info.csv")