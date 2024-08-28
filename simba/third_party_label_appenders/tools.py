from typing import Dict, List, Union, Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import os

from simba.utils.data import detect_bouts
from simba.utils.enums import Methods
from simba.utils.errors import ColumnNotFoundError, InvalidFileTypeError
from simba.utils.read_write import get_fn_ext, read_video_info, bento_file_reader, read_video_info_csv, find_files_of_filetypes_in_directory
from simba.utils.warnings import ThirdPartyAnnotationsInvalidFileFormatWarning
from simba.utils.checks import (check_valid_lst,
                                check_valid_dataframe,
                                check_all_file_names_are_represented_in_video_log,
                                check_str,
                                check_valid_boolean,
                                check_file_exist_and_readable,
                                check_if_dir_exists)

BENTO = "Bento"


def read_bento_files(data_paths: Union[List[str], str, os.PathLike],
                     video_info_df: Union[str, os.PathLike, pd.DataFrame],
                     error_setting: Literal[Union[None, Methods.ERROR.value, Methods.WARNING.value]] = None,
                     log_setting: Optional[bool] = False) -> Dict[str, pd.DataFrame]:

    """
    Reads multiple BENTO annotation files and processes them into a dictionary of DataFrames, each representing the
    combined annotations for a corresponding video. The function verifies that all files exist and that the file names
    match the video information provided.

    :param Union[List[str], str, os.PathLike] data_paths: Paths to BENTO annotation files or a directory containing such files. If a directory is provided, all files with the extension '.annot' will be processed.
    :param Union[str, os.PathLike, pd.DataFrame] video_info_df: Path to a CSV file containing video information or a preloaded DataFrame with the same data.  This information is used to match BENTO files with their corresponding videos and extract the FPS.
    :param Literal[Union[None, Methods.ERROR.value, Methods.WARNING.value]] error_setting: Determines the error handling mode. If set to `Methods.ERROR.value`, errors will raise exceptions. If set to `Methods.WARNING.value`, errors will generate warnings instead. If None, no error handling modifications are applied.
    :param Optional[bool] = False) -> Dict[str, pd.DataFrame] log_setting: If True, logging will be enabled for the process, providing detailed information about the steps being executed.
    :return: A dictionary where the keys are video names and the values are DataFrames containing the combined annotations for each video.
    :rtype: Dict[str, pd.DataFrame]

    :example:
    >>> dfs = read_bento_files(data_paths=r"C:\troubleshooting\bento_test\bento_files", error_setting='WARNING', log_setting=False, video_info_df=r"C:\troubleshooting\bento_test\project_folder\logs\video_info.csv")
    """

    if error_setting is not None:
        check_str(name=f'{read_bento_files.__name__} error_setting', value=error_setting, options=(Methods.ERROR.value, Methods.WARNING.value))
    check_valid_boolean(value=log_setting, source=f'{read_bento_files.__name__} log_setting')
    raise_error = False
    if error_setting == Methods.ERROR.value:
        raise_error = True
    if isinstance(video_info_df, str):
        check_file_exist_and_readable(file_path=video_info_df)
        video_info_df = read_video_info_csv(file_path=video_info_df)
    if isinstance(data_paths, list):
        check_valid_lst(data=data_paths, source=f'{read_bento_files.__name__} data_paths', min_len=1, valid_dtypes=(str,))
    elif isinstance(data_paths, str):
        check_if_dir_exists(in_dir=data_paths, source=f'{read_bento_files.__name__} data_paths')
        data_paths = find_files_of_filetypes_in_directory(directory=data_paths, extensions=['.annot'], raise_error=True)
    check_all_file_names_are_represented_in_video_log(video_info_df=video_info_df, data_paths=data_paths)
    check_valid_dataframe(df=video_info_df, source=read_bento_files.__name__)
    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, ext = get_fn_ext(filepath=file_path)
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        bento_dict = bento_file_reader(file_path=file_path, fps=fps, orient='columns', save_path=None, raise_error=raise_error, log_setting=log_setting)
        dfs[video_name] = pd.concat(bento_dict.values(), ignore_index=True)

    return dfs

def observer_timestamp_corrector(timestamps: List[str]) -> List[str]:
    corrected_ts = []
    for timestamp in timestamps:
        h, m, s = timestamp.split(":", 3)
        missing_fractions = 9 - len(s)
        if missing_fractions == 0:
            corrected_ts.append(timestamp)
        else:
            corrected_ts.append(f'{h}:{m}:{s}.{"0" * missing_fractions}')
    return corrected_ts


def is_new_boris_version(pd_df: pd.DataFrame):
    """
    Check the format of a boris annotation file.

    In the new version, additional column names are present, while
    others have slightly different name. Here, we check for the presence
    of a column name present only in the newer version.

    :return: True if newer version
    """
    return "Media file name" in list(pd_df.columns)


def read_boris_annotation_files(data_paths: List[str], error_setting: str, video_info_df: pd.DataFrame, log_setting: bool = False) -> Dict[str, pd.DataFrame]:
    MEDIA_FILE_PATH = "Media file path"
    OBSERVATION_ID = "Observation id"
    TIME = "Time"
    BEHAVIOR = "Behavior"
    STATUS = "Status"

    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(file_path)
        boris_df = pd.read_csv(file_path)
        try:
            if not is_new_boris_version(boris_df):
                expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
                start_idx = boris_df[boris_df[OBSERVATION_ID] == TIME].index.values
                df = pd.read_csv(file_path, skiprows=range(0, int(start_idx + 1)))[
                    expected_headers
                ]
            else:
                # Adjust column names to newer BORIS annotation format
                MEDIA_FILE_PATH = "Media file name"
                STATUS = "Behavior type"
                expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
                df = pd.read_csv(file_path)[expected_headers]
            _, video_base_name, _ = get_fn_ext(df.loc[0, MEDIA_FILE_PATH])
            df.drop(MEDIA_FILE_PATH, axis=1, inplace=True)
            df.columns = ["TIME", "BEHAVIOR", "EVENT"]
            df["TIME"] = df["TIME"].astype(float)
            dfs[video_base_name] = df.sort_values(by="TIME")
        except Exception as e:
            print(e)
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(
                    annotation_app="BORIS", file_path=file_path, log_status=log_setting
                )
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(
                    msg=f"{file_path} is not a valid BORIS file. See the docs for expected file format."
                )
            else:
                pass
    for video_name, video_df in dfs.items():
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        video_df["FRAME"] = (video_df["TIME"] * fps).astype(int)
        video_df.drop("TIME", axis=1, inplace=True)
    return dfs


# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_boris_annotation_files(data_paths=['/Users/simon/Downloads/FIXED/c_oxt27_190816_122013_s_trimmcropped.csv'],
#                                 error_setting='WARNING',
#                                  log_setting=False,
#                                  video_info_df=video_info_df)


def read_ethovision_files(
    data_paths: List[str],
    error_setting: str,
    video_info_df: pd.DataFrame,
    log_setting: bool = False,
) -> Dict[str, pd.DataFrame]:
    VIDEO_FILE = "Video file"
    HEADER_LINES = "Number of header lines:"
    RECORDING_TIME = "Recording time"
    BEHAVIOR = "Behavior"
    EVENT = "Event"
    POINT_EVENT = "point event"
    STATE_START = "state start"
    STATE_STOP = "state stop"
    START = "START"
    STOP = "STOP"

    EXPECTED_FIELDS = [RECORDING_TIME, BEHAVIOR, EVENT]

    dfs = {}
    data_paths = [x for x in data_paths if "~$" not in x]
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(
            f"Reading ETHOVISION annotation file ({str(file_cnt + 1)} / {str(len(data_paths))}) ..."
        )
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            sheet_name = list(df.keys())[-1]
            df = pd.read_excel(
                file_path, sheet_name=sheet_name, index_col=0, header=None
            )
            video_path = df.loc[VIDEO_FILE].values[0]
            _, video_name, ext = get_fn_ext(video_path)
            header_n = int(df.loc[HEADER_LINES].values[0]) - 2
            df = df.iloc[header_n:].reset_index(drop=True)
            df.columns = list(df.iloc[0])
            df = df.iloc[2:].reset_index(drop=True)[EXPECTED_FIELDS]
            df.columns = ["TIME", "BEHAVIOR", "EVENT"]
            df = df[df["EVENT"] != POINT_EVENT].reset_index(drop=True)
            df["EVENT"] = df["EVENT"].replace({STATE_START: START, STATE_STOP: STOP})
            dfs[video_name] = df

        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(
                    annotation_app="ETHOVISION",
                    file_path=file_path,
                    log_status=log_setting,
                )
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(
                    msg=f"{file_path} is not a valid ETHOVISION file. See the docs for expected file format."
                )
            else:
                pass

    for video_name, video_df in dfs.items():
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        video_df["FRAME"] = (video_df["TIME"] * fps).astype(int)
        video_df.drop("TIME", axis=1, inplace=True)

    return dfs


# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_ethovision_files(data_paths=['/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/ethovision_data/correct.xlsx'],
#                                 error_setting='WARNING',
#                                  log_setting=False,
#                                  video_info_df=video_info_df)
#


def read_observer_files(
    data_paths: List[str],
    error_setting: str,
    video_info_df: pd.DataFrame,
    log_setting: bool = False,
) -> Dict[str, pd.DataFrame]:
    TIME_FIELD = "Time_Relative_hmsf"
    VIDEO_NAME_FIELD = "Observation"
    BEHAVIOR_FIELD = "Behavior"
    EVENT_TYPE_FIELD = "Event_Type"
    POINT_EVENT = "Point"
    START = "State start"
    STOP = "State stop"
    EXPECTED_FIELDS = [TIME_FIELD, VIDEO_NAME_FIELD, BEHAVIOR_FIELD, EVENT_TYPE_FIELD]

    dfs = {}
    for file_path in data_paths:
        try:
            df = pd.read_excel(
                file_path, sheet_name=None, usecols=EXPECTED_FIELDS
            ).popitem(last=False)[1]
        except KeyError:
            raise ColumnNotFoundError(
                file_name=file_path, column_name=", ".join(EXPECTED_FIELDS)
            )
        try:
            for video_name in df[VIDEO_NAME_FIELD].unique():
                video_df = df[df[VIDEO_NAME_FIELD] == video_name].reset_index(drop=True)
                video_df = video_df[video_df[EVENT_TYPE_FIELD] != POINT_EVENT]
                video_name = video_df[VIDEO_NAME_FIELD].iloc[0]
                video_df.drop(VIDEO_NAME_FIELD, axis=1, inplace=True)
                video_df[TIME_FIELD] = observer_timestamp_corrector(
                    timestamps=list(video_df[TIME_FIELD].astype(str))
                )
                video_df[TIME_FIELD] = pd.to_timedelta(video_df[TIME_FIELD])
                video_df[EVENT_TYPE_FIELD] = video_df[EVENT_TYPE_FIELD].replace(
                    {START: "START", STOP: "STOP"}
                )
                video_df.columns = ["TIME", "BEHAVIOR", "EVENT"]
                if video_name in list(dfs.keys()):
                    dfs[video_name] = pd.concat(
                        [dfs[video_name], video_df], axis=0
                    ).reset_index(drop=True)
                else:
                    dfs[video_name] = video_df

        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(
                    annotation_app="OBSERVER",
                    file_path=file_path,
                    log_status=log_setting,
                )
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(
                    msg=f"{file_path} is not a valid OBSERVER file. See the docs for expected file format."
                )
            else:
                pass

    for video_name, video_df in dfs.items():
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        video_df["FRAME"] = video_df["TIME"].dt.total_seconds() * fps
        video_df["FRAME"] = video_df["FRAME"].apply(np.floor).astype(int)
        video_df.drop("TIME", axis=1, inplace=True)

    return dfs


# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_observer_files(data_paths=['/Users/simon/Desktop/envs/troubleshooting/Gosia/source/behaviours/Exp_38/03+11WT_20171010-120856_4_no_dupl_no_audio_fps4_grey-simba_crop_frame_n.xlsx'],
#                          error_setting='WARNING',
#                          log_setting=False,
#                         video_info_df=video_info_df)


def read_solomon_files(
    data_paths: List[str],
    error_setting: str,
    video_info_df: pd.DataFrame,
    log_setting: bool = False,
) -> Dict[str, pd.DataFrame]:
    BEHAVIOR = "Behaviour"
    TIME = "Time"
    EXPECTED_COLUMNS = [TIME, BEHAVIOR]

    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, file_name, _ = get_fn_ext(file_path)
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=file_name)
        try:
            df = pd.read_csv(file_path)[EXPECTED_COLUMNS]
            df = df[~df.isnull().any(axis=1)].reset_index(drop=True)
            df["FRAME"] = df[TIME] * fps
            df["FRAME"] = df["FRAME"].apply(np.floor).astype(int)
            video_df = pd.DataFrame()
            for behavior in df[BEHAVIOR].unique():
                behavior_arr = (
                    df["FRAME"][df[BEHAVIOR] == behavior].reset_index(drop=True).values
                )
                new_arr = np.full((np.max(behavior_arr) + 2), 0)
                for i in behavior_arr:
                    new_arr[i] = 1
                bouts = detect_bouts(
                    data_df=pd.DataFrame(new_arr, columns=[behavior]),
                    target_lst=[behavior],
                    fps=1,
                )[["Event", "Start_frame", "End_frame"]].values
                results = []
                for obs in bouts:
                    results.append([obs[0], "START", obs[1]])
                    results.append([obs[0], "STOP", obs[2]])
                video_df = pd.concat(
                    [
                        video_df,
                        pd.DataFrame(
                            results, columns=["BEHAVIOR", "EVENT", "FRAME"]
                        ).sort_values(by=["FRAME"]),
                    ],
                    axis=0,
                )
            dfs[file_name] = video_df.reset_index(drop=True)

        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(
                    annotation_app="SOLOMON",
                    file_path=file_path,
                    log_status=log_setting,
                )
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(
                    msg=f"{file_path} is not a valid SOLOMON file. See the docs for expected file format."
                )
            else:
                pass

    return dfs


# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_solomon_files(data_paths=['/Users/simon/Desktop/envs/simba_dev/tests/test_data/solomon_import/solomon_import/Together_1.csv'],
#                          error_setting='WARNING',
#                          log_setting=False,
#                          video_info_df=video_info_df)




def read_deepethogram_files(
    data_paths: List[str], error_setting: str, log_setting: bool = False
) -> Dict[str, pd.DataFrame]:
    BACKGROUND = "background"
    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(file_path)
        try:
            data_df = pd.read_csv(file_path, index_col=0)
            data_df.drop(BACKGROUND, axis=1, inplace=True)
            bouts = detect_bouts(
                data_df=data_df, target_lst=list(data_df.columns), fps=1
            )[["Event", "Start_frame", "End_frame"]].values
            results = []
            for obs in bouts:
                results.append([obs[0], "START", obs[1]])
                results.append([obs[0], "STOP", obs[2]])
            dfs[video_name] = (
                pd.DataFrame(results, columns=["BEHAVIOR", "EVENT", "FRAME"])
                .sort_values(by=["FRAME"])
                .reset_index(drop=True)
            )
        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(
                    annotation_app="DEEPETHOGRAM",
                    file_path=file_path,
                    log_status=log_setting,
                )
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(
                    msg=f"{file_path} is not a valid BORIS file. See the docs for expected file format."
                )
            else:
                pass

    return dfs


def fix_uneven_start_stop_count(data: pd.DataFrame) -> pd.DataFrame:
    starts = data["FRAME"][data["EVENT"] == "START"].values
    stops = data["FRAME"][data["EVENT"] == "STOP"].values
    if starts.shape[0] < stops.shape[0]:
        sorted_stops = np.sort(stops)
        for start in starts:
            stop_idx = np.argwhere(sorted_stops > start)[0][0]
            sorted_stops = np.delete(sorted_stops, stop_idx)
        for remove_val in sorted_stops:
            remove_idx = np.argwhere(stops == remove_val)[0][0]
            stops = np.delete(stops, remove_idx)

    if stops.shape[0] < starts.shape[0]:
        sorted_starts = np.sort(starts)
        for stop in stops:
            start_idx = np.argwhere(sorted_starts < stop)[-1][0]
            sorted_starts = np.delete(sorted_starts, start_idx)
        for remove_val in sorted_starts:
            remove_idx = np.argwhere(starts == remove_val)[0][0]
            starts = np.delete(starts, remove_idx)

    return pd.DataFrame({"START": starts, "STOP": stops})


def check_stop_events_prior_to_start_events(df: pd.DataFrame) -> List[int]:
    overlaps_idx = []
    for obs_cnt, obs in enumerate(df.values):
        if obs[0] > obs[1]:
            overlaps_idx.append(obs_cnt)
    return overlaps_idx
