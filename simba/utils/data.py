__author__ = "Simon Nilsson"

import ast
import configparser
import io
import os
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from numba import jit, prange
from pylab import *
from scipy import stats
from scipy.signal import savgol_filter

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_if_module_has_import,
                                check_if_string_value_is_valid_video_timestamp,
                                check_instance, check_int, check_str,
                                check_that_column_exist,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_array, check_valid_dataframe)
from simba.utils.enums import ConfigKey, Dtypes, Keys, Options
from simba.utils.errors import (BodypartColumnNotFoundError, CountError,
                                InvalidFileTypeError, InvalidInputError,
                                NoFilesFoundError)
from simba.utils.printing import stdout_success, stdout_warning
from simba.utils.read_write import (find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_config_file, read_df,
                                    read_project_path_and_file_type,
                                    read_roi_data, write_df)


def detect_bouts(
    data_df: pd.DataFrame, target_lst: List[str], fps: int
) -> pd.DataFrame:
    """
    Detect behavior "bouts" (e.g., continous sequence of classified behavior-present frames) for specified classifiers.

    .. note::
       Can be any field of boolean type. E.g., target_lst = ['Inside_ROI_1`] also works for bouts inside ROI shape.

    :param pd.DataFrame data_df: Dataframe with fields representing classifications in boolean type.
    :param List[str] target_lst: Classifier names. E.g., ['Attack', 'Sniffing', 'Grooming'] or ROIs
    :param int fps: The fps of the input video.
    :return pd.DataFrame: Dataframe where bouts are represented by rows and fields are represented by 'Event type ', 'Start time', 'End time', 'Start frame', 'End frame', 'Bout time'

    :example:
    >>> data_df = read_df(file_path='tests/data/test_projects/two_c57/project_folder/csv/machine_results/Together_1.csv', file_type='csv')
    >>> detect_bouts(data_df=data_df, target_lst=['Attack', 'Sniffing'], fps=25)
    >>>     'Event'  'Start_time'  'End Time'  'Start_frame'  'End_frame'  'Bout_time'
    >>> 0   'Attack'    5.03          5.33          151        159            0.30
    >>> 1   'Attack'    5.87          6.23          176        186            0.37
    >>> 2  'Sniffing'   3.47          3.83          104        114            0.37
    """

    boutsList, nameList, startTimeList, endTimeList, startFrameLst, endFrameList = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for target_name in target_lst:
        groupDf = pd.DataFrame()
        v = (data_df[target_name] != data_df[target_name].shift()).cumsum()
        u = data_df.groupby(v)[target_name].agg(["all", "count"])
        m = u["all"] & u["count"].ge(1)
        groupDf["groups"] = data_df.groupby(v).apply(
            lambda x: (x.index[0], x.index[-1])
        )[m]
        for _, row in groupDf.iterrows():
            bout = list(row["groups"])
            bout_time = ((bout[-1] - bout[0]) + 1) / fps
            bout_start = (bout[0]) / fps
            bout_end = (bout[1] + 1) / fps
            bout_start_frm = bout[0] + 1
            endFrame = bout[1]
            endTimeList.append(bout_end)
            startTimeList.append(bout_start)
            boutsList.append(bout_time)
            nameList.append(target_name)
            endFrameList.append(endFrame)
            startFrameLst.append(bout_start_frm)

    startFrameLst = [x - 1 for x in startFrameLst]
    return pd.DataFrame(
        list(
            zip(
                nameList,
                startTimeList,
                endTimeList,
                startFrameLst,
                endFrameList,
                boutsList,
            )
        ),
        columns=[
            "Event",
            "Start_time",
            "End Time",
            "Start_frame",
            "End_frame",
            "Bout_time",
        ],
    )


def detect_bouts_multiclass(
    data: pd.DataFrame, target: str, fps: int = 1, classifier_map: Dict[int, str] = None
) -> pd.DataFrame:
    """
    Detect bouts in a multiclass time series dataset and return the bout event types, their start times, end times and duration.

    :param pd.DataFrame data: A Pandas DataFrame containing multiclass time series data.
    :param str target: Name of the target column in ``data``.
    :param int fps: Frames per second of the video used to collect ``data``. Default is 1.
    :param Dict[int, str] classifier_map: A dictionary mapping class labels to their names. Used to replace numeric labels with descriptive names. If None, then numeric event labels are kept.

    :example:
    >>> df = pd.DataFrame({'value': [0, 0, 0, 2, 2, 1, 1, 1, 3, 3]})
    >>> detect_bouts_multiclass(data=df, target='value', fps=3, classifier_map={0: 'None', 1: 'sharp', 2: 'track', 3: 'sync'})
    >>>    'Event'  'Start_time'  'End_time'  'Start_frame'  'End_frame'  'Bout_time'
    >>> 0   'None'    0.000000  1.000000          0.0        2.0   1.000000
    >>> 1   'sharp'   1.666667  2.666667          5.0        7.0   1.000000
    >>> 2   'track'   1.000000  1.666667          3.0        4.0   0.666667
    >>> 3   'sync '   2.666667  3.333333          8.0        9.0   0.666667
    """

    check_int(name="FPS", value=fps, min_value=1.0)
    results = pd.DataFrame(
        columns=[
            "Event",
            "Start_time",
            "End_time",
            "Start_frame",
            "End_frame",
            "Bout_time",
        ]
    )
    data["is_new_bout"] = (data[target] != data[target].shift(1)).cumsum()
    bouts = data.groupby([target, "is_new_bout"]).apply(
        lambda x: (x.index[0], x.index[-1])
    )
    for start_idx, end_idx in bouts:
        if start_idx != 0:
            start_time = start_idx / fps
        else:
            start_time = 0
        if end_idx != 0:
            end_time = (end_idx + 1) / fps
        else:
            end_time = 0
        bout_time = end_time - start_time
        event = data.at[start_idx, target]
        results.loc[len(results)] = [
            event,
            start_time,
            end_time,
            start_idx,
            end_idx,
            bout_time,
        ]

    if classifier_map:
        results["Event"] = results["Event"].map(classifier_map)

    return results


def plug_holes_shortest_bout(
    data_df: pd.DataFrame, clf_name: str, fps: int, shortest_bout: int
) -> pd.DataFrame:
    """
    Removes behavior "bouts" that are shorter than the minimum user-specified length within a dataframe.

    :param pd.DataFrame data_df: Pandas Dataframe with classifier prediction data.
    :param str clf_name: Name of the classifier field.
    :param int fps: The fps of the input video.
    :param int shortest_bout: The shortest valid behavior boat in milliseconds.
    :return pd.DataFrame data_df: Dataframe where behavior bouts with invalid lengths have been removed (< shortest_bout)

    :example:
    >>>  data_df = pd.DataFrame(data=[1, 0, 1, 1, 1], columns=['target'])
    >>>  plug_holes_shortest_bout(data_df=data_df, clf_name='target', fps=10, shortest_bout=2000)
    >>>         target
    >>>    0       1
    >>>    1       1
    >>>    2       1
    >>>    3       1
    >>>    4       1
    """

    frames_to_plug = int(int(fps) * int(shortest_bout) / 1000)
    frames_to_plug_lst = list(range(1, frames_to_plug + 1))
    frames_to_plug_lst.reverse()
    patternListofLists, negPatternListofList = [], []
    for k in frames_to_plug_lst:
        zerosInList, oneInlist = [0] * k, [1] * k
        currList = [1]
        currList.extend(zerosInList)
        currList.extend([1])
        currListNeg = [0]
        currListNeg.extend(oneInlist)
        currListNeg.extend([0])
        patternListofLists.append(currList)
        negPatternListofList.append(currListNeg)
    fill_patterns = np.asarray(patternListofLists)
    remove_patterns = np.asarray(negPatternListofList)

    for currPattern in fill_patterns:
        n_obs = len(currPattern)
        data_df["rolling_match"] = (
            data_df[clf_name]
            .rolling(window=n_obs, min_periods=n_obs)
            .apply(lambda x: (x == currPattern).all())
            .mask(lambda x: x == 0)
            .bfill(limit=n_obs - 1)
            .fillna(0)
            .astype(bool)
        )
        data_df.loc[data_df["rolling_match"] == True, clf_name] = 1
        data_df = data_df.drop(["rolling_match"], axis=1)

    for currPattern in remove_patterns:
        n_obs = len(currPattern)
        data_df["rolling_match"] = (
            data_df[clf_name]
            .rolling(window=n_obs, min_periods=n_obs)
            .apply(lambda x: (x == currPattern).all())
            .mask(lambda x: x == 0)
            .bfill(limit=n_obs - 1)
            .fillna(0)
            .astype(bool)
        )
        data_df.loc[data_df["rolling_match"] == True, clf_name] = 0
        data_df = data_df.drop(["rolling_match"], axis=1)

    return data_df


def create_color_palettes(
    no_animals: int, map_size: int, cmaps: Optional[List[str]] = None
) -> List[List[int]]:
    """
    Create list of lists of bgr colors, one for each animal. Each list is pulled from a different palette
    matplotlib color map.

    :param int no_animals: Number of different palette lists
    :param int map_size: Number of colors in each created palette.
    :return List[List[int]]:  BGR colors

    :example:
    >>> create_color_palettes(no_animals=2, map_size=2)
    >>> [[[255.0, 0.0, 255.0], [0.0, 255.0, 255.0]], [[102.0, 127.5, 0.0], [102.0, 255.0, 255.0]]]
    """
    colorListofList = []
    if cmaps is None:
        cmaps = [
            "spring",
            "summer",
            "autumn",
            "cool",
            "Wistia",
            "Pastel1",
            "Set1",
            "winter",
            "afmhot",
            "gist_heat",
            "copper",
        ]
    for colormap in range(no_animals):
        currColorMap = cm.get_cmap(cmaps[colormap], map_size)
        currColorList = []
        for i in range(currColorMap.N):
            rgb = list((currColorMap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            currColorList.append(rgb)
        colorListofList.append(currColorList)
    return colorListofList


def create_color_palette(
    pallete_name: str,
    increments: int,
    as_rgb_ratio: Optional[bool] = False,
    as_hex: Optional[bool] = False,
) -> list:
    """
    Create a list of colors in RGB from specified color palette.

    :param str pallete_name: Palette name (e.g., ``jet``)
    :param int increments: Numbers of colors in the color palette to create.
    :param Optional[bool] as_rgb_ratio: Return RGB to ratios. Default: False
    :param Optional[bool] as_hex: Return values as HEX. Default: False

    .. note::
       If **both** as_rgb_ratio and as_hex, HEX values will be returned.

    :return list: Color palette values.

    :example:
    >>> create_color_palette(pallete_name='jet', increments=3)
    >>> [[127.5, 0.0, 0.0], [255.0, 212.5, 0.0], [0.0, 229.81481481481478, 255.0], [0.0, 0.0, 127.5]]
    >>> create_color_palette(pallete_name='jet', increments=3, as_rgb_ratio=True)
    >>> [[0.5, 0.0, 0.0], [1.0, 0.8333333333333334, 0.0], [0.0, 0.0.9012345679012345, 1.0], [0.0, 0.0, 0.5]]
    >>> create_color_palette(pallete_name='jet', increments=3, as_hex=True)
    >>> ['#800000', '#ffd400', '#00e6ff', '#000080']
    """
    if as_hex:
        as_rgb_ratio = True
    cmap = cm.get_cmap(pallete_name, increments + 1)
    color_lst = []
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        if not as_rgb_ratio:
            rgb = [i * 255 for i in rgb]
        rgb.reverse()
        if as_hex:
            rgb = matplotlib.colors.to_hex(rgb)
        color_lst.append(rgb)
    return color_lst


def smooth_data_savitzky_golay(
    config: configparser.ConfigParser,
    file_path: Union[str, os.PathLike],
    time_window_parameter: int,
    overwrite: Optional[bool] = True,
) -> None:
    """
    Perform Savitzky-Golay smoothing of pose-estimation data within a file.

    .. important::
       Overwrites the input data with smoothened data.

    :param configparser.ConfigParser config: Parsed SimBA project_config.ini file.
    :param str file_path: Path to pose estimation data.
    :param int time_window_parameter: Savitzky-Golay rolling window size in milliseconds.
    :param bool overwrite: If True, overwrites the input data. If False, returns the smoothened dataframe.

    :example:
    >>> config = read_config_file(config_path='Tests_022023/project_folder/project_config.ini')
    >>> smooth_data_savitzky_golay(config=config, file_path='Tests_022023/project_folder/csv/input_csv/Together_1.csv', time_window_parameter=500)
    """

    check_int(name="Savitzky-Golay time window", value=time_window_parameter)
    check_file_exist_and_readable(file_path)
    _, filename, _ = get_fn_ext(file_path)
    project_dir, file_format = read_project_path_and_file_type(config=config)
    video_dir = os.path.join(project_dir, "videos")
    video_file_path = find_video_of_file(video_dir, filename)
    if not video_file_path:
        raise NoFilesFoundError(
            msg=f"SIMBA ERROR: Import video for {filename} to perform Savitzky-Golay smoothing",
            source=smooth_data_savitzky_golay.__name__,
        )
    video_meta_data = get_video_meta_data(video_path=video_file_path)
    pose_df = read_df(file_path=file_path, file_type=file_format, check_multiindex=True)
    idx_names = ["scorer", "bodyparts", "coords"]
    frames_in_time_window = int(time_window_parameter / (1000 / video_meta_data["fps"]))
    if (frames_in_time_window % 2) == 0:
        frames_in_time_window = frames_in_time_window - 1
    if (frames_in_time_window % 2) <= 3:
        frames_in_time_window = 5
    new_df = deepcopy(pose_df)
    new_df.columns.names = idx_names

    for c in new_df:
        new_df[c] = savgol_filter(
            x=new_df[c].to_numpy(),
            window_length=frames_in_time_window,
            polyorder=3,
            mode="nearest",
        )
        new_df[c] = new_df[c].abs()
    print(f"Savitzky-Golay smoothing complete for {filename}...")
    if overwrite:
        write_df(df=new_df, file_type=file_format, save_path=file_path)
    else:
        return new_df


def smooth_data_gaussian(
    config: configparser.ConfigParser, file_path: str, time_window_parameter: int
) -> None:
    """
    Perform Gaussian smoothing of pose-estimation data.

    .. important::
       Overwrites the input data with smoothened data.

    :param configparser.ConfigParser config: Parsed SimBA project_config.ini file.
    :param str file_path: Path to pose estimation data.
    :param int time_window_parameter: Gaussian rolling window size in milliseconds.

    Example
    ----------
    >>> config = read_config_file(ini_path='/Users/simon/Desktop/envs/troubleshooting/Tests_022023/project_folder/project_config.ini')
    >>> smooth_data_gaussian(config=config, file_path='/Users/simon/Desktop/envs/troubleshooting/Tests_022023/project_folder/csv/input_csv/Together_1.csv', time_window_parameter=500)
    """

    check_int(name="Gaussian time window", value=time_window_parameter)
    _, filename, _ = get_fn_ext(file_path)
    project_dir = config.get(
        ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value
    )
    video_dir = os.path.join(project_dir, "videos")
    video_file_path = find_video_of_file(video_dir, filename)
    file_format = read_config_entry(
        config=config,
        section=ConfigKey.GENERAL_SETTINGS.value,
        option=ConfigKey.FILE_TYPE.value,
        data_type=Dtypes.STR.value,
        default_value="csv",
    )
    video_meta_data = get_video_meta_data(video_path=video_file_path)
    pose_df = read_df(file_path=file_path, file_type=file_format, check_multiindex=True)
    idx_names = ["scorer", "bodyparts", "coords"]
    frames_in_time_window = int(time_window_parameter / (1000 / video_meta_data["fps"]))
    new_df = deepcopy(pose_df)
    new_df.columns.names = idx_names

    for c in new_df:
        new_df[c] = (
            new_df[c]
            .rolling(
                window=int(frames_in_time_window), win_type="gaussian", center=True
            )
            .mean(std=5)
            .fillna(new_df[c])
            .abs()
        )
    write_df(df=new_df, file_type=file_format, save_path=file_path)
    print(f"Gaussian smoothing complete for file {filename}...")


def add_missing_ROI_cols(shape_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing ROI definitions in ROI info dataframes created by the first version of the SimBA ROI
    user-interface but analyzed using newer versions of SimBA.

    :param pd.DataFrame shape_df: Dataframe holding ROI definitions.
    :returns DataFrame
    """

    if not "Color BGR" in shape_df.columns:
        shape_df["Color BGR"] = [(255, 255, 255)] * len(shape_df)
    if not "Thickness" in shape_df.columns:
        shape_df["Thickness"] = [5] * len(shape_df)
    if not "Color name" in shape_df.columns:
        shape_df["Color name"] = "White"

    return shape_df


def find_bins(
    data: Dict[str, List[int]],
    bracket_type: Literal["QUANTILE", "QUANTIZE"],
    bracket_cnt: int,
    normalization_method: Literal["ALL VIDEOS", "BY VIDEO"],
) -> Dict[str, np.ndarray]:
    """
    Helper to find bin cut-off points.

    :param dict data: Dictionary with video names as keys and list of values of size len(frames).
    :param Literal[str] bracket_type: 'QUANTILE' or 'QUANTIZE'
    :param str bracket_cnt: Number of bins.
    :param str normalization_method: Create bins based on data in all videos ("ALL VIDEOS") or create different bins per video ('BY VIDEO')
    :returns dict: The videos as keys and bin cut off points as array of size len(bracket_cnt) x 2.
    """

    print("Finding bracket cut off points...")
    video_bins_info = {}
    if normalization_method == "ALL VIDEOS":
        m = []
        [m.extend((d.tolist())) for d in data.values()]
        if bracket_type == "QUANTILE":
            _, bins = pd.qcut(
                x=m, q=bracket_cnt, labels=list(range(1, bracket_cnt + 1)), retbins=True
            )
        else:
            _, bins = pd.cut(
                x=m,
                bins=bracket_cnt,
                labels=list(range(1, bracket_cnt + 1)),
                retbins=True,
            )
        bins = bins.clip(min=0)
        for video_name, video_movements in data.items():
            bin_array = np.full((len(bins) - 1, 2), np.nan)
            for i in range(len(bins) - 1):
                bin_array[i] = [bins[i].astype(int), bins[i + 1].astype(int)]
            video_bins_info[video_name] = bin_array
    else:
        for video_name, video_movements in data.items():
            if bracket_type == "QUANTILE":
                _, bins = pd.qcut(
                    x=video_movements,
                    q=bracket_cnt,
                    labels=list(range(1, bracket_cnt + 1)),
                    retbins=True,
                )
            else:
                _, bins = pd.cut(
                    x=video_movements,
                    bins=bracket_cnt,
                    labels=list(range(1, bracket_cnt + 1)),
                    retbins=True,
                )
            bins = bins.clip(min=0)
            bin_array = np.full((len(bins) - 1, 2), np.nan)
            for i in range(len(bins) - 1):
                bin_array[i] = [bins[i].astype(int), bins[i + 1].astype(int)]
            video_bins_info[video_name] = bin_array

    return video_bins_info


def find_frame_numbers_from_time_stamp(
    start_time: str, end_time: str, fps: int
) -> List[int]:
    """
    Given start and end timestamps in HH:MM:SS formats and the fps, return the frame numbers representing
    the time period.

    :param str start_time: Period start time in HH:MM:SS format.
    :param str end_time: Period end time in HH:MM:SS format.
    :param int fps: Framerate of the video.
    :returns List[int]: Frame numbers within the period.

    :example:
    >>> find_frame_numbers_from_time_stamp(start_time='00:00:00', end_time='00:00:01', fps=10)
    >>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    check_if_string_value_is_valid_video_timestamp(value=start_time, name="Start time")
    check_if_string_value_is_valid_video_timestamp(value=start_time, name="End time")
    check_that_hhmmss_start_is_before_end(
        start_time=start_time, end_time=end_time, name="Time period"
    )
    start_h, start_m, start_s = start_time.split(":")
    end_h, end_m, end_s = end_time.split(":")
    start_in_s = int(start_h) * 3600 + int(start_m) * 60 + float(start_s)
    end_in_s = int(end_h) * 3600 + int(end_m) * 60 + float(end_s)
    return list(range(int(start_in_s * fps), int(end_in_s * fps) + 2))


# def slp_to_df_convert(file_path: Union[str, os.PathLike],
#                       headers: List[str],
#                       joined_tracks: Optional[bool] = False,
#                       multi_index: Optional[bool] = True) -> pd.DataFrame:
#     """
#     Helper to convert .slp pose-estimation data to pandas dataframe.
#
#     .. note::
#        Written by Toshea111 - `see jupyter notebook <https://colab.research.google.com/drive/1EpyTKFHVMCqcb9Lj9vjMrriyaG9SvrPO?usp=sharing>`__.
#
#     :param Union[str, os.PathLike] file_path: Path to .slp file on disk.
#     :param List[str] headers: List of strings representing output dataframe headers.
#     :param bool joined_tracks: If True, the .slp file has been created by joining multiple .slp files.
#     :param bool multi_index: If True, inserts multi-index place-holders in the output dataframe (used in SimBA data import).
#     :raises InvalidFileTypeError: If ``file_path`` is not a valid SLEAP H5 pose-estimation file.
#     :raises DataHeaderError: If sleap file contains more or less body-parts than suggested by len(headers)
#
#     :return pd.DataFrame: With animal ID, Track ID and body-part names as colums.
#     """
#
#     try:
#         with h5py.File(file_path, "r") as sleap_dict:
#             data = {k: v[()] for k, v in sleap_dict.items()}
#             data["node_names"] = [s.decode() for s in data["node_names"].tolist()]
#             data["point_scores"] = np.transpose(data["point_scores"][0])
#             data["track_names"] = [s.decode() for s in data["track_names"].tolist()]
#             data["tracks"] = np.transpose(data["tracks"])
#             data["track_occupancy"] = data["track_occupancy"].astype(bool)
#     except OSError as e:
#         print(e.args)
#         raise InvalidFileTypeError(msg=f'{file_path} is not a valid SLEAP H5 file', source=slp_to_df_convert.__name__)
#     valid_frame_idxs = np.argwhere(data["track_occupancy"].any(axis=1)).flatten()
#     tracks = []
#     for frame_idx in valid_frame_idxs:
#         frame_tracks = data["tracks"][frame_idx]
#         for i in range(frame_tracks.shape[-1]):
#             pts = frame_tracks[..., i]
#             if np.isnan(pts).all():
#                 continue
#             detection = {"track": data["track_names"][i], "frame_idx": frame_idx}
#             for node_name, (x, y) in zip(data["node_names"], pts):
#                 detection[f"{node_name}.x"] = x
#                 detection[f"{node_name}.y"] = y
#             tracks.append(detection)
#     if joined_tracks:
#         df = pd.DataFrame(tracks).set_index('frame_idx').groupby(level=0).sum().astype(int).reset_index(drop=True)
#     else:
#         df = pd.DataFrame(tracks).fillna(0)
#     df.columns = list(range(0, len(df.columns)))
#     p_df = pd.DataFrame(data['point_scores'], index=df.index, columns=df.columns[1::2] + .5).fillna(0).clip(0.0, 1.0)
#     df = pd.concat([df, p_df], axis=1).sort_index(axis=1)
#     if len(headers) != len(df.columns):
#         raise DataHeaderError(msg=f'The SimBA project suggest the data should have {len(headers)} columns, but the input data has {len(df.columns)} columns', source=slp_to_df_convert.__name__)
#     df.columns = headers
#     if multi_index:
#         multi_idx_cols = []
#         for col_idx in range(len(df.columns)):
#             multi_idx_cols.append(tuple(('IMPORTED_POSE', 'IMPORTED_POSE', df.columns[col_idx])))
#         df.columns = pd.MultiIndex.from_tuples(multi_idx_cols, names=('scorer', 'bodypart', 'coords'))
#     return df


def convert_roi_definitions(
    roi_definitions_path: Union[str, os.PathLike], save_dir: Union[str, os.PathLike]
) -> None:
    """
    Helper to convert SimBA `ROI_definitions.h5` file into human-readable format.

    :param Union[str, os.PathLike] roi_definitions_path: Path to SimBA `ROI_definitions.h5` on disk.
    :param Union[str, os.PathLike] save_dir: Directory location where the output data should be stored
    """

    datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    check_file_exist_and_readable(file_path=roi_definitions_path)
    check_if_dir_exists(in_dir=save_dir)
    rectangle_df, circle_df, polygon_df = read_roi_data(roi_path=roi_definitions_path)
    for df, shape_name in zip(
        [rectangle_df, circle_df, polygon_df], ["rectangles", "circles", "polygons"]
    ):
        if len(df) > 0:
            file_save_path = os.path.join(save_dir, f"{shape_name}_{datetime_str}.csv")
            df.to_csv(file_save_path)
            stdout_success(
                msg=f"SIMBA COMPLETE: {file_save_path} successfully saved!",
                source=convert_roi_definitions.__name__,
            )


def slice_roi_dict_for_video(
    data: Dict[str, pd.DataFrame], video_name: str
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Given a dictionary of dataframes representing different ROIs (created by ``simba.mixins.config_reader.ConfigReader.read_roi_data``),
    retain only the ROIs belonging to the specified video.
    """
    check_if_keys_exist_in_dict(
        data=data,
        key=[
            Keys.ROI_RECTANGLES.value,
            Keys.ROI_CIRCLES.value,
            Keys.ROI_POLYGONS.value,
        ],
        name=slice_roi_dict_for_video.__name__,
    )
    new_data, shape_names = {}, []
    for k, v in data.items():
        check_instance(
            source=f"{slice_roi_dict_for_video.__name__} {k}",
            instance=v,
            accepted_types=(pd.DataFrame,),
        )
        check_that_column_exist(
            df=v, column_name="Video", file_name=slice_roi_dict_for_video.__name__
        )
        check_that_column_exist(
            df=v, column_name="Name", file_name=slice_roi_dict_for_video.__name__
        )
        v = v[v["Video"] == video_name]
        new_data[k] = v.reset_index(drop=True)
        shape_names.extend((list(v["Name"].unique())))
    return new_data, shape_names


def freedman_diaconis(data: np.array) -> (float, int):
    """
    Use Freedman-Diaconis rule to compute optimal count of histogram bins and their width.

    .. note::
       Can also use ``simba.utils.data.bucket_data`` passing method ``fd``.

    :references:
       .. [2] `Reference freedman_diaconis <http://www.jtrive.com/determining-histogram-bin-width-using-the-freedman-diaconis-rule.html>`_.

    """

    IQR = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    bin_width = (2 * IQR) / np.power(data.shape[0], 1 / 3)
    bin_count = int((np.max(data) - np.min(data) / bin_width) + 1)
    return bin_width, bin_count


@jit(nopython=True)
def hist_1d(data: np.ndarray, bins: int, range: np.ndarray):
    return np.histogram(data, bins, (range[0], range[1]))[0]


def bucket_data(
    data: np.ndarray,
    method: Literal[
        "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
    ] = "auto",
) -> Tuple[float, int]:
    """
    Computes the optimal bin count and bin width non-heuristically using specified method.

    :param np.ndarray data: 1D array of numerical data.
    :param np.ndarray data: The method to compute optimal bin count and bin width. These methods differ in how they estimate the optimal bin count and width. Defaults to 'auto', which represents the maximum of the Sturges and Freedman-Diaconis estimators. Available methods are 'fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'.
    :returns Tuple[float, int]: A tuple containing the optimal bin width and bin count.

    :example:
    >>> data = np.random.randint(low=1, high=1000, size=(1, 100))
    >>> bucket_data(data=data, method='fd')
    >>> (190.8, 6)
    >>> bucket_data(data=data, method='doane')
    >>> (106.0, 10)
    """

    check_valid_array(data=data, source=bucket_data.__name__, accepted_ndims=(1,))
    check_str(
        name=f"{bucket_data.__name__} method",
        value=method,
        options=Options.BUCKET_METHODS.value,
    )
    bin_edges = np.histogram_bin_edges(a=data, bins=method)
    bin_counts = bin_edges.shape[0]
    bin_width = bin_edges[1] - bin_edges[0]

    return bin_width, bin_counts


@jit(nopython=True)
def fast_minimum_rank(data: np.ndarray, descending: bool = True):
    """
    Jitted helper to rank values in 1D array using ``minimum`` method.

    :param np.ndarray data: 1D array of feature values.
    :param bool descending: If True, ranks returned where low values get a high rank. If False, low values get a low rank. Default: True.

    :references:
        `Jérôme Richard on StackOverflow <https://stackoverflow.com/a/69869255>`__.

    :example:
    >>> data = np.array([1, 1, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> fast_minimum_rank(data=data, descending=True)
    >>> [9, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> fast_minimum_rank(data=data, descending=False)
    >>> [ 1,  1,  3,  4,  5,  6,  7,  8,  9, 10]
    """

    result = np.empty((data.shape[0]), dtype=np.int32)
    if descending:
        order = np.argsort(-data)
    else:
        order = np.argsort(data)
    previous_value, previous_rank, result[order[0]] = data[order[0]], 1, 1
    for idx in prange(1, data.shape[0]):
        current_value = data[order[idx]]
        if current_value == previous_value:
            result[order[idx]] = previous_rank
        else:
            result[order[idx]] = idx + 1
            previous_value = current_value
            previous_rank = idx + 1
    return result


@jit(nopython=True)
def fast_mean_rank(data: np.ndarray, descending: bool = True):
    """
    Jitted helper to rank values in 1D array using ``mean`` method.

    :param np.ndarray data: 1D array of feature values.
    :param bool descending: If True, ranks returned where low values get a high rank. If False, low values get a low rank. Default: True.

    :references:
        `Modified from James Webber gist on GitHub <https://gist.github.com/jamestwebber/38ab26d281f97feb8196b3d93edeeb7b>`__.

    :example:
    >>> data = np.array([1, 1, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> fast_mean_rank(data=data, descending=True)
    >>> [9.5, 9.5, 8. , 7. , 6. , 5. , 4. , 3. , 2. , 1. ]
    """

    if descending:
        sorter = np.argsort(-data)
    else:
        sorter = np.argsort(data)
    data = data[sorter]
    obs = np.concatenate((np.array([True]), data[1:] != data[:-1]))
    dense = np.empty(obs.size, dtype=np.int64)
    dense[sorter] = obs.cumsum()
    count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))
    results = 0.5 * (count[dense] + count[dense - 1] + 1)
    return results


def slp_to_df_convert(
    file_path: Union[str, os.PathLike],
    headers: List[str],
    joined_tracks: Optional[bool] = False,
    multi_index: Optional[bool] = True,
    drop_body_parts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Helper to convert .slp pose-estimation data in h5 format to pandas dataframe.

    :param Union[str, os.PathLike] file_path: Path to SLEAP H5 file on disk.
    :param List[str] headers: List of strings representing output dataframe headers.
    :param bool joined_tracks: If True, the h5 file has been created by joining multiple .slp files.
    :param bool multi_index: If True, inserts multi-index place-holders in the output dataframe (used in SimBA data import).
    :param Optional[List[str]] drop_body_parts: Body-parts that should be removed from the SLEAP H5 dataset before import into SimBA. Use the body-part names as defined in SLEAP. Default: None.
    :raises InvalidFileTypeError: If ``file_path`` is not a valid SLEAP H5 pose-estimation file.
    :raises DataHeaderError: If sleap file contains more or less body-parts than suggested by len(headers)

    :return pd.DataFrame: With animal ID, Track ID and body-part names as columns.

    :example:
    >>> headers = ['d_nose_1', 'd_neck_1', 'd_back_1', 'd_tail_1', 'nest_s_2', 'nest_cc_2', 'nest_cv_2', 'nest_cc_2', 'nest_csc_2', 'nest_cscd_2']
    >>> new_headers = []
    >>> for h in headers: new_headers.append(h + '_x'); new_headers.append(h + '_y'); new_headers.append(h + '_p')
    >>> df = slp_to_df_convert(file_path='/Users/simon/Desktop/envs/troubleshooting/ryan/LBN4a_Ctrl_P05_1_2022-01-15_08-16-20c.h5', headers=new_headers, joined_tracks=True)
    """

    video_name = get_fn_ext(filepath=file_path)[1]
    print(f"Importing {video_name}...")
    with h5py.File(file_path, "r") as f:
        missing_keys = [
            x
            for x in ["tracks", "point_scores", "node_names", "track_names"]
            if not x in list(f.keys())
        ]
        if missing_keys:
            raise InvalidFileTypeError(
                msg=f"{file_path} is not a valid SLEAP H5 file. Missing keys: {missing_keys}",
                source=slp_to_df_convert.__name__,
            )
        tracks = f["tracks"][:].T
        point_scores = f["point_scores"][:].T
        node_names = [n.decode() for n in f["node_names"][:].tolist()]
        track_names = [n.decode() for n in f["track_names"][:].tolist()]

    csv_rows = []
    n_frames, n_nodes, _, n_tracks = tracks.shape

    for frame_ind in range(n_frames):
        csv_row = []
        for track_ind in range(n_tracks):
            for node_ind in range(n_nodes):
                for xyp in range(3):
                    if xyp == 0 or xyp == 1:
                        data = tracks[frame_ind, node_ind, xyp, track_ind]
                    else:
                        data = point_scores[frame_ind, node_ind, track_ind]

                    csv_row.append(f"{data:.3f}")
        csv_rows.append(" ".join(csv_row))

    sleap_header = []
    sleap_header_unique = []
    for track_ind in range(n_tracks):
        for node_ind in range(n_nodes):
            sleap_header_unique.append(f"{node_names[node_ind]}_{track_ind + 1}")
            for suffix in ["x", "y", "p"]:
                sleap_header.append(f"{node_names[node_ind]}_{track_ind + 1}_{suffix}")

    csv_rows = "\n".join(csv_rows)
    data_df = pd.read_csv(
        io.StringIO(csv_rows), delim_whitespace=True, header=None
    ).fillna(0)
    if len(data_df.columns) != len(sleap_header):
        raise BodypartColumnNotFoundError(
            msg=f"The number of body-parts in data file {file_path} do not match the number of body-parts in your SimBA project. "
            f"The number of of body-parts expected by your SimBA project is {int(len(sleap_header) / 3)}. "
            f"The number of of body-parts contained in data file {file_path} is {int(len(sleap_header) / 3)}. "
            f"Make sure you have specified the correct number of animals and body-parts in your project.",
            source=slp_to_df_convert.__name__,
        )

    if len(drop_body_parts) > 0:
        data_df.columns = sleap_header
        headers_to_drop = []
        for h in drop_body_parts:
            headers_to_drop.append(h + "_x")
            headers_to_drop.append(h + "_y")
            headers_to_drop.append(h + "_p")
        missing_headers = list(set(headers_to_drop) - set(list(data_df.columns)))
        if len(missing_headers) > 0:
            raise InvalidInputError(
                msg=f"Some of the body-part data that you specified to remove does not exist (e.g., {missing_headers[0][:-2]}) in the dataset: {sleap_header_unique}"
            )
        data_df = data_df.drop(headers_to_drop, axis=1)

    data_df.columns = headers
    if multi_index:
        multi_idx_cols = []
        for col_idx in range(len(data_df.columns)):
            multi_idx_cols.append(
                tuple(("IMPORTED_POSE", "IMPORTED_POSE", data_df.columns[col_idx]))
            )
        data_df.columns = pd.MultiIndex.from_tuples(
            multi_idx_cols, names=("scorer", "bodypart", "coords")
        )

    return data_df


def find_ranked_colors(
    data: Dict[str, float], palette: str, as_hex: Optional[bool] = False
) -> Dict[str, Union[Tuple[int], str]]:
    """
    Find ranked colors for a given data dictionary values based on a specified color palette.

    The key with the highest value in the data dictionary is assigned the most intense palette color, while
    the key with the lowest value in the data dictionary is assigned the least intense palette color.

    :param data: A dictionary where keys are labels and values are numerical scores.
    :param palette: A string representing the name of the color palette to use (e.g., 'magma').
    :param as_hex: If True, return colors in hexadecimal format; if False, return as RGB tuples. Default is False.
    :return: A dictionary where keys are labels and values are corresponding colors based on ranking.

    :examples:
    >>> data = {'Animal_1': 0.34786870380536705, 'Animal_2': 0.4307923198152757, 'Animal_3': 0.221338976379357}
    >>> find_ranked_colors(data=data, palette='magma', as_hex=True)
    >>> {'Animal_2': '#040000', 'Animal_1': '#7937b7', 'Animal_3': '#bffdfc'}
    """

    if palette not in Options.PALETTE_OPTIONS.value:
        raise InvalidInputError(
            msg=f"{palette} is not a valid palette. Options {Options.PALETTE_OPTIONS.value}",
            source=find_ranked_colors.__name__,
        )
    check_instance(
        source=find_ranked_colors.__name__, instance=data, accepted_types=dict
    )
    for k, v in data.items():
        check_str(name=k, value=k)
        check_float(name=v, value=v)
    clrs = create_color_palette(
        pallete_name=palette, increments=len(list(data.keys())) - 1, as_hex=as_hex
    )
    ranks, results = deepcopy(data), {}
    ranks = {
        key: rank
        for rank, key in enumerate(sorted(ranks, key=ranks.get, reverse=True), 1)
    }
    for k, v in ranks.items():
        results[k] = clrs[int(v) - 1]

    return results


def sample_df_n_by_unique(df: pd.DataFrame, field: str, n: int) -> pd.DataFrame:
    """
    Randomly sample at most N rows per unique value in specified field of a dataframe.

    For example, sample 100 observation from each inferred cluster assignment.

    :param pd.DataFramedf: The dataframe to sample from.
    :param str field: The column name in the DataFrame to use for sampling based on unique values.
    :param int n: The maximum number of rows to sample for each unique value in the specified column.
    :return pd.DataFrame: A dataframe containing randomly sampled rows.
    """

    check_instance(
        source=sample_df_n_by_unique.__name__,
        instance=df,
        accepted_types=(pd.DataFrame,),
    )
    check_str(
        name=f"{sample_df_n_by_unique.__name__} field",
        value=field,
        options=tuple(df.columns),
    )
    check_int(name=f"{sample_df_n_by_unique.__name__} n", value=n, min_value=1)
    check_that_column_exist(df=df, column_name=field, file_name="")
    unique_vals = df[field].unique()
    results = []
    for unique_val in unique_vals:
        sample = df[df[field] == unique_val]
        if (len(sample) <= n) or (n > len(sample)):
            results.append(sample)
        else:
            results.append(sample.sample(n=n, replace=False))
    return pd.concat(results, axis=0)


def get_mode(x: np.ndarray) -> Union[float, int]:
    """Get the mode (most frequent value) within an array"""
    check_valid_array(
        source=f"{get_mode.__name__} x",
        data=x,
        accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8),
    )
    values, counts = np.unique(x, return_counts=True)
    return counts.argmax()


def run_user_defined_feature_extraction_class(
    file_path: Union[str, os.PathLike], config_path: Union[str, os.PathLike]
) -> None:
    """
    Loads and executes user-defined feature extraction class within .py file.

    :param file_path: Path to .py file holding user-defined feature extraction class.
    :param str config_path: Path to SimBA project config file.


    .. warning::

       Legacy function. The GUI since 12/23 uses ``simba.utils.custom_feature_extractor.UserDefinedFeatureExtractor``.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/extractFeatures.md>`_.

       If the ``file_path`` contains multiple classes, then the first class will be used.

       The user defined class needs to contain a ``config_path`` init argument.

       If the feature extraction class contains a ``if __name__ == "__main__":`` entry point and uses argparse,
       then the custom feature extraction module will be executed through python subprocess.

       Else, will be executed using ``sys``.

       I recommend using the ``if __name__ == "__main__:`` and subprocess alternative, as the feature extraction clas will
       be executed in a different thread and any multicore parallel processes within the user feature extraction class will not be
       throttled by the graphical interface mainloop.

    :example:
    >>> run_user_defined_feature_extraction_class(config_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/project_config.ini', file_path='/Users/simon/Desktop/fish_feature_extractor_2023_version_5.py')
    >>> run_user_defined_feature_extraction_class(config_path='/Users/simon/Desktop/envs/troubleshooting/piotr/project_folder/train-20231108-sh9-frames-with-p-lt-2_plus3-&3_best-f1.ini', file_path='/simba/misc/piotr.py')
    """

    check_file_exist_and_readable(file_path=file_path)
    file_dir, file_name, file_extension = get_fn_ext(filepath=file_path)
    if file_extension != ".py":
        raise InvalidFileTypeError(
            msg=f"The user-defined feature extraction file ({file_path}) is not a .py file-extension",
            source=run_user_defined_feature_extraction_class.__name__,
        )
    parsed = ast.parse(Path(file_path).read_text())
    classes = [n for n in parsed.body if isinstance(n, ast.ClassDef)]
    class_name = [x.name for x in classes]
    if len(class_name) < 1:
        raise CountError(
            msg=f"The user-defined feature extraction file ({file_path}) contains no python classes",
            source=run_user_defined_feature_extraction_class.__name__,
        )
    if len(class_name) > 1:
        stdout_warning(
            msg=f"The user-defined feature extraction file ({file_path}) contains more than 1 python class. SimBA will use the first python class: {class_name[0]}."
        )
    class_name = class_name[0]
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    user_module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = user_module
    spec.loader.exec_module(user_module)
    user_class = getattr(user_module, class_name)
    if "config_path" not in inspect.signature(user_class).parameters:
        raise InvalidFileTypeError(
            msg=f"The user-defined class {class_name} does not contain a {config_path} init argument",
            source=run_user_defined_feature_extraction_class.__name__,
        )
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef)]
    function_names = [x.name for x in functions]
    has_argparse = check_if_module_has_import(
        parsed_file=parsed, import_name="argparse"
    )
    has_main = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.comparators[0], ast.Str)
        and node.test.comparators[0].s == "__main__"
        for node in parsed.body
    )

    if "main" in function_names and has_main and has_argparse:
        command = f'python "{file_path}" --config_path "{config_path}"'
        subprocess.call(command, shell=True)

    else:
        user_class(config_path)


# run_user_defined_feature_extraction_class(config_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/project_config.ini', file_path='/Users/simon/Desktop/fish_feature_extractor_2023_version_5.py')


# user_class(config_path=config_path)

# run_user_defined_feature_extraction_class(config_path='/Users/simon/Desktop/envs/troubleshooting/piotr/project_folder/train-20231108-sh9-frames-with-p-lt-2_plus3-&3_best-f1.ini',
#                                           file_path='/Users/simon/Desktop/envs/simba_dev/simba/feature_extractors/misc/piotr.py')

# data = {'Animal_1': 0.34786870380536705, 'Animal_2': 0.4307923198152757, 'Animal_3': 0.221338976379357}
# find_ranked_colors(data=data, palette='magma', as_hex=True)
# run_user_defined_feature_extraction_class(config_path='/Users/simon/Desktop/envs/troubleshooting/circular_features_zebrafish/project_folder/project_config.ini',
#                                           file_path='/Users/simon/Desktop/fish_feature_extractor_2023_version_5.py')
