__author__ = "Simon Nilsson"

import ast
import configparser
import glob
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

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

from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.errors import (CountError, DataHeaderError,
                                InvalidFileTypeError, NoFilesFoundError)
from simba.utils.lookups import get_bp_config_code_class_pairs
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
        patternListofLists.append(np.array(currList))
        negPatternListofList.append(np.array(currListNeg))
    fill_patterns = patternListofLists  # np.asarray(patternListofLists)
    remove_patterns = negPatternListofList  # np.asarray(negPatternListofList)

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


def create_color_palettes(no_animals: int, map_size: int) -> List[List[int]]:
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
            msg=f"SIMBA ERROR: Import video for {filename} to perform Savitzky-Golay smoothing"
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
    start_in_s = int(start_h) * 3600 + int(start_m) * 60 + int(start_s)
    end_in_s = int(end_h) * 3600 + int(end_m) * 60 + int(end_s)
    return list(range(int(start_in_s * fps), int(end_in_s * fps)))


def run_user_defined_feature_extraction_class(
    file_path: Union[str, os.PathLike], config_path: Union[str, os.PathLike]
) -> None:
    """
    Loads and executes user-defined feature extraction class.

    :param file_path: Path to .py file holding user-defined feature extraction class
    :param str config_path: Path to SimBA project config file.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/extractFeatures.md>`_.
       If the ``file_path`` contains multiple classes, then the first class will be used.

    """

    check_file_exist_and_readable(file_path=file_path)
    file_dir, file_name, file_extension = get_fn_ext(filepath=file_path)
    if file_extension != ".py":
        raise InvalidFileTypeError(
            msg=f"The user-defined feature extraction file ({file_path}) is not a .py file-extension"
        )
    parsed = ast.parse(Path(file_path).read_text())
    classes = [n for n in parsed.body if isinstance(n, ast.ClassDef)]
    class_name = [x.name for x in classes]
    if len(class_name) < 1:
        raise CountError(
            msg=f"The user-defined feature extraction file ({file_path}) contains no python classes"
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
    print(f"Running user-defined {class_name} feature extraction file...")
    user_class(config_path=config_path).run()


def slp_to_df_convert(
    file_path: Union[str, os.PathLike],
    headers: List[str],
    joined_tracks: Optional[bool] = False,
    multi_index: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Helper to convert .slp pose-estimation data to pandas dataframe.

    .. note::
       Written by Toshea111 - `see jupyter notebook <https://colab.research.google.com/drive/1EpyTKFHVMCqcb9Lj9vjMrriyaG9SvrPO?usp=sharing>`__.

    :param Union[str, os.PathLike] file_path: Path to .slp file on disk.
    :param List[str] headers: List of strings representing output dataframe headers.
    :param bool joined_tracks: If True, the .slp file has been created by joining multiple .slp files.
    :param bool multi_index: If True, inserts multi-index place-holders in the output dataframe (used in SimBA data import).
    :raises InvalidFileTypeError: If ``file_path`` is not a valid SLEAP H5 pose-estimation file.
    :raises DataHeaderError: If sleap file contains more or less body-parts than suggested by len(headers)

    :return pd.DataFrame: With animal ID, Track ID and body-part names as colums.
    """

    try:
        with h5py.File(file_path, "r") as sleap_dict:
            data = {k: v[()] for k, v in sleap_dict.items()}
            data["node_names"] = [s.decode() for s in data["node_names"].tolist()]
            data["point_scores"] = np.transpose(data["point_scores"][0])
            data["track_names"] = [s.decode() for s in data["track_names"].tolist()]
            data["tracks"] = np.transpose(data["tracks"])
            data["track_occupancy"] = data["track_occupancy"].astype(bool)
    except OSError as e:
        print(e.args)
        raise InvalidFileTypeError(msg=f"{file_path} is not a valid SLEAP H5 file")
    valid_frame_idxs = np.argwhere(data["track_occupancy"].any(axis=1)).flatten()
    tracks = []
    for frame_idx in valid_frame_idxs:
        frame_tracks = data["tracks"][frame_idx]
        for i in range(frame_tracks.shape[-1]):
            pts = frame_tracks[..., i]
            if np.isnan(pts).all():
                continue
            detection = {"track": data["track_names"][i], "frame_idx": frame_idx}
            for node_name, (x, y) in zip(data["node_names"], pts):
                detection[f"{node_name}.x"] = x
                detection[f"{node_name}.y"] = y
            tracks.append(detection)
    if joined_tracks:
        df = (
            pd.DataFrame(tracks)
            .set_index("frame_idx")
            .groupby(level=0)
            .sum()
            .astype(int)
            .reset_index(drop=True)
        )
    else:
        df = pd.DataFrame(tracks).fillna(0)
    df.columns = list(range(0, len(df.columns)))
    p_df = (
        pd.DataFrame(
            data["point_scores"], index=df.index, columns=df.columns[1::2] + 0.5
        )
        .fillna(0)
        .clip(0.0, 1.0)
    )
    df = pd.concat([df, p_df], axis=1).sort_index(axis=1)
    if len(headers) != len(df.columns):
        raise DataHeaderError(
            msg=f"The SimBA project suggest the data should have {len(headers)} columns, but the input data has {len(df.columns)} columns"
        )
    df.columns = headers
    if multi_index:
        multi_idx_cols = []
        for col_idx in range(len(df.columns)):
            multi_idx_cols.append(
                tuple(("IMPORTED_POSE", "IMPORTED_POSE", df.columns[col_idx]))
            )
        df.columns = pd.MultiIndex.from_tuples(
            multi_idx_cols, names=("scorer", "bodypart", "coords")
        )
    return df


def convert_roi_definitions(
    roi_definitions_path: Union[str, os.PathLike], save_dir: Union[str, os.PathLike]
) -> None:
    """
    Helper to convert SimBA `ROI_definitions.h5` file into human-readable format.

    :param Union[str, os.PathLike] roi_definitions_path: Path to SimBA `ROI_definitions.h5` on disk.
    :param Union[str, os.PathLike] save_dir: Directory location where the output data should be stored
    """

    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    check_file_exist_and_readable(file_path=roi_definitions_path)
    check_if_dir_exists(in_dir=save_dir)
    rectangle_df, circle_df, polygon_df = read_roi_data(roi_path=roi_definitions_path)
    for df, shape_name in zip(
        [rectangle_df, circle_df, polygon_df], ["rectangles", "circles", "polygons"]
    ):
        if len(df) > 0:
            file_save_path = os.path.join(save_dir, f"{shape_name}_{datetime_str}.csv")
            df.to_csv(file_save_path)
            stdout_success(msg=f"SIMBA COMPLETE: {file_save_path} successfully saved!")


def freedman_diaconis(data: np.array) -> (float, int):
    """
    Use Freedman-Diaconis rule to compute optimal count of histogram bins and their width.

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
    data: np.array,
    method: Literal[
        "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
    ] = "auto",
):
    """
    Use Freedman-Diaconis, Doane, Acott, Stone, Rice, Sturges or sqrt to compute the optimal bin count.
    'auto' represents the maximum of the Sturges and Freedman-Diaconis estimators.

    :example:
    >>> data = np.random.randint(low=1, high=1000, size=(1, 100))
    >>> bucket_data(data=data, method='fd')
    >>> (190.8, 6)
    >>> bucket_data(data=data, method='doane')
    >>> (106.0, 10)
    """

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
    >>> fast_rank(data=data, descending=True)
    >>> [9, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> fast_rank(data=data, descending=False)
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


# data = np.array([1, 1, 3, 4, 5, 6, 7, 8, 9, 10])
# fast_mean_rank(data=data, descending=True)
