__author__ = "Simon Nilsson"

import pandas as pd
from copy import deepcopy
import configparser
from pathlib import Path
import ast, os
from scipy.signal import savgol_filter
from pylab import *
from simba.utils.read_write import (get_fn_ext,
                                    read_project_path_and_file_type,
                                    find_video_of_file,
                                    read_df,
                                    write_df,
                                    read_config_entry,
                                    get_video_meta_data)
from simba.utils.checks import check_file_exist_and_readable, check_int
from simba.utils.errors import NoFilesFoundError,InvalidFileTypeError, CountError
from simba.utils.printing import stdout_warning
from simba.utils.enums import ConfigKey, Dtypes




def detect_bouts(data_df: pd.DataFrame,
                 target_lst: list,
                 fps: int):
    """
    Helper to detect all behavior bouts for a list of classifiers, or e.g., bouts of being within ROIs.

    Parameters
    ----------
    data_df: pd.DataFrame
        Pandas Dataframe with classifier prediction data.
    target_lst: list
        Classifier names. E.g., ['Attack', 'Sniffing', 'Grooming'] or ROIs
    fps: int
        The fps of the input video.

    Returns
    -------
    pd.DataFrame:
        Holding the start time, end time, end frame, bout time etc. of each classified bout.
    """

    boutsList, nameList, startTimeList, endTimeList, startFrameLst, endFrameList = [], [], [], [], [], []
    for target_name in target_lst:
        groupDf = pd.DataFrame()
        v = (data_df[target_name] != data_df[target_name].shift()).cumsum()
        u = data_df.groupby(v)[target_name].agg(['all', 'count'])
        m = u['all'] & u['count'].ge(1)
        groupDf['groups'] = data_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
        for _, row in groupDf.iterrows():
            bout = list(row['groups'])
            bout_time = ((bout[-1] - bout[0]) + 1) / fps
            bout_start = (bout[0] + 1) / fps
            bout_end = (bout[1]) / fps
            bout_start_frm = bout[0] + 1
            endFrame = (bout[1])
            endTimeList.append(bout_end)
            startTimeList.append(bout_start)
            boutsList.append(bout_time)
            nameList.append(target_name)
            endFrameList.append(endFrame)
            startFrameLst.append(bout_start_frm)

    startFrameLst = [x-1 for x in startFrameLst]
    return pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, startFrameLst, endFrameList, boutsList)),
                        columns=['Event', 'Start_time', 'End Time', 'Start_frame', 'End_frame', 'Bout_time'])


def plug_holes_shortest_bout(data_df: pd.DataFrame,
                             clf_name: str,
                             fps: int,
                             shortest_bout: int):
    """

    Parameters
    ----------
    data_df: pd.DataFrame
        Pandas Dataframe with classifier prediction data.
    clf_name: str
        Name of the classifier
    fps: int
        The fps of the input video
    shortest_bout: int
        The shortest valid behavior boat in milliseconds

    Returns
    -------
    data_df: pd.DataFrame
        Dataframe where behavior bouts with invalid lengths have been removed (< shortest_bout).
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
        data_df['rolling_match'] = (data_df[clf_name].rolling(window=n_obs, min_periods=n_obs)
                                    .apply(lambda x: (x == currPattern).all())
                                    .mask(lambda x: x == 0)
                                    .bfill(limit=n_obs - 1)
                                    .fillna(0)
                                    .astype(bool)
                                    )
        data_df.loc[data_df['rolling_match'] == True, clf_name] = 1
        data_df = data_df.drop(['rolling_match'], axis=1)

    for currPattern in remove_patterns:
        n_obs = len(currPattern)
        data_df['rolling_match'] = (data_df[clf_name].rolling(window=n_obs, min_periods=n_obs)
                                    .apply(lambda x: (x == currPattern).all())
                                    .mask(lambda x: x == 0)
                                    .bfill(limit=n_obs - 1)
                                    .fillna(0)
                                    .astype(bool)
                                    )
        data_df.loc[data_df['rolling_match'] == True, clf_name] = 0
        data_df = data_df.drop(['rolling_match'], axis=1)

    return data_df


def create_color_palettes(no_animals: int,
                          map_size: int):
    """
    Helper to return a list of lists of bgr colors. Each list is pulled from a different palette
    matplotlib color map.

    Parameters
    ----------
    no_animals: int
        Number of different palette lists
    map_size: int
        Number of colors in each created palette.

    Returns
     ----------
     colorListofList: list
        Bgr colors

    Notes
    -----

    Examples
    -----
    >>> colorListofList = createColorListofList(no_animals=2, map_size=8)
    """

    colorListofList = []
    cmaps = ['spring',
            'summer',
            'autumn',
            'cool',
            'Wistia',
            'Pastel1',
            'Set1',
            'winter',
            'afmhot',
            'gist_heat',
            'copper']
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



def create_color_palette(pallete_name: str,
                         increments: int,
                         as_rgb_ratio: bool = False,
                         as_hex: bool = False):
    """
    Helper to create a color palette of bgr colors in a list.

    Parameters
    ----------
    pallete_name: str
        Palette name (e.g., 'jet')
    increments: int
        Numbers of colors in the color palette to create.

    Returns
    -------
    color_lst: list

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



def smooth_data_savitzky_golay(config: configparser.ConfigParser,
                               file_path: str,
                               time_window_parameter=int):
    """
    Helper to perform Savitzky-Golay smoothing of pose-estimation data. Overwrites the input data with smoothened data.

    Parameters
    ----------
    config: configparser.ConfigParser
        Parsed SimBA project_config.ini file
    file_path: str
        Path to pose estimation data.
    time_window_parameter: int
        Savitzky-Golay rolling window size in milliseconds.

    Returns
    -------
    None

    Example
    ----------

    >>> config = read_config_file(ini_path='/Users/simon/Desktop/envs/troubleshooting/Tests_022023/project_folder/project_config.ini')
    >>> smooth_data_savitzky_golay(config=config, file_path='/Users/simon/Desktop/envs/troubleshooting/Tests_022023/project_folder/csv/input_csv/Together_1.csv', time_window_parameter=500)
    """

    check_int(name='Savitzky-Golay time window', value=time_window_parameter)
    check_file_exist_and_readable(file_path)
    _, filename, _ = get_fn_ext(file_path)
    project_dir, file_format = read_project_path_and_file_type(config=config)
    video_dir = os.path.join(project_dir, 'videos')
    video_file_path = find_video_of_file(video_dir, filename)
    if not video_file_path:
        raise NoFilesFoundError(msg=f'SIMBA ERROR: Import video for {filename} to perform Savitzky-Golay smoothing')
    video_meta_data = get_video_meta_data(video_path=video_file_path)
    pose_df = read_df(file_path=file_path, file_type=file_format, check_multiindex=True)
    idx_names = ['scorer', 'bodyparts', 'coords']
    frames_in_time_window = int(time_window_parameter / (1000 / video_meta_data['fps']))
    if (frames_in_time_window % 2) == 0:
        frames_in_time_window = frames_in_time_window - 1
    if (frames_in_time_window % 2) <= 3:
        frames_in_time_window = 5
    new_df = deepcopy(pose_df)
    new_df.columns.names = idx_names

    for c in new_df:
        new_df[c] = savgol_filter(x=new_df[c].to_numpy(), window_length=frames_in_time_window, polyorder=3, mode='nearest')
        new_df[c] = new_df[c].abs()
    write_df(df=new_df, file_type=file_format, save_path=file_path)
    print(f'Savitzky-Golay smoothing complete for {filename}...')


def smooth_data_gaussian(config: configparser.ConfigParser,
                         file_path: str,
                         time_window_parameter: int):
    """
    Helper to perform Gaussian smoothing of pose-estimation data. Overwrites the input data with smoothened data.

    Parameters
    ----------
    config: configparser.ConfigParser
        Parsed SimBA project_config.ini file
    file_path: str
        Path to pose estimation data.
    time_window_parameter: int
        Gaussian rolling window size in milliseconds.

    Returns
    -------
    None


    Example
    ----------
    >>> config = read_config_file(ini_path='/Users/simon/Desktop/envs/troubleshooting/Tests_022023/project_folder/project_config.ini')
    >>> smooth_data_gaussian(config=config, file_path='/Users/simon/Desktop/envs/troubleshooting/Tests_022023/project_folder/csv/input_csv/Together_1.csv', time_window_parameter=500)
    """

    check_int(name='Gaussian time window', value=time_window_parameter)
    _, filename, _ = get_fn_ext(file_path)
    project_dir = config.get(ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value)
    video_dir = os.path.join(project_dir, 'videos')
    video_file_path = find_video_of_file(video_dir, filename)
    file_format = read_config_entry(config=config, section=ConfigKey.GENERAL_SETTINGS.value, option=ConfigKey.FILE_TYPE.value, data_type=Dtypes.STR.value, default_value='csv')
    video_meta_data = get_video_meta_data(video_path=video_file_path)
    pose_df = read_df(file_path=file_path, file_type=file_format, check_multiindex=True)
    idx_names = ['scorer', 'bodyparts', 'coords']
    frames_in_time_window = int(time_window_parameter / (1000 / video_meta_data['fps']))
    new_df = deepcopy(pose_df)
    new_df.columns.names = idx_names

    for c in new_df:
        new_df[c] = new_df[c].rolling(window=int(frames_in_time_window), win_type='gaussian', center=True).mean(std=5).fillna(new_df[c]).abs()
    write_df(df=new_df, file_type=file_format, save_path=file_path)
    print(f'Gaussian smoothing complete for file {filename}...')


def add_missing_ROI_cols(shape_df: pd.DataFrame):
    """
    Helper to add missing ROI definitions in ROI info dataframes created by the first version of the SimBA ROI
    user-interface but analyzed using newer versions of SimBA.

    Parameters
    ----------
    shape_df: pd.DataFrame
        Dataframe holding ROI definitions.

    Returns
    -------
    pd.DataFrame
    """

    if not 'Color BGR' in shape_df.columns:
        shape_df['Color BGR'] = [(255, 255, 255)] * len(shape_df)
    if not 'Thickness' in shape_df.columns:
        shape_df['Thickness'] = [5] * len(shape_df)
    if not 'Color name' in shape_df.columns:
        shape_df['Color name'] = 'White'

    return shape_df


def run_user_defined_feature_extraction_class(file_path: str,
                                              config_path: str):
    """
    Parameters
    ----------
    file_path: str
        Path to .py file holding user-defined feature extraction class
    config_path: str
        Path to SimBA project config file
    """

    check_file_exist_and_readable(file_path=file_path)
    file_dir, file_name, file_extension = get_fn_ext(filepath=file_path)
    if file_extension != '.py':
        raise InvalidFileTypeError(msg=f'The user-defined feature extraction file ({file_path}) is not a .py file-extension')
    parsed = ast.parse(Path(file_path).read_text())
    classes = [n for n in parsed.body if isinstance(n, ast.ClassDef)]
    class_name = [x.name for x in classes]
    if len(class_name) < 1:
        raise CountError(msg=f'The user-defined feature extraction file ({file_path}) contains no python classes')
    if len(class_name) > 1:
        stdout_warning(msg=f'The user-defined feature extraction file ({file_path}) contains more than 1 python class. SimBA will use the first python class: {class_name[0]}.')
    class_name = class_name[0]
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    user_module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = user_module
    spec.loader.exec_module(user_module)
    user_class = getattr(user_module, class_name)
    print(f'Running user-defined {class_name} feature extraction file...')
    user_class(config_path=config_path)