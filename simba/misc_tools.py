from simba.drop_bp_cords import get_fn_ext, get_workflow_file_format
from simba.rw_dfs import read_df, save_df
import cv2
from pylab import *
import os, glob
from copy import deepcopy
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from pathlib import Path
import io
import PIL
import imutils
from numba import jit
from configparser import NoSectionError, ConfigParser, MissingSectionHeaderError
import multiprocessing
from simba.read_config_unit_tests import check_file_exist_and_readable
import shutil
import platform
import pickle
from datetime import datetime

def get_video_meta_data(video_path=None):
    vdata = {}
    cap = cv2.VideoCapture(video_path)
    _, vdata['video_name'], _ = get_fn_ext(video_path)
    vdata['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    vdata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vdata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vdata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k, v in vdata.items():
        if v == 0:
            raise ValueError('SIMBA WARNING: Video {} has {} of {}.'.format(vdata['video_name'], k, str(v)))
    vdata['resolution_str'] = str('{} x {}'.format(vdata['width'], vdata['height']))

    return vdata

def check_directionality_viable(animalBpDict):

    directionalitySetting = True
    NoseCoords = []
    EarLeftCoords = []
    EarRightCoords = []
    for animal in animalBpDict:
        for bp_cord in ['X_bps', 'Y_bps']:
            bp_list = animalBpDict[animal][bp_cord]
            for bp_name in bp_list:
                bp_name_components = bp_name.split('_')
                bp_name_components = [x.lower() for x in bp_name_components]
                if ('nose' in bp_name_components):
                    NoseCoords.append(bp_name)
                if ('ear' in bp_name_components) and ('left' in bp_name_components):
                    EarLeftCoords.append(bp_name)
                if ('ear' in bp_name_components) and ('right' in bp_name_components):
                    EarRightCoords.append(bp_name)

    for cord in [NoseCoords, EarLeftCoords, EarRightCoords]:
        if len(cord) != len(animalBpDict.keys()) * 2:
            directionalitySetting = False

    if directionalitySetting:
        NoseCoords = [NoseCoords[i * 2:(i + 1) * 2] for i in range((len(NoseCoords) + 2 - 1) // 2)]
        EarLeftCoords = [EarLeftCoords[i * 2:(i + 1) * 2] for i in range((len(EarLeftCoords) + 2 - 1) // 2)]
        EarRightCoords = [EarRightCoords[i * 2:(i + 1) * 2] for i in range((len(EarRightCoords) + 2 - 1) // 2)]

    return directionalitySetting, NoseCoords, EarLeftCoords, EarRightCoords


def check_multi_animal_status(config, noAnimals):
    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if (multiAnimalIDList[0] != '') and (noAnimals > 1):
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
            return multiAnimalStatus, multiAnimalIDList

        else:
            multiAnimalStatus = False
            multiAnimalIDList = []
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal + 1))
            print('Applying settings for classical tracking...')
            return multiAnimalStatus, multiAnimalIDList

    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1))
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')
        return multiAnimalStatus, multiAnimalIDList

def line_length(p, q, n, M, coord):
    Px = np.abs(p[0] - M[0])
    Py = np.abs(p[1] - M[1])
    Qx = np.abs(q[0] - M[0])
    Qy = np.abs(q[1] - M[1])
    Nx = np.abs(n[0] - M[0])
    Ny = np.abs(n[1] - M[1])
    Ph = np.sqrt(Px*Px + Py*Py)
    Qh = np.sqrt(Qx*Qx + Qy*Qy)
    Nh = np.sqrt(Nx*Nx + Ny*Ny)
    if (Nh < Ph and Nh < Qh and Qh < Ph):
        coord.extend((q[0], q[1]))
        return True, coord
    elif (Nh < Ph and Nh < Qh and Ph < Qh):
        coord.extend((p[0], p[1]))
        return True, coord
    else:
        return False, coord

@jit(nopython=True)
def line_length_numba(left_ear_array=None, right_ear_array=None, nose_array=None, target_array=None):
    results_array =  np.zeros((left_ear_array.shape[0], 4))
    for frame_no in range(results_array.shape[0]):
        Px = np.abs(left_ear_array[frame_no][0] - target_array[frame_no][0])
        Py = np.abs(left_ear_array[frame_no][1] - target_array[frame_no][1])
        Qx = np.abs(right_ear_array[frame_no][0] - target_array[frame_no][0])
        Qy = np.abs(right_ear_array[frame_no][1] - target_array[frame_no][1])
        Nx = np.abs(nose_array[frame_no][0] - target_array[frame_no][0])
        Ny = np.abs(nose_array[frame_no][1] - target_array[frame_no][1])
        Ph = np.sqrt(Px * Px + Py * Py)
        Qh = np.sqrt(Qx * Qx + Qy * Qy)
        Nh = np.sqrt(Nx * Nx + Ny * Ny)
        if (Nh < Ph and Nh < Qh and Qh < Ph):
            results_array[frame_no] = [0, right_ear_array[frame_no][0], right_ear_array[frame_no][1], True]
        elif (Nh < Ph and Nh < Qh and Ph < Qh):
            results_array[frame_no] = [1, left_ear_array[frame_no][0],  left_ear_array[frame_no][1], True]
        else:
            results_array[frame_no] = [2, -1, -1, False]

    return results_array


@jit(nopython=True)
def line_length_numba_to_static_location(left_ear_array=None, right_ear_array=None, nose_array=None, target_array=None):
    results_array =  np.zeros((left_ear_array.shape[0], 4))
    for frame_no in range(results_array.shape[0]):
        Px = np.abs(left_ear_array[frame_no][0] - target_array[0])
        Py = np.abs(left_ear_array[frame_no][1] - target_array[1])
        Qx = np.abs(right_ear_array[frame_no][0] - target_array[0])
        Qy = np.abs(right_ear_array[frame_no][1] - target_array[1])
        Nx = np.abs(nose_array[frame_no][0] - target_array[0])
        Ny = np.abs(nose_array[frame_no][1] - target_array[1])
        Ph = np.sqrt(Px * Px + Py * Py)
        Qh = np.sqrt(Qx * Qx + Qy * Qy)
        Nh = np.sqrt(Nx * Nx + Ny * Ny)
        if (Nh < Ph and Nh < Qh and Qh < Ph):
            results_array[frame_no] = [0, right_ear_array[frame_no][0], right_ear_array[frame_no][1], True]
        elif (Nh < Ph and Nh < Qh and Ph < Qh):
            results_array[frame_no] = [1, left_ear_array[frame_no][0],  left_ear_array[frame_no][1], True]
        else:
            results_array[frame_no] = [2, -1, -1, False]

    return results_array


def find_video_of_file(video_dir, filename):
    all_files_in_video_folder = [f for f in next(os.walk(video_dir))[2] if not f[0] == '.']
    all_files_in_video_folder = [os.path.join(video_dir, x) for x in all_files_in_video_folder]
    return_path = ''
    for file_path in all_files_in_video_folder:
        _, video_filename, ext = get_fn_ext(file_path)
        if ((video_filename == filename) and ((ext.lower() == '.mp4') or (ext.lower() == '.avi'))):
            return_path = file_path

    if return_path == '':
        print('FAILED: SimBA could not find a video file resenting {} in the project_folder/videos directory'.format(str(filename)))
    else:
        return return_path

def smooth_data_gaussian(config=None, file_path=None, time_window_parameter=None):
    try:
        time_window_parameter = int(time_window_parameter)
    except:
        print('Gaussian smoothing failed for video {}. Time window parameter {} can not be interpreted as an integer.'.format(str(os.path.basename(file_path)), str(time_window_parameter)))
    _, filename, _ = get_fn_ext(file_path)
    project_dir = config.get('General settings', 'project_path')
    video_dir = os.path.join(project_dir, 'videos')
    video_file_path = find_video_of_file(video_dir, filename)
    file_format = get_workflow_file_format(config)
    cap = cv2.VideoCapture(video_file_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if file_format == 'csv':
        pose_df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
    elif file_format == 'parquet':
        pose_df = pd.read_parquet(file_path)
    frames_in_time_window = int(time_window_parameter / (1000 / video_fps))
    new_df = deepcopy(pose_df)
    new_df.columns.names = ['scorer', 'bodyparts', 'coords']

    for c in new_df:
        new_df[c] = new_df[c].rolling(window=int(frames_in_time_window), win_type='gaussian', center=True).mean(std=5).fillna(new_df[c])
        new_df[c] = new_df[c].abs()

    if file_format == 'csv':
        save_df(new_df,file_format, file_path)
    elif file_format == 'parquet':
        table = pa.Table.from_pandas(new_df)
        pq.write_table(table, file_path)

    print('Gaussian smoothing complete for file {}'.format(filename), '...')


def smooth_data_savitzky_golay(config=None, file_path=None, time_window_parameter=None):
    try:
        time_window_parameter = int(time_window_parameter)
    except:
        print(
            'Savitzky-Golay smoothing failed for video {}. Time window parameter {} can not be interpreted as an integer.'.format(
                str(os.path.basename(file_path)), str(time_window_parameter)))
    _, filename, _ = get_fn_ext(file_path)
    project_dir = config.get('General settings', 'project_path')
    video_dir = os.path.join(project_dir, 'videos')
    video_file_path = find_video_of_file(video_dir, filename)
    file_format = get_workflow_file_format(config)
    cap = cv2.VideoCapture(video_file_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if file_format == 'csv':
        pose_df = pd.read_csv(file_path, header=[0, 1, 2], index_col=0)
    elif file_format == 'parquet':
        pose_df = pd.read_parquet(file_path)
    frames_in_time_window = int(time_window_parameter / (1000 / video_fps))
    if (frames_in_time_window % 2) == 0:
        frames_in_time_window = frames_in_time_window - 1
    if (frames_in_time_window % 2) <= 3:
        frames_in_time_window = 5
    new_df = deepcopy(pose_df)
    new_df.columns.names = ['scorer', 'bodyparts', 'coords']

    for c in new_df:
        new_df[c] = savgol_filter(x=new_df[c].to_numpy(), window_length=frames_in_time_window, polyorder=3, mode='nearest')
        new_df[c] = new_df[c].abs()

    if file_format == 'csv':
        save_df(new_df,file_format, file_path)
    elif file_format == 'parquet':
        table = pa.Table.from_pandas(new_df)
        pq.write_table(table, file_path)

    print('Savitzky-Golay smoothing complete for file {}'.format(filename), '...')


def add_missing_ROI_cols(shape_df):
    if not 'Color BGR' in shape_df.columns:
        shape_df['Color BGR'] = [(255, 255, 255)] * len(shape_df)
    if not 'Thickness' in shape_df.columns:
        shape_df['Thickness'] = [5] * len(shape_df)
    if not 'Color name' in shape_df.columns:
        shape_df['Color name'] = 'White'

    return shape_df

def get_file_path_parts(file_path):
    file_name = Path(file_path).stem
    file_directory = Path(file_path).parents[0]
    extension = Path(file_path).suffix

    return file_directory, file_name, extension


def create_gantt_img(bouts_df, clf_name, image_index, fps, gantt_img_title):
    fig, ax = plt.subplots()
    fig.suptitle(gantt_img_title, fontsize=24)
    relRows = bouts_df.loc[bouts_df['End_frame'] <= image_index]
    for i, event in enumerate(relRows.groupby("Event")):
        data_event = event[1][["Start_time", "Bout_time"]]
        ax.broken_barh(data_event.values, (4, 4), facecolors='red')
    xLength = (round(image_index / fps)) + 1
    if xLength < 10:
        xLength = 10
    ax.set_xlim(0, xLength)
    ax.set_ylim([0, 12])
    plt.ylabel(clf_name, fontsize=12)
    plt.yticks([])
    plt.xlabel('time(s)', fontsize=12)
    ax.yaxis.set_ticklabels([])
    ax.grid(True)
    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format="png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    ar = np.asarray(image)
    open_cv_image = ar[:, :, ::-1]
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    open_cv_image = cv2.resize(open_cv_image, (640, 480))
    open_cv_image = np.uint8(open_cv_image)
    buffer_.close()
    plt.close(fig)

    return open_cv_image

def resize_gantt(gantt_img, img_height):
    return imutils.resize(gantt_img, height=img_height)


def plug_holes_shortest_bout(data_df=None, clf_name=None, fps=None, shortest_bout=None):
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



def get_bouts_for_gantt(data_df=None, clf_name=None, fps=None):
    boutsList, nameList, startTimeList, endTimeList, endFrameList = [], [], [], [], []
    groupDf = pd.DataFrame()
    v = (data_df[clf_name] != data_df[clf_name].shift()).cumsum()
    u = data_df.groupby(v)[clf_name].agg(['all', 'count'])
    m = u['all'] & u['count'].ge(1)
    groupDf['groups'] = data_df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
    for indexes, rows in groupDf.iterrows():
        currBout = list(rows['groups'])
        boutTime = ((currBout[-1] - currBout[0]) + 1) / fps
        startTime = (currBout[0] + 1) / fps
        endTime = (currBout[1]) / fps
        endFrame = (currBout[1])
        endTimeList.append(endTime)
        startTimeList.append(startTime)
        boutsList.append(boutTime)
        nameList.append(clf_name)
        endFrameList.append(endFrame)

    return pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, endFrameList, boutsList)), columns=['Event', 'Start_time', 'End Time', 'End_frame', 'Bout_time'])

def read_config_file(ini_path=None):
    config = ConfigParser()
    try:
        config.read(str(ini_path))
    except MissingSectionHeaderError:
        raise MissingSectionHeaderError('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    return config

def create_single_color_lst(pallete_name=None, increments=None):
    cmap = cm.get_cmap(pallete_name, increments + 1)
    color_lst = []
    for i in range(cmap.N):
        rgb = list((cmap(i)[:3]))
        rgb = [i * 255 for i in rgb]
        rgb.reverse()
        color_lst.append(rgb)
    return color_lst

def detect_bouts(data_df=None, target_lst=None, fps=None):
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

    return pd.DataFrame(list(zip(nameList, startTimeList, endTimeList, startFrameLst, endFrameList, boutsList)), columns=['Event', 'Start_time', 'End Time', 'Start_frame', 'End_frame', 'Bout_time'])

def find_core_cnt():
    cpu_cnt = multiprocessing.cpu_count()
    cpu_cnt_to_use = int(cpu_cnt / 2)
    return cpu_cnt, cpu_cnt_to_use

def remove_a_folder(folder_dir=None):
    shutil.rmtree(folder_dir, ignore_errors=True)

def determine_chunksize_for_imap(data_df_len: int=None,
                                 percent: float = 0.025):
    if data_df_len < 100:
        return 10
    else:
        ratio = int(data_df_len *  percent)
        if ratio > 200:
            ratio = 200
        return ratio

def tabulate_clf_info(clf_path=None):
    _, clf_name, _ = get_fn_ext(clf_path)
    check_file_exist_and_readable(file_path=clf_path)
    try:
        clf_obj = pickle.load(open(clf_path, 'rb'))
    except:
        print('SIMBA ERROR: The {} file is not a pickle file'.format(clf_path))
        raise AttributeError('SIMBA ERROR: The {} file is not a pickle file'.format(clf_path))
    try:
        clf_features_no = clf_obj.n_features_
        clf_criterion = clf_obj.criterion
        clf_estimators = clf_obj.n_estimators
        clf_min_samples_leaf = clf_obj.min_samples_split
        clf_n_jobs = clf_obj.n_jobs
        clf_verbose = clf_obj.verbose
        if clf_verbose == 1: clf_verbose = True
        if clf_verbose == 0: clf_verbose = False
    except:
        print('SIMBA ERROR: The {} file is not an scikit-learn RF classifier'.format(clf_path))
        raise AttributeError('SIMBA ERROR: The {} file is not an scikit-learn RF classifier'.format(clf_path))
    creation_time = 'Unknown'
    try:
        if platform.system() == 'Windows':
            creation_time = os.path.getctime(clf_path)
        elif platform.system() == 'Darwin':
            creation_time = os.stat(clf_path)
            creation_time = creation_time.st_birthtime
    except AttributeError:
        pass
    if creation_time != 'Unknown':
        creation_time = str(datetime.utcfromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S'))


    print(str(clf_name), "CLASSIFIER INFORMATION")
    for (name, val) in zip(['NUMBER OF FEATURES', 'NUMBER OF TREES', 'CLASSIFIER CRITERION', 'CLASSIFIER_MIN_SAMPLE_LEAF',
              'CLASSIFIER_N_JOBS', 'CLASSIFIER VERBOSE SETTING', 'CLASSIFIER PATH', 'CLASSIFIER CREATION TIME'], [clf_features_no, clf_estimators, clf_criterion, clf_min_samples_leaf,
                                                                   clf_n_jobs, clf_verbose, clf_path, str(creation_time)]):
        print(name + ': ' + str(val))


# #
# configFile = str(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\aasf1\project_folder\project_config.ini")
# config = ConfigParser()
# config.read(configFile)
# # smooth_data_gaussian(config, r"Z:\DeepLabCut\DLC_extract\Troubleshooting\aasf1\project_folder\csv\input_csv\CSDS01702.csv", 200)
# #
#
# smooth_data_savitzky_golay(config, r"Z:\DeepLabCut\DLC_extract\Troubleshooting\aasf1\project_folder\csv\input_csv\CSDS01702.csv", 200)

#results = find_video_of_file(r'/Users/simon/Desktop/troubleshooting/light_analyzer/project_folder/videos', '20220422_ALMEAG02_B0')