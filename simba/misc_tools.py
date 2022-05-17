from configparser import NoSectionError, ConfigParser
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
        if multiAnimalIDList[0] != '':
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

def find_video_of_file(video_dir, filename):
    all_files_in_video_folder = next(os.walk(video_dir))[2]
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

    if file_format == 'csv':
        save_df(new_df,file_format, file_path)
    elif file_format == 'parquet':
        table = pa.Table.from_pandas(new_df)
        pq.write_table(table, file_path)

    print('Savitzky-Golay smoothing complete for file {}'.format(filename), '...')


# #
# configFile = str(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\aasf1\project_folder\project_config.ini")
# config = ConfigParser()
# config.read(configFile)
# # smooth_data_gaussian(config, r"Z:\DeepLabCut\DLC_extract\Troubleshooting\aasf1\project_folder\csv\input_csv\CSDS01702.csv", 200)
# #
#
# smooth_data_savitzky_golay(config, r"Z:\DeepLabCut\DLC_extract\Troubleshooting\aasf1\project_folder\csv\input_csv\CSDS01702.csv", 200)
