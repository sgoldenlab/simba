import pandas as pd
from pathlib import Path
import os

def read_video_info(vidinfDf, currVidName):
    currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == currVidName]
    if len(currVideoSettings) > 1:
        print('SimBA found multiple rows in the project_folder/logs/video_info.csv named ' + currVidName + '. Please make sure that each video is represented once only in the video_info.csv')
    elif len(currVideoSettings) < 1:
        print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file. SimBA could not find ' + currVidName + ' in the video_info.csv table.')
    else:
        try:
            currPixPerMM = float(currVideoSettings['pixels/mm'])
            fps = float(currVideoSettings['fps'])
            return currVideoSettings, currPixPerMM, fps
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')

def check_minimum_roll_windows(roll_windows_values, minimum_fps):
    for win in range(len(roll_windows_values)):
        if minimum_fps < roll_windows_values[win]:
            roll_windows_values[win] = minimum_fps
        else:
            pass
    roll_windows_values = list(set(roll_windows_values))
    return roll_windows_values

def check_if_file_exist(file_path=None):
    path_file_path = Path(file_path)
    if path_file_path.is_file():
        return True
    else:
        return False

def check_if_file_is_readable(file_path=None):
    if os.access(file_path, os.R_OK):
        return True
    else:
        return False

def read_video_info_csv(file_path=None):
    if not check_if_file_exist(file_path):
        print('The project "project_folder/logs/video_info.csv" file does not exists. Please generate the file by completing the [Video parameters] step')
        raise FileNotFoundError
    if not check_if_file_is_readable(file_path):
        print('The project "project_folder/logs/video_info.csv" file does not readable/corrupted. Please re-create it by completing the [Video parameters] step')
        raise ValueError
    info_df = pd.read_csv(file_path)
    for c in ['Video', 'fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm']:
        if c not in info_df.columns:
            print('The project "project_folder/logs/video_info.csv" does not not have an anticipated {} header. Please re-create the file and make sure each video has a {} value'.format(str(c), str(c)))
            raise ValueError
    info_df['Video'] = info_df['Video'].astype(str)
    for c in ['fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm']:
        try:
            info_df[c] = info_df[c].astype(float)
        except:
            print('One or more values in the {} column of the "project_folder/logs/video_info.csv" file could not be interepreted as a number. Please re-create the file and make sure the entries in the {} column are all numeric.'.format(str(c), str(c)))
            raise ValueError
    if info_df['fps'].min() <= 1:
        print('SIMBA WARNING: Videos in your SimBA project have an FPS of 1 or less. Please use videos with more than one frame per second, or correct the inaccurate fps inside the `project_folder/logs/videos_info.csv` file')

    return info_df












