import pandas as pd

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

