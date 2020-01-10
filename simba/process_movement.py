import pandas as pd
import re
import os
from configparser import ConfigParser
from datetime import datetime
import statistics
import numpy as np

def analyze_process_movement(configini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'machine_results')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    filesFound = []
    VideoNo_list = []

    ########### logfile path ###########
    log_fn = 'Movement_log_' + dateTime + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    columns = ['Video', 'Mean_velocity_Animal_1_cm/s', 'Median_velocity_Animal_1_cm/s', 'Mean_velocity_Animal_2_cm/s', 'Median_velocity_Animal_2_cm/s', 'Total_movement_Animal_1_cm', 'Total_movement_Animal_2_cm', 'Mean_centroid_distance_cm', 'Median_centroid_distance_cm', 'Mean_nose_to_nose_distance_cm', 'Median_nose_to_nose_distance_cm']
    log_df = pd.DataFrame(columns=columns)

    ########### FIND CSV FILES ###########
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)
    print('Processing movement data for ' + str(len(filesFound)) + ' files...')

    frames_processed_list = []
    meanVeloM1 = []
    meanVeloM2 = []
    medianVeloM1 = []
    medianVeloM2 = []
    totMoveM1 = []
    totMoveM2 = []
    mean_centroid_distance_mm = []
    mean_nose_distance_mm = []
    median_centroid_distance_mm = []
    median_nose_distance_mm = []
    fileCounter = 0

    for i in filesFound:
        centroid_distance_mm_list = []
        nose_2_nose_dist_mm_list = []
        frameCounter = 0
        list_nose_movement_M1 = []
        list_nose_movement_M2 = []
        currentFile = i
        currVidName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(currVidName.replace('.csv', ''))]
        try:
            fps = int(videoSettings['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        csv_df = pd.read_csv(currentFile)
        VideoName = os.path.basename(currentFile).replace('.csv', '')
        VideoNo_list.append(VideoName)
        df_lists = [csv_df[i:i+fps] for i in range(0,csv_df.shape[0],fps)]
        for i in df_lists:
            currentDf = i
            mmMove_nose_M1 = currentDf["Movement_mouse_1_centroid"].mean()
            mmMove_nose_M2 = currentDf["Movement_mouse_2_centroid"].mean()
            list_nose_movement_M1.append(mmMove_nose_M1)
            list_nose_movement_M2.append(mmMove_nose_M2)

            for index, row in currentDf.iterrows():
                centroid_distance_px = int(row["Centroid_distance"])
                centroid_distance_mm = (centroid_distance_px) / 10
                centroid_distance_mm = round(centroid_distance_mm, 2)
                centroid_distance_mm_list.append(centroid_distance_mm)
                nose_2_nose_dist_px = int(row["Nose_to_nose_distance"])
                nose_2_nose_dist_mm = (nose_2_nose_dist_px ) / 10
                nose_2_nose_dist_mm = round(nose_2_nose_dist_mm, 2)
                nose_2_nose_dist_mm_list.append(nose_2_nose_dist_mm)
                frameCounter += 1

        frames_processed_list.append(frameCounter)
        mean_centroid_distance_mm.append(statistics.mean(centroid_distance_mm_list))
        mean_nose_distance_mm.append(statistics.mean(nose_2_nose_dist_mm_list))
        median_centroid_distance_mm.append(statistics.median(centroid_distance_mm_list))
        median_nose_distance_mm.append(statistics.median(nose_2_nose_dist_mm_list))
        meanVeloM1.append(statistics.mean(list_nose_movement_M1))
        meanVeloM2.append(statistics.mean(list_nose_movement_M2))
        medianVeloM1.append(statistics.median(list_nose_movement_M1))
        medianVeloM2.append(statistics.median(list_nose_movement_M2))
        totMoveM1.append(sum(list_nose_movement_M1))
        totMoveM2.append(sum(list_nose_movement_M2))
        fileCounter += 1
        print('Files # processed for movement data: ' + str(fileCounter) + '/' + str(len(filesFound)))

    log_df['Video'] = VideoNo_list
    log_df['Frames_processed'] = frames_processed_list
    log_df['Mean_velocity_Animal_1_cm/s'] = meanVeloM1
    log_df['Mean_velocity_Animal_2_cm/s'] = meanVeloM2
    log_df['Median_velocity_Animal_1_cm/s'] = medianVeloM1
    log_df['Median_velocity_Animal_2_cm/s'] = medianVeloM2
    log_df['Total_movement_Animal_1_cm'] = totMoveM1
    log_df['Total_movement_Animal_2_cm'] = totMoveM2
    log_df['Mean_centroid_distance_cm'] = mean_centroid_distance_mm
    log_df['Median_centroid_distance_cm'] = median_centroid_distance_mm
    log_df['Mean_nose_to_nose_distance_cm'] = mean_nose_distance_mm
    log_df['Median_nose_to_nose_distance_cm'] = median_nose_distance_mm
    log_df = np.round(log_df,decimals=4)

    print('All files processed for movement data. ' + 'Data saved @' + str(log_fn))
    log_df.to_csv(log_fn, index=False)