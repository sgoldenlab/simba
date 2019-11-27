import pandas as pd
import re
import os
from configparser import ConfigParser
from datetime import datetime
import statistics
import numpy as np
dateTime = datetime.now().strftime('%Y%m%d%H%M%S')

config = ConfigParser()
configFile = r"Z:\DeepLabCut\DLC_extract\New_082119\project_folder\project_config.ini"
config.read(configFile)
csv_dir = config.get('General settings', 'csv_path')
csv_dir_in = os.path.join(csv_dir, 'machine_results')
use_master = config.get('General settings', 'use_master_config')
vidInfPath = config.get('General settings', 'project_path')
vidInfPath = os.path.join(vidInfPath, 'project_folder', 'logs')
vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
vidinfDf = pd.read_csv(vidInfPath)


filesFound = []
loop = 0
loopy = 0
VideoNo_list = []

########### logfile path ###########
log_fn = config.get('General settings', 'project_name')
log_fn = 'Movement_log_' + dateTime + '.csv'
log_path = config.get('General settings', 'project_path')
log_path = os.path.join(log_path, 'project_folder', 'logs')
log_fn = os.path.join(log_path, log_fn)
if not os.path.exists(log_path):
    os.makedirs(log_path)

columns = ['Video', 'Mean_velocity_Animal_1_cm/s', 'Median_velocity_Animal_1_cm/s', 'Mean_velocity_Animal_2_cm/s', 'Median_velocity_Animal_2_cm/s', 'Total_movement_Animal_1_cm', 'Total_movement_Animal_2_cm', 'Mean_centroid_distance_cm', 'Median_centroid_distance_cm', 'Mean_nose_to_nose_distance_cm', 'Median_nose_to_nose_distance_cm']
log_df = pd.DataFrame(columns=columns)
configFilelist = []

########### FIND CSV FILES ###########
if use_master == 'yes':
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)
if use_master == 'no':
    config_folder_path = config.get('General settings', 'config_folder')
    for i in os.listdir(config_folder_path):
        if i.__contains__(".ini"):
            configFilelist.append(os.path.join(config_folder_path, i))
            iniVidName = i.split(".")[0]
            csv_fn = iniVidName + '.csv'
            file = os.path.join(csv_dir_in, csv_fn)
            filesFound.append(file)

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
OneMMpixel = config.getint('Frame settings', 'mm_per_pixel')
fileCounter = 0
for i in filesFound:
    centroid_distance_mm_list = []
    nose_2_nose_dist_mm_list = []
    frameCounter = 0
    meanVelocity_M1_list = []
    meanVelocity_M2_list = []
    medianVelocity_M1_list = []
    medianVelocity_M2_list = []
    total_Movement_M1_list = []
    total_Movement_M2_list = []
    list_nose_movement_M1 = []
    list_nose_movement_M2 = []
    loop = 0
    currentFile = i
    if use_master == 'no':
        configFile = configFilelist[loopy]
        config = ConfigParser()
        config.read(configFile)
        fps = config.getint('Frame settings', 'fps')
        OneMMpixel = config.getint('Frame settings', 'mm_per_pixel')
    currVidName = os.path.basename(currentFile)
    videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(currVidName.replace('.csv', ''))]
    fps = int(videoSettings['fps'])
    OneMMpixel = int(videoSettings['pixels/mm'])
    loopy+=1
    csv_df = pd.read_csv(currentFile)
    VideoNo = os.path.basename(currentFile)
    VideoNo = 'Video' + str(re.sub("[^0-9]", "", VideoNo))
    VideoNo_list.append(VideoNo)
    df_lists = [csv_df[i:i+fps] for i in range(0,csv_df.shape[0],fps)]

    for i in df_lists:
        currentDf = i
        mmMove_nose_M1 = currentDf["Movement_mouse_1_nose"].mean()
        mmMove_nose_M2 = currentDf["Movement_mouse_2_nose"].mean()
        list_nose_movement_M1.append(mmMove_nose_M1)
        list_nose_movement_M2.append(mmMove_nose_M2)
        current_velocity_M1_cm_sec = (mmMove_nose_M1)
        current_velocity_M2_cm_sec = (mmMove_nose_M2)
        current_velocity_M1_cm_sec = round(current_velocity_M1_cm_sec, 2)
        current_velocity_M2_cm_sec = round(current_velocity_M2_cm_sec, 2)
        meanVelocity_M1 = statistics.mean(list_nose_movement_M1)
        meanVelocity_M2 = statistics.mean(list_nose_movement_M2)
        meanVelocity_M1 = round(meanVelocity_M1,2)
        meanVelocity_M2 = round(meanVelocity_M2, 2)
        meanVelocity_M1_list.append(meanVelocity_M1)
        meanVelocity_M2_list.append(meanVelocity_M2)

        medianVelocity_M1 = statistics.median(list_nose_movement_M1)
        medianVelocity_M2 = statistics.median(list_nose_movement_M2)
        medianVelocity_M1 = round(meanVelocity_M1, 2)
        medianVelocity_M2 = round(meanVelocity_M2, 2)
        medianVelocity_M1_list.append(meanVelocity_M1)
        medianVelocity_M2_list.append(meanVelocity_M2)

        total_Movement_M1 = sum(list_nose_movement_M1)
        total_Movement_M2 = sum(list_nose_movement_M2)
        total_Movement_M1 = round(total_Movement_M1, 2)
        total_Movement_M2 = round(total_Movement_M2, 2)
        total_Movement_M1_list.append(total_Movement_M1)
        total_Movement_M2_list.append(total_Movement_M2)

        for index, row in currentDf.iterrows():
            centroid_distance_px = (int(row["Centroid_distance"]))
            centroid_distance_mm = (centroid_distance_px) / 10
            centroid_distance_mm = round(centroid_distance_mm, 2)
            centroid_distance_mm_list.append(centroid_distance_mm)
            nose_2_nose_dist_px = (int(row["Nose_to_nose_distance"]))
            nose_2_nose_dist_mm = (nose_2_nose_dist_px ) / 10
            nose_2_nose_dist_mm = round(nose_2_nose_dist_mm, 2)
            nose_2_nose_dist_mm_list.append(nose_2_nose_dist_mm)
            loop += 1
            frameCounter += 1
    frames_processed_list.append(frameCounter)
    mean_centroid_distance_mm.append(statistics.mean(centroid_distance_mm_list))
    mean_nose_distance_mm.append(statistics.mean(nose_2_nose_dist_mm_list))
    median_centroid_distance_mm.append(statistics.median(centroid_distance_mm_list))
    median_nose_distance_mm.append(statistics.median(nose_2_nose_dist_mm_list))
    meanVeloM1.append(statistics.mean(meanVelocity_M1_list))
    meanVeloM2.append(statistics.mean(meanVelocity_M2_list))
    medianVeloM1.append(statistics.median(medianVelocity_M1_list))
    medianVeloM2.append(statistics.median(medianVelocity_M2_list))
    totMoveM1.append(total_Movement_M1_list[-1])
    totMoveM2.append(total_Movement_M2_list[-1])
    fileCounter += 1
    print('Files# processed for velocity/movement data: ' + str(fileCounter))

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


log_df.to_csv(log_fn, index=False)