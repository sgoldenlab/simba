import pandas as pd
import os
from configparser import ConfigParser
from datetime import datetime
import statistics
import numpy as np
import glob
from drop_bp_cords import define_movement_cols

#def analyze_process_movement(configini):

configini = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Trouble_040720\project_folder\project_config.ini"
dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
config = ConfigParser()
configFile = str(configini)
config.read(configFile)
projectPath = config.get('General settings', 'project_path')
csv_dir_in = os.path.join(projectPath, 'csv', 'machine_results')
vidLogFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
vidinfDf = pd.read_csv(vidLogFilePath)
noAnimals = config.getint('process movements', 'no_of_animals')
Animal_1_Bp = config.get('process movements', 'animal_1_bp')
Animal_2_Bp = config.get('process movements', 'animal_2_bp')
pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
VideoNo_list, columnNames1, fileCounter = [], [], 0

########### logfile path ###########
log_fn = os.path.join(projectPath, 'logs', 'Movement_log_' + dateTime + '.csv')
if not os.path.exists(log_fn):
    os.makedirs(log_fn)
columnNames = define_movement_cols(pose_estimation_body_parts=pose_estimation_body_parts,columnNames=columnNames1)
log_df = pd.DataFrame(columns=columnNames)

########### FIND CSV FILES ###########
filesFound = glob.glob(csv_dir_in + '/*.csv')
print('Processing movement data for ' + str(len(filesFound)) + ' files...')
frames_processed_list, meanVeloM1, medianVeloM1, totMoveM1 = ([], [], [], [])
if noAnimals == 2:
    meanVeloM2, medianVeloM2, totMoveM2, mean_distance, median_distance = [], [], [], [], []


### CREATE SHIFTED COLUMNS
boutsDf['Shifted start'] = boutsDf['Start Time'].shift(-1)



########### SET MOVEMENT COLUMN IF USER DEFINED CONFIG ###########








if pose_estimation_body_parts == 'user_defined':
    animal_1_movement_column = str('movement_' + Animal_1_Bp)
if noAnimals == 2:
    animal_2_movement_column = str('movement_' + Animal_2_Bp)
    distance_column = 'distance_' + str(Animal_1_Bp) + '_to_' + str(Animal_2_Bp)
    bodyPart_distance_list, mean_bodypart_distance_list, median_bodypart_distance_list = ([], [], [])

for i in filesFound:
    list_nose_movement_M1 = []
    if noAnimals == 2:
        centroid_distance_cm_list, nose_2_nose_dist_cm_list, list_nose_movement_M2  = ([], [], [])
    frameCounter = 0
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
        if (pose_estimation_body_parts == '7') or (pose_estimation_body_parts == '8') or (pose_estimation_body_parts == '4') or (pose_estimation_body_parts == '9'):
            mmMove_nose_M1 = currentDf["Movement_mouse_1_centroid"].mean()
            list_nose_movement_M1.append(mmMove_nose_M1)
        if (pose_estimation_body_parts == '16') or (pose_estimation_body_parts == '14'):
            mmMove_nose_M1 = currentDf["Movement_mouse_1_centroid"].mean()
            mmMove_nose_M2 = currentDf["Movement_mouse_2_centroid"].mean()
            list_nose_movement_M1.append(mmMove_nose_M1)
            list_nose_movement_M2.append(mmMove_nose_M2)
        if (pose_estimation_body_parts == 'user_defined') and (noAnimals==1):
            mmMove_nose_M1 = currentDf[animal_1_movement_column].mean()
            list_nose_movement_M1.append(mmMove_nose_M1)
        if (pose_estimation_body_parts == 'user_defined') and (noAnimals==2):
            mmMove_nose_M1 = currentDf[animal_1_movement_column].mean()
            mmMove_nose_M2 = currentDf[animal_2_movement_column].mean()
            list_nose_movement_M1.append(mmMove_nose_M1)
            list_nose_movement_M2.append(mmMove_nose_M2)

        for index, row in currentDf.iterrows():
            if (pose_estimation_body_parts == '16') or (pose_estimation_body_parts == '14'):
                centroid_distance_cm = round((int(row["Centroid_distance"]) / 10), 2)
                centroid_distance_cm_list.append(centroid_distance_cm)
                nose_2_nose_dist_cm = round((int(row["Nose_to_nose_distance"]) / 10), 2)
                nose_2_nose_dist_cm_list.append(nose_2_nose_dist_cm)
            if (pose_estimation_body_parts == 'user_defined') and (noAnimals==2):
                bodypart_distance = round((int(row[distance_column]) / 10), 2)
                bodyPart_distance_list.append(bodypart_distance)
            frameCounter += 1

    frames_processed_list.append(frameCounter)
    meanVeloM1.append(statistics.mean(list_nose_movement_M1))
    medianVeloM1.append(statistics.median(list_nose_movement_M1))
    totMoveM1.append(sum(list_nose_movement_M1))
    if (pose_estimation_body_parts == '16') or (pose_estimation_body_parts == '14'):
        mean_centroid_distance_cm.append(statistics.mean(centroid_distance_cm_list))
        mean_nose_distance_cm.append(statistics.mean(nose_2_nose_dist_cm_list))
        median_centroid_distance_cm.append(statistics.median(centroid_distance_cm_list))
        median_nose_distance_cm.append(statistics.median(nose_2_nose_dist_cm_list))
        meanVeloM2.append(statistics.mean(list_nose_movement_M2))
        medianVeloM2.append(statistics.median(list_nose_movement_M2))
        totMoveM2.append(sum(list_nose_movement_M2))
    if (pose_estimation_body_parts == 'user_defined') and (noAnimals == 2):
        meanVeloM2.append(statistics.mean(list_nose_movement_M2))
        medianVeloM2.append(statistics.median(list_nose_movement_M2))
        totMoveM2.append(sum(list_nose_movement_M2))
        mean_bodypart_distance_list.append(statistics.mean(bodyPart_distance_list))
        median_bodypart_distance_list.append(statistics.median(bodyPart_distance_list))
    fileCounter += 1
    print('Files # processed for movement data: ' + str(fileCounter) + '/' + str(len(filesFound)) + '...')

log_df['Video'] = VideoNo_list
log_df['Frames_processed'] = frames_processed_list
log_df['Mean_velocity_Animal_1_cm/s'] = meanVeloM1
log_df['Median_velocity_Animal_1_cm/s'] = medianVeloM1
log_df['Total_movement_Animal_1_cm'] = totMoveM1
if (pose_estimation_body_parts == '16') or (pose_estimation_body_parts == '14'):
    print('xx')
    log_df['Mean_velocity_Animal_2_cm/s'] = meanVeloM2
    log_df['Median_velocity_Animal_2_cm/s'] = medianVeloM2
    log_df['Total_movement_Animal_2_cm'] = totMoveM2
    log_df['Mean_centroid_distance_cm'] = mean_centroid_distance_cm
    log_df['Median_centroid_distance_cm'] = median_centroid_distance_cm
    log_df['Mean_nose_to_nose_distance_cm'] = mean_nose_distance_cm
    log_df['Median_nose_to_nose_distance_cm'] = median_nose_distance_cm
if (pose_estimation_body_parts == 'user_defined') and (noAnimals == 2):
    log_df['Mean_velocity_Animal_2_cm/s'] = meanVeloM2
    log_df['Median_velocity_Animal_2_cm/s'] = medianVeloM2
    log_df['Total_movement_Animal_2_cm'] = totMoveM2
    log_df['Mean_animal_distance_cm'] = mean_centroid_distance_cm
    log_df['Median_animal_distance_cn'] = median_centroid_distance_cm

log_df = np.round(log_df,decimals=4)
print('All files processed for movement data. ' + 'Data saved @ project_folder\logs')
log_df.to_csv(log_fn, index=False)