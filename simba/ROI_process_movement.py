import pandas as pd
import os
from configparser import ConfigParser
from datetime import datetime
import statistics
import numpy as np
import glob
from simba.drop_bp_cords import define_movement_cols

def ROI_process_movement(configini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)

    ## get column headers
    colHeads, colHeadsShifted, VideoNo_list = [], [], []
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3, colHead1_shifted, colHead2_shifted, colHead3_shifted = (bodypart + '_x', bodypart + '_y', bodypart + '_p', bodypart + '_x_shifted', bodypart + '_y_shifted', bodypart + '_p_shifted')
        colHeads.append(colHead1)
        colHeads.append(colHead2)
        colHeads.append(colHead3)
        colHeadsShifted.append(colHead1_shifted)
        colHeadsShifted.append(colHead2_shifted)
        colHeadsShifted.append(colHead3_shifted)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    noAnimals = config.getint('process movements', 'no_of_animals')
    Animal_1_Bp = config.get('process movements', 'animal_1_bp')
    columns2keep = [Animal_1_Bp + '_x', Animal_1_Bp + '_y', Animal_1_Bp + '_x_shifted', Animal_1_Bp + '_y_shifted']
    Animal_2_Bp = config.get('process movements', 'animal_2_bp')

    ########### logfile path ###########
    log_fn = 'Movement_log_' + dateTime + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_df = pd.DataFrame()

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Processing movement data for ' + str(len(filesFound)) + ' files...')
    frames_processed_list, meanVeloM1, medianVeloM1, totMoveM1  = [], [], [], []
    if noAnimals == 2:
        meanVeloM2, medianVeloM2, totMoveM2, mean_distance_animals_cm, median_distance_animals_cm, = ([], [], [], [], [])
        columns2keep.extend([Animal_2_Bp + '_x', Animal_2_Bp + '_y', Animal_2_Bp + '_x_shifted', Animal_2_Bp + '_y_shifted'])
    fileCounter = 0

    for currentFile in filesFound:
        movement_list_bp_1 = []
        if noAnimals == 2:
            movement_list_bp_2, bodyPart_distance_list = [], []
        currVidName = os.path.basename(currentFile)
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == str(currVidName.replace('.csv', ''))]
        try:
            fps = int(videoSettings['fps'])
            currPixPerMM = float(videoSettings['pixels/mm'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        csv_df = pd.read_csv(currentFile, skiprows=[0], names=colHeads)
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = colHeadsShifted
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)
        csv_df_combined = csv_df_combined[columns2keep]

        ### calculate movement of body-part of interest
        csv_df_combined['Movement_bp_1'] = (np.sqrt((csv_df_combined[columns2keep[0]] - csv_df_combined[columns2keep[2]]) ** 2 + (csv_df_combined[columns2keep[1]] - csv_df_combined[columns2keep[3]]) ** 2)) / currPixPerMM
        if noAnimals == 2:
            csv_df_combined['Movement_bp_2'] = (np.sqrt((csv_df_combined[columns2keep[4]] - csv_df_combined[columns2keep[6]]) ** 2 + (csv_df_combined[columns2keep[5]] - csv_df_combined[columns2keep[7]]) ** 2)) / currPixPerMM
        VideoName = os.path.basename(currentFile).replace('.csv', '')
        VideoNo_list.append(VideoName)
        df_lists = [csv_df_combined[i:i+fps] for i in range(0,csv_df_combined.shape[0],fps)]
        for currentDf in df_lists:
            mean_mm_move_bp1 = currentDf['Movement_bp_1'].mean()
            movement_list_bp_1.append(mean_mm_move_bp1)
            if noAnimals == 2:
                mean_mm_move_bp2 = currentDf['Movement_bp_2'].mean()
                movement_list_bp_2.append(mean_mm_move_bp2)
                for index, row in currentDf.iterrows():
                    bodypart_distance = (np.sqrt((row[columns2keep[0]] - row[columns2keep[4]]) ** 2 + (row[columns2keep[1]] - row[columns2keep[5]]) ** 2)) / currPixPerMM
                    bodyPart_distance_list.append(bodypart_distance / 10)
        frames_processed_list.append(len(csv_df))
        meanVeloM1.append(statistics.mean(movement_list_bp_1))
        medianVeloM1.append(statistics.median(movement_list_bp_1))
        totMoveM1.append(sum(movement_list_bp_1))
        if noAnimals == 2:
            mean_distance_animals_cm.append(statistics.mean(bodyPart_distance_list))
            median_distance_animals_cm.append(statistics.median(bodyPart_distance_list))
            meanVeloM2.append(statistics.mean(movement_list_bp_2))
            medianVeloM2.append(statistics.median(movement_list_bp_2))
            totMoveM2.append(sum(movement_list_bp_2))
        fileCounter += 1
        print('Files # processed for movement data: ' + str(fileCounter) + '/' + str(len(filesFound)) + '...')

    log_df['Video'] = VideoNo_list
    log_df['Frames_processed'] = frames_processed_list
    log_df['Mean_velocity_Animal_1_cm/s'] = meanVeloM1
    log_df['Median_velocity_Animal_1_cm/s'] = medianVeloM1
    log_df['Total_movement_Animal_1_cm'] = totMoveM1
    if noAnimals == 2:
        log_df['Mean_velocity_Animal_2_cm/s'] = meanVeloM2
        log_df['Median_velocity_Animal_2_cm/s'] = medianVeloM2
        log_df['Total_movement_Animal_2_cm'] = totMoveM2
        log_df['Mean_animal_distance_cm'] = mean_distance_animals_cm
        log_df['Median_animal_distance_cm'] = median_distance_animals_cm
    log_df = np.round(log_df,decimals=4)
    log_df.to_csv(log_fn, index=False)
    print('All files processed for movement data. ' + 'Data saved @ project_folder\logs')