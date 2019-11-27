import pandas as pd
import os
import statistics
import numpy as np
import math
from configparser import ConfigParser
from datetime import datetime
dateTime = datetime.now().strftime('%Y%m%d%H%M%S')

configFile = r"Z:\DeepLabCut\DLC_extract\New_082119\project_folder\project_config.ini"

config = ConfigParser()
config.read(configFile)

use_master = config.get('General settings', 'use_master_config')
csv_dir = config.get('General settings', 'csv_path')
csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement')
csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement_location')
vidInfPath = config.get('General settings', 'project_path')
vidInfPath = os.path.join(vidInfPath, 'project_folder', 'logs')
vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
vidinfDf = pd.read_csv(vidInfPath)
bodyPart1_mouse1 = config.get('Outlier settings', 'movement_bodyPart1_mouse1')
bodyPart2_mouse1 = config.get('Outlier settings', 'movement_bodyPart2_mouse1')
bodyPart1_mouse2 = config.get('Outlier settings', 'movement_bodyPart1_mouse2')
bodyPart2_mouse2 = config.get('Outlier settings', 'movement_bodyPart2_mouse2')
distanceCalcSetting = config.get('Outlier settings', 'mean_or_median')
if not os.path.exists(csv_dir_out):
    os.makedirs(csv_dir_out)

filesFound = []
vNm_list = []
fixedPositions_M1_list = []
fixedPositions_M2_list = []
frames_processed_list = []
counts_total_M1 = [0]*7
counts_total_M2 = [0]*7
counts_total_M1_list = []
counts_total_M2_list = []
configFilelist = []
loopy = 0

reliableCoordinates = np.zeros((7,2))

criterion = config.getfloat('Outlier settings', 'location_criterion')

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

########### logfile path ###########
log_fn = config.get('General settings', 'project_name')
log_fn = 'Outliers_corrected_movement_location_' + str(dateTime) + '.csv'
log_path = config.get('General settings', 'project_path')
log_path = os.path.join(log_path, 'project_folder', 'logs')
log_fn = os.path.join(log_path, log_fn)
if not os.path.exists(log_path):
    os.makedirs(log_path)

columns = ['Video', 'Frames_processed', 'Mouse1_locations_corrected', 'Mouse2_locations_corrected']
log_df = pd.DataFrame(columns=columns)


########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
for i in filesFound:
    fixedPositions_M1 = 0
    fixedPositions_M2 = 0
    counts_total_M1 = [0] * 7
    counts_total_M2 = [0] * 7
    outputArray = np.array([0] * 14)
    outputArray_2 = np.array([0] * 14)
    tailCoords_out = np.array([0] * 4)
    if use_master == 'no':
        configFile = configFilelist[loopy]
        config = ConfigParser()
        config.read(configFile)
        criterion = config.getint('Outlier settings', 'location_criterion')
    loopy += 1
    currentFile = i
    fn = os.path.basename(currentFile)
    fn = fn.split('.')[0]
    currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == fn]
    currPixPerMM = float(currVideoSettings['pixels/mm'])
    csv_df = pd.read_csv(currentFile, header=0,
                            names=["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                                "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p", "Lat_left_1_x", "Lat_left_1_y",
                                "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                                "Tail_base_1_y", "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p", "Ear_left_2_x",
                                "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p",
                                "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y",
                                "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                                "Tail_base_2_y", "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p", "video_no", "frames"])
    csv_df = csv_df.apply(pd.to_numeric)

    vNm = csv_df['video_no'].iloc[0]
    vNm = 'Video' + str(vNm)
    vNm_list.append(vNm)


    ########### MEAN MOUSE SIZES ###########################################
    csv_df['Mouse_1_nose_to_tail'] = np.sqrt((csv_df.Nose_1_x - csv_df.Tail_base_1_x) ** 2 + (csv_df.Nose_1_y - csv_df.Tail_base_1_y) ** 2)
    csv_df['Mouse_2_nose_to_tail'] = np.sqrt((csv_df.Nose_2_x - csv_df.Tail_base_2_x) ** 2 + (csv_df.Nose_2_y - csv_df.Tail_base_2_y) ** 2)
    ########### MEAN MOUSE SIZES ###########################################
    if distanceCalcSetting == 'mean':
        mouse1size = (statistics.mean(csv_df['Mouse_1_nose_to_tail'])) / currPixPerMM
        mouse2size = (statistics.mean(csv_df['Mouse_2_nose_to_tail'])) / currPixPerMM
    if distanceCalcSetting == 'median':
        mouse1size = (statistics.median(csv_df['Mouse_1_nose_to_tail'])) / currPixPerMM
        mouse2size = (statistics.median(csv_df['Mouse_2_nose_to_tail'])) / currPixPerMM

    mouse1size = mouse1size * criterion
    mouse2size = mouse2size * criterion


    for index, row in csv_df.iterrows():
        currentArray = np.array(
                [[row['Ear_left_1_x'], row["Ear_left_1_y"]], [row['Ear_right_1_x'], row["Ear_right_1_y"]],
                [row['Nose_1_x'], row["Nose_1_y"]], [row['Center_1_x'], row["Center_1_y"]],
                [row['Lat_left_1_x'], row["Lat_left_1_y"]], [row['Lat_right_1_x'], row["Lat_right_1_y"]],
                [row['Tail_base_1_x'], row["Tail_base_1_y"]]]).astype(int)
        nbody_parts = len(currentArray)
        counts = [0] * nbody_parts
        for i in range(0, (nbody_parts - 1)):
            for j in range((i + 1), (nbody_parts)):
                dist_ij = (np.sqrt((currentArray[i][0] - currentArray[j][0]) ** 2 + (currentArray[i][1] - currentArray[j][1]) ** 2) / currPixPerMM)
                if dist_ij > mouse1size:
                    counts[i] += 1
                    counts[j] += 1
        positions = [i for i in range(len(counts)) if counts[i] > 1]
        for pos in positions:
            counts_total_M1[pos] += 1
        fixedPositions_M1 = fixedPositions_M1 + len(positions)
        if not positions:
            reliableCoordinates = currentArray
        else:
            for b in positions:
                currentPosition = b
                currentArray[currentPosition][0] = reliableCoordinates[currentPosition][0]
                currentArray[currentPosition][1] = reliableCoordinates[currentPosition][1]
            reliableCoordinates = currentArray
        currentArray = currentArray.flatten()
        outputArray = np.vstack((outputArray, currentArray))
    outputArray = np.delete(outputArray, 0, 0)

    for index, row in csv_df.iterrows():
        currentArray = np.array(
            [[row['Ear_left_2_x'], row["Ear_left_2_y"]], [row['Ear_right_2_x'], row["Ear_right_2_y"]],
             [row['Nose_2_x'], row["Nose_2_y"]], [row['Center_2_x'], row["Center_2_y"]],
             [row['Lat_left_2_x'], row["Lat_left_2_y"]], [row['Lat_right_2_x'], row["Lat_right_2_y"]],
             [row['Tail_base_2_x'], row["Tail_base_2_y"]]]).astype(int)
        nbody_parts = len(currentArray)
        counts = [0] * nbody_parts
        for i in range(0, (nbody_parts - 1)):
            for j in range((i + 1), (nbody_parts)):
                dist_ij = (np.sqrt((currentArray[i][0] - currentArray[j][0]) ** 2 + (currentArray[i][1] - currentArray[j][1]) ** 2) / currPixPerMM)
                if dist_ij > mouse2size:
                    counts[i] += 1
                    counts[j] += 1
        positions = [i for i in range(len(counts)) if counts[i] > 1]
        for pos in positions:
            counts_total_M2[pos] += 1
        fixedPositions_M2 = fixedPositions_M2 + len(positions)
        if not positions:
            reliableCoordinates = currentArray
        else:
            for b in positions:
                currentPosition = b
                currentArray[currentPosition][0] = reliableCoordinates[currentPosition][0]
                currentArray[currentPosition][1] = reliableCoordinates[currentPosition][1]
            reliableCoordinates = currentArray
        currentArray = currentArray.flatten()
        outputArray_2 = np.vstack((outputArray_2, currentArray))
    outputArray_2 = np.delete(outputArray_2, 0, 0)

    for index, row in csv_df.iterrows():
        tailCoords = np.array([[row['Tail_base_1_x'], row["Tail_base_1_y"]], [row['Tail_end_1_x'], row["Tail_end_1_y"]], [row['Tail_base_2_x'], row["Tail_base_2_y"]], [row['Tail_end_2_x'], row["Tail_end_2_y"]]])
        Tail_1_to_base_1 = (math.hypot(tailCoords[0][0] - tailCoords[1][0], tailCoords[0][1] - tailCoords[1][1]))
        Tail_1_to_base_2 = (math.hypot(tailCoords[2][0] - tailCoords[1][0], tailCoords[2][1] - tailCoords[1][1]))
        if Tail_1_to_base_2 > Tail_1_to_base_1:
            tailCoords = ([tailCoords[2][0], tailCoords[2][1]], [tailCoords[3][0], tailCoords[3][1]], [tailCoords[0][0], tailCoords[0][1]], [tailCoords[1][0], tailCoords[1][1]])
            tailCoords = np.asarray(tailCoords)
        tailCoords = np.delete(tailCoords, [0, 2], 0)
        tailCoords = tailCoords.flatten()
        tailCoords_out = np.vstack((tailCoords_out, tailCoords))
    tailCoords_out = np.delete(tailCoords_out, 0, 0)
    comb_out_array = np.hstack((outputArray,outputArray_2, tailCoords_out))
    csv_out = pd.DataFrame(comb_out_array)
    scorer = pd.read_csv(currentFile).scorer
    scorer = pd.to_numeric(scorer)
    scorer = scorer.reset_index()
    scorer = scorer.drop(['index'], axis=1)
    csv_out['scorer'] = scorer.values.astype(int)
    csv_df.index = csv_out.index
    csv_out.columns = ["Ear_left_1_x", "Ear_left_1_y", "Ear_right_1_x", "Ear_right_1_y", "Nose_1_x", "Nose_1_y", "Center_1_x", "Center_1_y", "Lat_left_1_x", "Lat_left_1_y", "Lat_right_1_x", "Lat_right_1_y", "Tail_base_1_x", "Tail_base_1_y",  "Ear_left_2_x", "Ear_left_2_y", "Ear_right_2_x", "Ear_right_2_y", "Nose_2_x", "Nose_2_y", "Center_2_x", "Center_2_y", "Lat_left_2_x", "Lat_left_2_y", "Lat_right_2_x", "Lat_right_2_y","Tail_base_2_x", "Tail_base_2_y", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_2_x", "Tail_end_2_y", "scorer"]
    csv_out[['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p', 'Tail_base_1_p', 'Tail_end_1_p','Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p', 'Lat_left_2_p', 'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p']] = csv_df[['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p', 'Tail_base_1_p', 'Tail_end_1_p','Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p', 'Lat_left_2_p', 'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p']]
    csv_out = csv_out[["scorer", "Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y", "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p", "Lat_left_1_x", "Lat_left_1_y", "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x", "Tail_base_1_y", "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p", "Ear_left_2_x", "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p", "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y", "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x", "Tail_base_2_y", "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p"]]
    df_headers = pd.read_csv(currentFile, nrows=0)
    df_headers['video_no'] = 0
    df_headers['frames'] = 0
    csv_out['video_no'] = csv_df['video_no']
    csv_out['frames'] = csv_df['frames']
    csv_out.columns = df_headers.columns
    csv_out = pd.concat([df_headers, csv_out])
    csv_out = csv_out.reset_index()
    csv_out = csv_out.drop('index', axis=1)
    fname = os.path.basename(currentFile)
    fileName, fileEnding = fname.split('.')
    printNm = fileName.split('_')[0]
    fileOut = str(fileName) + str('.csv')
    csvOutPath = os.path.join(csv_dir_out, fileOut)
    csv_out.to_csv(csvOutPath, index=False)
    frames_processed = len(comb_out_array)
    frames_processed_list.append(frames_processed)
    fixedPositions_M1_list.append(fixedPositions_M1)
    fixedPositions_M2_list.append(fixedPositions_M2)
    counts_total_M1_list.append(counts_total_M1)
    counts_total_M2_list.append(counts_total_M2)
    counts_total_M1_np = np.array(counts_total_M1_list)
    counts_total_M2_np = np.array(counts_total_M2_list)
    percentBDcorrected = (fixedPositions_M1 + fixedPositions_M2)/(frames_processed*14)

    print(str(printNm) + ' corrected data. ' + str(frames_processed) +' total frames processed.  Mouse 1 body parts corrected: ' + str(fixedPositions_M1) + ' Mouse 2 body parts corrected: ' + str(fixedPositions_M2) + '. Percent body parts corrected: ' + str(percentBDcorrected))

log_df['Video'] = vNm_list
log_df['Frames_processed'] = frames_processed_list
log_df['Mouse1_locations_corrected'] = fixedPositions_M1_list
log_df['Mouse2_locations_corrected'] = fixedPositions_M2_list
log_df['Mouse1_left_ear'] = counts_total_M1_np[:,0]
log_df['Mouse1_right_ear'] = counts_total_M1_np[:,1]
log_df['Mouse1_nose'] = counts_total_M1_np[:,2]
log_df['Mouse1_centroid'] = counts_total_M1_np[:,3]
log_df['Mouse1_lateral_left'] = counts_total_M1_np[:,4]
log_df['Mouse1_lateral_right'] = counts_total_M1_np[:,5]
log_df['Mouse1_tail_base'] = counts_total_M1_np[:,6]
log_df['Mouse2_left_ear'] = counts_total_M2_np[:,0]
log_df['Mouse2_right_ear'] = counts_total_M2_np[:,1]
log_df['Mouse2_nose'] = counts_total_M2_np[:,2]
log_df['Mouse2_centroid'] = counts_total_M2_np[:,3]
log_df['Mouse2_lateral_left'] = counts_total_M2_np[:,4]
log_df['Mouse2_lateral_right'] = counts_total_M2_np[:,5]
log_df['Mouse2_tail_base'] = counts_total_M2_np[:,6]
log_df['% bodyparts corrected'] = (log_df['Mouse1_locations_corrected'] + log_df['Mouse2_locations_corrected']) / (log_df['Frames_processed'] * 14)
log_df.to_csv(log_fn, index=False)