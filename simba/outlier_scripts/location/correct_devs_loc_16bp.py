import pandas as pd
import os, glob
import statistics
import numpy as np
import math
from configparser import ConfigParser
from datetime import datetime
from simba.rw_dfs import *
from simba.drop_bp_cords import *

def dev_loc_16(projectini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    configFile = str(projectini)
    config = ConfigParser()
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in, csv_dir_out = os.path.join(projectPath, 'csv', 'outlier_corrected_movement'), os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    vNm_list = []
    fixedPositions_M1_list, fixedPositions_M2_list = [], []
    frames_processed_list, counts_total_M1_list, counts_total_M2_list, loopy = [], [], [], 0
    reliableCoordinates = np.zeros((7, 2))
    wfileType = config.get('General settings', 'workflow_file_type')
    criterion = config.getfloat('Outlier settings', 'location_criterion')

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Processing ' + str(len(filesFound)) + ' files for location outliers...')

    ########### logfile path ###########
    log_path = os.path.join(projectPath, 'logs', 'Outliers_location_' + str(dateTime) + '.csv')

    columns = ['Video', 'Frames_processed']
    log_df = pd.DataFrame(columns=columns)

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for currentFile in filesFound:
        fixedPositions_M1, fixedPositions_M2 = 0, 0
        counts_total_M1, counts_total_M2 = [0] * 7, [0] * 7
        outputArray = np.array([0] * 14)
        outputArray_2 = np.array([0] * 14)
        tailCoords_out = np.array([0] * 4)
        loopy += 1
        videoFileBaseName = os.path.basename(currentFile).replace('.csv', '')
        headerNames = ["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                                        "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y",
                                        "Center_1_p", "Lat_left_1_x", "Lat_left_1_y",
                                        "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                                        "Tail_base_1_y", "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p",
                                        "Ear_left_2_x",
                                        "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p",
                                        "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p",
                                        "Lat_left_2_x", "Lat_left_2_y",
                                        "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                                        "Tail_base_2_y", "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p"]


        csv_df = read_df(currentFile, wfileType,idx=None)
        csv_df = csv_df.set_index('scorer')
        csv_df.columns = headerNames

        csv_df = csv_df.apply(pd.to_numeric)
        vNm_list.append(videoFileBaseName)

        ########### MEAN MOUSE SIZES ###########################################
        csv_df['Mouse_1_nose_to_tail'] = np.sqrt(
            (csv_df.Nose_1_x - csv_df.Tail_base_1_x) ** 2 + (csv_df.Nose_1_y - csv_df.Tail_base_1_y) ** 2)
        csv_df['Mouse_2_nose_to_tail'] = np.sqrt(
            (csv_df.Nose_2_x - csv_df.Tail_base_2_x) ** 2 + (csv_df.Nose_2_y - csv_df.Tail_base_2_y) ** 2)
        mean1size = (statistics.mean(csv_df['Mouse_1_nose_to_tail']))
        mean2size = (statistics.mean(csv_df['Mouse_2_nose_to_tail']))

        mean1size = mean1size * criterion
        mean2size = mean2size * criterion

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
                    dist_ij = np.sqrt(
                        (currentArray[i][0] - currentArray[j][0]) ** 2 + (currentArray[i][1] - currentArray[j][1]) ** 2)
                    if dist_ij > mean1size:
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
                    dist_ij = np.sqrt(
                        (currentArray[i][0] - currentArray[j][0]) ** 2 + (currentArray[i][1] - currentArray[j][1]) ** 2)
                    if dist_ij > mean2size:
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
            tailCoords = np.array(
                [[row['Tail_base_1_x'], row["Tail_base_1_y"]], [row['Tail_end_1_x'], row["Tail_end_1_y"]],
                 [row['Tail_base_2_x'], row["Tail_base_2_y"]], [row['Tail_end_2_x'], row["Tail_end_2_y"]]])
            Tail_1_to_base_1 = (math.hypot(tailCoords[0][0] - tailCoords[1][0], tailCoords[0][1] - tailCoords[1][1]))
            Tail_1_to_base_2 = (math.hypot(tailCoords[2][0] - tailCoords[1][0], tailCoords[2][1] - tailCoords[1][1]))
            if Tail_1_to_base_2 > Tail_1_to_base_1:
                tailCoords = ([tailCoords[2][0], tailCoords[2][1]], [tailCoords[3][0], tailCoords[3][1]],
                              [tailCoords[0][0], tailCoords[0][1]], [tailCoords[1][0], tailCoords[1][1]])
                tailCoords = np.asarray(tailCoords)
            tailCoords = np.delete(tailCoords, [0, 2], 0)
            tailCoords = tailCoords.flatten()
            tailCoords_out = np.vstack((tailCoords_out, tailCoords))
        tailCoords_out = np.delete(tailCoords_out, 0, 0)
        comb_out_array = np.hstack((outputArray, outputArray_2, tailCoords_out))
        csv_out = pd.DataFrame(comb_out_array)
        scorer = csv_df.index
        scorer = pd.to_numeric(scorer)
        #scorer = scorer.reset_index()
        #scorer = scorer.drop(['index'], axis=1)
        csv_out['scorer'] = scorer.values.astype(int)
        csv_df.index = csv_out.index
        csv_out.columns = ["Ear_left_1_x", "Ear_left_1_y", "Ear_right_1_x", "Ear_right_1_y", "Nose_1_x", "Nose_1_y",
                           "Center_1_x", "Center_1_y", "Lat_left_1_x", "Lat_left_1_y", "Lat_right_1_x", "Lat_right_1_y",
                           "Tail_base_1_x", "Tail_base_1_y", "Ear_left_2_x", "Ear_left_2_y", "Ear_right_2_x",
                           "Ear_right_2_y", "Nose_2_x", "Nose_2_y", "Center_2_x", "Center_2_y", "Lat_left_2_x",
                           "Lat_left_2_y", "Lat_right_2_x", "Lat_right_2_y", "Tail_base_2_x", "Tail_base_2_y",
                           "Tail_end_1_x", "Tail_end_1_y", "Tail_end_2_x", "Tail_end_2_y", "scorer"]
        csv_out[['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p',
                 'Tail_base_1_p', 'Tail_end_1_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p',
                 'Lat_left_2_p', 'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p']] = csv_df[
            ['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p',
             'Tail_base_1_p', 'Tail_end_1_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p', 'Lat_left_2_p',
             'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p']]
        csv_out = csv_out[["scorer", "Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                           "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y",
                           "Center_1_p", "Lat_left_1_x", "Lat_left_1_y", "Lat_left_1_p", "Lat_right_1_x",
                           "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x", "Tail_base_1_y", "Tail_base_1_p",
                           "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p", "Ear_left_2_x", "Ear_left_2_y",
                           "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p", "Nose_2_x", "Nose_2_y",
                           "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y",
                           "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                           "Tail_base_2_y", "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p"]]
        csv_out.set_index('scorer')
        fname = os.path.basename(currentFile)
        fnamePrint = fname.replace(wfileType, '')
        csvOutPath = os.path.join(csv_dir_out, fnamePrint + wfileType)
        save_df(csv_out, wfileType, csvOutPath)
        frames_processed = len(comb_out_array)
        frames_processed_list.append(frames_processed)
        fixedPositions_M1_list.append(fixedPositions_M1)
        fixedPositions_M2_list.append(fixedPositions_M2)
        counts_total_M1_list.append(counts_total_M1)
        counts_total_M2_list.append(counts_total_M2)
        counts_total_M1_np = np.array(counts_total_M1_list)
        counts_total_M2_np = np.array(counts_total_M2_list)
        percentBDcorrected = round((fixedPositions_M1 + fixedPositions_M2) / (frames_processed * 14), 6)

        print(str(fnamePrint) + ' Tot frames: ' + str(frames_processed) + '. Outliers animal 1: ' + str(fixedPositions_M1) + '. Outliers animal 2: ' + str(fixedPositions_M2) + '. % outliers: ' + str(round(percentBDcorrected, 3)))

    log_df['Video'] = vNm_list
    log_df['Frames_processed'] = frames_processed_list
    log_df['Animal1_centroid'] = counts_total_M1_np[:, 3]
    log_df['Animal1_left_ear'] = counts_total_M1_np[:, 0]
    log_df['Animal1_right_ear'] = counts_total_M1_np[:, 1]
    log_df['Animal1_lateral_left'] = counts_total_M1_np[:, 4]
    log_df['Animal1_lateral_right'] = counts_total_M1_np[:, 5]
    log_df['Animal1_nose'] = counts_total_M1_np[:, 2]
    log_df['Animal1_tail_base'] = counts_total_M1_np[:, 6]
    log_df['Animal2_centroid'] = counts_total_M2_np[:, 3]
    log_df['Animal2_left_ear'] = counts_total_M2_np[:, 0]
    log_df['Animal2_right_ear'] = counts_total_M2_np[:, 1]
    log_df['Animal2_lateral_left'] = counts_total_M2_np[:, 4]
    log_df['Animal2_lateral_right'] = counts_total_M2_np[:, 5]
    log_df['Animal2_nose'] = counts_total_M2_np[:, 2]
    log_df['Animal2_tail_base'] = counts_total_M2_np[:, 6]
    log_df['Sum'] = log_df['Animal1_centroid'] + log_df['Animal1_left_ear'] + log_df['Animal1_right_ear'] + log_df['Animal1_lateral_left'] + log_df['Animal1_lateral_right'] + log_df['Animal1_nose'] + log_df['Animal1_tail_base'] + log_df['Animal2_centroid'] + log_df['Animal2_left_ear'] + log_df['Animal2_right_ear'] + log_df['Animal2_lateral_left'] + log_df['Animal2_lateral_right'] + log_df['Animal2_nose'] + log_df['Animal2_tail_base']
    log_df['% body parts corrected'] = log_df['Sum'] / (log_df['Frames_processed'] * 14)
    log_df.to_csv(log_path, index=False)
    print('Log for corrected "location outliers" saved in project_folder/logs')