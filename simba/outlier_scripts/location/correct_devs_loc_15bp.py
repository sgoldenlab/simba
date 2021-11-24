import pandas as pd
import os
import statistics
import numpy as np
from configparser import ConfigParser
from datetime import datetime
from simba.drop_bp_cords import *

def dev_loc_14(projectini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    configFile = str(projectini)
    config = ConfigParser()
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement', 'Batch_3')
    csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    filesFound = []
    vNm_list = []
    fixedPositions_M1_list = []
    fixedPositions_M2_list = []
    frames_processed_list = []
    counts_total_M1_list = []
    counts_total_M2_list = []
    loopy = 0
    reliableCoordinates = np.zeros((7, 2))

    criterion = config.getfloat('Outlier settings', 'location_criterion')

    ########### FIND CSV FILES ###########
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)
    print('Processing ' + str(len(filesFound)) + ' files for location outliers...')

    ########### logfile path ###########
    log_fn = 'Outliers_location_' + str(dateTime) + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    columns = ['Video', 'Frames_processed']
    log_df = pd.DataFrame(columns=columns)

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for i in filesFound:
        fixedPositions_M1 = 0
        fixedPositions_M2 = 0
        counts_total_M1 = [0] * 7
        counts_total_M2 = [0] * 7
        outputArray = np.array([0] * 14)
        outputArray_2 = np.array([0] * 14)
        loopy += 1
        currentFile = i
        videoFileBaseName = os.path.basename(currentFile).replace('.csv', '')
        csv_df = pd.read_csv(currentFile, header=0,
                             names=["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                                    "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y",
                                    "Center_1_p", "Lat_left_1_x", "Lat_left_1_y",
                                    "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                                    "Tail_base_1_y", "Tail_base_1_p","Ear_left_2_x",
                                    "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p",
                                    "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p",
                                    "Lat_left_2_x", "Lat_left_2_y", "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                                    "Tail_base_2_y", "Tail_base_2_p"])
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

        comb_out_array = np.hstack((outputArray, outputArray_2))
        csv_out = pd.DataFrame(comb_out_array)
        scorer = pd.read_csv(currentFile).scorer
        scorer = pd.to_numeric(scorer)
        scorer = scorer.reset_index()
        scorer = scorer.drop(['index'], axis=1)
        csv_out['scorer'] = scorer.values.astype(int)
        csv_df.index = csv_out.index
        csv_out.columns = ["Ear_left_1_x", "Ear_left_1_y", "Ear_right_1_x", "Ear_right_1_y", "Nose_1_x", "Nose_1_y",
                           "Center_1_x", "Center_1_y", "Lat_left_1_x", "Lat_left_1_y", "Lat_right_1_x", "Lat_right_1_y",
                           "Tail_base_1_x", "Tail_base_1_y", "Ear_left_2_x", "Ear_left_2_y", "Ear_right_2_x",
                           "Ear_right_2_y", "Nose_2_x", "Nose_2_y", "Center_2_x", "Center_2_y", "Lat_left_2_x",
                           "Lat_left_2_y", "Lat_right_2_x", "Lat_right_2_y", "Tail_base_2_x", "Tail_base_2_y", "scorer"]
        csv_out[['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p', 'Tail_base_1_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p', 'Lat_left_2_p', 'Lat_right_2_p', 'Tail_base_2_p']] = csv_df[['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p','Tail_base_1_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p', 'Lat_left_2_p','Lat_right_2_p', 'Tail_base_2_p']]
        csv_out = csv_out[["scorer", "Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                           "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y",
                           "Center_1_p", "Lat_left_1_x", "Lat_left_1_y", "Lat_left_1_p", "Lat_right_1_x",
                           "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x", "Tail_base_1_y", "Tail_base_1_p", "Ear_left_2_x", "Ear_left_2_y",
                           "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p", "Nose_2_x", "Nose_2_y",
                           "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y",
                           "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                           "Tail_base_2_y", "Tail_base_2_p"]]
        df_headers = pd.read_csv(currentFile, nrows=0)
        csv_out.columns = df_headers.columns
        csv_out = pd.concat([df_headers, csv_out])
        csv_out = csv_out.reset_index()
        csv_out = csv_out.drop('index', axis=1)
        fname = os.path.basename(currentFile)
        fnamePrint = fname.replace('.csv', '')
        csvOutPath = os.path.join(csv_dir_out, fname)
        csv_out.to_csv(csvOutPath, index=False)
        frames_processed = len(comb_out_array)
        frames_processed_list.append(frames_processed)
        fixedPositions_M1_list.append(fixedPositions_M1)
        fixedPositions_M2_list.append(fixedPositions_M2)
        counts_total_M1_list.append(counts_total_M1)
        counts_total_M2_list.append(counts_total_M2)
        counts_total_M1_np = np.array(counts_total_M1_list)
        counts_total_M2_np = np.array(counts_total_M2_list)
        percentBDcorrected = round((fixedPositions_M1 + fixedPositions_M2) / (frames_processed * 14), 6)

        print(str(fnamePrint) + '. Tot frames: ' + str(frames_processed) + '. Outliers animal 1: ' + str(fixedPositions_M1) + '. Outliers animal 2: ' + str(fixedPositions_M2) + '. % outliers: ' + str(round(percentBDcorrected, 3)))

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
    log_df.to_csv(log_fn, index=False)
    print('Log for corrected "location outliers" saved in project_folder/logs')