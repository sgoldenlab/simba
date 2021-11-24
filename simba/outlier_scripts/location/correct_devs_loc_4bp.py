import pandas as pd
import os
import statistics
import numpy as np
from configparser import ConfigParser
from datetime import datetime
from simba.drop_bp_cords import *


def dev_loc_4(projectini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    configFile = str(projectini)
    config = ConfigParser()
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement')
    csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    filesFound = []
    vNm_list = []
    fixedPositions_M1_list = []
    frames_processed_list = []
    counts_total_M1_list = []
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
        counts_total_M1 = [0] * 4
        outputArray = np.array([0] * 8)
        loopy += 1
        currentFile = i
        videoFileBaseName = os.path.basename(currentFile).replace('.csv', '')

        csv_df = pd.read_csv(currentFile, header=0,
                             names=["Ear_left_x", "Ear_left_y", "Ear_left_p", "Ear_right_x", "Ear_right_y",
                                    "Ear_right_p", "Nose_x", "Nose_y", "Nose_p", "Tail_base_x",
                                    "Tail_base_y", "Tail_base_p"])
        csv_df = csv_df.apply(pd.to_numeric)

        vNm_list.append(videoFileBaseName)

        ########### MEAN MOUSE SIZES ###########################################
        csv_df['Mouse_1_nose_to_tail'] = np.sqrt(
            (csv_df.Nose_1_x - csv_df.Tail_base_1_x) ** 2 + (csv_df.Nose_1_y - csv_df.Tail_base_1_y) ** 2)
        mean1size = (statistics.mean(csv_df['Mouse_1_nose_to_tail']))

        mean1size = mean1size * criterion

        for index, row in csv_df.iterrows():
            currentArray = np.array(
                [[row['Ear_left_x'], row["Ear_left_y"]], [row['Ear_right_x'], row["Ear_right_y"]],
                 [row['Nose_x'], row["Nose_y"]], [row['Tail_base_x'], row["Tail_base_y"]]]).astype(int)
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

        csv_out = pd.DataFrame(outputArray)
        scorer = pd.read_csv(currentFile).scorer
        scorer = pd.to_numeric(scorer)
        scorer = scorer.reset_index()
        scorer = scorer.drop(['index'], axis=1)
        csv_out['scorer'] = scorer.values.astype(int)
        csv_df.index = csv_out.index
        csv_out.columns = ["Ear_left_1_x", "Ear_left_1_y", "Ear_right_1_x", "Ear_right_1_y", "Nose_1_x", "Nose_1_y",
                           "Tail_base_1_x", "Tail_base_1_y", "scorer"]
        csv_out[['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', "Tail_base_1_p"]] = csv_df[
            ['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', "Tail_base_1_p"]]
        csv_out = csv_out[["scorer", "Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                           "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Tail_base_1_x", "Tail_base_1_y", "Tail_base_1_p"]]
        df_headers = pd.read_csv(currentFile, nrows=0)
        csv_out.columns = df_headers.columns
        csv_out = pd.concat([df_headers, csv_out])
        csv_out = csv_out.reset_index()
        csv_out = csv_out.drop('index', axis=1)
        fname = os.path.basename(currentFile)
        fnamePrint = fname.replace('.csv', '')
        csvOutPath = os.path.join(csv_dir_out, fname)
        csv_out.to_csv(csvOutPath, index=False)

        frames_processed = len(csv_out)
        frames_processed_list.append(frames_processed)
        fixedPositions_M1_list.append(fixedPositions_M1)
        counts_total_M1_list.append(counts_total_M1)
        counts_total_M1_np = np.array(counts_total_M1_list)
        percentBDcorrected = round((fixedPositions_M1) / (frames_processed * 4), 10)

        print(str(fnamePrint) + '. Tot frames: ' + str(frames_processed) + '. Outliers animal 1: ' + str(fixedPositions_M1) + '. % outliers: ' + str(round(percentBDcorrected, 10)))

    log_df['Video'] = vNm_list
    log_df['Frames_processed'] = frames_processed_list
    log_df['Animal1_left_ear'] = counts_total_M1_np[:, 0]
    log_df['Animal1_right_ear'] = counts_total_M1_np[:, 1]
    log_df['Animal1_nose'] = counts_total_M1_np[:, 2]
    log_df['Animal1_tail_base'] = counts_total_M1_np[:, 3]
    log_df['Sum'] =  log_df['Animal1_left_ear'] + log_df['Animal1_right_ear'] + log_df['Animal1_nose'] + log_df['Animal1_tail_base']
    log_df['% body parts corrected'] = log_df['Sum'] / (log_df['Frames_processed'] * 7)
    log_df.to_csv(log_fn, index=False)
    print('Log for corrected "location outliers" saved in project_folder/logs')