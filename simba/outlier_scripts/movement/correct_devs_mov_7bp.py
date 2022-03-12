import os
import pandas as pd
import math
import numpy as np
import statistics
from configparser import ConfigParser
from datetime import datetime
from simba.drop_bp_cords import *

def dev_move_7(configini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    filesFound = []
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    loop = 0
    loopy = 0
    criterion = config.getfloat('Outlier settings', 'movement_criterion')
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'input_csv')
    csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement')
    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    headers = ['Video', "frames_processed", 'Animal1_centroid', "Animal1_left_ear", "Animal1_right_ear", "Animal1_lateral_left", "Animal1_lateral_right",
               "Animal1_nose", "Animal1_tail_base", "Sum"]
    log_df = pd.DataFrame(columns=headers)

    ########### logfile path ###########
    log_fn = 'Outliers_movement_' + str(dateTime) + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    if not os.path.exists(log_path):
        os.makedirs(log_path)


    def add_correction_prefix(col, bpcorrected_list):
        colc = 'Corrected_' + col
        bpcorrected_list.append(colc)
        return bpcorrected_list

    def correct_value_position(df, colx, coly, col_corr_x, col_corr_y, dict_pos):

        dict_pos[colx] = dict_pos.get(colx, 0)
        dict_pos[coly] = dict_pos.get(coly, 0)

        animalSize = mean1size

        currentCriterion = mean1size * criterion
        list_x = []
        list_y = []
        prev_x = df.iloc[0][colx]
        prev_y = df.iloc[0][coly]
        ntimes = 0
        live_prevx = df.iloc[0][colx]
        live_prevy = df.iloc[0][coly]
        NT = 6
        for index, row in df.iterrows():

            if index == 0:
                continue

            if (math.hypot(row[colx] - prev_x, row[coly] - prev_y) < (animalSize/4)): #the mouse is standing still
                currentCriterion = animalSize * 2

            if ((math.hypot(row[colx] - prev_x, row[coly] - prev_y) < currentCriterion) or (ntimes > NT and \
                                          math.hypot(row[colx] - live_prevx, row[coly] - live_prevy) < currentCriterion)):

                list_x.append(row[colx])
                list_y.append(row[coly])

                prev_x = row[colx]
                prev_y = row[coly]

                ntimes = 0

            else:
                #out of range
                list_x.append(prev_x)
                list_y.append(prev_y)
                dict_pos[colx] += 1
                dict_pos[coly] += 1
                ntimes += 1

            live_prevx = row[colx]
            live_prevy = row[coly]

        df[col_corr_x] = list_x
        df[col_corr_y] = list_y

        return df

    ########### FIND CSV FILES ###########
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            file = os.path.join(csv_dir_in, i)
            filesFound.append(file)
    print('Processing ' + str(len(filesFound)) + ' files for movement outliers...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for i in filesFound:
        loopy += 1
        currentFile = i
        baseNameFile = os.path.basename(currentFile).replace('.csv', '')
        csv_df = pd.read_csv(currentFile,
                                 names=["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                                        "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y",
                                        "Center_1_p", "Lat_left_1_x", "Lat_left_1_y",
                                        "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                                        "Tail_base_1_y", "Tail_base_1_p"], low_memory=False)

        csv_df = csv_df.drop(csv_df.index[[0, 1, 2]])
        csv_df = csv_df.apply(pd.to_numeric)
        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted = csv_df_shifted.rename(
            columns={'Ear_left_1_x': 'Ear_left_1_x_shifted', 'Ear_left_1_y': 'Ear_left_1_y_shifted',
                     'Ear_left_1_p': 'Ear_left_1_p_shifted', 'Ear_right_1_x': 'Ear_right_1_x_shifted', \
                     'Ear_right_1_y': 'Ear_right_1_y_shifted', 'Ear_right_1_p': 'Ear_right_1_p_shifted',
                     'Nose_1_x': 'Nose_1_x_shifted', 'Nose_1_y': 'Nose_1_y_shifted', \
                     'Nose_1_p': 'Nose_1_p_shifted', 'Center_1_x': 'Center_1_x_shifted',
                     'Center_1_y': 'Center_1_y_shifted', 'Center_1_p': 'Center_1_p_shifted', 'Lat_left_1_x': \
                         'Lat_left_1_x_shifted', 'Lat_left_1_y': 'Lat_left_1_y_shifted',
                     'Lat_left_1_p': 'Lat_left_1_p_shifted', 'Lat_right_1_x': 'Lat_right_1_x_shifted',
                     'Lat_right_1_y': 'Lat_right_1_y_shifted', \
                     'Lat_right_1_p': 'Lat_right_1_p_shifted', 'Tail_base_1_x': 'Tail_base_1_x_shifted',
                     'Tail_base_1_y': 'Tail_base_1_y_shifted', \
                     'Tail_base_1_p': 'Tail_base_1_p_shifted'})
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')

        ########### EUCLIDEAN DISTANCES ###########################################
        csv_df_combined['Mouse_nose_to_tail'] = np.sqrt((csv_df_combined.Nose_1_x - csv_df_combined.Tail_base_1_x) ** 2 + (csv_df_combined.Nose_1_y - csv_df_combined.Tail_base_1_y) ** 2)
        csv_df_combined = csv_df_combined.fillna(0)

        ########### MEAN MOUSE SIZES ###########################################
        mean1size = (statistics.mean(csv_df_combined['Mouse_nose_to_tail']))

        bps = ['Ear', 'Nose', 'Lat', 'Center', 'Tail_base']
        bplist1x = []
        bplist1y = []
        bpcorrected_list1x = []
        bpcorrected_list1y = []

        for bp in bps:
            # makes the list of body part columns
            if bp in ['Ear', 'Lat']:
                # iterate over left and right
                colx = bp + '_left_1_x'
                coly = bp + '_left_1_y'
                bplist1x.append(colx)
                bplist1y.append(coly)

                bpcorrected_list1x = add_correction_prefix(colx, bpcorrected_list1x)
                bpcorrected_list1y = add_correction_prefix(coly, bpcorrected_list1y)

                colx = bp + '_right_1_x'
                coly = bp + '_right_1_y'

                bplist1x.append(colx)
                bplist1y.append(coly)

                bpcorrected_list1x = add_correction_prefix(colx, bpcorrected_list1x)
                bpcorrected_list1y = add_correction_prefix(coly, bpcorrected_list1y)

            else:
                colx = bp + '_1_x'
                coly = bp + '_1_y'
                bplist1x.append(colx)
                bplist1y.append(coly)
                bpcorrected_list1x = add_correction_prefix(colx, bpcorrected_list1x)
                bpcorrected_list1y = add_correction_prefix(coly, bpcorrected_list1y)

        # this dictionary will count the number of times each body part position needs to be corrected
        dict_pos = {}

        for idx, col1x in enumerate(bplist1x):
            # apply function to all body part data
            col1y = bplist1y[idx]
            col_corr_1x = bpcorrected_list1x[idx]
            col_corr_1y = bpcorrected_list1y[idx]
            csv_df_combined = correct_value_position(csv_df_combined, col1x, col1y, col_corr_1x, col_corr_1y, dict_pos)

        scorer = pd.read_csv(currentFile, low_memory=False).scorer.iloc[2:]
        scorer = pd.to_numeric(scorer)
        scorer = scorer.reset_index()
        scorer = scorer.drop(['index'], axis=1)
        csv_df_combined['scorer'] = scorer.values.astype(int)
        csv_df_combined = csv_df_combined[
            ["scorer", "Corrected_Ear_left_1_x", "Corrected_Ear_left_1_y", "Ear_left_1_p", "Corrected_Ear_right_1_x",
                 "Corrected_Ear_right_1_y", "Ear_right_1_p", "Corrected_Nose_1_x", "Corrected_Nose_1_y", "Nose_1_p",
                 "Corrected_Center_1_x", "Corrected_Center_1_y", "Center_1_p", "Corrected_Lat_left_1_x",
                 "Corrected_Lat_left_1_y", "Lat_left_1_p", "Corrected_Lat_right_1_x", "Corrected_Lat_right_1_y",
                 "Lat_right_1_p", "Corrected_Tail_base_1_x", "Corrected_Tail_base_1_y", "Tail_base_1_p"]]

        # csv_df_combined = csv_df_combined.drop(csv_df_combined.index[0:2])
        df_headers = pd.read_csv(currentFile, nrows=0, low_memory=False)
        csv_df_combined['frames'] = np.arange(len(csv_df_combined))
        framesProcessed = csv_df_combined['frames'].max()
        csv_df_combined = csv_df_combined.drop(['frames'], axis=1)
        try:
            csv_df_combined.columns = df_headers.columns
        except ValueError:
            print('Error: Too many or too few bodyparts. Check that the number of trackined bodyparts matched the input in SimBA')
        csv_df_combined = pd.concat([df_headers, csv_df_combined])
        fileName = os.path.basename(currentFile)
        fileName, fileEnding = fileName.split('.')
        fileOut = str(fileName) + str('.csv')
        pathOut = os.path.join(csv_dir_out, fileOut)
        csv_df_combined.to_csv(pathOut, index=False)

        fixed_M1_pos = []
        currentFixedList = []
        currentFixedList.append(baseNameFile)
        currentFixedList.append(framesProcessed)
        for k in list(dict_pos):
            if k.endswith('_x'):
                del dict_pos[k]
        for y in list(dict_pos):
            if y.__contains__('_1_'):
                fixed_M1_pos.append(dict_pos[y])
        currentFixedList.extend(fixed_M1_pos)
        totalfixed = sum(fixed_M1_pos)
        currentFixedList.append(totalfixed)
        log_df.loc[loop] = currentFixedList
        loop = loop + 1
        print(str(baseNameFile) + '. Tot frames: ' + str(framesProcessed) + '. Outliers animal 1: ' + str(sum(fixed_M1_pos)) + '. % outliers: ' + str(round(totalfixed / (framesProcessed * 7), 10)) + '.')

    log_df['% body parts corrected'] = log_df['Sum'] / (log_df['frames_processed'] * 7)
    log_df['Video'] = log_df['Video'].apply(str)
    log_df.to_csv(log_fn, index=False)
    print('Log for corrected "movement outliers" saved in project_folder/logs')
