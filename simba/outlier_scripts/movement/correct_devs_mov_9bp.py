import os
import pandas as pd
import math
import numpy as np
import statistics
from configparser import ConfigParser
from datetime import datetime
from simba.drop_bp_cords import *

def dev_move_9(configini):
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
    csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    headers = ['Video', "frames_processed", 'Left_ear', "Right_ear", "Left_hand", "Right_hand", "Left_foot",
               "Right_foot", "Nose", "Tail", "Back", "Sum"]


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
                             names=["Mouse1_left_ear_x", "Mouse1_left_ear_y", "Mouse1_left_ear_p", "Mouse1_right_ear_x",
                                    "Mouse1_right_ear_y", "Mouse1_right_ear_p", "Mouse1_left_hand_x",
                                    "Mouse1_left_hand_y", "Mouse1_left_hand_p", \
                                    "Mouse1_right_hand_x", "Mouse1_right_hand_y", "Mouse1_right_hand_p",
                                    "Mouse1_left_foot_x", "Mouse1_left_foot_y", "Mouse1_left_foot_p",
                                    "Mouse1_right_foot_x", "Mouse1_right_foot_y", "Mouse1_right_foot_p",
                                    "Mouse1_nose_x", "Mouse1_nose_y", "Mouse1_nose_p", "Mouse1_tail_x", "Mouse1_tail_y",
                                    "Mouse1_tail_p", "Mouse1_back_x", "Mouse1_back_y", "Mouse1_back_p"], low_memory=False)

        csv_df = csv_df.drop(csv_df.index[[0, 1, 2]])
        csv_df = csv_df.apply(pd.to_numeric)
        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted = csv_df_shifted.rename(columns={'Mouse1_left_ear_x': 'Mouse1_left_ear_x_shifted', 'Mouse1_left_ear_y': 'Mouse1_left_ear_y_shifted',
                     'Mouse1_left_ear_p': 'Mouse1_left_ear_p_shifted', 'Mouse1_right_ear_x': 'Mouse1_right_ear_x_shifted', \
                     'Mouse1_right_ear_y': 'Mouse1_right_ear_y_shifted', 'Mouse1_right_ear_p': 'Mouse1_right_ear_p_shifted',
                     'Mouse1_left_hand_x': 'Mouse1_left_hand_x_shifted', 'Mouse1_left_hand_y': 'Mouse1_left_hand_y_shifted', \
                     'Mouse1_left_hand_p': 'Mouse1_left_hand_p_shifted', 'Mouse1_right_hand_x': 'Mouse1_right_hand_x_shifted',
                     'Mouse1_right_hand_y': 'Mouse1_right_hand_y_shifted', 'Mouse1_right_hand_p': 'Mouse1_right_hand_p_shifted', 'Mouse1_left_foot_x': \
                     'Mouse1_left_foot_x_shifted', 'Mouse1_left_foot_y': 'Mouse1_left_foot_y_shifted',
                     'Mouse1_left_foot_p': 'Mouse1_left_foot_p_shifted', 'Mouse1_right_foot_x': 'Mouse1_right_foot_x_shifted',
                     'Mouse1_right_foot_y': 'Mouse1_right_foot_y_shifted', \
                     'Mouse1_right_foot_p': 'Mouse1_right_foot_p_shifted', 'Mouse1_nose_x': 'Mouse1_nose_x_shifted',
                     'Mouse1_nose_y': 'Mouse1_nose_y_shifted', 'Mouse1_nose_p': 'Mouse1_nose_p_shifted', 'Mouse1_tail_x': 'Mouse1_tail_x_shifted',
                     'Mouse1_tail_y': 'Mouse1_tail_y_shifted', 'Mouse1_tail_p': 'Mouse1_tail_p_shifted',
                     'Mouse1_back_x': 'Mouse1_back_x_shifted', 'Mouse1_back_y': 'Mouse1_back_y_shifted',
                     'Mouse1_back_p': 'Mouse1_back_p_shifted'})
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')

        ########### EUCLIDEAN DISTANCES ###########################################
        csv_df_combined['Mouse_nose_to_tail'] = np.sqrt((csv_df_combined.Mouse1_nose_x - csv_df_combined.Mouse1_tail_x) ** 2 + (csv_df_combined.Mouse1_nose_y - csv_df_combined.Mouse1_tail_y) ** 2)
        csv_df_combined = csv_df_combined.fillna(0)

        ########### MEAN MOUSE SIZES ###########################################
        mean1size = (statistics.mean(csv_df_combined['Mouse_nose_to_tail']))

        bps = ['Mouse1_left_ear', 'Mouse1_right_ear', 'Mouse1_left_hand', 'Mouse1_right_hand', 'Mouse1_left_foot',
               'Mouse1_tail', 'Mouse1_right_foot', 'Mouse1_back', 'Mouse1_nose']
        bplist1x = []
        bplist1y = []
        bpcorrected_list1x = []
        bpcorrected_list1y = []

        for bp in bps:
            colx = bp + '_x'
            coly = bp + '_y'
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
        print(csv_df_combined)
        csv_df_combined = csv_df_combined[
            ["scorer", "Corrected_Mouse1_left_ear_x", "Corrected_Mouse1_left_ear_y", "Mouse1_left_ear_p",
             "Corrected_Mouse1_right_ear_x", "Corrected_Mouse1_right_ear_y", "Mouse1_right_ear_p",
             "Corrected_Mouse1_left_hand_x", "Corrected_Mouse1_left_hand_y", "Mouse1_left_hand_p", "Corrected_Mouse1_right_hand_x",
             "Corrected_Mouse1_right_hand_y", "Mouse1_right_hand_p", "Corrected_Mouse1_left_foot_x", "Corrected_Mouse1_left_foot_y", "Mouse1_left_foot_p",
             "Corrected_Mouse1_right_foot_x", "Corrected_Mouse1_right_foot_y", "Mouse1_right_foot_p",
             "Corrected_Mouse1_nose_x", "Corrected_Mouse1_nose_y", "Mouse1_nose_p", "Corrected_Mouse1_tail_x",
             "Corrected_Mouse1_tail_y", "Mouse1_tail_p", "Corrected_Mouse1_back_x", "Corrected_Mouse1_back_y",
             "Mouse1_back_p"]]
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
            if y.__contains__('1'):
                fixed_M1_pos.append(dict_pos[y])
        currentFixedList.extend(fixed_M1_pos)
        print(currentFixedList)
        totalfixed = sum(fixed_M1_pos)
        currentFixedList.append(totalfixed)
        log_df.loc[loop] = currentFixedList
        loop = loop + 1
        print(str(baseNameFile) + '. Tot frames: ' + str(framesProcessed) + '. Outliers animal 1: ' + str(sum(fixed_M1_pos)) + '. % outliers: ' + str(round(totalfixed / (framesProcessed * 7), 10)) + '.')

    log_df['% body parts corrected'] = log_df['Sum'] / (log_df['frames_processed'] * 7)
    log_df['Video'] = log_df['Video'].apply(str)
    log_df.to_csv(log_fn, index=False)
    print(log_fn)
    print('Log for corrected "movement outliers" saved in project_folder/logs')
