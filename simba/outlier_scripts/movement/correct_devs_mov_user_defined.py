import pandas as pd
import os
import numpy as np
import statistics
import math
from configparser import ConfigParser, NoSectionError, NoOptionError
from datetime import datetime
from simba.drop_bp_cords import *
import glob
from simba.rw_dfs import *
from simba.drop_bp_cords import *


def dev_move_user_defined(configini):
    dateTime, loop = datetime.now().strftime('%Y%m%d%H%M%S'), 0
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    try:
        criterion = config.getfloat('Outlier settings', 'movement_criterion')
    except:
        print('No movement criterion found in the project_config.ini file')
    animal_no = config.getint('General settings', 'animal_no')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'


    animalIDlist = config.get('Multi animal IDs', 'id_list')

    if not animalIDlist:
        animalIDlist = []
        for animal in range(animal_no):
            animalIDlist.append('Animal_' + str(animal + 1))
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    else:
        animalIDlist = animalIDlist.split(",")
        multiAnimalStatus = True
        print('Applying settings for multi-animal tracking...')

    bodyPartNameArrayForMeans = np.empty((animal_no, 5), dtype=object)
    for animal in range(animal_no):
        bodyPart_1_Name, bodyPart_2_Name = 'movement_bodypart1_' + str(animalIDlist[animal]), 'movement_bodypart2_' + str(animalIDlist[animal])
        bodyPart1, bodyPart2 = config.get('Outlier settings', bodyPart_1_Name), config.get('Outlier settings', bodyPart_2_Name)
        bodyPart_1_X, bodyPart_1_Y, bodyPart_2_X, bodyPart_2_Y = bodyPart1 + '_x', bodyPart1 + '_y', bodyPart2 + '_x', bodyPart2 + '_y'
        bodyPartNameArrayForMeans[animal] = [animalIDlist[animal], bodyPart_1_X, bodyPart_1_Y, bodyPart_2_X, bodyPart_2_Y]

    x_cols, y_cols, p_cols = getBpNames(configFile)
    bp_names =  list(map(lambda x: x.replace('_x',''),x_cols))
    colHeads = getBpHeaders(configFile)

    columnHeadersShifted = [s + '_shifted' for s in colHeads]

    #### CREATE DICT TO HOLD ANIMAL BPS AND NAMES
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, animalIDlist, animal_no, x_cols, y_cols, p_cols, [])
    csv_dir_in, csv_dir_out, log_fn = os.path.join(projectPath, 'csv', 'input_csv'), os.path.join(projectPath, 'csv', 'outlier_corrected_movement'), os.path.join(projectPath, 'logs', 'Outliers_movement_' + str(dateTime) + '.csv')

    def add_correction_prefix(col, bpcorrected_list):
        colc = 'Corrected_' + col
        bpcorrected_list.append(colc)
        return bpcorrected_list

    def correct_value_position(df, colx, coly, col_corr_x, col_corr_y, dict_pos):
        dict_pos[colx] = dict_pos.get(colx, 0)
        dict_pos[coly] = dict_pos.get(coly, 0)
        currentCriterion = meanSize * criterion
        list_x = []
        list_y = []
        prev_x = df.iloc[0][colx]
        prev_y = df.iloc[0][coly]
        ntimes = 0
        live_prevx = df.iloc[0][colx]
        live_prevy = df.iloc[0][coly]
        NT = 12
        for index, row in df.iterrows():
            if index == 0:
                list_x.append(row[colx]), list_y.append(row[coly])
                continue
            if ((math.hypot(row[colx] - prev_x, row[coly] - prev_y) < currentCriterion) or (ntimes > NT and  math.hypot(row[colx] - live_prevx, row[coly] - live_prevy) < currentCriterion)):
                list_x.append(row[colx])
                list_y.append(row[coly])
                prev_x = row[colx]
                prev_y = row[coly]
                ntimes = 0
            else:
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

    filesFound = glob.glob(csv_dir_in + '/*.' +wfileType)
    print('Processing ' + str(len(filesFound)) + ' files for movement outliers...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    logDfColumns = ['Video', 'Frames processed']
    logDfColumns.extend(bp_names)
    logDfColumns.append(str('% corrected'))
    log_df = pd.DataFrame(columns=logDfColumns)
    list_dict_count_corrections = {}
    for currentFile in filesFound:
        list_dict_count_corrections[currentFile] = {}
        baseNameFile = os.path.basename(currentFile).replace('.' + wfileType, '')
        csv_df = read_df(currentFile, wfileType)
        try:
            csv_df = csv_df.set_index('scorer')
        except KeyError:
            pass

        csv_df.columns = colHeads
        try:
            csv_df = csv_df.apply(pd.to_numeric)
        except ValueError:
            csv_df = csv_df.iloc[2:]
            csv_df.reset_index()
            csv_df = csv_df.apply(pd.to_numeric)

    ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = columnHeadersShifted
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        df_p_cols = pd.DataFrame([csv_df.pop(x) for x in p_cols]).T
        csv_out = pd.DataFrame()

        ########### MEAN MOUSE SIZES ###########################################
        dict_pos = {}
        for animal in range(len(bodyPartNameArrayForMeans)):
            print('Processing animal ' + str(animal + 1) + ' in video ' + str(loop + 1)  + ' / ' + str(len(filesFound)) + '...')
            meanSize = statistics.mean(np.sqrt((csv_df[bodyPartNameArrayForMeans[animal][1]] - csv_df[bodyPartNameArrayForMeans[animal][3]]) ** 2 + (csv_df[bodyPartNameArrayForMeans[animal][2]] - csv_df[bodyPartNameArrayForMeans[animal][4]]) ** 2))
            bplist1x, bplist1y, bpcorrected_list1x, bpcorrected_list1y = [], [], [], []
            currXcols, currYcols, currPcols = animalBpDict[animalIDlist[animal]]['X_bps'], animalBpDict[animalIDlist[animal]]['Y_bps'], animalBpDict[animalIDlist[animal]]['P_bps']

            for bp in currXcols:
                bplist1x.append(bp)
                bpcorrected_list1x = add_correction_prefix(bp, bpcorrected_list1x)
            for bp in currYcols:
                bplist1y.append(bp)
                bpcorrected_list1y = add_correction_prefix(bp, bpcorrected_list1y)

            for idx, col1x in enumerate(bplist1x):
                col1y = bplist1y[idx]
                col_corr_1x = bpcorrected_list1x[idx]
                col_corr_1y = bpcorrected_list1y[idx]
                csv_df_combined = correct_value_position(csv_df_combined, col1x, col1y, col_corr_1x, col_corr_1y, dict_pos)
            csv_df_combined.reset_index()

            for cols in range(len(currXcols)):
                csv_out = pd.concat([csv_out, csv_df_combined[bpcorrected_list1x[cols]], csv_df_combined[bpcorrected_list1y[cols]], df_p_cols[currPcols[cols]]], sort=False, axis=1)

        csv_out = csv_out.rename_axis('scorer')
        csv_out.columns = csv_out.columns.str.replace('Corrected_', '')

        fileOut = str(baseNameFile) + str('.csv')
        pathOut = os.path.join(csv_dir_out, fileOut)
        csv_out.to_csv(pathOut)
        fixed_M1_pos, currentFixedList = [], []
        currentFixedList.append(baseNameFile)
        currentFixedList.append(len(csv_out))
        for k in list(dict_pos):
            if k.endswith('_x'):
                del dict_pos[k]
        for y in list(dict_pos):
            fixed_M1_pos.append(dict_pos[y])
        currentFixedList.extend(fixed_M1_pos)
        percentCorrected = round(sum(fixed_M1_pos) / (len(csv_out) * len(x_cols)) * 100, 3)
        currentFixedList.append(percentCorrected)
        log_df.loc[loop] = currentFixedList
        loop = loop + 1
        print(str(baseNameFile) + ' movement outlier correction complete. Tot frames: '+ str(len(csv_out)) + '. Outliers detected: ' + str(sum(fixed_M1_pos)) + '. % corrected: ' + str(percentCorrected) + '.')
    log_df.to_csv(log_fn, index=False)
    print('Log for corrected "movement outliers" saved in project_folder/logs.')