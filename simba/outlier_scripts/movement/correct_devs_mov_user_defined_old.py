import pandas as pd
import os
import numpy as np
import statistics
import math
from configparser import ConfigParser, NoOptionError
from datetime import datetime
import glob
from simba.rw_dfs import *


def dev_move_user_defined(configini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    configFile = str(configini)
    config = ConfigParser()
    config.read(configFile)
    loop, loopy = 0, 0
    criterion = config.getfloat('Outlier settings', 'movement_criterion')
    bodyPart1 = config.get('Outlier settings', 'location_bodypart1_mouse1')
    bodyPart2 = config.get('Outlier settings', 'location_bodypart2_mouse1')
    bodyPart1x, bodyPart1y = (bodyPart1 + '_x', bodyPart1 + '_y')
    bodyPart2x, bodyPart2y = (bodyPart2 + '_x', bodyPart2 + '_y')
    projectPath = config.get('General settings', 'project_path')
    currentBodyPartFile = os.path.join(projectPath, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    bodyPartsFile = pd.read_csv(os.path.join(currentBodyPartFile), header=None)
    bodyPartsList = list(bodyPartsFile[0])
    bodyPartHeaders = []
    columnHeadersShifted = []
    xy_headers = []
    p_cols, x_cols, y_cols = [], [], []
    for i in bodyPartsList:
        col1, col2, col3 = (str(i) + '_x', str(i) + '_y', str(i) + '_p')
        col4, col5, col6 = (col1 + '_x_shifted', col2 + '_y_shifted', col3 + '_p_shifted')
        columnHeadersShifted.extend((col4, col5, col6))
        p_cols.append(col3)
        x_cols.append(col1)
        y_cols.append(col2)
        bodyPartHeaders.extend((col1, col2, col3))
        xy_headers.extend((col1, col2))
    csv_dir_in = os.path.join(projectPath, 'csv', 'input_csv')
    csv_dir_out = os.path.join(projectPath, 'csv', 'outlier_corrected_movement')

    ########### logfile path ###########
    log_fn = os.path.join(projectPath, 'logs', 'Outliers_movement_' + str(dateTime) + '.csv')

    def add_correction_prefix(col, bpcorrected_list):
        colc = 'Corrected_' + col
        bpcorrected_list.append(colc)
        return bpcorrected_list

    def correct_value_position(df, colx, coly, col_corr_x, col_corr_y, dict_pos):
        dict_pos[colx] = dict_pos.get(colx, 0)
        dict_pos[coly] = dict_pos.get(coly, 0)
        currentCriterion = mean1size * criterion
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
                continue
            if (math.hypot(row[colx] - prev_x, row[coly] - prev_y) < (mean1size / 4)):  # the mouse is standing still
                currentCriterion = mean1size * 2
            if ((math.hypot(row[colx] - prev_x, row[coly] - prev_y) < currentCriterion) or (ntimes > NT and  math.hypot( row[colx] - live_prevx,row[coly] - live_prevy) < currentCriterion)):
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

    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Processing ' + str(len(filesFound)) + ' files for movement outliers...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    logDfColumns = ['Video', 'Frames processed']
    logDfColumns.extend(bodyPartsList)
    logDfColumns.append(str('% corrected'))
    log_df = pd.DataFrame(columns=logDfColumns)
    for currentFile in filesFound:
        loopy += 1
        baseNameFile = os.path.basename(currentFile).replace('.' + wfileType, '')
        csv_df = read_df(currentFile, wfileType)
        csv_df.columns = bodyPartHeaders
        csv_df = csv_df.drop(csv_df.index[[0, 1, 2]])
        csv_df = csv_df.apply(pd.to_numeric)
    ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = columnHeadersShifted
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        df_p_cols = pd.DataFrame([csv_df.pop(x) for x in p_cols]).T
        df_p_cols = df_p_cols.reset_index()
    ########### MEAN MOUSE SIZES ###########################################
        csv_df['Reference_value'] = np.sqrt((csv_df[bodyPart1x] - csv_df[bodyPart2x]) ** 2 + (csv_df[bodyPart1y] - csv_df[bodyPart2y]) ** 2)
        mean1size = (statistics.mean(csv_df['Reference_value']))
        mean1size = mean1size * criterion
        bplist1x = []
        bplist1y = []
        bpcorrected_list1x = []
        bpcorrected_list1y = []
        for bp in x_cols:
            bplist1x.append(bp)
            bpcorrected_list1x = add_correction_prefix(bp, bpcorrected_list1x)
        for bp in y_cols:
            bplist1y.append(bp)
            bpcorrected_list1y = add_correction_prefix(bp, bpcorrected_list1y)
        dict_pos = {}
        for idx, col1x in enumerate(bplist1x):
            col1y = bplist1y[idx]
            col_corr_1x = bpcorrected_list1x[idx]
            col_corr_1y = bpcorrected_list1y[idx]
            csv_df_combined = correct_value_position(csv_df_combined, col1x, col1y, col_corr_1x, col_corr_1y, dict_pos)
        csv_df_combined = csv_df_combined.reset_index()
        csv_out = pd.DataFrame()
        for cols in range(len(x_cols)):
            csv_out = pd.concat([csv_out, csv_df_combined[x_cols[cols]], csv_df_combined[y_cols[cols]], df_p_cols[p_cols[cols]]], axis=1)
        csv_out = csv_out.rename_axis('scorer')
        fileOut = str(baseNameFile) + str('.') + wfileType
        pathOut = os.path.join(csv_dir_out, fileOut)
        save_df(csv_out, wfileType, pathOut)
        print(pathOut)
        fixed_M1_pos = []
        currentFixedList = []
        currentFixedList.append(baseNameFile)
        currentFixedList.append(len(csv_out))
        for k in list(dict_pos):
            if k.endswith('_x'):
                del dict_pos[k]
        for y in list(dict_pos):
            fixed_M1_pos.append(dict_pos[y])
        currentFixedList.extend(fixed_M1_pos)
        percentCorrected = round(sum(fixed_M1_pos) / (len(csv_out) * len(bodyPartsList)) * 100, 3)
        currentFixedList.append(percentCorrected)
        log_df.loc[loop] = currentFixedList
        loop = loop + 1
        print(str(baseNameFile) + '. Tot frames: '+ str(len(csv_out)) + '. Outliers: ' + str(sum(fixed_M1_pos)) + '. % corrected: ' + str(percentCorrected))
    log_df.to_csv(log_fn, index=False)
    print('Log for corrected "movement outliers" saved in project_folder/logs')