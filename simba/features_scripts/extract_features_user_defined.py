from __future__ import division
import os
import pandas as pd
import numpy as np
from configparser import ConfigParser
import glob

def extract_features_wotarget_user_defined(inifile):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(csv_dir, 'features_extracted')
    vidInfPath = config.get('General settings', 'project_path')
    logsPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(logsPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    poseConfigPath = os.path.join(logsPath, 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseConfigDf = pd.read_csv(poseConfigPath, header=None)
    poseConfigDf = list(poseConfigDf[0])

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)

    def count_values_in_range(series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    roll_windows = []
    roll_windows_values = [2, 5, 6, 7.5, 15]
    loopy = 0

    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Extracting features from ' + str(len(filesFound)) + ' files...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for i in filesFound:
        currentFile = i
        currVidName = os.path.basename(currentFile.replace('.csv', ''))
        currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == currVidName]
        try:
            currPixPerMM = float(currVideoSettings['pixels/mm'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        fps = float(currVideoSettings['fps'])
        print('Processing ' + '"' + str(currVidName) + '".' + ' Fps: ' + str(fps) + ". mm/ppx: " + str(currPixPerMM))
        for i in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[i]))
        loopy += 1
        bodypartNames = list(poseConfigDf)
        columnHeaders = []
        columnHeadersShifted = []
        p_cols = []
        for bodypart in bodypartNames:
            colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
            colHead4, colHead5, colHead6 = (bodypart + '_x_shifted', bodypart + '_y_shifted', bodypart + '_p_shifted')
            columnHeaders.extend((colHead1, colHead2, colHead3))
            columnHeadersShifted.extend((colHead4, colHead5, colHead6))
            p_cols.append(colHead3)

        csv_df = pd.read_csv(currentFile, names=columnHeaders, low_memory=False)
        csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        csv_df = csv_df.reset_index(drop=True)

        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = columnHeadersShifted
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)
        print('Calculating euclidean distances...')

        ########### EUCLIDEAN DISTANCES BETWEEN BODY PARTS###########################################
        distanceColNames = []
        for idx in range(len(bodypartNames)-1):
            for idy in range(idx+1, len(bodypartNames)):
                colName = 'distance_' + str(bodypartNames[idx]) + '_to_' + str(bodypartNames[idy])
                distanceColNames.append(colName)
                firstBpX , firstBpY = (bodypartNames[idx] + '_x', bodypartNames[idx] + '_y')
                secondBpX, secondBpY = (bodypartNames[idy] + '_x', bodypartNames[idy] + '_y')
                csv_df[colName] = (np.sqrt((csv_df[firstBpX] - csv_df[secondBpX]) ** 2 + (csv_df[firstBpY]- csv_df[secondBpY]) ** 2)) / currPixPerMM

        ########### MOVEMENTS OF ALL BODY PARTS ###########################################
        movementColNames = []
        for selectBp in bodypartNames:
            colName = 'movement_' + selectBp
            movementColNames.append(colName)
            selectBpX_1, selectBpY_1 = (selectBp + '_x', selectBp + '_y')
            selectBpX_2, selectBpY_2 = (selectBp + '_x_shifted', selectBp + '_y_shifted')
            csv_df[colName] = (np.sqrt((csv_df_combined[selectBpX_1] - csv_df_combined[selectBpX_2]) ** 2 + (csv_df_combined[selectBpY_1] - csv_df_combined[selectBpY_2]) ** 2)) / currPixPerMM
        movementDf = csv_df.filter(movementColNames, axis=1)
        descriptiveColNames = ['collapsed_sum_of_all_movements', 'collapsed_mean_of_all_movements', 'collapsed_median_of_all_movements', 'collapsed_min_of_all_movements', 'collapsed_max_of_all_movements']
        csv_df['collapsed_sum_of_all_movements'] = movementDf[movementColNames].sum(axis=1)
        csv_df['collapsed_mean_of_all_movements'] = movementDf[movementColNames].mean(axis=1)
        csv_df['collapsed_median_of_all_movements'] = movementDf[movementColNames].median(axis=1)
        csv_df['collapsed_min_of_all_movements'] = movementDf[movementColNames].min(axis=1)
        csv_df['collapsed_max_of_all_movements'] = movementDf[movementColNames].max(axis=1)

        print('Calculating rolling windows data...')

        ########### CALC MEAN, MEDIAN, AND SUM DISTANCES BETWEEN BODY PARTS IN ROLLING WINDOWS ###########################################
        combinedLists_1 = distanceColNames + movementColNames + descriptiveColNames
        for i in range(len(roll_windows_values)):
            for selectedCol in combinedLists_1:
                colName = 'Mean_' + str(selectedCol) + '_' + str(roll_windows_values[i])
                csv_df[colName] = csv_df[selectedCol].rolling(roll_windows[i], min_periods=1).mean()
                colName = 'Sum_' + str(selectedCol) + '_' + str(roll_windows_values[i])
                csv_df[colName] = csv_df[selectedCol].rolling(roll_windows[i], min_periods=1).sum()

        print('Calculating body part movements...')
        ########### BODY PART MOVEMENTS RELATIVE TO EACH OTHER ###########################################
        movementDiffcols = []
        for idx in range(len(movementColNames)-1):
            for idy in range(idx+1, len(movementColNames)):
                colName = 'Movement_difference_' + movementColNames[idx] + '_' + movementColNames[idy]
                movementDiffcols.append(colName)
                csv_df[colName] = abs(csv_df[movementColNames[idx]]-csv_df[movementColNames[idy]])
                movementDiffcols.append(colName)
                csv_df[colName] = abs(csv_df[movementColNames[idx]]-csv_df[movementColNames[idy]])

        print('Calculating deviations and rank...')

        ########### DEVIATIONS FROM MEAN ###########################################
        combinedLists_2 = combinedLists_1 + movementDiffcols
        for column in combinedLists_2:
            colName = str('Deviation_from_median_') + column
            csv_df[colName] = csv_df[column].mean() - csv_df[column]

        ########### PERCENTILE RANK ###########################################
        combinedLists_2 = combinedLists_1 + movementDiffcols
        for column in combinedLists_2:
            colName = 'Rank_' + column
            csv_df[colName] = csv_df[column].rank(pct=True)

        ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
        print('Calculating pose probability scores...')
        probabilityDf = csv_df.filter(p_cols, axis=1)
        csv_df['Sum_probabilities'] = probabilityDf.sum()
        csv_df['Mean_probabilities'] = probabilityDf.mean()
        values_in_range_min, values_in_range_max = 0.0, 0.1
        csv_df["Low_prob_detections_0.1"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.5
        csv_df["Low_prob_detections_0.5"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.75
        csv_df["Low_prob_detections_0.75"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)

        ########### SAVE DF ###########################################
        #csv_df = csv_df.loc[:, ~csv_df.T.duplicated(keep='first')]
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        #csv_df = csv_df.drop(columns=['index'])
        fileOutName = os.path.basename(currentFile)
        savePath = os.path.join(csv_dir_out, fileOutName)
        csv_df.to_csv(savePath)
        print('Feature extraction complete for ' + '"' + str(currVidName) + '".')

    print('All feature extraction complete.')