from __future__ import division
import os, glob
import numpy as np
from configparser import ConfigParser, NoOptionError, NoSectionError
import glob
from simba.rw_dfs import *
from simba.drop_bp_cords import *
from numba import jit
from simba.features_scripts.unit_tests import *

def extract_features_wotarget_user_defined(inifile):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(csv_dir, 'features_extracted')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    vidInfPath = config.get('General settings', 'project_path')
    noAnimals = config.getint('General settings', 'animal_no')
    logsPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(logsPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)

    #change videos name to str
    vidinfDf.Video = vidinfDf.Video.astype('str')

    Xcols, Ycols, Pcols = getBpNames(inifile)
    columnHeaders = getBpHeaders(inifile)
    columnHeadersShifted = [bp + '_shifted' for bp in columnHeaders]

    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
            print('Applying settings for classical tracking...')

    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, noAnimals, Xcols, Ycols, Pcols, [])

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    roll_windows = []
    roll_windows_values = [2, 5, 6, 7.5, 15]
    loopy = 0

    #REMOVE WINDOWS THAT ARE TOO SMALL
    roll_windows_values = check_minimum_roll_windows(roll_windows_values, vidinfDf['fps'].min())
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Extracting features from ' + str(len(filesFound)) + ' files...')

    def count_values_in_range(series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    @jit(nopython=True, cache=True)
    def EuclidianDistCalc(bp1xVals, bp1yVals, bp2xVals, bp2yVals, currPixPerMM):
        series = (np.sqrt((csv_df[bp1xVals] - csv_df[bp2xVals]) ** 2 + (csv_df[bp1yVals] - csv_df[bp2yVals]) ** 2)) / currPixPerMM
        return series

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for currentFile in filesFound:
        currVidName = os.path.basename(currentFile.replace('.' + wfileType, ''))
        currVideoSettings, currPixPerMM, fps = read_video_info(vidinfDf, currVidName)

        print('Processing ' + '"' + str(currVidName) + '".' + ' Fps: ' + str(fps) + ". mm/ppx: " + str(currPixPerMM))
        for i in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[i]))


        loopy += 1
        csv_df = read_df(currentFile, wfileType)
        try:
            csv_df = csv_df.set_index('scorer')
        except KeyError:
            pass
        csv_df.columns = columnHeaders
        csv_df = csv_df.fillna(0)
        csv_df = csv_df.apply(pd.to_numeric)

        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = columnHeadersShifted
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)

        print('Calculating euclidean distances...')
        ########### EUCLIDEAN DISTANCES BETWEEN BODY PARTS###########################################
        distanceColNames = []
        for currAnimal in animalBpDict:
            currentAnimalX, currentAnimalY = animalBpDict[currAnimal]['X_bps'], animalBpDict[currAnimal]['Y_bps']
            otherAnimals = {i: animalBpDict[i] for i in animalBpDict if i != currAnimal}
            for currBpX, currBpY in zip(currentAnimalX, currentAnimalY):
                for otherAnimal in otherAnimals:
                    otherAnimalBpX, otherAnimalBpY = animalBpDict[otherAnimal]['X_bps'], animalBpDict[otherAnimal]['Y_bps']
                    for otherBpX, otherBpY in zip(otherAnimalBpX, otherAnimalBpY):
                        bpName1, bpName2 =  currBpX.strip('_x'), otherBpX.strip('_x')
                        colName = 'Euclidean_distance_' + bpName1 + '_' + bpName2
                        reverseColName = 'Euclidean_distance_' + bpName2 + '_' + bpName1
                        if not reverseColName in csv_df.columns:
                            csv_df[colName] = (np.sqrt((csv_df[currBpX] - csv_df[otherBpX]) ** 2 + (csv_df[currBpY] - csv_df[otherBpY]) ** 2)) / currPixPerMM
                            distanceColNames.append(colName)

        print('Calculating movements of all bodyparts...')
        collapsedColNamesMean, collapsedColNamesSum = [], []
        for currAnimal in animalBpDict:
            animalCols = []
            currentAnimalX, currentAnimalY = animalBpDict[currAnimal]['X_bps'], animalBpDict[currAnimal]['Y_bps']
            for currBpX, currBpY in zip(currentAnimalX, currentAnimalY):
                shiftedBpX, shiftedBpY = currBpX + '_shifted', currBpY + '_shifted'
                colName = 'Movement_' + currBpX.strip('_x')
                csv_df[colName] = (np.sqrt((csv_df_combined[currBpX] - csv_df_combined[shiftedBpX]) ** 2 + (csv_df_combined[currBpY] - csv_df_combined[shiftedBpY]) ** 2)) / currPixPerMM
                animalCols.append(colName)
            sumColName, meanColName = 'All_bp_movements_' + currAnimal + '_sum', 'All_bp_movements_' + currAnimal + '_mean'
            csv_df[sumColName] = csv_df[animalCols].sum(axis=1)
            csv_df[meanColName] = csv_df[animalCols].mean(axis=1)
            csv_df['All_bp_movements_' + currAnimal + '_min'] = csv_df[animalCols].min(axis=1)
            csv_df['All_bp_movements_' + currAnimal + '_max'] = csv_df[animalCols].max(axis=1)
            collapsedColNamesMean.append(meanColName)
            collapsedColNamesSum.append(sumColName)


        print('Calculating rolling windows data: distances between body-parts')
        ########### CALC MEAN & SUM DISTANCES BETWEEN BODY PARTS IN ROLLING WINDOWS ###########################################
        for i in range(len(roll_windows_values)):
            for currDistanceCol in distanceColNames:
                colName = 'Mean_' + str(currDistanceCol) + '_' + str(roll_windows_values[i])
                csv_df[colName] = csv_df[currDistanceCol].rolling(roll_windows[i], min_periods=1).mean()
                colName = 'Sum_' + str(currDistanceCol) + '_' + str(roll_windows_values[i])
                csv_df[colName] = csv_df[currDistanceCol].rolling(roll_windows[i], min_periods=1).sum()

        print('Calculating rolling windows data: animal movements')
        for i in range(len(roll_windows_values)):
            for animal in collapsedColNamesMean:
                colName = 'Mean_' + str(animal) + '_' + str(roll_windows_values[i])
                csv_df[colName] = csv_df[animal].rolling(roll_windows[i], min_periods=1).mean()
                colName = 'Sum_' + str(animal) + '_' + str(roll_windows_values[i])
                csv_df[colName] = csv_df[animal].rolling(roll_windows[i], min_periods=1).sum()

        ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
        print('Calculating pose probability scores...')
        probabilityDf = csv_df.filter(Pcols, axis=1)
        csv_df['Sum_probabilities'] = probabilityDf.sum(axis=1)
        csv_df['Mean_probabilities'] = probabilityDf.mean(axis=1)
        values_in_range_min, values_in_range_max = 0.0, 0.1
        csv_df["Low_prob_detections_0.1"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.5
        csv_df["Low_prob_detections_0.5"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_mcreate_body_part_dictionaryin, values_in_range_max = 0.000000000, 0.75
        csv_df["Low_prob_detections_0.75"] = probabilityDf.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)

        ########### SAVE DF ###########################################
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        fileOutName = os.path.basename(currentFile)
        savePath = os.path.join(csv_dir_out, fileOutName)
        print('Saving features...')
        save_df(csv_df, wfileType, savePath)
        print('Feature extraction complete for ' + '"' + str(currVidName) + '".')

    print('All feature extraction complete.')

# extract_features_wotarget_user_defined(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Parquet_test\project_folder\project_config.ini")