__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
import os, glob
import numpy as np
from configparser import ConfigParser, NoSectionError, NoOptionError
from datetime import datetime
from simba.rw_dfs import *
from simba.features_scripts.unit_tests import read_video_info, read_video_info_csv
from simba.drop_bp_cords import *

def analyze_process_severity(configini,severitbrac,targetBehavior):
    print('Processing',targetBehavior, 'severity...')
    config = ConfigParser()
    config.read(configini)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'machine_results')
    severity_brackets = int(severitbrac)
    vidInfPath = config.get('General settings', 'project_path')
    log_path = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(log_path, 'video_info.csv')
    noAnimals = config.getint('General settings', 'animal_no')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    vidinfDf = read_video_info_csv(vidInfPath)
    severityGrades = list(np.arange(0, 1.0, ((10/severity_brackets)/10)))
    severityGrades.append(10)
    severityLogFrames = [0] * severity_brackets
    severityLogTime = [0] * severity_brackets

    log_fn = 'severity_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
    log_fn = os.path.join(log_path, log_fn)

    headers = ['Video']
    for i in range(severity_brackets):
        headers.append('Grade' + str(i) + '_frames')
    for i in range(severity_brackets):
        headers.append('Grade' + str(i) + '_time')
    log_df = pd.DataFrame(columns=headers)
    Xcols, Ycols, Pcols = getBpNames(configini)

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


    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    for counter, currentFile in enumerate(filesFound):
        CurrentVideoName = os.path.basename(currentFile).replace('.' + wfileType, '')
        videoSettings, _, fps = read_video_info(vidinfDf, CurrentVideoName)
        csv_df = read_df(currentFile, wfileType)

        if not 'Scaled_movement_M1_M2' in list(csv_df.columns):
            movement_df = pd.DataFrame()
            col_head_shifted = [x + '_shifted' for x in list(csv_df.columns)]
            csv_df_shifted = csv_df.shift(periods=1)
            csv_df_shifted.columns = col_head_shifted
            csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)
            for animal in animalBpDict:
                currentAnimalX, currentAnimalY = animalBpDict[animal]['X_bps'], animalBpDict[animal]['Y_bps']
                for currBpX, currBpY in zip(currentAnimalX, currentAnimalY):
                    shiftedBpX, shiftedBpY = currBpX + '_shifted', currBpY + '_shifted'
                    colName = 'Movement_' + currBpX.strip('_x')
                    movement_df[colName] = (np.sqrt((csv_df_combined[currBpX] - csv_df_combined[shiftedBpX]) ** 2 + (csv_df_combined[currBpY] - csv_df_combined[shiftedBpY]) ** 2))
            movement_df['Total_movement_M1_M2'] = movement_df.sum(axis=1)
            csv_df['Scaled_movement_M1_M2'] = (movement_df['Total_movement_M1_M2']-movement_df['Total_movement_M1_M2'].min())/(movement_df['Total_movement_M1_M2'].max()-movement_df['Total_movement_M1_M2'].min())

        for pp in range(severity_brackets):
            lowerBound = severityGrades[pp]
            upperBound = severityGrades[pp + 1]
            currGrade = len(csv_df[(csv_df[str(targetBehavior)] == 1) & (csv_df['Scaled_movement_M1_M2'] > lowerBound) & (csv_df['Scaled_movement_M1_M2'] <= upperBound)])
            severityLogFrames[pp] = currGrade
        log_list = []
        log_list.append(str(CurrentVideoName.replace('.' + wfileType, '')))
        for bb in range(len(severityLogFrames)):
            severityLogTime[bb] = round(severityLogFrames[bb] / fps, 4)
        log_list.extend(severityLogFrames)
        log_list.extend(severityLogTime)
        log_df.loc[counter] = log_list
        print('Files # processed for movement data: ' + str(counter + 1))
    log_df = log_df.replace('NaN', 0)
    log_df.to_csv(log_fn, index=False)
    print('All files processed for severity data: ' + 'data saved @' + str(log_fn))











