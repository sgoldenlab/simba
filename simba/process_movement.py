import pandas as pd
import os
from configparser import ConfigParser, NoSectionError, NoOptionError
from datetime import datetime
import statistics
import numpy as np
import glob
from simba.drop_bp_cords import define_movement_cols
from simba.rw_dfs import *


def analyze_process_movement(configini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    vidLogFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidLogFilePath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    noAnimals = config.getint('process movements', 'no_of_animals')
    Animal_1_Bp = config.get('process movements', 'animal_1_bp')
    Animal_2_Bp = config.get('process movements', 'animal_2_bp')
    VideoNo_list, columnNames1, fileCounter = [], [], 0

    ########### logfile path ###########
    log_fn = os.path.join(projectPath, 'logs', 'Movement_log_' + dateTime + '.csv')
    columnNames = define_movement_cols(noAnimals)
    log_df = pd.DataFrame(columns=columnNames)

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Processing movement data for ' + str(len(filesFound)) + ' files...')
    columnHeaders = [Animal_1_Bp + '_x', Animal_1_Bp + '_y']
    shifted_columnHeaders = ['shifted_' + Animal_1_Bp + '_x', 'shifted_' + Animal_1_Bp + '_y']
    if noAnimals == 2:
        columnHeaders.extend([Animal_2_Bp + '_x', Animal_2_Bp + '_y'])
        shifted_columnHeaders.extend(['shifted_' + Animal_2_Bp + '_x', 'shifted_' + Animal_2_Bp + '_y'])

    for currentFile in filesFound:
        frameCounter = 0
        currVideoName = os.path.basename(currentFile).replace('.' + wfileType, '')
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == currVideoName]
        try:
            fps = int(videoSettings['fps'])
            currPixPerMM = float(videoSettings['pixels/mm'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        csv_df = read_df(currentFile, wfileType)
        csv_df = csv_df[columnHeaders]
        csv_df_shifted = csv_df.shift(-1, axis=0)
        csv_df_shifted.columns = shifted_columnHeaders
        csv_df = pd.concat([csv_df, csv_df_shifted], axis=1)
        csv_df['Movement_animal_1'] = (np.sqrt((csv_df[columnHeaders[0]] - csv_df[shifted_columnHeaders[0]]) ** 2 + (csv_df[columnHeaders[1]] - csv_df[shifted_columnHeaders[1]]) ** 2)) / currPixPerMM
        if noAnimals == 2:
            csv_df['Movement_animal_2'] = (np.sqrt((csv_df[columnHeaders[2]] - csv_df[shifted_columnHeaders[2]]) ** 2 + (csv_df[columnHeaders[3]] - csv_df[shifted_columnHeaders[3]]) ** 2)) / currPixPerMM
            csv_df['Animal_distance'] = (np.sqrt((csv_df[columnHeaders[0]] - csv_df[columnHeaders[2]]) ** 2 + (csv_df[columnHeaders[1]] - csv_df[columnHeaders[3]]) ** 2)) / currPixPerMM

        df_lists = [csv_df[i:i + fps] for i in range(0, csv_df.shape[0], fps)]
        movementListAnimal1, movementListAnimal2, distanceList, velocityAnimal1List, velocityAnimal2List, currentVidList = [], [], [], [], [], []
        for currentDf in df_lists:
            if noAnimals == 1:
                movementListAnimal1.append(currentDf['Movement_animal_1'].mean())
                velocityAnimal1List.append(currentDf['Movement_animal_1'].mean() / 1)
            if noAnimals == 2:
                movementListAnimal1.append(currentDf['Movement_animal_1'].mean())
                movementListAnimal2.append(currentDf['Movement_animal_2'].mean())
                velocityAnimal1List.append(currentDf['Movement_animal_1'].mean() / 1)
                velocityAnimal2List.append(currentDf['Movement_animal_2'].mean() / 1)
                distanceList.append(currentDf['Animal_distance'].mean())
                print(currentDf['Animal_distance'].mean())
            frameCounter += fps

        totalMovement_animal1 = sum(movementListAnimal1)
        meanVelocity_animal_1 = statistics.mean(velocityAnimal1List)
        medianVelocity_animal_1 = statistics.median(velocityAnimal1List)
        currentVidList = [currVideoName, frameCounter, totalMovement_animal1, meanVelocity_animal_1, medianVelocity_animal_1]
        if noAnimals == 2:
            totalMovement_animal2 = sum(movementListAnimal2)
            meanVelocity_animal_2 = statistics.mean(velocityAnimal2List)
            medianVelocity_animal_2 = statistics.median(velocityAnimal2List)
            meanDistance = statistics.mean(distanceList) / 10
            medianDistance = statistics.median(distanceList) / 10
            currentVidList = [currVideoName, frameCounter, totalMovement_animal1, meanVelocity_animal_1, medianVelocity_animal_1, totalMovement_animal2, meanVelocity_animal_2, medianVelocity_animal_2, meanDistance, medianDistance]
        log_df.loc[fileCounter] = currentVidList
        fileCounter += 1
        print('Files # processed for movement data: ' + str(fileCounter) + '/' + str(len(filesFound)) + '...')
    log_df = np.round(log_df,decimals=4)
    log_df.to_csv(log_fn, index=False)
    print('All files processed for movement data. ' + 'Data saved @ project_folder\logs')
