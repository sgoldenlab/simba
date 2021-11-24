import pandas as pd
import os
from configparser import ConfigParser, NoOptionError, NoSectionError
from datetime import datetime
import statistics
import numpy as np
import glob
from simba.drop_bp_cords import *
from simba.rw_dfs import *


def ROI_process_movement(configini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    vidLogFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidLogFilePath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)


    noAnimals = config.getint('process movements', 'no_of_animals')
    animalBps = []
    for bp in range(noAnimals):
        animalName = 'animal_' + str(bp + 1) + '_bp'
        animalBpName = config.get('process movements', animalName)
        animalBpNameX, animalBpNameY, animalBpNameXshifted, animalBpNameYshifted = animalBpName + '_x', animalBpName + '_y', animalBpName + '_x' + '_shifted', animalBpName + '_y' + '_shifted'
        animalBps.append([animalBpNameX, animalBpNameY, animalBpNameXshifted, animalBpNameYshifted])
    columns2grab = [item[0:2] for item in animalBps]
    columns2grab = [item for sublist in columns2grab for item in sublist]
    shiftedColNames = [item[2:4] for item in animalBps]
    shiftedColNames = [item for sublist in shiftedColNames for item in sublist]
    VideoNo_list, columnNames1, fileCounter = [], [], 0
    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if multiAnimalIDList[0] != '':
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            multiAnimalIDList = []
            for animal in range(noAnimals):
                multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
            print('Applying settings for classical tracking...')

    except NoSectionError:
        multiAnimalIDList = []
        for animal in range(noAnimals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    ########### logfile path ###########
    log_fn = os.path.join(projectPath, 'logs', 'Movement_log_' + dateTime + '.csv')
    columnNames = define_movement_cols(multiAnimalIDList)
    log_df = pd.DataFrame(columns=columnNames)

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Processing movement data for ' + str(len(filesFound)) + ' files...')

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
        try:
            csv_df = csv_df.set_index('scorer')
        except KeyError:
            pass
        colHeads = getBpHeaders(configini)
        csv_df = csv_df.iloc[:, : len(colHeads)]
        csv_df.columns = colHeads
        csv_df = csv_df[columns2grab]

        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted.columns = shiftedColNames
        csv_df = pd.concat([csv_df, csv_df_shifted], axis=1)
        for bp in range(noAnimals):
            colName = 'Movement_animal_' + str(bp+1)
            csv_df[colName] = (np.sqrt((csv_df[animalBps[bp][0]] - csv_df[animalBps[bp][2]]) ** 2 + (csv_df[animalBps[bp][1]] - csv_df[animalBps[bp][3]]) ** 2)) / currPixPerMM
        if noAnimals == 2:
            csv_df['Animal_distance'] = (np.sqrt((csv_df[animalBps[0][0]] - csv_df[animalBps[1][0]]) ** 2 + (csv_df[animalBps[0][1]] - csv_df[animalBps[1][1]]) ** 2)) / currPixPerMM
        df_lists = [csv_df[i:i + fps] for i in range(0, csv_df.shape[0], fps)]
        movementList, distanceList, velocityList, currentVidList, totalMovement, meanVelocityList, medianVelocityList = [[] for i in range(noAnimals)], [], [[] for i in range(noAnimals)], [], [], [], []
        for currentDf in df_lists:
            frameCounter += fps
            for animal in range(noAnimals):
                currColName = 'Movement_animal_' + str(animal + 1)
                movementList[animal].append(currentDf[currColName].mean())
                velocityList[animal].append(currentDf[currColName].mean() / 1)
                if noAnimals == 2:
                    distanceList.append(currentDf['Animal_distance'].mean())
        for animal in range(noAnimals):
            totalMovement.append(sum(movementList[animal]))
            meanVelocityList.append(statistics.mean(velocityList[animal]))
            medianVelocityList.append(statistics.median(velocityList[animal]))
        currentVidList = []
        currentVidList.append(currVideoName)
        currentVidList.append(frameCounter-fps)
        out_lists = [totalMovement, meanVelocityList, medianVelocityList]
        # for current_list in out_lists:
        #     for value in current_list:
        #         currentVidList.append(value)
        for movement, mean_velocity, median_velocity in zip(totalMovement, meanVelocityList, medianVelocityList):
            currentVidList.extend((movement, mean_velocity, median_velocity))
        print(currentVidList)
        if noAnimals == 2:
            currentVidList.append(statistics.mean(distanceList) / 10)
            currentVidList.append(statistics.median(distanceList) / 10)
        log_df.loc[fileCounter] = currentVidList
        fileCounter += 1
        print('Files # processed for movement data: ' + str(fileCounter) + '/' + str(len(filesFound)) + '...')
    log_df = np.round(log_df,decimals=4)
    log_df.to_csv(log_fn, index=False)
    print('All files processed for movement data. ' + 'Data saved @ project_folder\logs')