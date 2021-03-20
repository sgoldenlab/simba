import pandas as pd
import os
from configparser import ConfigParser, NoOptionError, NoSectionError
from datetime import datetime
import statistics
import numpy as np
import glob
from simba.drop_bp_cords import *
from simba.rw_dfs import *



def time_bins_movement(configini,binLength):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    vidLogFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidLogFilePath)
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    noAnimals = config.getint('General settings', 'animal_no')
    columnHeaders, shifted_columnHeaders = [], []
    for animal in range(noAnimals):
        animalBp = config.get('process movements', 'animal_' + str(animal + 1) + '_bp')
        columnHeaders.append([animalBp + '_x', animalBp + '_y'])
        shifted_columnHeaders.append([animalBp + '_x_shifted', animalBp + '_y_shifted'])
    columnHeaders_flat = [item for sublist in columnHeaders for item in sublist]
    shifetcolheaders_flat = [item for sublist in shifted_columnHeaders for item in sublist]
    VideoNo_list, columnNames1, fileCounter = [], [], 0
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


    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Processing movement data for ' + str(len(filesFound)) + ' files...')

    for currentFile in filesFound:
        mov_and_vel_headers, distanceHeaders = [], []
        frameCounter, readHeadersList, finalList, concatList, finalList, distanceCols = 0, [], [], [], [], []
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
        colHeaders = getBpHeaders(configini)
        csv_df.columns = colHeaders
        csv_df = csv_df[columnHeaders_flat]
        csv_df_shifted = csv_df.shift(-1, axis=0)
        csv_df_shifted.columns = shifetcolheaders_flat
        csv_df = pd.concat([csv_df, csv_df_shifted], axis=1)
        for animal in range(noAnimals):
            colName1 = 'Movement_' + str(multiAnimalIDList[animal])
            csv_df[colName1] = (np.sqrt((csv_df[columnHeaders[animal][0]] - csv_df[shifted_columnHeaders[animal][0]]) ** 2 + (csv_df[columnHeaders[animal][1]] - csv_df[shifted_columnHeaders[animal][1]]) ** 2)) / currPixPerMM
            for animal1 in range(noAnimals):
                for animal2 in reversed(range(noAnimals)):
                    colName = 'Distance_' + str(multiAnimalIDList[animal1]) + '_' + str(multiAnimalIDList[animal2])
                    colNameReversed = 'Distance_' + str(multiAnimalIDList[animal2]) + '_' + str(multiAnimalIDList[animal1])
                    if not (colNameReversed in csv_df.columns) and not (colName in csv_df.columns) and animal1 != animal2:
                        csv_df[colName] = (np.sqrt((csv_df[columnHeaders[animal1][0]] - csv_df[columnHeaders[animal2][0]]) ** 2 + (csv_df[columnHeaders[animal1][1]] - csv_df[columnHeaders[animal2][1]]) ** 2)) / currPixPerMM
                        distanceCols.append(colName)
        df_lists = [csv_df[i:i + fps] for i in range(0, csv_df.shape[0], fps)]
        for animal in range(noAnimals):
            movementList, currAnimalID = [], multiAnimalIDList[animal]
            for currentDf in df_lists:
                movementList.append(currentDf['Movement_' + currAnimalID].mean())
            movementListChunks = [movementList[x:x + binLength] for x in range(0, len(movementList), binLength)]
            for L in movementListChunks: finalList.append(sum(L))
            for i in range(len(movementListChunks)): mov_and_vel_headers.append(currAnimalID + '_total_movement_' + str(i + 1))
        for animal in range(noAnimals):
            velocityList, currAnimalID = [], multiAnimalIDList[animal]
            for currentDf in df_lists:
                velocityList.append(currentDf['Movement_' + currAnimalID].mean() / 1)
            velocityListChunks = [velocityList[x:x + binLength] for x in range(0, len(velocityList), binLength)]
            for L in velocityListChunks: finalList.append(statistics.mean(L))
            for i in range(len(velocityListChunks)): mov_and_vel_headers.append(currAnimalID + '_mean_velocity_' + str(i + 1))
        for distanceCol in distanceCols:
            distanceList = []
            for currdf in df_lists:
                distanceList.append(currdf[distanceCol].mean())
            distanceListChunks = [distanceList[x:x + binLength] for x in range(0, len(distanceList), binLength)]
            for L in distanceListChunks: finalList.append(statistics.mean(L))
        for distanceCol in distanceCols:
            for currCol in range(len(distanceListChunks)):
                distanceHeaders.append(distanceCol + '_bin_' + str(currCol + 1))
        headerList = mov_and_vel_headers + distanceHeaders
        headerList.insert(0, 'Video_name')
        finalList = [round(num, 4) for num in finalList]
        finalList.insert(0, currVideoName)
        if currentFile == filesFound[0]:
            outputDf = pd.DataFrame(columns=headerList)
        outputDf.loc[len(outputDf)] = finalList
        fileCounter += 1
        print('Processed time-bins for file ' + str(fileCounter) + '/' + str(len(filesFound)))
    log_fn = os.path.join(projectPath, 'logs', 'Time_bins_movement_results_' + dateTime + '.csv')
    try:
        outputDf.to_csv(log_fn, index=False)
    except UnboundLocalError:
        print('Error: Check that files exist. Have you corrected, or indicated you want to skip, the outlier correction step?')
    print('Time-bin analysis for movement results complete.')