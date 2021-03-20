import pandas as pd
import os
from configparser import ConfigParser
from datetime import datetime
import statistics
import numpy as np
import glob
from simba.drop_bp_cords import getBpNames


def time_bins_movement(configini,binLength):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
    vidLogFilePath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidLogFilePath)
    noAnimals = config.getint('process movements', 'no_of_animals')
    Animal_1_Bp = config.get('process movements', 'animal_1_bp')
    Animal_2_Bp = config.get('process movements', 'animal_2_bp')
    VideoNo_list, columnNames1, fileCounter = [], [], 0
    move1Hlist, move2Hlist, vel1Hlist, vel2Hlist, distHlist = [], [], [], [], []

    ########### FIND CSV FILES ###########
    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Processing movement data for ' + str(len(filesFound)) + ' files...')
    columnHeaders = [Animal_1_Bp + '_x', Animal_1_Bp + '_y']
    shifted_columnHeaders = ['shifted_' + Animal_1_Bp + '_x', 'shifted_' + Animal_1_Bp + '_y']

    if noAnimals == 2:
        columnHeaders.extend([Animal_2_Bp + '_x', Animal_2_Bp + '_y'])
        shifted_columnHeaders.extend(['shifted_' + Animal_2_Bp + '_x', 'shifted_' + Animal_2_Bp + '_y'])

    for currentFile in filesFound:
        frameCounter, readHeadersList, finalList, concatList, finalList = 0, [], [], [], []
        currVideoName = os.path.basename(currentFile).replace('.csv', '')
        videoSettings = vidinfDf.loc[vidinfDf['Video'] == currVideoName]
        try:
            fps = int(videoSettings['fps'])
            currPixPerMM = float(videoSettings['pixels/mm'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        csv_df = pd.read_csv(currentFile)
        Xcols, Ycols, Pcols = getBpNames(configFile)
        for i in range(len(Xcols)):
            col1, col2, col3 = Xcols[i], Ycols[i], Pcols[i]
            readHeadersList.extend((col1, col2, col3))
        readHeadersList.insert(0, 'scorer')
        csv_df.columns = readHeadersList
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
            frameCounter += fps
            if noAnimals == 1:
                movementListAnimal1Chunks = [movementListAnimal1[x:x + binLength] for x in range(0, len(movementListAnimal1), binLength)]
                velocityAnimal1ListChunks = [velocityAnimal1List[x:x + binLength] for x in range(0, len(velocityAnimal1List), binLength)]
            if noAnimals == 2:
                movementListAnimal1Chunks = [movementListAnimal1[x:x + binLength] for x in range(0, len(movementListAnimal1), binLength)]
                movementListAnimal2Chunks = [movementListAnimal2[x:x + binLength] for x in range(0, len(movementListAnimal2), binLength)]
                velocityAnimal1ListChunks = [velocityAnimal1List[x:x + binLength] for x in range(0, len(velocityAnimal1List), binLength)]
                velocityAnimal2ListChunks = [velocityAnimal2List[x:x + binLength] for x in range(0, len(velocityAnimal2List), binLength)]
                distanceListChunks = [movementListAnimal1[x:x + binLength] for x in range(0, len(distanceList), binLength)]
        if noAnimals == 1:
            for i in range(len(movementListAnimal1Chunks)):
                move1Hlist.append('Animal_1_movement_bin_' + str(i))
                vel1Hlist.append('Animal_1_velocity_bin_' + str(i))
            headerList = move1Hlist + vel1Hlist
            headerList.insert(0, currVideoName)
            for L in movementListAnimal1Chunks: finalList.append(sum(L))
            for L in velocityAnimal1ListChunks: finalList.append(statistics.mean(L))
        if noAnimals == 2:
            for i in range(len(movementListAnimal1Chunks)):
                move1Hlist.append('Animal_1_movement_cm_bin_' + str(i+1))
                move2Hlist.append('Animal_2_movement_cm_bin_' + str(i+1))
                vel1Hlist.append('Animal_1_velocity_cm/s_bin_' + str(i+1))
                vel2Hlist.append('Animal_2_velocity_cm/s_bin_' + str(i+1))
                distHlist.append('Distance_cm_bin_' + str(i))
            headerList = move1Hlist + move2Hlist + vel1Hlist + vel2Hlist + distHlist
            headerList.insert(0, currVideoName)
            for L in movementListAnimal1Chunks: finalList.append(sum(L))
            for L in movementListAnimal2Chunks: finalList.append(sum(L))
            for L in velocityAnimal1ListChunks: finalList.append(statistics.mean(L))
            for L in velocityAnimal2ListChunks: finalList.append(statistics.mean(L))
            for L in distanceListChunks: finalList.append(statistics.mean(L))
        finalList = [round(num, 4) for num in finalList]
        finalList.insert(0, currVideoName)
        if currentFile == filesFound[0]:
            outputDf = pd.DataFrame(columns=headerList)
        outputDf.loc[len(outputDf)] = finalList
        fileCounter += 1
        print('Processed time-bins for file ' + str(fileCounter) + '/' + str(len(filesFound)))
    log_fn = os.path.join(projectPath, 'logs', 'Time_bins_movement_results_' + dateTime + '.csv')
    outputDf.to_csv(log_fn, index=False)
    print('Time-bin analysis for movement results complete.')