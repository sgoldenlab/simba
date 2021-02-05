import pandas as pd
import os
import statistics
import numpy as np
from configparser import ConfigParser, NoOptionError, NoSectionError
from simba.rw_dfs import *
from datetime import datetime
from simba.drop_bp_cords import *
import glob


def dev_loc_user_defined(projectini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    configFile = str(projectini)
    config = ConfigParser()
    config.read(configFile)
    animalNo = config.getint('General settings', 'animal_no')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    projectPath = config.get('General settings', 'project_path')
    try:
        criterion = config.getfloat('Outlier settings', 'location_criterion')
    except:
        print('No location criterion found recorded in the project_config.ini file')

    animalIDlist = config.get('Multi animal IDs', 'id_list')

    if not animalIDlist:
        animalIDlist = []
        for animal in range(animalNo):
            animalIDlist.append('Animal_' + str(animal + 1))
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    else:
        animalIDlist = animalIDlist.split(",")
        multiAnimalStatus = True
        print('Applying settings for multi-animal tracking...')

    bodyPartNameArrayForMeans = np.empty((animalNo, 5), dtype=object)
    for animal in range(animalNo):
        bodyPart_1_Name, bodyPart_2_Name = 'movement_bodypart1_' + str(animalIDlist[animal]), 'movement_bodypart2_' + str(animalIDlist[animal])
        bodyPart1, bodyPart2 = config.get('Outlier settings', bodyPart_1_Name), config.get('Outlier settings',bodyPart_2_Name)
        bodyPart_1_X, bodyPart_1_Y, bodyPart_2_X, bodyPart_2_Y = bodyPart1 + '_x', bodyPart1 + '_y', bodyPart2 + '_x', bodyPart2 + '_y'
        bodyPartNameArrayForMeans[animal] = [animalIDlist[animal], bodyPart_1_X, bodyPart_1_Y, bodyPart_2_X, bodyPart_2_Y]
    x_cols, y_cols, p_cols = getBpNames(configFile)
    bp_names = list(map(lambda x: x.replace('_x',''),x_cols))
    colHeads = getBpHeaders(configFile)


    #### CREATE DICT TO HOLD ANIMAL BPS AND NAMES
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, animalIDlist, animalNo, x_cols, y_cols, p_cols, [])
    csv_dir_in, csv_dir_out, log_fn = os.path.join(projectPath, 'csv', 'outlier_corrected_movement'), os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location'), os.path.join(projectPath, 'logs', 'Outliers_location_' + str(dateTime) + '.csv')
    vNm_list, fixedPositions_M1_list, frames_processed_list, counts_total_M1_list = ([], [], [], [])
    reliableCoordinates = np.zeros((len(bp_names), 2))
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Processing ' + str(len(filesFound)) + ' files for location outliers...')

    # ########### logfile path ###########
    log_fn = 'Outliers_location_' + str(dateTime) + '.csv'
    log_path = config.get('General settings', 'project_path')
    log_path = os.path.join(log_path, 'logs')
    log_fn = os.path.join(log_path, log_fn)
    logDf_cols = ['Video', 'Frames_processed']

    for animal in animalBpDict:
        currAnimal = animalBpDict[animal]['X_bps']
        for bp in currAnimal:
            logDf_cols.append(bp.replace('_x', ''))
    logDf_cols.append(str('% corrected'))
    log_df = pd.DataFrame(columns=logDf_cols)



    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    videoCounter=0
    for currentFile in filesFound:
        print('Analyzing video ' + str(videoCounter+1) + '...')
        videoFileBaseName = os.path.basename(currentFile).replace('.' + wfileType, '')

        csv_df = read_df(currentFile, wfileType,idx=None)
        csv_df = csv_df.set_index('scorer')

        csv_df = csv_df.apply(pd.to_numeric)
        vNm_list.append(videoFileBaseName)
        animalCounter = 0
        outputDf = pd.DataFrame()
        outputRow = []
        for animal in animalBpDict:
            print('Analyzing animal ' + str(animalCounter + 1) + '...')
            currentFixedList, currCols, fixedPositions_M1 = [], [], 0
            counts_total_M1 = [0] * len(animalBpDict[animal]['X_bps'])
            outputArray = np.array([0] * (2*len(animalBpDict[animal]['X_bps'])))
            meanSize = statistics.mean(np.sqrt((csv_df[bodyPartNameArrayForMeans[animalCounter][1]] - csv_df[bodyPartNameArrayForMeans[animalCounter][3]]) ** 2 + (csv_df[bodyPartNameArrayForMeans[animalCounter][2]] - csv_df[bodyPartNameArrayForMeans[animalCounter][4]]) ** 2))
            currentCriterion = meanSize * criterion
            currXcols, currYcols, currPcols = animalBpDict[animalIDlist[animalCounter]]['X_bps'], animalBpDict[animalIDlist[animalCounter]]['Y_bps'], animalBpDict[animalIDlist[animalCounter]]['P_bps']
            for Xcol, Ycol  in zip(currXcols, currYcols):
                currCols.extend((Xcol, Ycol))
            currAnimaldf = csv_df[currCols]
            for index, row in currAnimaldf.iterrows():
                currentArray = row.to_numpy()
                currentArray = currentArray.reshape((len(currXcols), -1))
                nbody_parts = len(currentArray)
                counts = [0] * nbody_parts
                for i in range(0, (nbody_parts - 1)):
                    for j in range((i + 1), (nbody_parts)):
                        dist_ij = np.sqrt((currentArray[i][0] - currentArray[j][0]) ** 2 + (currentArray[i][1] - currentArray[j][1]) ** 2)
                        if dist_ij > currentCriterion:
                            counts[i] += 1
                            counts[j] += 1
                positions = [i for i in range(len(counts)) if counts[i] > 1]
                for pos in positions:
                    counts_total_M1[pos] += 1
                fixedPositions_M1 = fixedPositions_M1 + len(positions)
                if not positions:
                    reliableCoordinates = currentArray
                else:
                    for currentPosition in positions:
                        currentArray[currentPosition][0] = reliableCoordinates[currentPosition][0]
                    currentArray[currentPosition][1] = reliableCoordinates[currentPosition][1]
                    reliableCoordinates = currentArray
                currentArray = currentArray.flatten()
                outputArray = np.vstack((outputArray, currentArray))
            outputArray = np.delete(outputArray, 0, 0)
            csvDf = pd.DataFrame(outputArray, columns=currCols)
            csvDf.reset_index(drop=True)
            outputDf = pd.concat([outputDf, csvDf], axis=1)
            outputDf.index.name = 'scorer'
            #csv out
            fname = os.path.basename(currentFile)
            fnamePrint = fname.replace(wfileType, '')
            csvOutPath = os.path.join(csv_dir_out, fnamePrint + wfileType)
            save_df(outputDf, wfileType, csvOutPath)

            outputRow.extend((counts_total_M1))
            animalCounter+=1
        videoCounter+=1
        totalBpsCorrected = sum(outputRow)
        outputRow.insert(0, len(csv_df))
        outputRow.insert(0, videoFileBaseName)
        outputRow.append(totalBpsCorrected)
        log_df.loc[len(log_df)] = outputRow
        print(str(videoFileBaseName) + '. Tot frames: ' + str(len(csv_df)) + '. Outliers: ' + str(totalBpsCorrected) + '.')
    log_df.to_csv(log_fn, index=False)
    print('Log for corrected "location outliers" saved in project_folder/logs')