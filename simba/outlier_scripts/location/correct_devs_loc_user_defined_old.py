import pandas as pd
import os
import statistics
import numpy as np
from configparser import ConfigParser
from datetime import datetime
import glob


def dev_loc_user_defined(projectini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    configFile = str(projectini)
    config = ConfigParser()
    config.read(configFile)
    animalNo = config.getint('General settings', 'animal_no')
    projectPath = config.get('General settings', 'project_path')
    currentBodyPartFile = os.path.join(projectPath, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    bp1_animal1 = config.get('Outlier settings', 'location_bodypart1_mouse1')
    bp2_animal1 = config.get('Outlier settings', 'location_bodypart2_mouse1')
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement')
    csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    bp1x_animal1, bp1y_animal1 = (bp1_animal1 + '_x', bp1_animal1 + '_y')
    bp2x_animal1, bp2x_animal1 = (bp2_animal1 + '_x', bp2_animal1 + '_y')
    if animalNo == 2:
        bp1_animal2 = config.get('Outlier settings', 'location_bodypart1_mouse2')
        bp2_animal2 = config.get('Outlier settings', 'location_bodypart2_mouse2')
        bp1x_animal2, bp1y_animal2 = (bp1_animal2 + '_x', bp1_animal2 + '_y')
        bp2x_animal2, bp2y_animal2 = (bp2_animal2 + '_x', bp2_animal2 + '_y')
    bodyPartsFile = pd.read_csv(os.path.join(currentBodyPartFile, currentBodyPartFile), header=None)
    bodyPartsList = list(bodyPartsFile[0])
    bodyPartHeaders = []
    xy_headers, p_cols, x_cols, y_cols, animal1_headers, animal2_headers, animalHeadersListofList = [], [], [], [], [], [], []
    for i in bodyPartsList:
        col1, col2, col3 = (str(i) + '_x', str(i) + '_y', str(i) + '_p')
        p_cols.append(col3)
        x_cols.append(col1)
        y_cols.append(col2)
        bodyPartHeaders.extend((col1, col2, col3))
        xy_headers.extend((col1, col2))
    if animalNo == 2:
        for element in xy_headers:
            animal1_headers.extend([element for element in element.split() if '_1_' in element])
            animal2_headers.extend([element for element in element.split() if '_2_' in element])
            animalHeadersListofList = [animal1_headers, animal2_headers]
    if animalNo == 1:
        animalHeadersListofList = xy_headers
    vNm_list, fixedPositions_M1_list, frames_processed_list, counts_total_M1_list = ([], [], [], [])
    loopy = 0
    reliableCoordinates = np.zeros((7, 2))
    criterion = config.getfloat('Outlier settings', 'location_criterion')
    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Processing ' + str(len(filesFound)) + ' files for location outliers...')

    # ########### logfile path ###########
    log_fn = os.path.join(projectPath, 'logs', 'Outliers_location_' + str(dateTime) + '.csv')
    logDfColumns = ['Video', 'Frames_processed']
    logDfColumns.extend(bodyPartsList)
    logDfColumns.append(str('% corrected'))
    log_df = pd.DataFrame(columns=logDfColumns)

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    loop=0
    for currentFile in filesFound:
        loopy += 1
        videoFileBaseName = os.path.basename(currentFile).replace('.csv', '')
        csv_df = pd.read_csv(currentFile, names=bodyPartHeaders, low_memory=False)
        csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        vNm_list.append(videoFileBaseName)


        ########### MEAN MOUSE SIZES ###########################################
        meanSizeList = []
        csv_df['Reference_value_1'] = np.sqrt((csv_df[bp1x_animal1] - csv_df[bp2x_animal1]) ** 2 + (csv_df[bp1y_animal1] - csv_df[bp1y_animal1]) ** 2)
        mean1size = (statistics.mean(csv_df['Reference_value_1']))
        meanSizeList.append(mean1size * criterion)
        csv_df = csv_df.drop('Reference_value_1', axis=1)
        if animalNo == 2:
            csv_df['Reference_value_2'] = np.sqrt((csv_df[bp1x_animal2] - csv_df[bp2x_animal2]) ** 2 + (csv_df[bp1y_animal2] - csv_df[bp2y_animal2]) ** 2)
            mean2size = (statistics.mean(csv_df['Reference_value_2']))
            meanSizeList.append(mean2size * criterion)
            csv_df = csv_df.drop('Reference_value_2', axis=1)
        df_p_cols = pd.DataFrame([csv_df.pop(x) for x in p_cols]).T
        df_p_cols = df_p_cols.reset_index()

        for animal in range(animalNo):
            currMeanSize = meanSizeList[animal]
            currentFixedList, fixedPositions_M1, fixedPositions_M2 = [], 0, 0
            counts_total_M1 = [0] * 32
            outputArray = np.array([0] * (len(bodyPartsList) / animalNo))
            print(outp)

            currDf = csv_df[animalHeadersListofList[animal]]
            for index, row in currDf.iterrows():
                print(row)
                currentArray = row.to_numpy()
                currentArray = currentArray[0:-2]
                currentArray = [[i, i + 1] for i in currentArray[0:-1:2]]
                nbody_parts = int(len(currentArray) / animalNo)
                counts = [0] * nbody_parts
                for i in range(0, (nbody_parts - 1)):
                    for j in range((i + 1), (nbody_parts)):
                        dist_ij = np.sqrt((currentArray[i][0] - currentArray[j][0]) ** 2 + (currentArray[i][1] - currentArray[j][1]) ** 2)
                        if dist_ij > currMeanSize:
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
                currentArray = np.array(currentArray)
                currentArray = currentArray.flatten()
                outputArray = np.vstack((outputArray, currentArray))
            outputArray = np.delete(outputArray, 0, 0)

        csvDf = pd.DataFrame(outputArray, columns=xy_headers)
        csvDf = csvDf.reset_index()
        scorer = pd.read_csv(currentFile).scorer
        scorer = list(scorer[0:])
        csv_df.index = csvDf.index
        csv_out = pd.DataFrame()
        for cols in range(len(x_cols)):
            csv_out = pd.concat([csv_out, csvDf[x_cols[cols]], csvDf[y_cols[cols]], df_p_cols[p_cols[cols]]], axis=1)
        csv_out.insert(loc=0, column='scorer', value=scorer)
        fname = os.path.basename(currentFile)
        fnamePrint = fname.replace('.csv', '')
        csvOutPath = os.path.join(csv_dir_out, fname)
        csv_out.to_csv(csvOutPath, index=False)
        frames_processed = len(csv_df)
        frames_processed_list.append(frames_processed)
        fixedPositions_M1_list.append(fixedPositions_M1)
        counts_total_M1_list.append(counts_total_M1)
        counts_total_M1_np = np.array(counts_total_M1_list)
        percentBDcorrected = round((fixedPositions_M1) / (frames_processed * len(bodyPartsList)), 6)
        currentFixedList.append(videoFileBaseName)
        currentFixedList.append(frames_processed)
        currentFixedList.extend(counts_total_M1)
        currentFixedList.append(percentBDcorrected)
        print(str(fnamePrint) + '. Tot frames: ' + str(frames_processed) + '. Outliers: ' + str(fixedPositions_M1) + '. % outliers: ' + str(round(percentBDcorrected, 3)))
        log_df.loc[loop] = currentFixedList
        loop = loop + 1

    log_df.to_csv(log_fn, index=False)
    print('Log for corrected "location outliers" saved in project_folder/logs')