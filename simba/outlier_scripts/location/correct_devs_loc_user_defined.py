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

    if animalNo == 1:
        bodyPart1 = config.get('Outlier settings', 'location_bodypart1_mouse1')
        bodyPart2 = config.get('Outlier settings', 'location_bodypart2_mouse1')
        bodyPart1x, bodyPart1y = (bodyPart1 + '_x', bodyPart1 + '_y')
        bodyPart2x, bodyPart2y = (bodyPart2 + '_x', bodyPart2 + '_y')
        projectPath = config.get('General settings', 'project_path')
        currentBodyPartFile = os.path.join(projectPath, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
        bodyPartsFile = pd.read_csv(os.path.join(currentBodyPartFile, currentBodyPartFile), header=None)
        bodyPartsList = list(bodyPartsFile[0])
        bodyPartHeaders, x_cols, xy_headers, p_cols, y_cols = [], [], [], [], []
        for i in bodyPartsList:
            col1, col2, col3 = (str(i) + '_x', str(i) + '_y', str(i) + '_p')
            p_cols.append(col3)
            x_cols.append(col1)
            y_cols.append(col2)
            bodyPartHeaders.extend((col1, col2, col3))
            xy_headers.extend((col1, col2))
        csv_dir = config.get('General settings', 'csv_path')
        csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement')
        csv_dir_out = os.path.join(csv_dir, 'outlier_corrected_movement_location')
        if not os.path.exists(csv_dir_out):
            os.makedirs(csv_dir_out)
        vNm_list, fixedPositions_M1_list, frames_processed_list, counts_total_M1_list = ([], [], [], [])
        loopy = 0
        reliableCoordinates = np.zeros((len(bodyPartsList), 2))
        criterion = config.getfloat('Outlier settings', 'location_criterion')
        filesFound = glob.glob(csv_dir_in + '/*.csv')
        print('Processing ' + str(len(filesFound)) + ' files for location outliers...')
        # ########### logfile path ###########
        log_fn = 'Outliers_location_' + str(dateTime) + '.csv'
        log_path = config.get('General settings', 'project_path')
        log_path = os.path.join(log_path, 'logs')
        log_fn = os.path.join(log_path, log_fn)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logDfColumns = ['Video', 'Frames_processed']
        logDfColumns.extend(bodyPartsList)
        logDfColumns.append(str('% corrected'))
        log_df = pd.DataFrame(columns=logDfColumns)


        ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
        loop=0
        for currentFile in filesFound:
            currentFixedList = []
            fixedPositions_M1 = 0
            counts_total_M1 = [0] * len(bodyPartsList)
            outputArray = np.array([0] * (2*len(bodyPartsList)))
            loopy += 1
            videoFileBaseName = os.path.basename(currentFile).replace('.csv', '')
            csv_df = pd.read_csv(currentFile, names=bodyPartHeaders, low_memory=False)
            csv_df = csv_df.drop(csv_df.index[[0]])
            csv_df = csv_df.apply(pd.to_numeric)
            vNm_list.append(videoFileBaseName)
            ########### MEAN MOUSE SIZES ###########################################
            csv_df['Reference_value'] = np.sqrt((csv_df[bodyPart1x] - csv_df[bodyPart2x]) ** 2 + (csv_df[bodyPart1y] - csv_df[bodyPart2y]) ** 2)
            mean1size = (statistics.mean(csv_df['Reference_value']))
            mean1size = mean1size * criterion
            csv_df = csv_df.drop('Reference_value', axis=1)
            df_p_cols = pd.DataFrame([csv_df.pop(x) for x in p_cols]).T
            df_p_cols = df_p_cols.reset_index()
            for index, row in csv_df.iterrows():
                currentArray = row.to_numpy()
                currentArray = currentArray.reshape((len(bodyPartsList), -1))
                nbody_parts = len(currentArray)
                counts = [0] * nbody_parts
                for i in range(0, (nbody_parts - 1)):
                    for j in range((i + 1), (nbody_parts)):
                        dist_ij = np.sqrt((currentArray[i][0] - currentArray[j][0]) ** 2 + (currentArray[i][1] - currentArray[j][1]) ** 2)
                        if dist_ij > mean1size:
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

    if animalNo != 1:
        print('SimBAs outlier correction tools is currently *not* supported for user-defined pose-configurations when using multiple animals. To proceed, consider skipping outlier correction for now.')

