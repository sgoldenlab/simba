import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pandas as pd
import pickle
import numpy as np
import statistics
import os
from configparser import ConfigParser, MissingSectionHeaderError
from simba.drop_bp_cords import drop_bp_cords
import glob
warnings.simplefilter(action='ignore', category=FutureWarning)

def rfmodel(inifile):
    config = ConfigParser()
    configFile = str(inifile)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'features_extracted')
    csv_dir_out = os.path.join(csv_dir, 'machine_results')
    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    model_dir = config.get('SML settings', 'model_dir')
    model_nos = config.getint('SML settings', 'No_targets')
    pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    model_paths, target_names, DTList, min_bout_list = ([], [], [], [])
    target_names = []
    loop = 1
    loopy = 0

    ########### GET MODEL PATHS, NAMES, AND DISCRIMIINATION THRESHOLDS ###########
    for i in range(model_nos):
        currentModelPaths = 'model_path_' + str(loop)
        currentModelNames = 'target_name_' + str(loop)
        currentDT = 'threshold_' + str(loop)
        currMinBoutName = 'min_bout_' + str(loop)
        currentModelPaths = config.get('SML settings', currentModelPaths)
        currentModelNames = config.get('SML settings', currentModelNames)
        currentDT = config.getfloat('threshold_settings', currentDT)
        currMinBout = config.getfloat('Minimum_bout_lengths', currMinBoutName)
        DTList.append(currentDT)
        min_bout_list.append(currMinBout)
        model_paths.append(currentModelPaths)
        target_names.append(currentModelNames)
        loop += 1

    filesFound = glob.glob(csv_dir_in + '/*.csv')
    print('Running ' + str(len(target_names)) + ' model(s) on ' + str(len(filesFound)) + ' video file(s).')

    for i in filesFound:
        currFile = i
        currentFileName = os.path.basename(currFile)
        loopy += 1
        print('Analyzing video ' + str(loopy) + '/' + str(len(filesFound)) + '...')
        inputFile = pd.read_csv(currFile)
        inputFile = inputFile.loc[:, ~inputFile.columns.str.contains('^Unnamed')]
        inputFile = inputFile.drop(["scorer"], axis=1, errors='ignore')
        inputFileOrganised = drop_bp_cords(inputFile, inifile)
        currVidInfoDf = vidinfDf.loc[vidinfDf['Video'] == str(currentFileName.replace('.csv', ''))]
        try:
            currVidFps = int(currVidInfoDf['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        outputDf = inputFile.copy()
        outputDf.reset_index()

        for b in range(model_nos):

            # CREATE LIST OF GAPS BASED ON SHORTEST BOUT
            shortest_bout = min_bout_list[b]
            framesToPlug = int(currVidFps * (shortest_bout / 1000))
            framesToPlugList = list(range(1, framesToPlug + 1))
            framesToPlugList.reverse()
            patternListofLists = []
            for k in framesToPlugList:
                zerosInList = [0] * k
                currList = [1]
                currList.extend(zerosInList)
                currList.extend([1])
                patternListofLists.append(currList)
            patternListofLists.append([0, 1, 1, 0])
            patternListofLists.append([0, 1, 0])
            patterns = np.asarray(patternListofLists)


            currentModelPath = model_paths[b]
            model = os.path.join(model_dir, currentModelPath)
            currModelName = target_names[b]
            discrimination_threshold = DTList[b]
            currProbName = 'Probability_' + currModelName
            clf = pickle.load(open(model, 'rb'))
            predictions = clf.predict_proba(inputFileOrganised)

            outputDf[currProbName] = predictions[:, 1]
            outputDf[currModelName] = np.where(outputDf[currProbName] > discrimination_threshold, 1, 0)

            ########## FIX  'GAPS' ###########################################
            for l in patterns:
                currPattern = l
                n_obs = len(currPattern)
                outputDf['rolling_match'] = (outputDf[currModelName].rolling(window=n_obs, min_periods=n_obs)
                                             .apply(lambda x: (x == currPattern).all())
                                             .mask(lambda x: x == 0)
                                             .bfill(limit=n_obs - 1)
                                             .fillna(0)
                                             .astype(bool)
                                             )
                if (currPattern == patterns[-2]) or (currPattern == patterns[-1]):
                    outputDf.loc[outputDf['rolling_match'] == True, currModelName] = 0
                else:
                    outputDf.loc[outputDf['rolling_match'] == True, currModelName] = 1
                outputDf = outputDf.drop(['rolling_match'], axis=1)

        if pose_estimation_body_parts == '16' or pose_estimation_body_parts == '14':
            mouse1size = (statistics.mean(outputDf['Mouse_1_nose_to_tail']))
            mouse2size = (statistics.mean(outputDf['Mouse_2_nose_to_tail']))
            mouse1Max = mouse1size * 8
            mouse2Max = mouse2size * 8
            outputDf['Scaled_movement_M1'] = (outputDf['Total_movement_all_bodyparts_M1'] / (mouse1Max))
            outputDf['Scaled_movement_M2'] = (outputDf['Total_movement_all_bodyparts_M2'] / (mouse2Max))
            outputDf['Scaled_movement_M1_M2'] = (outputDf['Scaled_movement_M1'] + outputDf['Scaled_movement_M2']) / 2
            outputDf['Scaled_movement_M1_M2'] = outputDf['Scaled_movement_M1_M2'].round(decimals=2)
        fileBaseName = os.path.basename(currFile)
        outFname = os.path.join(csv_dir_out, fileBaseName)
        outputDf.to_csv(outFname)
        print('Predictions generated for ' + str(fileBaseName) + '...')
    print('Predictions complete. Saved @ project_folder/csv/machine_results')