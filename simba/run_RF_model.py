import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pandas as pd
import pickle
import numpy as np
import statistics
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError, NoSectionError
from simba.drop_bp_cords import *
import glob, os
from simba.rw_dfs import *

warnings.simplefilter(action='ignore', category=FutureWarning)

def rfmodel(inifile):
    config = ConfigParser()
    configFile = str(inifile)
    try:
        config.read(configFile)
    except MissingSectionHeaderError:
        print('ERROR:  Not a valid project_config file. Please check the project_config.ini path.')
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in, csv_dir_out = os.path.join(projectPath, 'csv', 'features_extracted'), os.path.join(projectPath, 'csv', 'machine_results')
    model_dir = config.get('SML settings', 'model_dir')
    model_nos = config.getint('SML settings', 'No_targets')
    poseEstimationBps = config.get('create ensemble settings', 'pose_estimation_body_parts')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)

    try:
        multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
        multiAnimalIDList = multiAnimalIDList.split(",")
        if (multiAnimalIDList[0] != '') and (poseEstimationBps == 'user_defined'):
            multiAnimalStatus = True
            print('Applying settings for multi-animal tracking...')
        else:
            multiAnimalStatus = False
            print('Applying settings for classical tracking...')

    except NoSectionError:
        multiAnimalIDList = ['']
        multiAnimalStatus = False
        print('Applying settings for classical tracking...')

    bpHeaders = getBpHeaders(inifile)
    model_paths, target_names, DTList, min_bout_list = ([], [], [], [])
    target_names = []
    fileCounter = 0

    ########### GET MODEL PATHS, NAMES, AND DISCRIMIINATION THRESHOLDS ###########
    for i in range(model_nos):
        try:
            currentModelPaths = 'model_path_' + str(i+1)
            currentModelNames = 'target_name_' + str(i+1)
            currentDT = 'threshold_' + str(i+1)
            currMinBoutName = 'min_bout_' + str(i+1)
            currentModelPaths = config.get('SML settings', currentModelPaths)
            if currentModelPaths == '':
                print('Skipping ' + str(currentModelNames) + ' classifications: ' + ' no path set to .sav file.')
                continue
            currentModelNames = config.get('SML settings', currentModelNames)
            currentDT = config.getfloat('threshold_settings', currentDT)
            currMinBout = config.getfloat('Minimum_bout_lengths', currMinBoutName)
            DTList.append(currentDT)
            min_bout_list.append(currMinBout)
            model_paths.append(currentModelPaths)
            target_names.append(currentModelNames)
        except ValueError:
            print('Skipping ' + str(currentModelNames) + ' classifications: ' + ' no discrimination threshold and/or minimum bout set.')
            continue

    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Running ' + str(len(target_names)) + ' model(s) on ' + str(len(filesFound)) + ' video file(s).')

    for currFile in filesFound:
        currentFileName = os.path.basename(currFile)
        fileCounter+=1
        print('Analyzing video ' + str(fileCounter) + '/' + str(len(filesFound)) + '...')
        inputFile = read_df(currFile, wfileType)
        try:
            inputFile = inputFile.set_index('scorer')
        except KeyError:
            pass
        inputFile = inputFile.loc[:, ~inputFile.columns.str.contains('^Unnamed')]
        inputFileOrganised = drop_bp_cords(inputFile, inifile)
        currVidInfoDf = vidinfDf.loc[vidinfDf['Video'] == str(currentFileName.replace('.' + wfileType, ''))]
        try:
            currVidFps = int(currVidInfoDf['fps'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        outputDf = inputFile.copy(deep=True)

        for b in range(model_nos):
            shortest_bout = min_bout_list[b]
            framesToPlug = int(currVidFps * (shortest_bout / 1000))
            framesToPlugList = list(range(1, framesToPlug + 1))
            framesToPlugList.reverse()
            patternListofLists, negPatternListofList = [], []
            for k in framesToPlugList:
                zerosInList, oneInlist = [0] * k, [1] * k
                currList = [1]
                currList.extend(zerosInList)
                currList.extend([1])
                currListNeg = [0]
                currListNeg.extend(oneInlist)
                currListNeg.extend([0])
                patternListofLists.append(currList)
                negPatternListofList.append(currListNeg)
            fillPatterns = np.asarray(patternListofLists)
            remPatterns = np.asarray(negPatternListofList)
            currentModelPath = model_paths[b]
            model = os.path.join(model_dir, currentModelPath)
            currModelName = target_names[b]
            discrimination_threshold = DTList[b]
            currProbName = 'Probability_' + currModelName
            clf = pickle.load(open(model, 'rb'))
            try:
                predictions = clf.predict_proba(inputFileOrganised)
            except ValueError:
                print('Mismatch in the number of features in input file and what is expected from the model in file ' + str(currentFileName) + ' and model ' + str(currModelName))

            try:
                outputDf[currProbName] = predictions[:, 1]
            except IndexError:
                print('IndexError: Your classifier has not been created properly. See The SimBA GitHub FAQ page for more information and suggested fixes.')
            outputDf[currModelName] = np.where(outputDf[currProbName] > discrimination_threshold, 1, 0)

            ########## FILL PATTERNS ###########################################
            for currPattern in fillPatterns:
                n_obs = len(currPattern)
                outputDf['rolling_match'] = (outputDf[currModelName].rolling(window=n_obs, min_periods=n_obs)
                                             .apply(lambda x: (x == currPattern).all())
                                             .mask(lambda x: x == 0)
                                             .bfill(limit=n_obs - 1)
                                             .fillna(0)
                                             .astype(bool)
                                             )
                outputDf.loc[outputDf['rolling_match'] == True, currModelName] = 1
                outputDf = outputDf.drop(['rolling_match'], axis=1)
            for currPattern in remPatterns:
                n_obs = len(currPattern)
                outputDf['rolling_match'] = (outputDf[currModelName].rolling(window=n_obs, min_periods=n_obs)
                                             .apply(lambda x: (x == currPattern).all())
                                             .mask(lambda x: x == 0)
                                             .bfill(limit=n_obs - 1)
                                             .fillna(0)
                                             .astype(bool)
                                             )
                outputDf.loc[outputDf['rolling_match'] == True, currModelName] = 0
                outputDf = outputDf.drop(['rolling_match'], axis=1)

        if poseEstimationBps == '4' or poseEstimationBps == '7' or poseEstimationBps == '8' or poseEstimationBps == '7':
            #sketchy fix due to version compatability
            try:
                mouse1size = (statistics.mean(outputDf['Mouse_1_nose_to_tail']))
                mouse1Max = mouse1size * 8
                outputDf['Scaled_movement_M1'] = (outputDf['Total_movement_all_bodyparts_M1'] / (mouse1Max))
            except:
                mouse1size = (statistics.mean(outputDf['Mouse_nose_to_tail']))
                mouse1Max = mouse1size * 8
                outputDf['Scaled_movement_M1'] = (outputDf['Total_movement_all_bodyparts_M1'] / (mouse1Max))

        if poseEstimationBps == '16' or poseEstimationBps == '14':
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
        save_df(outputDf, wfileType, outFname)
        print('Predictions generated for ' + str(fileBaseName) + '...')
    print('Predictions complete. Saved @ project_folder/csv/machine_results')