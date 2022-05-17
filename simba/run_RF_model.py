import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import pandas as pd
import pickle
import numpy as np
import statistics
from configparser import ConfigParser, MissingSectionHeaderError, NoOptionError, NoSectionError
from simba.drop_bp_cords import *
from simba.drop_bp_cords import get_workflow_file_format
import glob, os
from simba.rw_dfs import *
from simba.features_scripts.unit_tests import read_video_info
from simba.misc_tools import check_multi_animal_status

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

    wfileType = get_workflow_file_format(config)
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    vidinfDf["Video"] = vidinfDf["Video"].astype(str)

    noAnimals = config.getint('General settings', 'animal_no')
    multiAnimalStatus, multiAnimalIDList = check_multi_animal_status(config, noAnimals)

    ########### GET MODEL PATHS, NAMES, AND DISCRIMIINATION THRESHOLDS ###########
    model_dict = {}
    for i in range(model_nos):
        try:
            model_dict[i] = {}
            if config.get('SML settings', 'model_path_' + str(i+1)) == '':
                print('Skipping ' + str(config.get('SML settings', 'target_name_' + str(i+1))) + ' classifications: ' + ' no path set to model file.')
                continue
            model_dict[i]['model_path'] = config.get('SML settings', 'model_path_' + str(i+1))
            model_dict[i]['model_name'] = config.get('SML settings', 'target_name_' + str(i+1))
            model_dict[i]['threshold'] = config.getfloat('threshold_settings', 'threshold_' + str(i+1))
            model_dict[i]['minimum_bout_length'] = config.getfloat('Minimum_bout_lengths', 'min_bout_' + str(i+1))
        except ValueError:
            print('Skipping ' + str(config.get('SML settings', 'target_name_' + str(i+1))) + ' classifications: ' + ' no discrimination threshold and/or minimum bout set.')
            continue

    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Running ' + str(len(model_dict.keys())) + ' model(s) on ' + str(len(filesFound)) + ' video file(s).')

    for file_cnt, currFile in enumerate(filesFound):
        print('Analyzing video ' + str(file_cnt + 1) + '/' + str(len(filesFound)) + '...')
        file_cnt+=1
        dir_name, file_name, ext = get_fn_ext(currFile)
        inputFile = read_df(currFile, wfileType)
        try:
            inputFile = inputFile.set_index('scorer')
        except KeyError:
            pass
        inputFile = inputFile.loc[:, ~inputFile.columns.str.contains('^Unnamed')]
        inputFileOrganised = drop_bp_cords(inputFile, inifile)

        currVidInfoDf, currPixPerMM, currVidFps = read_video_info(vidinfDf, str(file_name))
        outputDf = inputFile.copy(deep=True)

        for model in model_dict:
            shortest_bout = model_dict[model]['minimum_bout_length']
            framesToPlugList = list(range(1, int(currVidFps * (shortest_bout / 1000)) + 1))
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

            currentModelPath = model_dict[model]['model_path']
            currModelName = model_dict[model]['model_name']
            discrimination_threshold = model_dict[model]['threshold']
            currProbName = 'Probability_' + currModelName
            clf = pickle.load(open(currentModelPath, 'rb'))

            try:
                predictions = clf.predict_proba(inputFileOrganised)
            except ValueError as e:
                print(e.args)
                print('Mismatch in the number of features in input file and what is expected from the model in file ' + str(file_name) + ' and model ' + str(currModelName))

            try:
                outputDf[currProbName] = predictions[:, 1]
            except IndexError as e:
                print(e.args)
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


#rfmodel(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_2_black_060320\project_folder\project_config.ini")