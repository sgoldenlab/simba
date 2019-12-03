import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import statistics
import os
from configparser import ConfigParser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def rfmodel(inifile):
    print('Running RF model...')
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'features_extracted')
    csv_dir_out = os.path.join(csv_dir, 'machine_results')
    use_master = config.get('General settings', 'use_master_config')
    discrimination_threshold = config.getfloat('validation/run model', 'discrimination_threshold')
    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    model_dir = config.get('SML settings', 'model_dir')
    model_nos = config.getint('SML settings', 'No_targets')
    shortest_bout = config.getint('validation/run model', 'shortest_bout')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    currentAttackGapList = []
    filesFound = []
    model_paths = []
    target_names = []
    loop = 1
    vNm_list = []
    log_df = pd.DataFrame()
    configFilelist = []
    loopy = 0

    ########### GET MODEL PATHS AND NAMES ###########
    for i in range(model_nos):
        currentModelPaths = 'model_path_' + str(loop)
        currentModelNames = 'target_name_' + str(loop)
        currentModelPaths = config.get('SML settings', currentModelPaths)
        currentModelNames = config.get('SML settings', currentModelNames)
        model_paths.append(currentModelPaths)
        target_names.append(currentModelNames)
        loop += 1
    loop = 0
    target_frames_found = []

    ########### FIND CSV FILES ###########
    if use_master == 'yes':
        for i in os.listdir(csv_dir_in):
            if i.__contains__(".csv"):
                file = os.path.join(csv_dir_in, i)
                filesFound.append(file)
    if use_master == 'no':
        config_folder_path = config.get('General settings', 'config_folder')
        for i in os.listdir(config_folder_path):
            if i.__contains__(".ini"):
                configFilelist.append(os.path.join(config_folder_path, i))
                iniVidName = i.split(".")[0]
                csv_fn = iniVidName + '.csv'
                file = os.path.join(csv_dir_in, csv_fn)
                filesFound.append(file)

    for i in filesFound:
        currFile = i
        currentFileName = os.path.basename(currFile)
        loopy += 1
        inputFile = pd.read_csv(currFile, index_col=0)
        inputFileOrganised = inputFile.drop(
            ["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y", "Ear_right_1_p",
             "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p", "Lat_left_1_x",
             "Lat_left_1_y",
             "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x", "Tail_base_1_y",
             "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p", "Ear_left_2_x",
             "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p", "Nose_2_x", "Nose_2_y",
             "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x", "Lat_left_2_y",
             "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x", "Tail_base_2_y",
             "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p", ], axis=1)
        video_no = inputFileOrganised.pop('video_no').values
        frame_number = inputFileOrganised.pop('frames').values
        currVidInfoDf = vidinfDf.loc[vidinfDf['Video'] == str(currentFileName.replace('.csv', ''))]
        currVidFps = int(currVidInfoDf['fps'])
        outputDf = inputFile.copy()
        outputDf.reset_index()
        outputDf['frames'] = outputDf.index
        video_no = outputDf.pop('video_no').values

        # CREATE LIST OF GAPS BASED ON SHORTEST BOUT
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

        for b in range(model_nos):
            currentModelPath = model_paths[b]
            model = os.path.join(model_dir, currentModelPath)
            currModelName = os.path.basename(model)
            currModelName = currModelName.split('.')
            currModelName = str(currModelName[0])
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

        mouse1size = (statistics.mean(outputDf['Mouse_1_nose_to_tail']))
        mouse2size = (statistics.mean(outputDf['Mouse_2_nose_to_tail']))
        mouse1Max = mouse1size * 8
        mouse2Max = mouse2size * 8
        outputDf['Scaled_movement_M1'] = (outputDf['Total_movement_all_bodyparts_M1'] / (mouse1Max))
        outputDf['Scaled_movement_M2'] = (outputDf['Total_movement_all_bodyparts_M2'] / (mouse2Max))
        outputDf['Scaled_movement_M1_M2'] = (outputDf['Scaled_movement_M1'] + outputDf['Scaled_movement_M2']) / 2
        outputDf['Scaled_movement_M1_M2'] = outputDf['Scaled_movement_M1_M2'].round(decimals=2)

        outFname = os.path.basename(currFile)
        outFname = os.path.join(csv_dir_out, outFname)
        outputDf.to_csv(outFname)
        print(str(outFname) + str(' completed'))