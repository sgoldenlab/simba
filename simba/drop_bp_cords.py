import pandas as pd
import os
from pathlib import Path
from configparser import ConfigParser, NoOptionError, NoSectionError
import glob
import re
import sys
from pylab import cm
import shutil
from datetime import datetime
from simba.rw_dfs import *


def drop_bp_cords(dataFrame, inifile):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    columnHeaders = []
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        columnHeaders.extend((colHead1, colHead2, colHead3))
    try:
        dataFrame = dataFrame.drop(columnHeaders, axis=1)
    except KeyError:
        print('Could not drop bodypart coordinates, bodypart coordinates are missing in dataframe')

    return dataFrame

def define_bp_drop_down(configini):
    config = ConfigParser()
    configFile = str(configini)
    config.read(configFile)
    animalno = config.getint('General settings', 'animal_no')
    try:
        IDList = config.get('Multi animal IDs', 'id_list')
    except NoSectionError:
        IDList = []

    # get list
    bpcsv = (os.path.join(os.path.dirname(configini), 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv'))
    bplist = []
    with open(bpcsv) as f:
        for row in f:
            bplist.append(row)
    bplist = list(map(lambda x: x.replace('\n', ''), bplist))

    if not IDList:
        if animalno != 1:
            animal1bp = [f for f in bplist if '_1' in f]
            animal2bp = [f for f in bplist if '_2' in f]
            return animal1bp,animal2bp
        else:
            animal1bp = bplist
            return animal1bp,['No body parts']

    #multianimal
    if IDList:
        if animalno != 1:
            IDList = IDList.split(",")
            animalBpLists = []
            for animal in IDList:
                animalBpLists.append([f for f in bplist if animal in f])
            if not animalBpLists[0]:
                animalBpLists = []
                for animal in range(animalno):
                    currStr = '_' + str(animal + 1)
                    currAnimalBp = [f for f in bplist if currStr in f]
                    animalBpLists.append(currAnimalBp)
            return animalBpLists
        else:
            animal1bp = bplist
            return animal1bp,['No body parts']

def define_movement_cols(multiAnimalIDList):
    columnNames = ['Video', 'Frames processed']
    if len(multiAnimalIDList) == 1:
        columnNames = ['Video', 'Frames processed', 'Total movement (cm)', 'Mean velocity (cm / s)', 'Median velocity (cm/s)']
    if len(multiAnimalIDList) == 2:
        movementCols, meanVelCols, MedianVelCols = [], [], []
        for animal in multiAnimalIDList:
            movementCols.append( 'Total movement (cm) ' + animal)
            meanVelCols.append('Mean velocity (cm/s) ' + animal)
            MedianVelCols.append('Median velocity (cm/s) ' + animal)
        columnNames = columnNames + movementCols + meanVelCols + MedianVelCols
        str4, str5 = 'Mean animal distance (cm)', 'Median animal distance (cm)'
        columnNames.extend((str4, str5))
    if len(multiAnimalIDList) > 2:
        for animal in multiAnimalIDList:
            str1, str2, str3, = 'Total movement (cm) ' + animal, 'Mean velocity (cm / s) ' + animal, 'Median velocity (cm / s) ' + animal
            columnNames.extend((str1, str2, str3))

    return columnNames

def bodypartConfSchematic():
    optionsBaseListImagesPath = os.path.join(os.path.dirname(__file__), 'pose_configurations', 'schematics')
    optionsBaseListNamesPath = os.path.join(os.path.dirname(__file__), 'pose_configurations', 'configuration_names', 'pose_config_names.csv')
    optionsBaseNameList = pd.read_csv(optionsBaseListNamesPath, header=None)
    optionsBaseNameList = list(optionsBaseNameList[0])
    optionsBaseNameList.append('Create pose config...')
    optionsBasePhotosList = glob.glob(optionsBaseListImagesPath + '/*.png')
    optionsBasePhotosList.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return optionsBaseNameList, optionsBasePhotosList


def GenerateMetaDataFileHeaders():
    metaDataHeaders = ["Classifier_name", "RF_criterion", "RF_max_features", "RF_min_sample_leaf",
     "RF_n_estimators", "compute_feature_permutation_importance",
     "generate_classification_report", "generate_example_decision_tree",
     "generate_features_importance_bar_graph", "generate_features_importance_log",
     "generate_precision_recall_curves", "generate_rf_model_meta_data_file",
     "generate_sklearn_learning_curves", "learning_curve_data_splits",
     "learning_curve_k_splits", "n_feature_importance_bars",
     "over_sample_ratio", "over_sample_setting", "train_test_size", "under_sample_ratio",
     "under_sample_setting"]

    return metaDataHeaders

def getBpNames(inifile):
    Xcols, Ycols, Pcols = ([],[],[])
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = str(os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv'))
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        Xcols.append(colHead1)
        Ycols.append(colHead2)
        Pcols.append(colHead3)
    return Xcols, Ycols, Pcols

def getBpHeaders(inifile):
    colHeads = []
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names','project_bp_names.csv')
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        colHeads.extend((colHead1, colHead2, colHead3))
    return colHeads


def create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, animalsNo, Xcols, Ycols, Pcols, colorListofList):
    animalBpDict = {}
    if multiAnimalStatus == True:
        for animal in range(animalsNo):
            animalBpDict[multiAnimalIDList[animal]] = {}
            animalBpDict[multiAnimalIDList[animal]]['X_bps'] = [i for i in Xcols if multiAnimalIDList[animal] in i]
            animalBpDict[multiAnimalIDList[animal]]['Y_bps'] = [i for i in Ycols if multiAnimalIDList[animal] in i]
            if colorListofList:
                animalBpDict[multiAnimalIDList[animal]]['colors'] = colorListofList[animal]
            if Pcols:
                animalBpDict[multiAnimalIDList[animal]]['P_bps'] = [i for i in Pcols if multiAnimalIDList[animal] in i]
            if not animalBpDict[multiAnimalIDList[animal]]['X_bps']:
                multiAnimalStatus = False
                break

    if multiAnimalStatus == False:
        if animalsNo > 1:
            for animal in range(animalsNo):
                currAnimalName = 'Animal_' + str(animal + 1)
                search_string_x = '_' + str(animal + 1) + '_x'
                search_string_y = '_' + str(animal + 1) + '_y'
                search_string_p = '_' + str(animal + 1) + '_p'
                animalBpDict[currAnimalName] = {}
                animalBpDict[currAnimalName]['X_bps'] = [i for i in Xcols if i.endswith(search_string_x)]
                animalBpDict[currAnimalName]['Y_bps'] = [i for i in Ycols if i.endswith(search_string_y)]
                if colorListofList:
                    animalBpDict[currAnimalName]['colors'] = colorListofList[animal]
                if Pcols:
                    animalBpDict[currAnimalName]['P_bps'] = [i for i in Pcols if i.endswith(search_string_p)]
            if multiAnimalIDList[0] != '':
                for animal in range(len(multiAnimalIDList)):
                    currAnimalName = 'Animal_' + str(animal + 1)
                    animalBpDict[multiAnimalIDList[animal]] = animalBpDict.pop(currAnimalName)
            # else:
            #     for animal in range(len(multiAnimalIDList)):
            #         animalBpDict[multiAnimalIDList[animal]] = animalBpDict.pop(currAnimalName)

        else:
            animalBpDict['Animal_1'] = {}
            animalBpDict['Animal_1']['X_bps'] = [i for i in Xcols]
            animalBpDict['Animal_1']['Y_bps'] = [i for i in Ycols]
            if colorListofList:
                animalBpDict['Animal_1']['colors'] = colorListofList[0]
            if Pcols:
                animalBpDict['Animal_1']['P_bps'] = [i for i in Pcols]
    return animalBpDict


def checkDirectionalityCords(animalBpDict):
    directionalityDict = {}
    for animal in animalBpDict:
        directionalityDict[animal] = {}
        directionalityDict[animal]['Nose'] = {}
        directionalityDict[animal]['Ear_left'] = {}
        directionalityDict[animal]['Ear_right'] = {}
        for cord in animalBpDict[animal]:
            for columnName in animalBpDict[animal][cord]:
                if ("Nose".lower() in columnName.lower()) and ("X".lower() in columnName.lower()):
                    directionalityDict[animal]['Nose']['X_bps'] = columnName
                if ("Nose".lower() in columnName.lower()) and ("Y".lower() in columnName.lower()):
                    directionalityDict[animal]['Nose']['Y_bps'] = columnName
                if ("Left".lower() in columnName.lower()) and ("X".lower() in columnName.lower()) and ("ear".lower() in columnName.lower()):
                    directionalityDict[animal]['Ear_left']['X_bps'] = columnName
                if ("Left".lower() in columnName.lower()) and ("Y".lower() in columnName.lower()) and ("ear".lower() in columnName.lower()):
                    directionalityDict[animal]['Ear_left']['Y_bps'] = columnName
                if ("Right".lower() in columnName.lower()) and ("X".lower() in columnName.lower()) and ("ear".lower() in columnName.lower()):
                    directionalityDict[animal]['Ear_right']['X_bps'] = columnName
                if ("Right".lower() in columnName.lower()) and ("Y".lower() in columnName.lower()) and ("ear".lower() in columnName.lower()):
                    directionalityDict[animal]['Ear_right']['Y_bps'] = columnName
    return directionalityDict

def createColorListofList(noAnimals, cMapSize):
    colorListofList = []
    cmaps = ['spring', 'summer', 'autumn', 'cool', 'Wistia', 'Pastel1', 'Set1', 'winter']
    for colormap in range(noAnimals):
        currColorMap = cm.get_cmap(cmaps[colormap], cMapSize)
        currColorList = []
        for i in range(currColorMap.N):
            rgb = list((currColorMap(i)[:3]))
            rgb = [i * 255 for i in rgb]
            rgb.reverse()
            currColorList.append(rgb)
        colorListofList.append(currColorList)
    return colorListofList


def reverse_dlc_input_files(configini):
    dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    config = ConfigParser()
    config.read(configini)
    projectPath = config.get('General settings', 'project_path')

    input_folder_path = os.path.join(projectPath, 'csv', 'input_csv')
    animalsNo = config.getint('General settings', 'animal_no')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
    pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
    if not multiAnimalIDList:
        multiAnimalIDList = []
        for animal in range(animalsNo):
            multiAnimalIDList.append('Animal_' + str(animal + 1))
            multiAnimalStatus = False
    else:
        multiAnimalIDList, multiAnimalStatus = multiAnimalIDList.split(","), True
    Xcols, Ycols, Pcols = getBpNames(configini)
    animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, animalsNo, Xcols, Ycols, [], [])
    bps_columns_numbers_per_animal = [[0]]
    for cur_animal, animal in enumerate(multiAnimalIDList):
        if cur_animal == 0: bps_columns_numbers_per_animal[cur_animal].append(len(animalBpDict[animal]['X_bps'])*3)
        else: bps_columns_numbers_per_animal.append([bps_columns_numbers_per_animal[-1][1], bps_columns_numbers_per_animal[-1][1] + len(animalBpDict[animal]['X_bps'])*3])
    filesFound = glob.glob(input_folder_path + '/*.' + wfileType)
    store_original_files_folder = os.path.join(input_folder_path, 'Original_tracking_files_' +str(dateTime))
    if not os.path.exists(store_original_files_folder): os.makedirs(store_original_files_folder)

    for file in filesFound:
        print('Reversing ', os.path.basename(file) + '...')
        currentDf = read_df(file, wfileType)
        df_list = []
        reversed_df = pd.DataFrame()
        for animal in range(animalsNo): df_list.append(currentDf[list(currentDf.columns[bps_columns_numbers_per_animal[animal][0]:bps_columns_numbers_per_animal[animal][1]])])
        for curr_df in reversed(df_list): reversed_df = pd.concat([reversed_df, curr_df], axis=1)
        reversed_df.columns = currentDf.columns
        reversed_df.iloc[0:1] = currentDf.iloc[0:1]
        shutil.move(file, os.path.join(store_original_files_folder, os.path.basename(file)))
        save_df(reversed_df, wfileType, file)
    print('All reversals complete.')

def get_fn_ext(filepath):
    ext = Path(filepath).suffix
    file_name = os.path.basename(filepath.rsplit(ext, 1)[0])
    dir_name = os.path.dirname(filepath)
    return dir_name, file_name, ext


def get_workflow_file_format(config):
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
        return wfileType
    except NoOptionError:
        return 'csv'

