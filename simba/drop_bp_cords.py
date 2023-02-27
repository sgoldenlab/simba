__author__ = "Simon Nilsson", "JJ Choong"

import configparser
import pandas as pd
import os
from pathlib import Path
from configparser import (ConfigParser,
                          NoOptionError,
                          NoSectionError)
import glob
import re
from pylab import cm
import shutil
from datetime import datetime
from simba.rw_dfs import read_df, save_df
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_file_exist_and_readable)
from simba.enums import ReadConfig, Paths


def create_body_part_dictionary(multiAnimalStatus: bool,
                                multiAnimalIDList: list,
                                animalsNo: int,
                                Xcols: list,
                                Ycols: list,
                                Pcols: list,
                                colorListofList=None):
    """
    Helper to create dict of dict lookup of body-parts where the keys are animal names, and
    values are the body-part names.

    Parameters
    ----------
    multiAnimalStatus: bool
        If True, it is a multi-animal SimBA project
    multiAnimalIDList: list
        List of animal names. Eg., ['Animal_1, 'Animals_2']
    animalsNo: int
        Number of animals in the SimBA project.
    Xcols: list
        list of column names for body-part coordinates on x-axis
    Ycols: list
        list of column names for body-part coordinates on y-axis
    Pcols: list
        list of column names for body-part pose-estimation probability values
    colorListofList: list or None
        List of list of bgr colors.
    Returns
    -------
    animalBpDict: dict
    """

    animalBpDict = {}
    if multiAnimalStatus:
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

    if not multiAnimalStatus:
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

        else:
            animalBpDict['Animal_1'] = {}
            animalBpDict['Animal_1']['X_bps'] = [i for i in Xcols]
            animalBpDict['Animal_1']['Y_bps'] = [i for i in Ycols]
            if colorListofList:
                animalBpDict['Animal_1']['colors'] = colorListofList[0]
            if Pcols:
                animalBpDict['Animal_1']['P_bps'] = [i for i in Pcols]
    return animalBpDict

def getBpNames(inifile: str):
    """
    Helper to extract pose-estimation data field names (x, y, p) .

    Parameters
    ----------
    inifile: str
        Path to SimBA project_config.ini

    Returns
    -------
    x_cols: list
        list of column names for body-part coordinates on x-axis
    y_cols: list
        list of column names for body-part coordinates on y-axis
    p_cols: list
        list of column names for body-part pose-estimation probability values
    """

    x_cols, y_cols, p_cols = [], [], []
    config = read_config_file(ini_path=inifile)
    project_path = read_config_entry(config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.PROJECT_PATH.value, data_type=ReadConfig.FOLDER_PATH.value)
    body_part_lst_path = str(os.path.join(project_path, Paths.BP_NAMES.value))
    pose_config_lst = pd.read_csv(body_part_lst_path, header=None).iloc[:, 0].to_list()
    pose_config_lst = [x for x in pose_config_lst if str(x) != 'nan']
    for bodypart in pose_config_lst:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        x_cols.append(colHead1)
        y_cols.append(colHead2)
        p_cols.append(colHead3)
    return x_cols, y_cols, p_cols


def define_bp_drop_down(configini: str):
    """
    Helper to create list of animal body-parts for Tkinter drop-down menus.

    Parameters
    ----------
    configini

    Returns
    -------
    animal_bp_lists: list (list of list of str)
    """

    from simba.misc_tools import check_multi_animal_status
    config = read_config_file(ini_path=configini)
    no_animals = read_config_entry(config=config, section=ReadConfig.GENERAL_SETTINGS.value,option=ReadConfig.ANIMAL_CNT.value,data_type='int')
    multi_animal_status, multi_animal_id_lst = check_multi_animal_status(config, no_animals)
    x_cols, y_cols, pcols = getBpNames(configini)
    animal_bp_dict = create_body_part_dictionary(multi_animal_status, multi_animal_id_lst, no_animals, x_cols, y_cols, [], [])
    animal_bp_lists = []
    for animal_name, animal_data in animal_bp_dict.items():
        animal_bp_lists.append([x[0:-2] for x in animal_data['X_bps']])
    return animal_bp_lists



# def define_bp_drop_down(configini):
#     config = ConfigParser()
#     configFile = str(configini)
#     config.read(configFile)
#     animalno = config.getint('General settings', 'animal_no')
#     try:
#         IDList = config.get('Multi animal IDs', 'id_list')
#     except NoSectionError:
#         IDList = []
#
#     # get list
#     bpcsv = (os.path.join(os.path.dirname(configini), 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv'))
#     bplist = []
#     with open(bpcsv) as f:
#         for row in f:
#             bplist.append(row)
#     bplist = list(map(lambda x: x.replace('\n', ''), bplist))
#
#     if not IDList:
#         if animalno != 1:
#             animal1bp = [f for f in bplist if '_1' in f]
#             animal2bp = [f for f in bplist if '_2' in f]
#             return animal1bp,animal2bp
#         else:
#             animal1bp = bplist
#             return animal1bp,['No body parts']
#
#     #multianimal
#     if IDList:
#         if animalno != 1:
#             IDList = IDList.split(",")
#             animalBpLists = []
#             for animal in IDList:
#                 animalBpLists.append([f for f in bplist if animal in f])
#             if not animalBpLists[0]:
#                 animalBpLists = []
#                 for animal in range(animalno):
#                     currStr = '_' + str(animal + 1)
#                     currAnimalBp = [f for f in bplist if currStr in f]
#                     animalBpLists.append(currAnimalBp)
#             return animalBpLists
#         else:
#             animal1bp = bplist
#             return animal1bp, ['No body parts']
#
#
#
#



def drop_bp_cords(df: pd.DataFrame,
                  config_path: str):

    """
    Helper to remove pose-estimation data from dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        pandas dataframe containing pose-estimation data
    config_path: bool
        path to SimBA project config file in Configparser format

    Returns
     ----------
     out_df: pd.DataFrame
        Dataframe w/o pose-estimation data

    Examples
    -----
    >>> df_wo_pose = drop_bp_cords(df='DataFrameWithPose', config_path='MySimBAConfigfile')
    """

    config = read_config_file(config_path)
    project_path = read_config_entry(config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.PROJECT_PATH.value, data_type='folder_path')
    body_part_list_path = os.path.join(project_path, Paths.BP_NAMES.value)
    check_file_exist_and_readable(body_part_list_path)
    pose_df = pd.read_csv(body_part_list_path, header=None)
    pose_lst = list(pose_df[0])
    bp_headers = []
    for bodypart in pose_lst:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        bp_headers.extend((colHead1, colHead2, colHead3))
    try:
        out_df = df.drop(bp_headers, axis=1)
        return out_df
    except KeyError as e:
        print('SIMBA WARNING: SimBA could not drop bodypart coordinates, some bodypart names are missing in dataframe. SimBA expected the following body-parts, that could not be found inside the file:')
        print(e.args[0])

def define_movement_cols(multi_animal_id_list: list):
    """
    Helper to create column names representing aggregate movement and velocity for each animal in the video.

    Parameters
    ----------
    multi_animal_id_list: list
        list of animal names

    Examples
    -----
    >>> column_names = define_movement_cols(multi_animal_id_list=['Animal_1', 'Animal_2' 'Animal_3'])
    """

    columnNames = ['Video', 'Frames processed']
    if len(multi_animal_id_list) == 1:
        columnNames = ['Video', 'Frames processed', 'Total movement (cm)', 'Mean velocity (cm / s)', 'Median velocity (cm/s)']
    if len(multi_animal_id_list) == 2:
        movementCols, meanVelCols, MedianVelCols = [], [], []
        for animal in multi_animal_id_list:
            movementCols.append( 'Total movement (cm) ' + animal)
            meanVelCols.append('Mean velocity (cm/s) ' + animal)
            MedianVelCols.append('Median velocity (cm/s) ' + animal)
        columnNames = columnNames + movementCols + meanVelCols + MedianVelCols
        str4, str5 = 'Mean animal distance (cm)', 'Median animal distance (cm)'
        columnNames.extend((str4, str5))
    if len(multi_animal_id_list) > 2:
        for animal in multi_animal_id_list:
            str1, str2, str3, = 'Total movement (cm) ' + animal, 'Mean velocity (cm / s) ' + animal, 'Median velocity (cm / s) ' + animal
            columnNames.extend((str1, str2, str3))

    return columnNames

def bodypartConfSchematic():
    """Helper to return (i) named body-part schematics of all pose-estimation schemas in SimBA installation, and
    (ii) the paths to the images representing those  body-part schematics in SimBA installation
    """

    optionsBaseListImagesPath = os.path.join(os.path.dirname(__file__), Paths.SCHEMATICS.value)
    optionsBaseListNamesPath = os.path.join(os.path.dirname(__file__), 'pose_configurations', 'configuration_names', 'pose_config_names.csv')
    optionsBaseNameList = pd.read_csv(optionsBaseListNamesPath, header=None)
    optionsBaseNameList = list(optionsBaseNameList[0])
    optionsBaseNameList.append('Create pose config...')
    optionsBasePhotosList = glob.glob(optionsBaseListImagesPath + '/*.png')
    optionsBasePhotosList.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    return optionsBaseNameList, optionsBasePhotosList


def GenerateMetaDataFileHeaders():
    """Helper to return the headings for the SimBA RF model config files."""


    meta_data_headers = ["Classifier_name",
                         "RF_criterion",
                         "RF_max_features",
                         "RF_min_sample_leaf",
                         "RF_n_estimators",
                         "compute_feature_permutation_importance",
                         "generate_classification_report",
                         "generate_example_decision_tree",
                         "generate_features_importance_bar_graph",
                         "generate_features_importance_log",
                         "generate_precision_recall_curves",
                         "generate_rf_model_meta_data_file",
                         "generate_sklearn_learning_curves",
                         "learning_curve_data_splits",
                         "learning_curve_k_splits",
                         "n_feature_importance_bars",
                         "over_sample_ratio",
                         "over_sample_setting",
                         "train_test_size",
                         "train_test_split_type",
                         "under_sample_ratio",
                         "under_sample_setting",
                         "class_weights"]

    return meta_data_headers


def getBpHeaders(inifile: str):
    """
    Helper to create ordered list of all column header fields for SimBA project dataframes.

    Parameters
    ----------
    inifile: str
        Path to SimBA project_config.ini

    Returns
    -------
    column_headers: list
    """

    column_headers = []
    config = read_config_file(inifile)
    project_path = config.get('General settings', 'project_path')
    bodyparthListPath = os.path.join(project_path, 'logs', 'measures', 'pose_configs', 'bp_names','project_bp_names.csv')
    poseConfigDf = pd.read_csv(bodyparthListPath, header=None)
    poseConfigList = list(poseConfigDf[0])
    for bodypart in poseConfigList:
        colHead1, colHead2, colHead3 = (bodypart + '_x', bodypart + '_y', bodypart + '_p')
        column_headers.extend((colHead1, colHead2, colHead3))
    return column_headers


def checkDirectionalityCords(animal_bp_dict: dict):
    """
    Helper to check if ear and nose body-parts are present within the pose-estimation data.

    Parameters
    ----------
    animal_bp_dict: dict
        Python dictionary created by ``create_body_part_dictionary``.

    Returns
    -------
    directionalityDict: dict
        Python dictionary populated with body-part names of ear and nose body-parts. If empty,
        ear and nose body-parts are not present within the pose-estimation data
    """

    directionalityDict = {}
    for animal in animal_bp_dict:
        directionalityDict[animal] = {}
        directionalityDict[animal]['Nose'] = {}
        directionalityDict[animal]['Ear_left'] = {}
        directionalityDict[animal]['Ear_right'] = {}
        for cord in animal_bp_dict[animal]:
            for columnName in animal_bp_dict[animal][cord]:
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

def createColorListofList(no_animals: int,
                          map_size: int):
    """
    Helper to return a list of lists of bgr colors. Each list is pulled from a different palette
    matplotlib color map.

    Parameters
    ----------
    no_animals: int
        Number of different palette lists
    map_size: int
        Number of colors in each created palette.

    Returns
     ----------
     colorListofList: list
        List of lists holding bgr colors

    Notes
    -----

    Examples
    -----
    >>> colorListofList = createColorListofList(no_animals=2, map_size=8)
    """

    colorListofList = []
    cmaps = ['spring', 'summer', 'autumn', 'cool', 'Wistia', 'Pastel1', 'Set1', 'winter', 'afmhot', 'gist_heat', 'copper']
    for colormap in range(no_animals):
        currColorMap = cm.get_cmap(cmaps[colormap], map_size)
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

def get_fn_ext(filepath: str):
    """
    Helper to split file path into three components: (i) directory, (ii) file name, and (iii) file extension.

    Parameters
    ----------
    filepath: str
        Path to file.

    Returns
    -------
    dir_name: str
    file_name: str
    file_extension: str

    """
    file_extension = Path(filepath).suffix
    try:
        file_name = os.path.basename(filepath.rsplit(file_extension, 1)[0])
    except ValueError:
        print('SIMBA ERROR: {} is not a valid filepath'.format(filepath))
        raise ValueError('SIMBA ERROR: {} is not a valid filepath'.format(filepath))
    dir_name = os.path.dirname(filepath)
    return dir_name, file_name, file_extension


def get_workflow_file_format(config: configparser.ConfigParser):
    """
    Helper to read the workflow file format in SimBA project. If missing, then defaults to CSV.

    Parameters
    ----------
    config: configparser.ConfigParser

    Returns
    -------
    wfileType: str
    """

    try:
        wfileType = config.get('General settings', 'workflow_file_type')
        return wfileType
    except NoOptionError:
        return 'csv'

