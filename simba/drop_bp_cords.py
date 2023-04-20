__author__ = "Simon Nilsson", "JJ Choong"

from simba.utils.warnings import BodypartColumnNotFoundWarning
import pandas as pd
import os
from pathlib import Path
from configparser import (ConfigParser,
                          NoOptionError)
import glob
from pylab import cm
import shutil
from datetime import datetime
from simba.rw_dfs import read_df, save_df
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_file_exist_and_readable)
from simba.utils.errors import InvalidFilepathError
from simba.enums import Paths, ReadConfig, Dtypes


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
         column names for body-part coordinates on y-axis
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
        BodypartColumnNotFoundWarning(msg=f'SimBA could not drop body-part coordinates, some body-part names are missing in dataframe. SimBA expected the following body-parts, that could not be found inside the file: {e.args[0]}')

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


def getBpHeaders(inifile: str):
    """
    Helper to create ordered list of all column header fields for SimBA project dataframes.

    Parameters
    ----------
    inifile: str
        Path to SimBA project_config.ini

    Returns
    -------
    list
    """

    config = read_config_file(inifile)
    project_path = read_config_entry(config=config, section=ReadConfig.GENERAL_SETTINGS.value, option=ReadConfig.PROJECT_PATH.value, data_type=Dtypes.STR.value)
    bp_path = os.path.join(project_path, Paths.BP_NAMES.value)
    check_file_exist_and_readable(file_path=bp_path)
    bp_df = pd.read_csv(bp_path, header=None)
    bp_lst = list(bp_df[0])
    results = []
    for bp in bp_lst:
        c1, c2, c3 = (f'{bp}_x', f'{bp}_y', f'{bp}_p')
        results.extend((c1, c2, c3))
    return results

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
        raise InvalidFilepathError(msg='{} is not a valid filepath'.format(filepath))
    dir_name = os.path.dirname(filepath)
    return dir_name, file_name, file_extension




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
    cmaps = ['spring',
            'summer',
            'autumn',
            'cool',
            'Wistia',
            'Pastel1',
            'Set1',
            'winter',
            'afmhot',
            'gist_heat',
            'copper']
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



