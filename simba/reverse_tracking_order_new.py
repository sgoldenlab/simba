__author__ = "Simon Nilsson", "JJ Choong"

import os, glob, shutil
import pandas as pd
from datetime import datetime
from configparser import ConfigParser, NoOptionError
from simba.rw_dfs import *
from simba.drop_bp_cords import *
from simba.features_scripts.extract_features_16bp import extract_features_wotarget_16
from simba.features_scripts.extract_features_14bp import extract_features_wotarget_14
from simba.features_scripts.extract_features_14bp_from_16bp import extract_features_wotarget_14_from_16
from simba.features_scripts.extract_features_9bp import extract_features_wotarget_9
from simba.features_scripts.extract_features_8bp import extract_features_wotarget_8
from simba.features_scripts.extract_features_7bp import extract_features_wotarget_7
from simba.features_scripts.extract_features_4bp import extract_features_wotarget_4
from simba.features_scripts.extract_features_user_defined import extract_features_wotarget_user_defined


configini = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\reverse_classifier\project_folder\project_config.ini"


config = ConfigParser()
config.read(configini)
projectPath = config.get('General settings', 'project_path')
model_nos = config.getint('SML settings', 'No_targets')
input_folder_path = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location')
dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
store_path_outliers = os.path.join(input_folder_path, 'Non_reversed_files_at_' +str(dateTime))
if not os.path.exists(store_path_outliers): os.makedirs(store_path_outliers)
features_extracted_path = os.path.join(projectPath, 'csv', 'features_extracted')
target_inserted_path = os.path.join(projectPath, 'csv', 'targets_inserted')
store_path_features = os.path.join(features_extracted_path, 'Non_reversed_files_at_' +str(dateTime))
if not os.path.exists(store_path_features): os.makedirs(store_path_features)
store_path_targets = os.path.join(target_inserted_path, 'Non_reversed_files_at_' +str(dateTime))
if not os.path.exists(store_path_targets): os.makedirs(store_path_targets)
animalsNo = config.getint('General settings', 'animal_no')
try: wfileType = config.get('General settings', 'workflow_file_type')
except NoOptionError: wfileType = 'csv'
multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
pose_estimation_body_parts = config.get('create ensemble settings', 'pose_estimation_body_parts')
if not multiAnimalIDList:
    multiAnimalIDList = []
    for animal in range(animalsNo): multiAnimalIDList.append('Animal_' + str(animal + 1))
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

#REVERSE THE DATAFRAMES
for file in filesFound:
    print('Reversing ', os.path.basename(file) + '...')
    currentDf = read_df(file, wfileType)
    df_list  = []
    reversed_df = pd.DataFrame()
    for animal in range(animalsNo): df_list.append(currentDf[list(currentDf.columns[bps_columns_numbers_per_animal[animal][0]:bps_columns_numbers_per_animal[animal][1]])])
    for curr_df in reversed(df_list): reversed_df = pd.concat([reversed_df, curr_df], axis=1)
    shutil.move(file, os.path.join(store_path_outliers, os.path.basename(file)))
    save_df(reversed_df, wfileType, file)
reversed_column_names = list(reversed_df.columns)

#RECALCULATE FEATURES
print('Re-calculating features...')
old_feature_files = glob.glob(features_extracted_path + '/*.' + wfileType)
for file in old_feature_files: shutil.move(file, os.path.join(store_path_features, os.path.basename(file)))
if pose_estimation_body_parts == '16':
    extract_features_wotarget_16(configini)
if (pose_estimation_body_parts == '14'):
    extract_features_wotarget_14(configini)
if (pose_estimation_body_parts == '987'):
    extract_features_wotarget_14_from_16(configini)
if pose_estimation_body_parts == '9':
    extract_features_wotarget_9(configini)
if pose_estimation_body_parts == '8':
    extract_features_wotarget_8(configini)
if pose_estimation_body_parts == '7':
    extract_features_wotarget_7(configini)
if pose_estimation_body_parts == '4':
    extract_features_wotarget_4(configini)
if pose_estimation_body_parts == 'user_defined':
    extract_features_wotarget_user_defined(configini)

#RE-APPEND TARGETS
target_names_list = []
print('Re-appending_human annotations...')
old_target_files = glob.glob(target_inserted_path + '/*.' + wfileType)
for file in old_target_files: shutil.move(file, os.path.join(store_path_targets, os.path.basename(file)))
for i in range(model_nos): target_names_list.append(config.get('SML settings', 'target_name_' + str(i+1)))
new_feature_files = glob.glob(features_extracted_path + '/*.' + wfileType)
for file in new_feature_files:
    currentDf = read_df(file, wfileType)
    try:
        old_target_df = read_df(os.path.join(store_path_targets, os.path.basename(file)), wfileType)
        for target_col in target_names_list:
            currentDf[target_col] = old_target_df[target_col]
        save_file_path = os.path.join(target_inserted_path, os.path.basename(file))
        save_df(currentDf, wfileType, save_file_path)
    except TypeError:
        print('Could not locate the prior human annotations file ' + str(os.path.basename(file)) + ' inside the project_folder/csv/targets_inserted directory.')
        continue

print('Re-ordering complete. Reversed annotations saves inside the project_folder/csv/targets_inserted folder. You can now go a head and train a new classifier with your reversed annotations. See the SimBA Github repository for more information')