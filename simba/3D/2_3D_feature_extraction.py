from configparser import ConfigParser, NoOptionError, NoSectionError
import os
from simba.rw_dfs import *
from simba.drop_bp_cords import *
import math
import numpy as np
from numba import jit


INIFILE = '/Volumes/GoogleDrive/My Drive/GitHub/SimBA_DANNCE/project_config.ini'
config = ConfigParser()
configFile = str(INIFILE)
config.read(configFile)
project_path = config.get('General settings', 'project_path')
csv_dir_in = os.path.join(project_path, 'csv', 'outlier_corrected_movement_location')
csv_dir_out = os.path.join(project_path, 'csv', 'features_extracted')
logs_path = os.path.join(project_path, 'logs')
video_info_path = os.path.join(logs_path, 'video_info.csv')
no_animals = config.getint('General settings', 'animal_no')
try:
    wfileType = config.get('General settings', 'workflow_file_type')
except NoOptionError:
    wfileType = 'csv'
video_info_df = pd.read_csv(video_info_path)
video_info_df.Video = video_info_df.Video.astype('str')
Xcols, Ycols, Zcols, Pcols = getBpNames(INIFILE, three_d_data=True)
columnHeaders = getBpHeaders(INIFILE)
columnHeadersShifted = [bp + '_shifted' for bp in columnHeaders]

if not os.path.exists(csv_dir_out):
    os.makedirs(csv_dir_out)
roll_windows = []
roll_windows_values = [2, 5, 6, 7.5, 15]

#REMOVE WINDOWS THAT ARE TOO SMALL
minimum_fps = video_info_df['fps'].min()
for win in range(len(roll_windows_values)):
    if minimum_fps < roll_windows_values[win]:
        roll_windows_values[win] = minimum_fps
    else:
        pass
roll_windows_values = list(set(roll_windows_values))
files_found = glob.glob(csv_dir_in + '/*.' + wfileType)

try:
    multiAnimalIDList = config.get('Multi animal IDs', 'id_list')
    multiAnimalIDList = multiAnimalIDList.split(",")
    if multiAnimalIDList[0] != '':
        multiAnimalStatus = True
        print('Applying settings for multi-animal tracking...')
    else:
        multiAnimalStatus = False
        for animal in range(no_animals):
            multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
        print('Applying settings for classical tracking...')
except NoSectionError:
    multiAnimalIDList = []
    for animal in range(no_animals):
        multiAnimalIDList.append('Animal_' + str(animal + 1) + '_')
    multiAnimalStatus = False
    print('Applying settings for classical tracking...')

animalBpDict = create_body_part_dictionary(multiAnimalStatus, multiAnimalIDList, no_animals, Xcols, Ycols, Pcols, [], Zcols, three_d_data=True)
print('Extracting features from ' + str(len(files_found)) + ' files...')

@jit(nopython=True)
def distance_btw_bps(bp_1_array, bp_2_array):
    dist_arr = np.zeros((0))
    for origin, dest in zip(bp_1_array, bp_2_array):
        dist_arr = np.append(dist_arr, np.linalg.norm(origin - dest))
    return dist_arr

for file in files_found:
    vid_name = os.path.basename(file.replace('.' + wfileType, ''))
    video_settings = video_info_df.loc[video_info_df['Video'] == vid_name]
    px_per_mm = float(video_settings['pixels/mm'])
    fps = float(video_settings['fps'])
    print('Processing ' + '"' + str(vid_name) + '".' + ' Fps: ' + str(fps) + ". mm/ppx: " + str(px_per_mm))
    for i in range(len(roll_windows_values)):
        roll_windows.append(int(fps / roll_windows_values[i]))
    csv_df = read_df(file, wfileType).fillna(0).apply(pd.to_numeric)

    columnHeadersShifted = [x + '_shifted' for x in list(csv_df.columns)]
    csv_df_shifted = csv_df.shift(periods=1)
    csv_df_shifted.columns = columnHeadersShifted
    csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)

    print('Calculating euclidean distances...')
    ########### EUCLIDEAN DISTANCES BETWEEN BODY PARTS###########################################
    for currAnimal in animalBpDict:
        c_anim_X_list, c_anim_Y_list, c_anim_Z_list = animalBpDict[currAnimal]['X_bps'], animalBpDict[currAnimal]['Y_bps'], animalBpDict[currAnimal]['Z_bps']
        for c_bp_x, c_bp_y, c_bp_z in zip(c_anim_X_list, c_anim_Y_list, c_anim_Z_list):
            other_bp_x, other_bp_y, other_bp_z  = [x for x in c_anim_X_list if x != c_bp_x], [x for x in c_anim_Y_list if x != c_bp_y],  [x for x in c_anim_Z_list if x != c_bp_z]
            for o_bp_x, o_bp_y, o_bp_z in zip(other_bp_x, other_bp_y, other_bp_z):
                bpName1, bpName2 = c_bp_x.strip('_x'), o_bp_x.strip('_x')
                col_name = 'Euclidean_distance_' + bpName1 + '_' + bpName2
                reverse_col_name = 'Euclidean_distance_' + bpName2 + '_' + bpName1
                if not reverse_col_name in csv_df.columns:
                    bp_1_array = np.float32(csv_df[[c_bp_x, c_bp_y, c_bp_z]].to_numpy())
                    bp_2_array = np.float32(csv_df[[o_bp_x, o_bp_y, o_bp_z]].to_numpy())
                    csv_df[col_name] = pd.Series(distance_btw_bps(bp_1_array, bp_2_array))

    print('Calculating movements of all bodyparts...')
    collapsedColNamesMean, collapsedColNamesSum = [], []
    for currAnimal in animalBpDict:
        movement_cols = []
        c_anim_X_list, c_anim_Y_list, c_anim_Z_list = animalBpDict[currAnimal]['X_bps'], animalBpDict[currAnimal]['Y_bps'], animalBpDict[currAnimal]['Z_bps']
        for c_bp_x, c_bp_y, c_bp_z in zip(c_anim_X_list, c_anim_Y_list, c_anim_Z_list):
            shif_bp_x_name, shif_bp_y_name, shif_bp_z_name = c_bp_x + '_shifted', c_bp_y + '_shifted', c_bp_z + '_shifted'
            col_name = 'Movement_' + c_bp_x.strip('_x')
            bp_1_array = np.float32(csv_df_combined[[c_bp_x, c_bp_y, c_bp_z]].to_numpy())
            bp_2_array = np.float32(csv_df_combined[[shif_bp_x_name, shif_bp_y_name, shif_bp_z_name]].to_numpy())
            csv_df[col_name] = pd.Series(distance_btw_bps(bp_1_array, bp_2_array))
            movement_cols.append(col_name)
        sumColName, meanColName = 'All_bp_movements_' + currAnimal + '_sum', 'All_bp_movements_' + currAnimal + '_mean'
        csv_df[sumColName] = csv_df[movement_cols].sum(axis=1)
        csv_df[meanColName] = csv_df[movement_cols].mean(axis=1)
        csv_df['All_bp_movements_' + currAnimal + '_min'] = csv_df[movement_cols].min(axis=1)
        csv_df['All_bp_movements_' + currAnimal + '_max'] = csv_df[movement_cols].max(axis=1)
        collapsedColNamesMean.append(meanColName)
        collapsedColNamesSum.append(sumColName)

    print('Calculating rolling windows data: distances between body-parts')
    ########### CALC MEAN & SUM DISTANCES BETWEEN BODY PARTS IN ROLLING WINDOWS ###########################################
    for i in range(len(roll_windows_values)):
        for distance_col in movement_cols:
            colName = 'Mean_' + str(distance_col) + '_' + str(roll_windows_values[i])
            csv_df[colName] = csv_df[distance_col].rolling(roll_windows[i], min_periods=1).mean()
            colName = 'Sum_' + str(distance_col) + '_' + str(roll_windows_values[i])
            csv_df[colName] = csv_df[distance_col].rolling(roll_windows[i], min_periods=1).sum()

    print('Calculating rolling windows data: animal movements')
    for i in range(len(roll_windows_values)):
        for animal in collapsedColNamesMean:
            colName = 'Mean_' + str(animal) + '_' + str(roll_windows_values[i])
            csv_df[colName] = csv_df[animal].rolling(roll_windows[i], min_periods=1).mean()
            colName = 'Sum_' + str(animal) + '_' + str(roll_windows_values[i])
            csv_df[colName] = csv_df[animal].rolling(roll_windows[i], min_periods=1).sum()

    ########### SAVE DF ###########################################
    csv_df = csv_df.reset_index(drop=True).fillna(0)
    savePath = os.path.join(csv_dir_out, os.path.basename(file))
    print('Saving features...')
    save_df(csv_df, wfileType, savePath)
    print('Feature extraction complete for ' + '"' + str(vid_name) + '".')