from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from configparser import ConfigParser

def extract_features_wotarget_14_dlstream(inifile):
    """Adapted version of extract_features_wotargat_14bp to work with fast feature extraction code within DLStream
    Reduced feature selection based on recursive feature elimination for social behavior between 2 mice (anogenital approach, attack, etc.)
    """
    configFile = str(inifile)
    config = ConfigParser()
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(csv_dir, 'features_extracted')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)
    #change videos name to str
    vidinfDf.Video = vidinfDf.Video.astype('str')

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)

    def euclidean_distance(x1,x2,y1,y2):
        result = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return result

    filesFound = []
    roll_windows = []
    roll_windows_values = [2, 5, 6, 7.5, 15]
    loopy = 0

    ########### FIND CSV FILES ###########
    for i in os.listdir(csv_dir_in):
        if i.__contains__(".csv"):
            fname = os.path.join(csv_dir_in, i)
            filesFound.append(fname)
    print('Extracting features from ' + str(len(filesFound)) + ' files...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for i in filesFound:
        currentFile = i
        currVidName = os.path.basename(currentFile)
        currVidName = currVidName.replace('.csv', '')

        # get current pixels/mm
        currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == currVidName]
        try:
            currPixPerMM = float(currVideoSettings['pixels/mm'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        try:
            fps = float(currVideoSettings['fps'])
        except TypeError:
            print('No file found.')
            continue
        print('Processing ' + '"' + str(currVidName) + '".' + ' Fps: ' + str(fps) + ". mm/ppx: " + str(currPixPerMM))

        for i in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[i]))
        loopy += 1
        columnHeaders = ["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                         "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p",
                         "Lat_left_1_x", "Lat_left_1_y",
                         "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                         "Tail_base_1_y", "Tail_base_1_p",
                         "Ear_left_2_x",
                         "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p",
                         "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x",
                         "Lat_left_2_y",
                         "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                         "Tail_base_2_y", "Tail_base_2_p"]
        csv_df = pd.read_csv(currentFile, names=columnHeaders, low_memory=False)
        csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        csv_df = csv_df.reset_index()
        csv_df = csv_df.reset_index(drop=True)

        ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
        M1_hull_mean_euclidean_list = []
        M2_hull_mean_euclidean_list = []
        #print('Creating shifted dataframes for distance calculations')
        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted = csv_df_shifted.rename(
            columns={'Ear_left_1_x': 'Ear_left_1_x_shifted','Ear_left_1_y': 'Ear_left_1_y_shifted',
                     'Ear_left_1_p': 'Ear_left_1_p_shifted','Ear_right_1_x': 'Ear_right_1_x_shifted', \
                     'Ear_right_1_y': 'Ear_right_1_y_shifted','Ear_right_1_p': 'Ear_right_1_p_shifted',
                     'Nose_1_x': 'Nose_1_x_shifted','Nose_1_y': 'Nose_1_y_shifted', \
                     'Nose_1_p': 'Nose_1_p_shifted','Center_1_x': 'Center_1_x_shifted',
                     'Center_1_y': 'Center_1_y_shifted','Center_1_p': 'Center_1_p_shifted','Lat_left_1_x': \
                         'Lat_left_1_x_shifted','Lat_left_1_y': 'Lat_left_1_y_shifted',
                     'Lat_left_1_p': 'Lat_left_1_p_shifted','Lat_right_1_x': 'Lat_right_1_x_shifted',
                     'Lat_right_1_y': 'Lat_right_1_y_shifted', \
                     'Lat_right_1_p': 'Lat_right_1_p_shifted','Tail_base_1_x': 'Tail_base_1_x_shifted',
                     'Tail_base_1_y': 'Tail_base_1_y_shifted', \
                     'Tail_base_1_p': 'Tail_base_1_p_shifted',
                     'Ear_left_2_x': 'Ear_left_2_x_shifted','Ear_left_2_y': 'Ear_left_2_y_shifted',
                     'Ear_left_2_p': 'Ear_left_2_p_shifted','Ear_right_2_x': 'Ear_right_2_x_shifted', \
                     'Ear_right_2_y': 'Ear_right_2_y_shifted','Ear_right_2_p': 'Ear_right_2_p_shifted',
                     'Nose_2_x': 'Nose_2_x_shifted','Nose_2_y': 'Nose_2_y_shifted', \
                     'Nose_2_p': 'Nose_2_p_shifted','Center_2_x': 'Center_2_x_shifted',
                     'Center_2_y': 'Center_2_y_shifted','Center_2_p': 'Center_2_p_shifted','Lat_left_2_x': \
                         'Lat_left_2_x_shifted','Lat_left_2_y': 'Lat_left_2_y_shifted',
                     'Lat_left_2_p': 'Lat_left_2_p_shifted','Lat_right_2_x': 'Lat_right_2_x_shifted',
                     'Lat_right_2_y': 'Lat_right_2_y_shifted', \
                     'Lat_right_2_p': 'Lat_right_2_p_shifted','Tail_base_2_x': 'Tail_base_2_x_shifted',
                     'Tail_base_2_y': 'Tail_base_2_y_shifted', \
                     'Tail_base_2_p': 'Tail_base_2_p_shifted'
                     })
        csv_df_combined = pd.concat([csv_df,csv_df_shifted],axis=1,join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)
        #print('Calculating euclidean distances...')
        ########### EUCLIDEAN DISTANCES ###########################################
        # within mice

        eucl_distance_dict_wm = dict(
            nose_to_tail=('Nose','Tail_base')
            ,width=('Lat_left','Lat_right')
            ,Ear_distance=('Ear_right','Ear_left')
        )

        mice = [1,2]
        for mouse in mice:
            for distance_measurement,bodyparts in eucl_distance_dict_wm.items():
                # skip mouse 2 "ear_distance" measurement
                if mouse == 2 and distance_measurement == 'Ear_distance':
                    continue

                x1 = csv_df[f'{bodyparts[0]}_{mouse}_x'].to_numpy()
                y1 = csv_df[f'{bodyparts[0]}_{mouse}_y'].to_numpy()
                x2 = csv_df[f'{bodyparts[1]}_{mouse}_x'].to_numpy()
                y2 = csv_df[f'{bodyparts[1]}_{mouse}_y'].to_numpy()
                csv_df[f'Mouse_{mouse}_{distance_measurement}'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM

        # between mice

        eucl_distance_dict_bm = dict(
            Centroid_distance=('Center_1','Center_2')
            ,Nose_to_nose_distance=('Nose_1','Nose_2')
            ,M1_Nose_to_M2_lat_left=('Nose_1','Lat_left_2')
            ,M1_Nose_to_M2_lat_right=('Nose_1','Lat_right_2')
            ,M2_Nose_to_M1_lat_left=('Nose_2','Lat_left_1')
            ,M2_Nose_to_M1_lat_right=('Nose_2','Lat_right_1')
            ,M1_Nose_to_M2_tail_base=('Nose_1','Tail_base_2')
            ,M2_Nose_to_M1_tail_base=('Nose_2','Tail_base_1')
        )

        for distance_measurement,bodyparts in eucl_distance_dict_bm.items():
            x1 = csv_df[f'{bodyparts[0]}_x'].to_numpy()
            y1 = csv_df[f'{bodyparts[0]}_y'].to_numpy()
            x2 = csv_df[f'{bodyparts[1]}_x'].to_numpy()
            y2 = csv_df[f'{bodyparts[1]}_y'].to_numpy()
            csv_df[f'{distance_measurement}'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM

        # Movement

        bp_list = ('Center','Nose','Lat_left','Lat_right','Tail_base','Ear_left','Ear_right')

        mice = [1,2]
        for mouse in mice:
            for bp in bp_list:
                x1 = csv_df_combined[f'{bp}_{mouse}_x_shifted'].to_numpy()
                y1 = csv_df_combined[f'{bp}_{mouse}_y_shifted'].to_numpy()
                x2 = csv_df_combined[f'{bp}_{mouse}_x'].to_numpy()
                y2 = csv_df_combined[f'{bp}_{mouse}_y'].to_numpy()
                'Movement_mouse_1_centroid'
                if bp == 'Center':
                    csv_df[f'Movement_mouse_{mouse}_centroid'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                elif bp == 'Ear_left':
                    csv_df[f'Movement_mouse_{mouse}_left_ear'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                elif bp == 'Ear_right':
                    csv_df[f'Movement_mouse_{mouse}_right_ear'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                elif bp == 'Lat_left':
                    csv_df[f'Movement_mouse_{mouse}_lateral_left'] = euclidean_distance(x1,x2,y1,
                                                                                        y2) / self._currPixPerMM
                elif bp == 'Lat_right':
                    csv_df[f'Movement_mouse_{mouse}_lateral_right'] = euclidean_distance(x1,x2,y1,
                                                                                         y2) / self._currPixPerMM
                else:
                    csv_df[f'Movement_mouse_{mouse}_{bp.lower()}'] = euclidean_distance(x1,x2,y1,
                                                                                        y2) / self._currPixPerMM

        # print('Calculating hull variables...')
        ########### HULL - EUCLIDEAN DISTANCES ###########################################

        for index,row in csv_df.iterrows():
            M1_np_array = np.array(
                [[row['Ear_left_1_x'],row["Ear_left_1_y"]],[row['Ear_right_1_x'],row["Ear_right_1_y"]],
                 [row['Nose_1_x'],row["Nose_1_y"]],[row['Center_1_x'],row["Center_1_y"]],
                 [row['Lat_left_1_x'],row["Lat_left_1_y"]],[row['Lat_right_1_x'],row["Lat_right_1_y"]],
                 [row['Tail_base_1_x'],row["Tail_base_1_y"]]]).astype(int)
            M2_np_array = np.array(
                [[row['Ear_left_2_x'],row["Ear_left_2_y"]],[row['Ear_right_2_x'],row["Ear_right_2_y"]],
                 [row['Nose_2_x'],row["Nose_2_y"]],[row['Center_2_x'],row["Center_2_y"]],
                 [row['Lat_left_2_x'],row["Lat_left_2_y"]],[row['Lat_right_2_x'],row["Lat_right_2_y"]],
                 [row['Tail_base_2_x'],row["Tail_base_2_y"]]]).astype(int)
            M1_dist_euclidean = cdist(M1_np_array,M1_np_array,metric='euclidean')
            M1_dist_euclidean = M1_dist_euclidean[M1_dist_euclidean != 0]
            M1_hull_mean_euclidean = np.mean(M1_dist_euclidean)
            M1_hull_mean_euclidean_list.append(M1_hull_mean_euclidean)
            M2_dist_euclidean = cdist(M2_np_array,M2_np_array,metric='euclidean')
            M2_dist_euclidean = M2_dist_euclidean[M2_dist_euclidean != 0]
            M2_hull_mean_euclidean = np.mean(M2_dist_euclidean)
            M2_hull_mean_euclidean_list.append(M2_hull_mean_euclidean)
        csv_df['M1_mean_euclidean_distance_hull'] = list(
            map(lambda x: x / self._currPixPerMM,M1_hull_mean_euclidean_list))
        csv_df['M2_mean_euclidean_distance_hull'] = list(
            map(lambda x: x / self._currPixPerMM,M2_hull_mean_euclidean_list))

        ########### COLLAPSED MEASURES ###########################################
        # print('Collapsed measures')
        csv_df['Total_movement_centroids'] = csv_df['Movement_mouse_1_centroid'] + csv_df['Movement_mouse_2_centroid']
        csv_df['Total_movement_all_bodyparts_M1'] = csv_df['Movement_mouse_1_centroid'] + csv_df[
            'Movement_mouse_1_nose'] + csv_df['Movement_mouse_1_tail_base'] + \
                                                    csv_df['Movement_mouse_1_left_ear'] + csv_df[
                                                        'Movement_mouse_1_right_ear'] + csv_df[
                                                        'Movement_mouse_1_lateral_left'] + csv_df[
                                                        'Movement_mouse_1_lateral_right']
        csv_df['Total_movement_all_bodyparts_M2'] = csv_df['Movement_mouse_2_centroid'] + csv_df[
            'Movement_mouse_2_nose'] + csv_df['Movement_mouse_2_tail_base'] + \
                                                    csv_df['Movement_mouse_2_left_ear'] + csv_df[
                                                        'Movement_mouse_2_right_ear'] + csv_df[
                                                        'Movement_mouse_2_lateral_left'] + csv_df[
                                                        'Movement_mouse_2_lateral_right']
        csv_df['Total_movement_all_bodyparts_both_mice'] = csv_df['Total_movement_all_bodyparts_M1'] + csv_df[
            'Total_movement_all_bodyparts_M2']


        ########### CALC ROLLING WINDOWS MEDIANS AND MEANS ###########################################
        #print('Calculating rolling windows: medians, medians, and sums...')
        # full version for distance (keeping style for future adaptation), reduced version for the rest

        for num,roll_value in enumerate(roll_windows_values):

            parameter_dict1 = dict(Distance='Centroid_distance'
                                   ,Mouse1_width='Mouse_1_width'
                                   ,Mouse2_width='Mouse_2_width'
                                   ,Movement='Total_movement_centroids'
                                   )

            # adapted from:
            # csv_df[currentColName] = csv_df['Sum_euclidean_distance_hull_M1_M2'].rolling(roll_windows[i],
            #                                                                              min_periods=1).median()

            for key,clm_name in parameter_dict1.items():

                if roll_value != 2 and key != 'Distance':
                    continue

                currentcolname = f'{key}_mean_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).mean()

                currentcolname = f'{key}_median_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).median()

                currentcolname = f'{key}_sum_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).sum()

            clm_name1 = 'euclidean_distance_hull'
            clm_name2 = 'euclid_distances'
            measure = 'mean'

            if roll_value == 2:
                for mouse in mice:
                    # keeping style for future adaptation
                    clm_name = f'M{mouse}_{measure}_{clm_name1}'

                    currentcolname = f'Mouse{mouse}_{measure}_{clm_name2}_mean_' + str(roll_value)
                    csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).mean()

                    currentcolname = f'Mouse{mouse}_{measure}_{clm_name2}_median_' + str(roll_value)
                    csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).median()

            clm_list = (
                'Total_movement_all_bodyparts_both_mice'
                ,'Total_movement_centroids'
            )
            if roll_value in [2,5]:
                for clm_name in clm_list:
                    currentcolname = clm_name + '_mean_' + str(roll_value)
                    csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).mean()

                    currentcolname = clm_name + '_sum_' + str(roll_value)
                    csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).sum()

                    if roll_value == 2:
                        currentcolname = clm_name + '_median_' + str(roll_value)
                        csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],
                                                                          min_periods=1).median()

            parameter_dict2 = dict(
                Nose_movement='nose'
                ,Centroid_movement='centroid'
                ,Tail_base_movement='tail_base'
            )

            if roll_value == 2:
                for mouse in mice:
                    for key,bp in parameter_dict2.items():
                        clm_name = f'Movement_mouse_{mouse}_{bp.lower()}'

                        currentcolname = f'{key}_M{mouse}_mean_' + str(roll_value)
                        csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).mean()

                        currentcolname = f'{key}_M{mouse}_median_' + str(roll_value)
                        csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],
                                                                          min_periods=1).median()

                        currentcolname = f'{key}_M{mouse}_sum_' + str(roll_value)
                        csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).sum()

        if roll_value == 2:

            for mouse in mice:
                # Tail_base_movement
                clm_name = f'Movement_mouse_{mouse}_tail_base'
                # median
                currentcolname = f'Tail_base_movement_M{mouse}_median_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).median()
                # only for 2nd mouse
                if mouse == 2:
                    # mean
                    currentcolname = f'Tail_base_movement_M{mouse}_mean_' + str(roll_value)
                    csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).mean()
                    # sum
                    currentcolname = f'Tail_base_movement_M{mouse}_sum_' + str(roll_value)
                    csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).sum()
                # Centroid_movement
                clm_name = f'Movement_mouse_{mouse}_centroid'
                # mean
                currentcolname = f'Centroid_movement_M{mouse}_mean_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).mean()
                # sum
                currentcolname = f'Centroid_movement_M{mouse}_sum_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).sum()

                # Nose_movement
                clm_name = f'Movement_mouse_{mouse}_nose'
                # mean
                currentcolname = f'Nose_movement_M{mouse}_mean_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).mean()
                # sum
                currentcolname = f'Nose_movement_M{mouse}_sum_' + str(roll_value)
                csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).sum()
                # only for 1st mouse
                if mouse == 1:
                    # median
                    currentcolname = f'Nose_movement_M{mouse}_median_' + str(roll_value)
                    csv_df[currentcolname] = csv_df[clm_name].rolling(roll_windows[num],min_periods=1).median()

        ########### BODY PARTS RELATIVE TO EACH OTHER ##################

        ################# EMPETY #########################################
        #
        # ########### ANGLES ###########################################
        # not used in this version
        # ########### DEVIATIONS ###########################################
        # not used in this version
        # ########### PERCENTILE RANK ###########################################
        # not used in this version
        # ########### CALCULATE STRAIGHTNESS OF POLYLINE PATH: tortuosity  ###########################################
        # not used in this version
        # ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
        # not used in this version
        ########### DROP CALCULATION COLUMNS THAT ARE NOT USED IN THE FINAL VERSION#########
        clms_to_drop = ['Nose_1_x','Nose_1_y','Ear_left_1_x','Ear_left_1_y','Ear_right_1_x','Ear_right_1_y',
                        'Center_1_x','Center_1_y','Lat_left_1_x','Lat_left_1_y','Lat_right_1_x','Lat_right_1_y',
                        'Tail_base_1_x','Tail_base_1_y','Nose_2_x','Nose_2_y','Ear_left_2_x','Ear_left_2_y',
                        'Ear_right_2_x','Ear_right_2_y','Center_2_x','Center_2_y','Lat_left_2_x','Lat_left_2_y',
                        'Lat_right_2_x','Lat_right_2_y','Tail_base_2_x','Tail_base_2_y','Nose_1_p','Ear_left_1_p',
                        'Ear_right_1_p','Center_1_p','Lat_left_1_p','Lat_right_1_p','Tail_base_1_p','Nose_2_p',
                        'Ear_left_2_p','Ear_right_2_p','Center_2_p','Lat_left_2_p','Lat_right_2_p','Tail_base_2_p',
                        'Mouse_1_width','Mouse_2_width','Movement_mouse_1_centroid','Movement_mouse_1_nose',
                        'Movement_mouse_1_lateral_left','Movement_mouse_1_lateral_right','Movement_mouse_1_tail_base',
                        'Movement_mouse_1_left_ear','Movement_mouse_1_right_ear','Movement_mouse_2_centroid',
                        'Movement_mouse_2_nose','Movement_mouse_2_lateral_left','Movement_mouse_2_lateral_right',
                        'Movement_mouse_2_tail_base','Movement_mouse_2_left_ear','Movement_mouse_2_right_ear',
                        'M1_mean_euclidean_distance_hull','M2_mean_euclidean_distance_hull','Total_movement_centroids',
                        'Total_movement_all_bodyparts_M1','Total_movement_all_bodyparts_M2',
                        'Total_movement_all_bodyparts_both_mice','Movement_mean_2','Movement_median_2','Movement_sum_2',
                        'Total_movement_centroids_mean_2','Total_movement_centroids_sum_2',
                        'Total_movement_centroids_median_2','Centroid_movement_M1_median_2',
                        'Tail_base_movement_M1_mean_2','Tail_base_movement_M1_sum_2','Nose_movement_M2_median_2',
                        'Centroid_movement_M2_median_2','Total_movement_centroids_mean_5',
                        'Total_movement_centroids_sum_5']
        csv_df = csv_df.drop(columns=clms_to_drop)

        ########### DROP COORDINATE COLUMNS ###########################################
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(columns=['index'])
        fileName = os.path.basename(currentFile)
        fileName = fileName.split('.')
        fileOut = str(fileName[0]) + str('.csv')
        saveFN = os.path.join(csv_dir_out, fileOut)
        csv_df.to_csv(saveFN)
        print('Feature extraction complete for ' + '"' + str(currVidName) + '".')
    print('All feature extraction complete.')