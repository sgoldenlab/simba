from __future__ import division
import os
import pandas as pd
import math
import numpy as np
from scipy.spatial import ConvexHull
import scipy
from configparser import ConfigParser

def extract_features_wotarget_16(inifile):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(csv_dir, 'features_extracted')
    vidInfPath = config.get('General settings', 'project_path')
    vidInfPath = os.path.join(vidInfPath, 'logs')
    vidInfPath = os.path.join(vidInfPath, 'video_info.csv')
    vidinfDf = pd.read_csv(vidInfPath)

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    def count_values_in_range(series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    def angle3pt(ax, ay, bx, by, cx, cy):
        ang = math.degrees(
            math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx))
        return ang + 360 if ang < 0 else ang

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
        M1_hull_large_euclidean_list = []
        M1_hull_small_euclidean_list = []
        M1_hull_mean_euclidean_list = []
        M1_hull_sum_euclidean_list = []
        M2_hull_large_euclidean_list = []
        M2_hull_small_euclidean_list = []
        M2_hull_mean_euclidean_list = []
        M2_hull_sum_euclidean_list = []
        currentFile = i
        currVidName = os.path.basename(currentFile)
        currVidName = currVidName.replace('.csv', '')

        # get current pixels/mm
        currVideoSettings = vidinfDf.loc[vidinfDf['Video'] == currVidName]
        try:
            currPixPerMM = float(currVideoSettings['pixels/mm'])
        except TypeError:
            print('Error: make sure all the videos that are going to be analyzed are represented in the project_folder/logs/video_info.csv file')
        fps = float(currVideoSettings['fps'])
        print('Processing ' + '"' + str(currVidName) + '".' + ' Fps: ' + str(fps) + ". mm/ppx: " + str(currPixPerMM))

        for i in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[i]))
        loopy += 1
        columnHeaders = ["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                         "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p",
                         "Lat_left_1_x", "Lat_left_1_y",
                         "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                         "Tail_base_1_y", "Tail_base_1_p", "Tail_end_1_x", "Tail_end_1_y", "Tail_end_1_p",
                         "Ear_left_2_x",
                         "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p",
                         "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x",
                         "Lat_left_2_y",
                         "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                         "Tail_base_2_y", "Tail_base_2_p", "Tail_end_2_x", "Tail_end_2_y", "Tail_end_2_p"]
        csv_df = pd.read_csv(currentFile, names=columnHeaders, low_memory=False)
        csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        csv_df = csv_df.reset_index()
        csv_df = csv_df.reset_index(drop=True)

        print('Evaluating convex hulls...')
        ########### MOUSE AREAS ###########################################
        csv_df['Mouse_1_poly_area'] = csv_df.apply(lambda x: ConvexHull(np.array(
            [[x['Ear_left_1_x'], x["Ear_left_1_y"]],
             [x['Ear_right_1_x'], x["Ear_right_1_y"]],
             [x['Nose_1_x'], x["Nose_1_y"]],
             [x['Lat_left_1_x'], x["Lat_left_1_y"]], \
             [x['Lat_right_1_x'], x["Lat_right_1_y"]],
             [x['Tail_base_1_x'], x["Tail_base_1_y"]],
             [x['Center_1_x'], x["Center_1_y"]]])).area, axis=1)
        csv_df['Mouse_1_poly_area'] = csv_df['Mouse_1_poly_area'] / currPixPerMM
        csv_df['Mouse_2_poly_area'] = csv_df.apply(lambda x: ConvexHull(np.array(
            [[x['Ear_left_2_x'], x["Ear_left_2_y"]],
             [x['Ear_right_2_x'], x["Ear_right_2_y"]],
             [x['Nose_2_x'], x["Nose_2_y"]],
             [x['Lat_left_2_x'], x["Lat_left_2_y"]], \
             [x['Lat_right_2_x'], x["Lat_right_2_y"]],
             [x['Tail_base_2_x'], x["Tail_base_2_y"]],
             [x['Center_2_x'], x["Center_2_y"]]])).area, axis=1)
        csv_df['Mouse_2_poly_area'] = csv_df['Mouse_2_poly_area'] / currPixPerMM

        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted = csv_df_shifted.rename(
            columns={'Ear_left_1_x': 'Ear_left_1_x_shifted', 'Ear_left_1_y': 'Ear_left_1_y_shifted',
                     'Ear_left_1_p': 'Ear_left_1_p_shifted', 'Ear_right_1_x': 'Ear_right_1_x_shifted', \
                     'Ear_right_1_y': 'Ear_right_1_y_shifted', 'Ear_right_1_p': 'Ear_right_1_p_shifted',
                     'Nose_1_x': 'Nose_1_x_shifted', 'Nose_1_y': 'Nose_1_y_shifted', \
                     'Nose_1_p': 'Nose_1_p_shifted', 'Center_1_x': 'Center_1_x_shifted',
                     'Center_1_y': 'Center_1_y_shifted', 'Center_1_p': 'Center_1_p_shifted', 'Lat_left_1_x': \
                         'Lat_left_1_x_shifted', 'Lat_left_1_y': 'Lat_left_1_y_shifted',
                     'Lat_left_1_p': 'Lat_left_1_p_shifted', 'Lat_right_1_x': 'Lat_right_1_x_shifted',
                     'Lat_right_1_y': 'Lat_right_1_y_shifted', \
                     'Lat_right_1_p': 'Lat_right_1_p_shifted', 'Tail_base_1_x': 'Tail_base_1_x_shifted',
                     'Tail_base_1_y': 'Tail_base_1_y_shifted', \
                     'Tail_base_1_p': 'Tail_base_1_p_shifted', 'Tail_end_1_x': 'Tail_end_1_x_shifted',
                     'Tail_end_1_y': 'Tail_end_1_y_shifted', 'Tail_end_1_p': 'Tail_end_1_p_shifted',
                     'Ear_left_2_x': 'Ear_left_2_x_shifted', 'Ear_left_2_y': 'Ear_left_2_y_shifted',
                     'Ear_left_2_p': 'Ear_left_2_p_shifted', 'Ear_right_2_x': 'Ear_right_2_x_shifted', \
                     'Ear_right_2_y': 'Ear_right_2_y_shifted', 'Ear_right_2_p': 'Ear_right_2_p_shifted',
                     'Nose_2_x': 'Nose_2_x_shifted', 'Nose_2_y': 'Nose_2_y_shifted', \
                     'Nose_2_p': 'Nose_2_p_shifted', 'Center_2_x': 'Center_2_x_shifted',
                     'Center_2_y': 'Center_2_y_shifted', 'Center_2_p': 'Center_2_p_shifted', 'Lat_left_2_x': \
                         'Lat_left_2_x_shifted', 'Lat_left_2_y': 'Lat_left_2_y_shifted',
                     'Lat_left_2_p': 'Lat_left_2_p_shifted', 'Lat_right_2_x': 'Lat_right_2_x_shifted',
                     'Lat_right_2_y': 'Lat_right_2_y_shifted', \
                     'Lat_right_2_p': 'Lat_right_2_p_shifted', 'Tail_base_2_x': 'Tail_base_2_x_shifted',
                     'Tail_base_2_y': 'Tail_base_2_y_shifted', \
                     'Tail_base_2_p': 'Tail_base_2_p_shifted', 'Tail_end_2_x': 'Tail_end_2_x_shifted',
                     'Tail_end_2_y': 'Tail_end_2_y_shifted', 'Tail_end_2_p': 'Tail_end_2_p_shifted',
                     'Mouse_1_poly_area': 'Mouse_1_poly_area_shifted',
                     'Mouse_2_poly_area': 'Mouse_2_poly_area_shifted'})
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)

        print('Calculating euclidean distances...')
        ########### EUCLIDEAN DISTANCES ###########################################
        csv_df['Mouse_1_nose_to_tail'] = (np.sqrt((csv_df.Nose_1_x - csv_df.Tail_base_1_x) ** 2 + (
                    csv_df.Nose_1_y - csv_df.Tail_base_1_y) ** 2)) / currPixPerMM
        csv_df['Mouse_2_nose_to_tail'] = (np.sqrt((csv_df.Nose_2_x - csv_df.Tail_base_2_x) ** 2 + (
                    csv_df.Nose_2_y - csv_df.Tail_base_2_y) ** 2)) / currPixPerMM
        csv_df['Mouse_1_width'] = (np.sqrt((csv_df.Lat_left_1_x - csv_df.Lat_right_1_x) ** 2 + (
                    csv_df.Lat_left_1_y - csv_df.Lat_right_1_y) ** 2)) / currPixPerMM
        csv_df['Mouse_2_width'] = (np.sqrt((csv_df.Lat_left_2_x - csv_df.Lat_right_2_x) ** 2 + (
                    csv_df.Lat_left_2_y - csv_df.Lat_right_2_y) ** 2)) / currPixPerMM
        csv_df['Mouse_1_Ear_distance'] = (np.sqrt((csv_df.Ear_left_1_x - csv_df.Ear_right_1_x) ** 2 + (
                    csv_df.Ear_left_1_y - csv_df.Ear_right_1_y) ** 2)) / currPixPerMM
        csv_df['Mouse_2_Ear_distance'] = (np.sqrt((csv_df.Ear_left_2_x - csv_df.Ear_right_2_x) ** 2 + (
                    csv_df.Ear_left_2_y - csv_df.Ear_right_2_y) ** 2)) / currPixPerMM
        csv_df['Mouse_1_Nose_to_centroid'] = (np.sqrt(
            (csv_df.Nose_1_x - csv_df.Center_1_x) ** 2 + (csv_df.Nose_1_y - csv_df.Center_1_y) ** 2)) / currPixPerMM
        csv_df['Mouse_2_Nose_to_centroid'] = (np.sqrt(
            (csv_df.Nose_2_x - csv_df.Center_2_x) ** 2 + (csv_df.Nose_2_y - csv_df.Center_2_y) ** 2)) / currPixPerMM
        csv_df['Centroid_distance'] = (np.sqrt(
            (csv_df.Center_2_x - csv_df.Center_1_x) ** 2 + (csv_df.Center_2_y - csv_df.Center_1_y) ** 2)) / currPixPerMM
        csv_df['Nose_to_nose_distance'] = (np.sqrt(
            (csv_df.Nose_2_x - csv_df.Nose_1_x) ** 2 + (csv_df.Nose_2_y - csv_df.Nose_1_y) ** 2)) / currPixPerMM
        csv_df['M1_Nose_to_M2_lat_left'] = (np.sqrt(
            (csv_df.Nose_1_x - csv_df.Lat_left_2_x) ** 2 + (csv_df.Nose_1_y - csv_df.Lat_left_2_y) ** 2)) / currPixPerMM
        csv_df['M1_Nose_to_M2_lat_right'] = (np.sqrt((csv_df.Nose_1_x - csv_df.Lat_right_2_x) ** 2 + (
                    csv_df.Nose_1_y - csv_df.Lat_right_2_y) ** 2)) / currPixPerMM
        csv_df['M2_Nose_to_M1_lat_left'] = (np.sqrt(
            (csv_df.Nose_2_x - csv_df.Lat_left_1_x) ** 2 + (csv_df.Nose_2_y - csv_df.Lat_left_1_y) ** 2)) / currPixPerMM
        csv_df['M2_Nose_to_M1_lat_right'] = (np.sqrt((csv_df.Nose_2_x - csv_df.Lat_right_1_x) ** 2 + (
                    csv_df.Nose_2_y - csv_df.Lat_right_1_y) ** 2)) / currPixPerMM
        csv_df['M1_Nose_to_M2_tail_base'] = (np.sqrt((csv_df.Nose_1_x - csv_df.Tail_base_2_x) ** 2 + (
                    csv_df.Nose_1_y - csv_df.Tail_base_2_y) ** 2)) / currPixPerMM
        csv_df['M2_Nose_to_M1_tail_base'] = (np.sqrt((csv_df.Nose_2_x - csv_df.Tail_base_1_x) ** 2 + (
                    csv_df.Nose_2_y - csv_df.Tail_base_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_centroid'] = (np.sqrt(
            (csv_df_combined.Center_1_x_shifted - csv_df_combined.Center_1_x) ** 2 + (
                        csv_df_combined.Center_1_y_shifted - csv_df_combined.Center_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_centroid'] = (np.sqrt(
            (csv_df_combined.Center_2_x_shifted - csv_df_combined.Center_2_x) ** 2 + (
                        csv_df_combined.Center_2_y_shifted - csv_df_combined.Center_2_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_nose'] = (np.sqrt(
            (csv_df_combined.Nose_1_x_shifted - csv_df_combined.Nose_1_x) ** 2 + (
                        csv_df_combined.Nose_1_y_shifted - csv_df_combined.Nose_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_nose'] = (np.sqrt(
            (csv_df_combined.Nose_2_x_shifted - csv_df_combined.Nose_2_x) ** 2 + (
                        csv_df_combined.Nose_2_y_shifted - csv_df_combined.Nose_2_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_tail_base'] = (np.sqrt((csv_df_combined.Tail_base_1_x_shifted - csv_df_combined.Tail_base_1_x) ** 2 + (
                        csv_df_combined.Tail_base_1_y_shifted - csv_df_combined.Tail_base_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_tail_base'] = (np.sqrt(
            (csv_df_combined.Tail_base_2_x_shifted - csv_df_combined.Tail_base_2_x) ** 2 + (
                        csv_df_combined.Tail_base_2_y_shifted - csv_df_combined.Tail_base_2_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_tail_end'] = (np.sqrt(
            (csv_df_combined.Tail_end_1_x_shifted - csv_df_combined.Tail_end_1_x) ** 2 + (
                        csv_df_combined.Tail_end_1_y_shifted - csv_df_combined.Tail_end_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_tail_end'] = (np.sqrt(
            (csv_df_combined.Tail_end_2_x_shifted - csv_df_combined.Tail_end_2_x) ** 2 + (
                        csv_df_combined.Tail_end_2_y_shifted - csv_df_combined.Tail_end_2_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_left_ear'] = (np.sqrt(
            (csv_df_combined.Ear_left_1_x_shifted - csv_df_combined.Ear_left_1_x) ** 2 + (
                        csv_df_combined.Ear_left_1_y_shifted - csv_df_combined.Ear_left_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_left_ear'] = (np.sqrt(
            (csv_df_combined.Ear_left_2_x_shifted - csv_df_combined.Ear_left_2_x) ** 2 + (
                        csv_df_combined.Ear_left_2_y_shifted - csv_df_combined.Ear_left_2_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_right_ear'] = (np.sqrt(
            (csv_df_combined.Ear_right_1_x_shifted - csv_df_combined.Ear_right_1_x) ** 2 + (
                        csv_df_combined.Ear_right_1_y_shifted - csv_df_combined.Ear_right_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_right_ear'] = (np.sqrt(
            (csv_df_combined.Ear_right_2_x_shifted - csv_df_combined.Ear_right_2_x) ** 2 + (
                        csv_df_combined.Ear_right_2_y_shifted - csv_df_combined.Ear_right_2_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_lateral_left'] = (np.sqrt(
            (csv_df_combined.Lat_left_1_x_shifted - csv_df_combined.Lat_left_1_x) ** 2 + (
                        csv_df_combined.Lat_left_1_y_shifted - csv_df_combined.Lat_left_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_lateral_left'] = (np.sqrt(
            (csv_df_combined.Lat_left_2_x_shifted - csv_df_combined.Lat_left_2_x) ** 2 + (
                        csv_df_combined.Lat_left_2_y_shifted - csv_df_combined.Lat_left_2_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_1_lateral_right'] = (np.sqrt(
            (csv_df_combined.Lat_right_1_x_shifted - csv_df_combined.Lat_right_1_x) ** 2 + (
                        csv_df_combined.Lat_right_1_y_shifted - csv_df_combined.Lat_right_1_y) ** 2)) / currPixPerMM
        csv_df['Movement_mouse_2_lateral_right'] = (np.sqrt(
            (csv_df_combined.Lat_right_2_x_shifted - csv_df_combined.Lat_right_2_x) ** 2 + (
                        csv_df_combined.Lat_right_2_y_shifted - csv_df_combined.Lat_right_2_y) ** 2)) / currPixPerMM
        csv_df['Mouse_1_polygon_size_change'] = (
                    csv_df_combined['Mouse_1_poly_area_shifted'] - csv_df_combined['Mouse_1_poly_area'])
        csv_df['Mouse_2_polygon_size_change'] = (
                    csv_df_combined['Mouse_2_poly_area_shifted'] - csv_df_combined['Mouse_2_poly_area'])

        print('Calculating hull variables...')
        ########### HULL - EUCLIDEAN DISTANCES ###########################################
        for index, row in csv_df.iterrows():
            M1_np_array = np.array(
                [[row['Ear_left_1_x'], row["Ear_left_1_y"]], [row['Ear_right_1_x'], row["Ear_right_1_y"]],
                 [row['Nose_1_x'], row["Nose_1_y"]], [row['Center_1_x'], row["Center_1_y"]],
                 [row['Lat_left_1_x'], row["Lat_left_1_y"]], [row['Lat_right_1_x'], row["Lat_right_1_y"]],
                 [row['Tail_base_1_x'], row["Tail_base_1_y"]]]).astype(int)
            M2_np_array = np.array(
                [[row['Ear_left_2_x'], row["Ear_left_2_y"]], [row['Ear_right_2_x'], row["Ear_right_2_y"]],
                 [row['Nose_2_x'], row["Nose_2_y"]], [row['Center_2_x'], row["Center_2_y"]],
                 [row['Lat_left_2_x'], row["Lat_left_2_y"]], [row['Lat_right_2_x'], row["Lat_right_2_y"]],
                 [row['Tail_base_2_x'], row["Tail_base_2_y"]]]).astype(int)
            M1_dist_euclidean = scipy.spatial.distance.cdist(M1_np_array, M1_np_array, metric='euclidean')
            M1_dist_euclidean = M1_dist_euclidean[M1_dist_euclidean != 0]
            M1_hull_large_euclidean = np.amax(M1_dist_euclidean)
            M1_hull_small_euclidean = np.min(M1_dist_euclidean)
            M1_hull_mean_euclidean = np.mean(M1_dist_euclidean)
            M1_hull_sum_euclidean = np.sum(M1_dist_euclidean)
            M1_hull_large_euclidean_list.append(M1_hull_large_euclidean)
            M1_hull_small_euclidean_list.append(M1_hull_small_euclidean)
            M1_hull_mean_euclidean_list.append(M1_hull_mean_euclidean)
            M1_hull_sum_euclidean_list.append(M1_hull_sum_euclidean)
            M2_dist_euclidean = scipy.spatial.distance.cdist(M2_np_array, M2_np_array, metric='euclidean')
            M2_dist_euclidean = M2_dist_euclidean[M2_dist_euclidean != 0]
            M2_hull_large_euclidean = np.amax(M2_dist_euclidean)
            M2_hull_small_euclidean = np.min(M2_dist_euclidean)
            M2_hull_mean_euclidean = np.mean(M2_dist_euclidean)
            M2_hull_sum_euclidean = np.sum(M2_dist_euclidean)
            M2_hull_large_euclidean_list.append(M2_hull_large_euclidean)
            M2_hull_small_euclidean_list.append(M2_hull_small_euclidean)
            M2_hull_mean_euclidean_list.append(M2_hull_mean_euclidean)
            M2_hull_sum_euclidean_list.append(M2_hull_sum_euclidean)
        csv_df['M1_largest_euclidean_distance_hull'] = list(
            map(lambda x: x / currPixPerMM, M1_hull_large_euclidean_list))
        csv_df['M1_smallest_euclidean_distance_hull'] = list(
            map(lambda x: x / currPixPerMM, M1_hull_small_euclidean_list))
        csv_df['M1_mean_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_mean_euclidean_list))
        csv_df['M1_sum_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_sum_euclidean_list))
        csv_df['M2_largest_euclidean_distance_hull'] = list(
            map(lambda x: x / currPixPerMM, M2_hull_large_euclidean_list))
        csv_df['M2_smallest_euclidean_distance_hull'] = list(
            map(lambda x: x / currPixPerMM, M2_hull_small_euclidean_list))
        csv_df['M2_mean_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M2_hull_mean_euclidean_list))
        csv_df['M2_sum_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M2_hull_sum_euclidean_list))
        csv_df['Sum_euclidean_distance_hull_M1_M2'] = (
                    csv_df['M1_sum_euclidean_distance_hull'] + csv_df['M2_sum_euclidean_distance_hull'])


        ########### COLLAPSED MEASURES ###########################################
        csv_df['Total_movement_centroids'] = csv_df['Movement_mouse_1_centroid'] + csv_df['Movement_mouse_2_centroid']
        csv_df['Total_movement_tail_ends'] = csv_df['Movement_mouse_1_tail_end'] + csv_df['Movement_mouse_2_tail_end']
        csv_df['Total_movement_all_bodyparts_M1'] = csv_df['Movement_mouse_1_centroid'] + csv_df[
            'Movement_mouse_1_nose'] + csv_df['Movement_mouse_1_tail_end'] + csv_df['Movement_mouse_1_tail_base'] + \
                                                    csv_df['Movement_mouse_1_left_ear'] + csv_df[
                                                        'Movement_mouse_1_right_ear'] + csv_df[
                                                        'Movement_mouse_1_lateral_left'] + csv_df[
                                                        'Movement_mouse_1_lateral_right']
        csv_df['Total_movement_all_bodyparts_M2'] = csv_df['Movement_mouse_2_centroid'] + csv_df[
            'Movement_mouse_2_nose'] + csv_df['Movement_mouse_2_tail_end'] + csv_df['Movement_mouse_2_tail_base'] + \
                                                    csv_df['Movement_mouse_2_left_ear'] + csv_df[
                                                        'Movement_mouse_2_right_ear'] + csv_df[
                                                        'Movement_mouse_2_lateral_left'] + csv_df[
                                                        'Movement_mouse_2_lateral_right']
        csv_df['Total_movement_all_bodyparts_both_mice'] = csv_df['Total_movement_all_bodyparts_M1'] + csv_df[
            'Total_movement_all_bodyparts_M2']

        ########### CALC ROLLING WINDOWS MEDIANS AND MEANS ###########################################
        print('Calculating rolling windows: medians, medians, and sums...')

        for i in range(len(roll_windows_values)):
            currentColName = 'Sum_euclid_distances_hull_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Sum_euclidean_distance_hull_M1_M2'].rolling(roll_windows[i],
                                                                                         min_periods=1).median()
            currentColName = 'Sum_euclid_distances_hull_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Sum_euclidean_distance_hull_M1_M2'].rolling(roll_windows[i],
                                                                                         min_periods=1).mean()
            currentColName = 'Sum_euclid_distances_hull_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Sum_euclidean_distance_hull_M1_M2'].rolling(roll_windows[i],
                                                                                         min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_centroids'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_centroids'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_centroids'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Centroid_distance'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Distance_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Centroid_distance'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Distance_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Centroid_distance'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_width_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mouse_1_width'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Mouse1_width_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mouse_1_width'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Mouse1_width_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mouse_1_width'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse2_width_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mouse_2_width'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Mouse2_width_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mouse_2_width'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Mouse2_width_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mouse_2_width'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_mean_euclid_distances_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_mean_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                       min_periods=1).median()
            currentColName = 'Mouse1_mean_euclid_distances_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_mean_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                       min_periods=1).mean()
            currentColName = 'Mouse1_mean_euclid_distances_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_mean_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                       min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse2_mean_euclid_distances_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_mean_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                       min_periods=1).median()
            currentColName = 'Mouse2_mean_euclid_distances_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_mean_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                       min_periods=1).mean()
            currentColName = 'Mouse2_mean_euclid_distances_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_mean_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                       min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_smallest_euclid_distances_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_smallest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                           min_periods=1).median()
            currentColName = 'Mouse1_smallest_euclid_distances_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_smallest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                           min_periods=1).mean()
            currentColName = 'Mouse1_smallest_euclid_distances_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_smallest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                           min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse2_smallest_euclid_distances_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_smallest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                           min_periods=1).median()
            currentColName = 'Mouse2_smallest_euclid_distances_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_smallest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                           min_periods=1).mean()
            currentColName = 'Mouse2_smallest_euclid_distances_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_smallest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                           min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_largest_euclid_distances_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_largest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                          min_periods=1).median()
            currentColName = 'Mouse1_largest_euclid_distances_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_largest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                          min_periods=1).mean()
            currentColName = 'Mouse1_largest_euclid_distances_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M1_largest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                          min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse2_largest_euclid_distances_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_largest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                          min_periods=1).median()
            currentColName = 'Mouse2_largest_euclid_distances_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_largest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                          min_periods=1).mean()
            currentColName = 'Mouse2_largest_euclid_distances_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['M2_largest_euclidean_distance_hull'].rolling(roll_windows[i],
                                                                                          min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_movement_all_bodyparts_both_mice_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_all_bodyparts_both_mice'].rolling(roll_windows[i],
                                                                                              min_periods=1).median()
            currentColName = 'Total_movement_all_bodyparts_both_mice_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_all_bodyparts_both_mice'].rolling(roll_windows[i],
                                                                                              min_periods=1).mean()
            currentColName = 'Total_movement_all_bodyparts_both_mice_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_all_bodyparts_both_mice'].rolling(roll_windows[i],
                                                                                              min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_movement_centroids_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_centroids'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Total_movement_centroids_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_centroids'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Total_movement_centroids_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_centroids'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Tail_base_movement_M1_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_tail_base'].rolling(roll_windows[i],
                                                                                  min_periods=1).median()
            currentColName = 'Tail_base_movement_M1_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_tail_base'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Tail_base_movement_M1_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_tail_base'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Tail_base_movement_M2_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_tail_base'].rolling(roll_windows[i],
                                                                                  min_periods=1).median()
            currentColName = 'Tail_base_movement_M2_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_tail_base'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Tail_base_movement_M2_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_tail_base'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Centroid_movement_M1_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_centroid'].rolling(roll_windows[i],
                                                                                 min_periods=1).median()
            currentColName = 'Centroid_movement_M1_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_centroid'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Centroid_movement_M1_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_centroid'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Centroid_movement_M2_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_centroid'].rolling(roll_windows[i],
                                                                                 min_periods=1).median()
            currentColName = 'Centroid_movement_M2_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_centroid'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Centroid_movement_M2_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_centroid'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Tail_end_movement_M1_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_tail_end'].rolling(roll_windows[i],
                                                                                 min_periods=1).median()
            currentColName = 'Tail_end_movement_M1_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_tail_end'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Tail_end_movement_M1_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_tail_end'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Tail_end_movement_M2_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_tail_end'].rolling(roll_windows[i],
                                                                                 min_periods=1).median()
            currentColName = 'Tail_end_movement_M2_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_tail_end'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Tail_end_movement_M2_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_tail_end'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Nose_movement_M1_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_nose'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Nose_movement_M1_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_nose'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Nose_movement_M1_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_1_nose'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Nose_movement_M2_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_nose'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Nose_movement_M2_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_nose'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Nose_movement_M2_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_mouse_2_nose'].rolling(roll_windows[i], min_periods=1).sum()

        ########### BODY PARTS RELATIVE TO EACH OTHER ##################
        csv_df['Tail_end_relative_to_tail_base_centroid_nose'] = csv_df['Movement_mouse_1_tail_end'] - (
                    csv_df['Movement_mouse_1_tail_base'] + csv_df['Movement_mouse_1_centroid'] + csv_df[
                'Movement_mouse_1_nose'])
        for i in range(len(roll_windows_values)):
            currentColName_M1 = 'Tail_end_relative_to_tail_base_centroid_nose_M1_' + str(roll_windows_values[i])
            tail_end_col_name = 'Tail_end_movement_M1_mean_' + str(roll_windows_values[i])
            tail_base_col_name = 'Tail_base_movement_M1_mean_' + str(roll_windows_values[i])
            centroid_col_name = 'Centroid_movement_M1_mean_' + str(roll_windows_values[i])
            nose_col_name = 'Nose_movement_M1_mean_' + str(roll_windows_values[i])
            currentColName_M2 = 'Tail_end_relative_to_tail_base_centroid_nose_M2_mean_' + str(roll_windows_values[i])
            tail_end_col_name_M2 = 'Tail_end_movement_M2_mean_' + str(roll_windows_values[i])
            tail_base_col_name_M2 = 'Tail_base_movement_M2_mean_' + str(roll_windows_values[i])
            centroid_col_name_M2 = 'Centroid_movement_M2_mean_' + str(roll_windows_values[i])
            nose_col_name_M2 = 'Nose_movement_M2_mean_' + str(roll_windows_values[i])
            csv_df[currentColName_M1] = csv_df[tail_end_col_name] - (
                    csv_df[tail_base_col_name] + csv_df[centroid_col_name] + csv_df[nose_col_name])
            csv_df[currentColName_M2] = csv_df[tail_end_col_name_M2] - (
                    csv_df[tail_base_col_name_M2] + csv_df[centroid_col_name_M2] + csv_df[nose_col_name_M2])

        ########### ANGLES ###########################################
        print('Calculating angles...')
        csv_df['Mouse_1_angle'] = csv_df.apply(
            lambda x: angle3pt(x['Nose_1_x'], x['Nose_1_y'], x['Center_1_x'], x['Center_1_y'], x['Tail_base_1_x'],
                               x['Tail_base_1_y']), axis=1)
        csv_df['Mouse_2_angle'] = csv_df.apply(
            lambda x: angle3pt(x['Nose_2_x'], x['Nose_2_y'], x['Center_2_x'], x['Center_2_y'], x['Tail_base_2_x'],
                               x['Tail_base_2_y']), axis=1)
        csv_df['Total_angle_both_mice'] = csv_df['Mouse_1_angle'] + csv_df['Mouse_2_angle']
        for i in range(len(roll_windows_values)):
            currentColName = 'Total_angle_both_mice_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_angle_both_mice'].rolling(roll_windows[i], min_periods=1).sum()

        ########### DEVIATIONS ###########################################
        print('Calculating deviations...')
        csv_df['Total_movement_all_bodyparts_both_mice_deviation'] = (
                    csv_df['Total_movement_all_bodyparts_both_mice'].mean() - csv_df[
                'Total_movement_all_bodyparts_both_mice'])
        csv_df['Sum_euclid_distances_hull_deviation'] = (
                    csv_df['Sum_euclidean_distance_hull_M1_M2'].mean() - csv_df['Sum_euclidean_distance_hull_M1_M2'])
        csv_df['M1_smallest_euclid_distances_hull_deviation'] = (
                    csv_df['M1_smallest_euclidean_distance_hull'].mean() - csv_df[
                'M1_smallest_euclidean_distance_hull'])
        csv_df['M1_largest_euclid_distances_hull_deviation'] = (
                    csv_df['M1_largest_euclidean_distance_hull'].mean() - csv_df['M1_largest_euclidean_distance_hull'])
        csv_df['M1_mean_euclid_distances_hull_deviation'] = (
                    csv_df['M1_mean_euclidean_distance_hull'].mean() - csv_df['M1_mean_euclidean_distance_hull'])
        csv_df['Centroid_distance_deviation'] = (csv_df['Centroid_distance'].mean() - csv_df['Centroid_distance'])
        csv_df['Total_angle_both_mice_deviation'] = (
                    csv_df['Total_angle_both_mice'].mean() - csv_df['Total_angle_both_mice'])
        csv_df['Movement_mouse_1_deviation_centroid'] = (
                    csv_df['Movement_mouse_1_centroid'].mean() - csv_df['Movement_mouse_1_centroid'])
        csv_df['Movement_mouse_2_deviation_centroid'] = (
                    csv_df['Movement_mouse_2_centroid'].mean() - csv_df['Movement_mouse_2_centroid'])
        csv_df['Mouse_1_polygon_deviation'] = (csv_df['Mouse_1_poly_area'].mean() - csv_df['Mouse_1_poly_area'])
        csv_df['Mouse_2_polygon_deviation'] = (csv_df['Mouse_2_poly_area'].mean() - csv_df['Mouse_2_poly_area'])

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_movement_all_bodyparts_both_mice_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Sum_euclid_distances_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_smallest_euclid_distances_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_largest_euclid_distances_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_mean_euclid_distances_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_angle_both_mice_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        ########### PERCENTILE RANK ###########################################
        print('Calculating percentile ranks...')
        csv_df['Movement_percentile_rank'] = csv_df['Total_movement_centroids'].rank(pct=True)
        csv_df['Distance_percentile_rank'] = csv_df['Centroid_distance'].rank(pct=True)
        csv_df['Movement_mouse_1_percentile_rank'] = csv_df['Movement_mouse_1_centroid'].rank(pct=True)
        csv_df['Movement_mouse_2_percentile_rank'] = csv_df['Movement_mouse_1_centroid'].rank(pct=True)
        csv_df['Movement_mouse_1_deviation_percentile_rank'] = csv_df['Movement_mouse_1_deviation_centroid'].rank(
            pct=True)
        csv_df['Movement_mouse_2_deviation_percentile_rank'] = csv_df['Movement_mouse_2_deviation_centroid'].rank(
            pct=True)
        csv_df['Centroid_distance_percentile_rank'] = csv_df['Centroid_distance'].rank(pct=True)
        csv_df['Centroid_distance_deviation_percentile_rank'] = csv_df['Centroid_distance_deviation'].rank(pct=True)

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_movement_all_bodyparts_both_mice_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Sum_euclid_distances_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_mean_euclid_distances_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_smallest_euclid_distances_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Mouse1_largest_euclid_distances_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        ########### CALCULATE STRAIGHTNESS OF POLYLINE PATH: tortuosity  ###########################################
        print('Calculating path tortuosities...')
        as_strided = np.lib.stride_tricks.as_strided
        win_size = 3
        centroidList_Mouse1_x = as_strided(csv_df.Center_1_x, (len(csv_df) - (win_size - 1), win_size),
                                           (csv_df.Center_1_x.values.strides * 2))
        centroidList_Mouse1_y = as_strided(csv_df.Center_1_y, (len(csv_df) - (win_size - 1), win_size),
                                           (csv_df.Center_1_y.values.strides * 2))
        centroidList_Mouse2_x = as_strided(csv_df.Center_2_x, (len(csv_df) - (win_size - 1), win_size),
                                           (csv_df.Center_2_x.values.strides * 2))
        centroidList_Mouse2_y = as_strided(csv_df.Center_2_y, (len(csv_df) - (win_size - 1), win_size),
                                           (csv_df.Center_2_y.values.strides * 2))

        for k in range(len(roll_windows_values)):
            start = 0
            end = start + int(roll_windows_values[k])
            tortuosity_M1 = []
            tortuosity_M2 = []
            for y in range(len(csv_df)):
                tortuosity_List_M1 = []
                tortuosity_List_M2 = []
                CurrCentroidList_Mouse1_x = centroidList_Mouse1_x[start:end]
                CurrCentroidList_Mouse1_y = centroidList_Mouse1_y[start:end]
                CurrCentroidList_Mouse2_x = centroidList_Mouse2_x[start:end]
                CurrCentroidList_Mouse2_y = centroidList_Mouse2_y[start:end]
                for i in range(len(CurrCentroidList_Mouse1_x)):
                    currMovementAngle_mouse1 = (
                        angle3pt(CurrCentroidList_Mouse1_x[i][0], CurrCentroidList_Mouse1_y[i][0],
                                 CurrCentroidList_Mouse1_x[i][1], CurrCentroidList_Mouse1_y[i][1],
                                 CurrCentroidList_Mouse1_x[i][2], CurrCentroidList_Mouse1_y[i][2]))
                    currMovementAngle_mouse2 = (
                        angle3pt(CurrCentroidList_Mouse2_x[i][0], CurrCentroidList_Mouse2_y[i][0],
                                 CurrCentroidList_Mouse2_x[i][1], CurrCentroidList_Mouse2_y[i][1],
                                 CurrCentroidList_Mouse2_x[i][2], CurrCentroidList_Mouse2_y[i][2]))
                    tortuosity_List_M1.append(currMovementAngle_mouse1)
                    tortuosity_List_M2.append(currMovementAngle_mouse2)
                tortuosity_M1.append(sum(tortuosity_List_M1) / (2 * math.pi))
                tortuosity_M2.append(sum(tortuosity_List_M2) / (2 * math.pi))
                start += 1
                end += 1
            currentColName1 = str('Tortuosity_Mouse1_') + str(roll_windows_values[k])
            #currentColName2 = str('Tortuosity_Mouse2_') + str(roll_windows_values[k])
            csv_df[currentColName1] = tortuosity_M1
            #csv_df[currentColName2] = tortuosity_M2

        ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
        print('Calculating pose probability scores...')
        csv_df['Sum_probabilities'] = (
                    csv_df['Ear_left_1_p'] + csv_df['Ear_right_1_p'] + csv_df['Nose_1_p'] + csv_df['Center_1_p'] +
                    csv_df['Lat_left_1_p'] + csv_df['Lat_right_1_p'] + csv_df['Tail_base_1_p'] + csv_df[
                        'Tail_end_1_p'] + csv_df['Ear_left_2_p'] + csv_df['Ear_right_2_p'] + csv_df['Nose_2_p'] +
                    csv_df['Center_2_p'] + csv_df['Lat_left_2_p'] + csv_df['Lat_right_2_p'] + csv_df['Tail_base_2_p'] +
                    csv_df['Tail_end_2_p'])
        csv_df['Sum_probabilities_deviation'] = (csv_df['Sum_probabilities'].mean() - csv_df['Sum_probabilities'])
        csv_df['Sum_probabilities_deviation_percentile_rank'] = csv_df['Sum_probabilities_deviation'].rank(pct=True)
        csv_df['Sum_probabilities_percentile_rank'] = csv_df['Sum_probabilities_deviation_percentile_rank'].rank(
            pct=True)
        csv_df_probability = csv_df.filter(
            ['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p',
             'Tail_base_1_p', 'Tail_end_1_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p', 'Lat_left_2_p',
             'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p'])
        values_in_range_min, values_in_range_max = 0.0, 0.1
        csv_df["Low_prob_detections_0.1"] = csv_df_probability.apply(
            func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.5
        csv_df["Low_prob_detections_0.5"] = csv_df_probability.apply(
            func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.75
        csv_df["Low_prob_detections_0.75"] = csv_df_probability.apply(
            func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)

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