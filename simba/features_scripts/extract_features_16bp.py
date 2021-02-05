from __future__ import division
import os, glob
import pandas as pd
import math
import numpy as np
from scipy.spatial import ConvexHull
import scipy
from configparser import ConfigParser, NoOptionError, NoSectionError
from numba import jit
from simba.rw_dfs import *
import re


def extract_features_wotarget_16(inifile):
    config = ConfigParser()
    configFile = str(inifile)
    config.read(configFile)
    projectPath = config.get('General settings', 'project_path')
    csv_dir_in, csv_dir_out = os.path.join(projectPath, 'csv', 'outlier_corrected_movement_location'), os.path.join(projectPath,'csv', 'features_extracted')
    vidInfPath = os.path.join(projectPath, 'logs', 'video_info.csv')
    try:
        wfileType = config.get('General settings', 'workflow_file_type')
    except NoOptionError:
        wfileType = 'csv'
    vidinfDf = pd.read_csv(vidInfPath)
    #change videos name to str
    vidinfDf.Video = vidinfDf.Video.astype('str')

    def count_values_in_range(series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    def angle3pt(ax, ay, bx, by, cx, cy):
        ang = math.degrees(
            math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx))
        return ang + 360 if ang < 0 else ang

    @jit(nopython=True, cache=True)
    def EuclidianDistCald(bp1xVals, bp1yVals, bp2xVals, bp2yVals, currPixPerMM):
        series = (np.sqrt((bp1xVals - bp2xVals) ** 2 + (bp1yVals - bp2yVals) ** 2)) / currPixPerMM
        return series

    roll_windows, loopy = [], 0
    roll_windows_values = [2, 5, 6, 7.5, 15]

    ########### FIND CSV FILES ###########
    print(csv_dir_in)
    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Extracting features from ' + str(len(filesFound)) + ' files...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for currentFile in filesFound:
        M1_hull_large_euclidean_list, M1_hull_small_euclidean_list, M1_hull_mean_euclidean_list, M1_hull_sum_euclidean_list, M2_hull_large_euclidean_list, M2_hull_small_euclidean_list, M2_hull_mean_euclidean_list, M2_hull_sum_euclidean_list = [], [], [], [], [], [], [], []
        currVidName = os.path.basename(currentFile).replace('.'  +wfileType, '')

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
        csv_df = read_df(currentFile, wfileType)
        try:
            csv_df = csv_df.set_index('scorer')
        except KeyError:
            pass
        csv_df.columns = columnHeaders
        csv_df = csv_df.fillna(0)
        #csv_df = csv_df.drop(csv_df.index[[0]])
        csv_df = csv_df.apply(pd.to_numeric)
        csv_df = csv_df.reset_index()
        csv_df = csv_df.reset_index(drop=True)

        print('Evaluating convex hulls...')
        ########### MOUSE AREAS ###########################################

        try:
            csv_df['Mouse_1_poly_area'] = csv_df.apply(lambda x: ConvexHull(np.array(
                [[x['Ear_left_1_x'], x["Ear_left_1_y"]],
                 [x['Ear_right_1_x'], x["Ear_right_1_y"]],
                 [x['Nose_1_x'], x["Nose_1_y"]],
                 [x['Lat_left_1_x'], x["Lat_left_1_y"]], \
                 [x['Lat_right_1_x'], x["Lat_right_1_y"]],
                 [x['Tail_base_1_x'], x["Tail_base_1_y"]],
                 [x['Center_1_x'], x["Center_1_y"]]])).area, axis=1)
        except scipy.spatial.qhull.QhullError as e:
            print(e)
            print('ERROR: For more information, go to https://github.com/sgoldenlab/simba/blob/SimBA_no_TF/docs/FAQ.md#i-get-a-qhull-eg-qh6154-or-6013-error-when-extracting-the-features')

        csv_df['Mouse_1_poly_area'] = csv_df.eval('Mouse_1_poly_area / @currPixPerMM')

        try:
            csv_df['Mouse_2_poly_area'] = csv_df.apply(lambda x: ConvexHull(np.array(
                [[x['Ear_left_2_x'], x["Ear_left_2_y"]],
                 [x['Ear_right_2_x'], x["Ear_right_2_y"]],
                 [x['Nose_2_x'], x["Nose_2_y"]],
                 [x['Lat_left_2_x'], x["Lat_left_2_y"]], \
                 [x['Lat_right_2_x'], x["Lat_right_2_y"]],
                 [x['Tail_base_2_x'], x["Tail_base_2_y"]],
                 [x['Center_2_x'], x["Center_2_y"]]])).area, axis=1)
        except scipy.spatial.qhull.QhullError as e:
            print(e)
            print('ERROR: For more information, check https://github.com/sgoldenlab/simba/blob/SimBA_no_TF/docs/FAQ.md#i-get-a-qhull-eg-qh6154-or-6013-error-when-extracting-the-features')

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
        csv_df['Mouse_1_nose_to_tail'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values, csv_df['Tail_base_1_x'].values, csv_df['Tail_base_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_nose_to_tail'] = EuclidianDistCald(csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values, csv_df['Tail_base_2_x'].values, csv_df['Tail_base_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_width'] = EuclidianDistCald(csv_df['Lat_left_1_x'].values, csv_df['Lat_left_1_y'].values, csv_df['Lat_right_1_x'].values, csv_df['Lat_right_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_width'] = EuclidianDistCald(csv_df['Lat_left_2_x'].values, csv_df['Lat_left_2_y'].values, csv_df['Lat_right_2_x'].values, csv_df['Lat_right_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_Ear_distance'] = EuclidianDistCald(csv_df['Ear_left_1_x'].values, csv_df['Ear_left_1_y'].values, csv_df['Ear_right_1_x'].values, csv_df['Ear_right_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_Ear_distance'] = EuclidianDistCald(csv_df['Ear_left_2_x'].values, csv_df['Ear_left_2_y'].values, csv_df['Ear_right_2_x'].values, csv_df['Ear_right_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_Nose_to_centroid'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values, csv_df['Center_1_x'].values, csv_df['Center_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_Nose_to_centroid'] = EuclidianDistCald(csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values,csv_df['Center_2_x'].values, csv_df['Center_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_Nose_to_lateral_left'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values,csv_df['Lat_left_1_x'].values, csv_df['Lat_left_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_Nose_to_lateral_left'] = EuclidianDistCald(csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values,csv_df['Lat_left_2_x'].values, csv_df['Lat_left_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_Nose_to_lateral_right'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values,csv_df['Lat_right_1_x'].values, csv_df['Lat_right_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_Nose_to_lateral_right'] = EuclidianDistCald(csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values,csv_df['Lat_right_2_x'].values, csv_df['Lat_right_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_Centroid_to_lateral_left'] = EuclidianDistCald(csv_df['Center_1_x'].values, csv_df['Center_1_y'].values,csv_df['Lat_left_1_x'].values, csv_df['Lat_left_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_Centroid_to_lateral_left'] = EuclidianDistCald(csv_df['Center_2_x'].values, csv_df['Center_2_y'].values,csv_df['Lat_left_2_x'].values, csv_df['Lat_left_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_Centroid_to_lateral_right'] = EuclidianDistCald(csv_df['Center_1_x'].values, csv_df['Center_1_y'].values,csv_df['Lat_right_1_x'].values, csv_df['Lat_right_1_y'].values, currPixPerMM)
        csv_df['Mouse_2_Centroid_to_lateral_right'] = EuclidianDistCald(csv_df['Center_2_x'].values, csv_df['Center_2_y'].values,csv_df['Lat_right_2_x'].values, csv_df['Lat_right_2_y'].values, currPixPerMM)
        csv_df['Centroid_distance'] = EuclidianDistCald(csv_df['Center_2_x'].values, csv_df['Center_2_y'].values,csv_df['Center_1_x'].values, csv_df['Center_1_y'].values, currPixPerMM)
        csv_df['Nose_to_nose_distance'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values,csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values, currPixPerMM)
        csv_df['M1_Nose_to_M2_lat_left'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values,csv_df['Lat_left_2_x'].values, csv_df['Lat_left_2_y'].values, currPixPerMM)
        csv_df['M1_Nose_to_M2_lat_right'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values,csv_df['Lat_right_2_x'].values, csv_df['Lat_right_2_y'].values, currPixPerMM)
        csv_df['M2_Nose_to_M1_lat_left'] = EuclidianDistCald(csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values,csv_df['Lat_left_1_x'].values, csv_df['Lat_left_1_y'].values, currPixPerMM)
        csv_df['M2_Nose_to_M1_lat_right'] = EuclidianDistCald(csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values,csv_df['Lat_right_1_x'].values, csv_df['Lat_right_1_y'].values, currPixPerMM)
        csv_df['M1_Nose_to_M2_tail_base'] = EuclidianDistCald(csv_df['Nose_1_x'].values, csv_df['Nose_1_y'].values,csv_df['Tail_base_2_x'].values, csv_df['Tail_base_2_y'].values, currPixPerMM)
        csv_df['M2_Nose_to_M1_tail_base'] = EuclidianDistCald(csv_df['Nose_2_x'].values, csv_df['Nose_2_y'].values,csv_df['Tail_base_1_x'].values, csv_df['Tail_base_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_centroid'] = EuclidianDistCald(csv_df_combined['Center_1_x_shifted'].values, csv_df_combined['Center_1_y_shifted'].values,csv_df_combined['Center_1_x'].values, csv_df_combined['Center_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_centroid'] = EuclidianDistCald(csv_df_combined['Center_2_x_shifted'].values, csv_df_combined['Center_2_y_shifted'].values,csv_df_combined['Center_2_x'].values, csv_df_combined['Center_2_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_nose'] = EuclidianDistCald(csv_df_combined['Nose_1_x_shifted'].values, csv_df_combined['Nose_1_y_shifted'].values,csv_df_combined['Nose_1_x'].values, csv_df_combined['Nose_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_nose'] = EuclidianDistCald(csv_df_combined['Nose_2_x_shifted'].values, csv_df_combined['Nose_2_y_shifted'].values,csv_df_combined['Nose_2_x'].values, csv_df_combined['Nose_2_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_tail_base'] = EuclidianDistCald(csv_df_combined['Tail_base_1_x_shifted'].values, csv_df_combined['Tail_base_1_y_shifted'].values,csv_df_combined['Tail_base_1_x'].values, csv_df_combined['Tail_base_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_tail_base'] = EuclidianDistCald(csv_df_combined['Tail_base_2_x_shifted'].values, csv_df_combined['Tail_base_2_y_shifted'].values,csv_df_combined['Tail_base_2_x'].values, csv_df_combined['Tail_base_2_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_tail_end'] = EuclidianDistCald(csv_df_combined['Tail_end_1_x_shifted'].values, csv_df_combined['Tail_end_1_y_shifted'].values,csv_df_combined['Tail_end_1_x'].values, csv_df_combined['Tail_end_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_tail_end'] = EuclidianDistCald(csv_df_combined['Tail_end_2_x_shifted'].values, csv_df_combined['Tail_end_2_y_shifted'].values,csv_df_combined['Tail_end_2_x'].values, csv_df_combined['Tail_end_2_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_left_ear'] = EuclidianDistCald(csv_df_combined['Ear_left_1_x_shifted'].values, csv_df_combined['Ear_left_1_y_shifted'].values,csv_df_combined['Ear_left_1_x'].values, csv_df_combined['Ear_left_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_left_ear'] = EuclidianDistCald(csv_df_combined['Ear_left_2_x_shifted'].values, csv_df_combined['Ear_left_2_y_shifted'].values,csv_df_combined['Ear_left_2_x'].values, csv_df_combined['Ear_left_2_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_right_ear'] = EuclidianDistCald(csv_df_combined['Ear_right_1_x_shifted'].values, csv_df_combined['Ear_right_1_y_shifted'].values,csv_df_combined['Ear_right_1_x'].values, csv_df_combined['Ear_right_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_right_ear'] = EuclidianDistCald(csv_df_combined['Ear_right_2_x_shifted'].values, csv_df_combined['Ear_right_2_y_shifted'].values,csv_df_combined['Ear_right_2_x'].values, csv_df_combined['Ear_right_2_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_lateral_left'] = EuclidianDistCald(csv_df_combined['Lat_left_1_x_shifted'].values, csv_df_combined['Lat_left_1_y_shifted'].values,csv_df_combined['Lat_left_1_x'].values, csv_df_combined['Lat_left_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_lateral_left'] = EuclidianDistCald(csv_df_combined['Lat_left_2_x_shifted'].values, csv_df_combined['Lat_left_2_y_shifted'].values,csv_df_combined['Lat_left_2_x'].values, csv_df_combined['Lat_left_2_y'].values, currPixPerMM)
        csv_df['Movement_mouse_1_lateral_right'] = EuclidianDistCald(csv_df_combined['Lat_right_1_x_shifted'].values, csv_df_combined['Lat_right_1_y_shifted'].values,csv_df_combined['Lat_right_1_x'].values, csv_df_combined['Lat_right_1_y'].values, currPixPerMM)
        csv_df['Movement_mouse_2_lateral_right'] = EuclidianDistCald(csv_df_combined['Lat_right_2_x_shifted'].values, csv_df_combined['Lat_right_2_y_shifted'].values,csv_df_combined['Lat_right_2_x'].values, csv_df_combined['Lat_right_2_y'].values, currPixPerMM)
        csv_df['Mouse_1_polygon_size_change'] = pd.eval("csv_df_combined.Mouse_1_poly_area_shifted - csv_df_combined.Mouse_1_poly_area")
        csv_df['Mouse_2_polygon_size_change'] = pd.eval("csv_df_combined.Mouse_2_poly_area_shifted - csv_df_combined.Mouse_2_poly_area")

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
        csv_df['M1_largest_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_large_euclidean_list))
        csv_df['M1_smallest_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_small_euclidean_list))
        csv_df['M1_mean_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_mean_euclidean_list))
        csv_df['M1_sum_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_sum_euclidean_list))
        csv_df['M2_largest_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M2_hull_large_euclidean_list))
        csv_df['M2_smallest_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M2_hull_small_euclidean_list))
        csv_df['M2_mean_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M2_hull_mean_euclidean_list))
        csv_df['M2_sum_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M2_hull_sum_euclidean_list))
        csv_df['Sum_euclidean_distance_hull_M1_M2'] = (csv_df['M1_sum_euclidean_distance_hull'] + csv_df['M2_sum_euclidean_distance_hull'])


        ########### COLLAPSED MEASURES ###########################################
        csv_df['Total_movement_centroids'] = csv_df.eval("Movement_mouse_1_centroid + Movement_mouse_2_centroid")
        csv_df['Total_movement_tail_ends'] = csv_df.eval('Movement_mouse_1_tail_end + Movement_mouse_2_tail_end')
        csv_df['Total_movement_all_bodyparts_M1'] = csv_df.eval('Movement_mouse_1_nose + Movement_mouse_1_tail_end + Movement_mouse_1_tail_base + Movement_mouse_1_left_ear + Movement_mouse_1_right_ear + Movement_mouse_1_lateral_left + Movement_mouse_1_lateral_right')
        csv_df['Total_movement_all_bodyparts_M2'] = csv_df.eval('Movement_mouse_2_nose + Movement_mouse_2_tail_end + Movement_mouse_2_tail_base + Movement_mouse_2_left_ear + Movement_mouse_2_right_ear + Movement_mouse_2_lateral_left + Movement_mouse_2_lateral_right')
        csv_df['Total_movement_all_bodyparts_both_mice'] = csv_df.eval('Total_movement_all_bodyparts_M1 + Total_movement_all_bodyparts_M2')

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
            csv_df[currentColName] = csv_df['Sum_euclidean_distance_hull_M1_M2'].rolling(roll_windows[i], min_periods=1).sum()

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
        csv_df['Total_movement_all_bodyparts_both_mice_deviation'] = csv_df.eval('Total_movement_all_bodyparts_both_mice.mean() - Total_movement_all_bodyparts_both_mice')
        csv_df['Sum_euclid_distances_hull_deviation'] = csv_df.eval('Sum_euclidean_distance_hull_M1_M2.mean() - Sum_euclidean_distance_hull_M1_M2')
        csv_df['M1_smallest_euclid_distances_hull_deviation'] = csv_df.eval('M1_smallest_euclidean_distance_hull.mean() - M1_smallest_euclidean_distance_hull')
        csv_df['M1_largest_euclid_distances_hull_deviation'] = csv_df.eval('M1_largest_euclidean_distance_hull.mean() - M1_largest_euclidean_distance_hull')
        csv_df['M1_mean_euclid_distances_hull_deviation'] = csv_df.eval('M1_mean_euclidean_distance_hull.mean() - M1_mean_euclidean_distance_hull')
        csv_df['Centroid_distance_deviation'] = csv_df.eval('Centroid_distance.mean() - Centroid_distance')
        csv_df['Total_angle_both_mice_deviation'] = csv_df.eval('Total_angle_both_mice - Total_angle_both_mice')
        csv_df['Movement_mouse_1_deviation_centroid'] = csv_df.eval('Movement_mouse_1_centroid.mean() - Movement_mouse_1_centroid')
        csv_df['Movement_mouse_2_deviation_centroid'] = csv_df.eval('Movement_mouse_2_centroid.mean() - Movement_mouse_2_centroid')
        csv_df['Mouse_1_polygon_deviation'] = csv_df.eval('Mouse_1_poly_area.mean() - Mouse_1_poly_area')
        csv_df['Mouse_2_polygon_deviation'] = csv_df.eval('Mouse_2_poly_area.mean() - Mouse_2_poly_area')

        for i in roll_windows_values:
            currentColName = 'Total_movement_all_bodyparts_both_mice_mean_' + str(i)
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
        csv_df['Sum_probabilities'] = csv_df.eval('Ear_left_1_p + Ear_right_1_p + Nose_1_p + Center_1_p + Lat_left_1_p + Lat_right_1_p + Tail_base_1_p + Tail_end_1_p + Ear_left_2_p + Ear_right_2_p + Nose_2_p + Center_2_p + Lat_left_2_p + Lat_right_2_p + Tail_base_2_p + Tail_end_2_p')
        csv_df['Sum_probabilities_deviation'] =  csv_df.eval('Sum_probabilities.mean() - Sum_probabilities')
        csv_df['Sum_probabilities_deviation_percentile_rank'] = csv_df['Sum_probabilities_deviation'].rank(pct=True)
        csv_df['Sum_probabilities_percentile_rank'] = csv_df['Sum_probabilities_deviation_percentile_rank'].rank(pct=True)
        csv_df_probability = csv_df.filter(
            ['Ear_left_1_p', 'Ear_right_1_p', 'Nose_1_p', 'Center_1_p', 'Lat_left_1_p', 'Lat_right_1_p',
             'Tail_base_1_p', 'Tail_end_1_p', 'Ear_left_2_p', 'Ear_right_2_p', 'Nose_2_p', 'Center_2_p', 'Lat_left_2_p',
             'Lat_right_2_p', 'Tail_base_2_p', 'Tail_end_2_p'])
        values_in_range_min, values_in_range_max = 0.0, 0.1
        csv_df["Low_prob_detections_0.1"] = csv_df_probability.apply(func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.5
        csv_df["Low_prob_detections_0.5"] = csv_df_probability.apply(
            func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)
        values_in_range_min, values_in_range_max = 0.000000000, 0.75
        csv_df["Low_prob_detections_0.75"] = csv_df_probability.apply(
            func=lambda row: count_values_in_range(row, values_in_range_min, values_in_range_max), axis=1)

        ########### DROP COORDINATE COLUMNS ###########################################
        csv_df = csv_df.reset_index(drop=True)
        csv_df = csv_df.fillna(0)
        csv_df = csv_df.drop(columns=['index'], axis=1, errors='ignore')
        fileName = os.path.basename(currentFile)
        saveFN = os.path.join(csv_dir_out, fileName)
        save_df(csv_df, wfileType, saveFN)
        print('Feature extraction complete for ' + '"' + str(currVidName) + '".')
    print('All feature extraction complete.')