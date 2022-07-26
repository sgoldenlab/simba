from __future__ import division
import os, glob
import pandas as pd
import math
import numpy as np
from scipy.spatial import ConvexHull
import scipy
from simba.features_scripts.unit_tests import check_minimum_roll_windows, read_video_info_csv, read_video_info
from configparser import ConfigParser, NoOptionError, NoSectionError
from simba.features_scripts.unit_tests import read_video_info
from simba.drop_bp_cords import get_workflow_file_format
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df, save_df

def extract_features_wotarget_9(inifile):
    config = ConfigParser()
    config.read(str(inifile))
    csv_dir = config.get('General settings', 'csv_path')
    csv_dir_in = os.path.join(csv_dir, 'outlier_corrected_movement_location')
    csv_dir_out = os.path.join(csv_dir, 'features_extracted')
    project_path = config.get('General settings', 'project_path')
    vidinfDf = read_video_info_csv(os.path.join(project_path, 'logs', 'video_info.csv'))

    if not os.path.exists(csv_dir_out):
        os.makedirs(csv_dir_out)
    def count_values_in_range(series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    def angle3pt(ax, ay, bx, by, cx, cy):
        ang = math.degrees(
            math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx))
        return ang + 360 if ang < 0 else ang

    roll_windows_values = [2, 5, 6, 7.5, 15]
    roll_windows_values = check_minimum_roll_windows(roll_windows_values, vidinfDf['fps'].min())

    wfileType = get_workflow_file_format(config)

    filesFound = glob.glob(csv_dir_in + '/*.' + wfileType)
    print('Extracting features from ' + str(len(filesFound)) + ' file(s)...')

    ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
    for file_path in filesFound:
        M1_hull_large_euclidean_list = []
        M1_hull_small_euclidean_list = []
        M1_hull_mean_euclidean_list = []
        M1_hull_sum_euclidean_list = []

        _, currVidName, _ = get_fn_ext(file_path)

        currVideoSettings, currPixPerMM, fps = read_video_info(vidinfDf,currVidName)
        print('Processing ' + '"' + str(currVidName) + '".' + ' Fps: ' + str(fps) + ". mm/ppx: " + str(currPixPerMM))

        roll_windows = []
        for i in range(len(roll_windows_values)):
            roll_windows.append(int(fps / roll_windows_values[i]))


        columnHeaders = ["Mouse1_left_ear_x", "Mouse1_left_ear_y", "Mouse1_left_ear_p", \
                         "Mouse1_right_ear_x", "Mouse1_right_ear_y", "Mouse1_right_ear_p", \
                         "Mouse1_left_hand_x", "Mouse1_left_hand_y", "Mouse1_left_hand_p", \
                         "Mouse1_right_hand_x", "Mouse1_right_hand_y", "Mouse1_right_hand_p", \
                         "Mouse1_left_foot_x", "Mouse1_left_foot_y", "Mouse1_left_foot_p", \
                         "Mouse1_right_foot_x", "Mouse1_right_foot_y", "Mouse1_right_foot_p", \
                         "Mouse1_nose_x", "Mouse1_nose_y", "Mouse1_nose_p", \
                         "Mouse1_tail_x", "Mouse1_tail_y", "Mouse1_tail_p", \
                         "Mouse1_back_x", "Mouse1_back_y", "Mouse1_back_p"]

        csv_df = read_df(file_path, wfileType).fillna(0)
        csv_df = csv_df.apply(pd.to_numeric).reset_index(drop=True)
        csv_df.columns = columnHeaders

        print('Evaluating convex hulls...')
        ########### MOUSE AREAS ###########################################
        csv_df['Mouse_poly_area'] = csv_df.apply(lambda x: ConvexHull(np.array(
            [[x['Mouse1_left_ear_x'], x["Mouse1_left_ear_y"]],
             [x['Mouse1_right_ear_x'], x["Mouse1_right_ear_y"]],
             [x['Mouse1_left_hand_x'], x["Mouse1_left_hand_y"]],
             [x['Mouse1_right_hand_x'], x["Mouse1_right_hand_y"]], \
             [x['Mouse1_left_foot_x'], x["Mouse1_left_foot_y"]],
             [x['Mouse1_tail_x'], x["Mouse1_tail_y"]],
             [x['Mouse1_right_foot_x'], x["Mouse1_right_foot_y"]],
             [x['Mouse1_back_x'], x["Mouse1_back_y"]],
             [x['Mouse1_nose_x'], x["Mouse1_nose_y"]]])).area, axis=1)
        csv_df['Mouse_poly_area'] = csv_df['Mouse_poly_area'] / currPixPerMM

        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        csv_df_shifted = csv_df.shift(periods=1)
        csv_df_shifted = csv_df_shifted.rename(
            columns={'Mouse1_left_ear_x': 'Mouse1_left_ear_x_shifted', 'Mouse1_left_ear_y': 'Mouse1_left_ear_y_shifted',
                     'Mouse1_left_ear_p': 'Mouse1_left_ear_p_shifted', 'Mouse1_right_ear_x': 'Mouse1_right_ear_x_shifted', \
                     'Mouse1_right_ear_y': 'Mouse1_right_ear_y_shifted', 'Mouse1_right_ear_p': 'Mouse1_right_ear_p_shifted',
                     'Mouse1_left_hand_x': 'Mouse1_left_hand_x_shifted', 'Mouse1_left_hand_y': 'Mouse1_left_hand_y_shifted', \
                     'Mouse1_left_hand_p': 'Mouse1_left_hand_p_shifted', 'Mouse1_right_hand_x': 'Mouse1_right_hand_x_shifted',
                     'Mouse1_right_hand_y': 'Mouse1_right_hand_y_shifted', 'Mouse1_right_hand_p': 'Mouse1_right_hand_p_shifted', 'Mouse1_left_foot_x': \
                         'Mouse1_left_foot_x_shifted', 'Mouse1_left_foot_y': 'Mouse1_left_foot_y_shifted',
                     'Mouse1_left_foot_p': 'Mouse1_left_foot_p_shifted', 'Mouse1_right_foot_x': 'Mouse1_right_foot_x_shifted',
                     'Mouse1_right_foot_y': 'Mouse1_right_foot_y_shifted', \
                     'Mouse1_right_foot_p': 'Mouse1_right_foot_p_shifted', 'Mouse1_nose_x': 'Mouse1_nose_x_shifted',
                     'Mouse1_nose_y': 'Mouse1_nose_y_shifted', 'Mouse1_nose_p': 'Mouse1_nose_p_shifted', 'Mouse1_tail_x': 'Mouse1_tail_x_shifted',
                     'Mouse1_tail_y': 'Mouse1_tail_y_shifted', 'Mouse1_tail_p': 'Mouse1_tail_p_shifted',
                     'Mouse1_back_x': 'Mouse1_back_x_shifted', 'Mouse1_back_y': 'Mouse1_back_y_shifted',
                     'Mouse1_back_p': 'Mouse1_back_p_shifted', 'Mouse_poly_area': 'Mouse_poly_area_shifted'})
        csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner')
        csv_df_combined = csv_df_combined.fillna(0)
        csv_df_combined = csv_df_combined.reset_index(drop=True)

        print('Calculating euclidean distances...')
        ########### EUCLIDEAN DISTANCES ###########################################
        csv_df['Nose_to_tail'] = (np.sqrt((csv_df.Mouse1_nose_x - csv_df.Mouse1_tail_x) ** 2 + (csv_df.Mouse1_nose_y - csv_df.Mouse1_tail_y) ** 2)) / currPixPerMM
        csv_df['Distance_feet'] = (np.sqrt((csv_df.Mouse1_left_foot_x - csv_df.Mouse1_right_foot_x) ** 2 + (csv_df.Mouse1_left_foot_y - csv_df.Mouse1_right_foot_y) ** 2)) / currPixPerMM
        csv_df['Distance_hands'] = (np.sqrt((csv_df.Mouse1_left_hand_x - csv_df.Mouse1_right_hand_x) ** 2 + (csv_df.Mouse1_left_hand_y - csv_df.Mouse1_right_hand_y) ** 2)) / currPixPerMM
        csv_df['Distance_ears'] = (np.sqrt((csv_df.Mouse1_left_ear_x - csv_df.Mouse1_right_ear_x) ** 2 + (csv_df.Mouse1_left_ear_y - csv_df.Mouse1_right_ear_y) ** 2)) / currPixPerMM
        csv_df['Distance_unilateral_left_hands_feet'] = (np.sqrt((csv_df.Mouse1_left_foot_x - csv_df.Mouse1_left_hand_x) ** 2 + (csv_df.Mouse1_left_foot_y - csv_df.Mouse1_left_hand_y) ** 2)) / currPixPerMM
        csv_df['Distance_unilateral_right_hands_feet'] = (np.sqrt((csv_df.Mouse1_right_foot_x - csv_df.Mouse1_right_hand_x) ** 2 + (csv_df.Mouse1_right_foot_y - csv_df.Mouse1_right_hand_y) ** 2)) / currPixPerMM
        csv_df['Distance_bilateral_left_foot_right_hand'] = (np.sqrt((csv_df.Mouse1_left_foot_x - csv_df.Mouse1_right_hand_x) ** 2 + (csv_df.Mouse1_left_foot_y - csv_df.Mouse1_right_hand_y) ** 2)) / currPixPerMM
        csv_df['Distance_bilateral_right_foot_left_hand'] = (np.sqrt((csv_df.Mouse1_right_foot_x - csv_df.Mouse1_left_hand_x) ** 2 + (csv_df.Mouse1_right_foot_y - csv_df.Mouse1_left_hand_y) ** 2)) / currPixPerMM
        csv_df['Distance_back_tail'] = (np.sqrt((csv_df.Mouse1_back_x - csv_df.Mouse1_tail_x) ** 2 + (csv_df.Mouse1_back_y - csv_df.Mouse1_tail_y) ** 2)) / currPixPerMM
        csv_df['Distance_back_nose'] = (np.sqrt((csv_df.Mouse1_back_x - csv_df.Mouse1_nose_x) ** 2 + (csv_df.Mouse1_back_y - csv_df.Mouse1_nose_y) ** 2)) / currPixPerMM

        csv_df['Movement_nose'] = (np.sqrt((csv_df_combined.Mouse1_nose_x_shifted - csv_df_combined.Mouse1_nose_x) ** 2 + (csv_df_combined.Mouse1_nose_y_shifted - csv_df_combined.Mouse1_nose_y) ** 2)) / currPixPerMM
        csv_df['Movement_back'] = (np.sqrt((csv_df_combined.Mouse1_back_x_shifted - csv_df_combined.Mouse1_back_x) ** 2 + (csv_df_combined.Mouse1_back_y_shifted - csv_df_combined.Mouse1_back_y) ** 2)) / currPixPerMM
        csv_df['Movement_left_ear'] = (np.sqrt((csv_df_combined.Mouse1_left_ear_x_shifted - csv_df_combined.Mouse1_left_ear_x) ** 2 + (csv_df_combined.Mouse1_left_ear_y_shifted - csv_df_combined.Mouse1_left_ear_y) ** 2)) / currPixPerMM
        csv_df['Movement_right_ear'] = (np.sqrt((csv_df_combined.Mouse1_right_ear_x_shifted - csv_df_combined.Mouse1_right_ear_x) ** 2 + (csv_df_combined.Mouse1_right_ear_y_shifted - csv_df_combined.Mouse1_right_ear_y) ** 2)) / currPixPerMM
        csv_df['Movement_left_foot'] = (np.sqrt((csv_df_combined.Mouse1_left_foot_x_shifted - csv_df_combined.Mouse1_left_foot_x) ** 2 + (csv_df_combined.Mouse1_left_foot_y_shifted - csv_df_combined.Mouse1_left_foot_y) ** 2)) / currPixPerMM
        csv_df['Movement_right_foot'] = (np.sqrt((csv_df_combined.Mouse1_right_foot_x_shifted - csv_df_combined.Mouse1_right_foot_x) ** 2 + (csv_df_combined.Mouse1_right_foot_y_shifted - csv_df_combined.Mouse1_right_foot_y) ** 2)) / currPixPerMM
        csv_df['Movement_tail'] = (np.sqrt((csv_df_combined.Mouse1_tail_x_shifted - csv_df_combined.Mouse1_tail_x) ** 2 + (csv_df_combined.Mouse1_tail_y_shifted - csv_df_combined.Mouse1_tail_y) ** 2)) / currPixPerMM
        csv_df['Movement_left_hand'] = (np.sqrt((csv_df_combined.Mouse1_left_hand_x_shifted - csv_df_combined.Mouse1_left_hand_x) ** 2 + (csv_df_combined.Mouse1_left_hand_y_shifted - csv_df_combined.Mouse1_left_hand_y) ** 2)) / currPixPerMM
        csv_df['Movement_right_hand'] = (np.sqrt((csv_df_combined.Mouse1_right_hand_x_shifted - csv_df_combined.Mouse1_right_hand_x) ** 2 + (csv_df_combined.Mouse1_right_hand_y_shifted - csv_df_combined.Mouse1_right_hand_y) ** 2)) / currPixPerMM
        csv_df['Mouse_polygon_size_change'] = (csv_df_combined['Mouse_poly_area_shifted']) - (csv_df_combined['Mouse_poly_area'])



        print('Calculating hull variables...')
        ########### HULL - EUCLIDEAN DISTANCES ###########################################
        for index, row in csv_df.iterrows():
            M1_np_array = np.array(
                [[row['Mouse1_left_ear_x'], row["Mouse1_left_ear_y"]], [row['Mouse1_right_ear_x'], row["Mouse1_right_ear_y"]],
                 [row['Mouse1_left_hand_x'], row["Mouse1_left_hand_y"]], [row['Mouse1_right_hand_x'], row["Mouse1_right_hand_y"]],
                 [row['Mouse1_left_foot_x'], row["Mouse1_left_foot_y"]], [row['Mouse1_tail_x'], row["Mouse1_tail_y"]], [row['Mouse1_back_x'], row["Mouse1_back_y"]], [row['Mouse1_nose_x'], row["Mouse1_nose_y"]],
                 [row['Mouse1_right_foot_x'], row["Mouse1_right_foot_y"]]]).astype(int)
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
        csv_df['Largest_euclidean_distance_hull'] = list(
            map(lambda x: x / currPixPerMM, M1_hull_large_euclidean_list))
        csv_df['Smallest_euclidean_distance_hull'] = list(
            map(lambda x: x / currPixPerMM, M1_hull_small_euclidean_list))
        csv_df['Mean_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_mean_euclidean_list))
        csv_df['Sum_euclidean_distance_hull'] = list(map(lambda x: x / currPixPerMM, M1_hull_sum_euclidean_list))

        ########### COLLAPSED MEASURES ###########################################
        csv_df['Total_movement_all_bodyparts'] = csv_df['Movement_nose'] + csv_df['Movement_back'] + csv_df['Movement_left_ear'] + csv_df['Movement_right_ear'] + csv_df['Movement_left_foot'] + csv_df['Movement_right_foot'] + csv_df['Movement_tail'] + csv_df['Movement_left_hand'] + csv_df['Movement_right_hand']


        ########### CALC ROLLING WINDOWS MEDIANS AND MEANS ###########################################
        print('Calculating rolling windows descriptives: medians, medians, and sums...')

        for i in range(len(roll_windows_values)):
            currentColName = 'Nose_to_tail_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Nose_to_tail'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Nose_to_tail_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Nose_to_tail'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Nose_to_tail_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Nose_to_tail'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_feet_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_feet'].rolling(roll_windows[i],min_periods=1).median()
            currentColName = 'Distance_feet_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_feet'].rolling(roll_windows[i],min_periods=1).mean()
            currentColName = 'Distance_feet_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_feet'].rolling(roll_windows[i],min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_ears_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_ears'].rolling(roll_windows[i],min_periods=1).median()
            currentColName = 'Distance_ears_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_ears'].rolling(roll_windows[i],min_periods=1).mean()
            currentColName = 'Distance_ears_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_ears'].rolling(roll_windows[i],min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_unilateral_left_hands_feet_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_unilateral_left_hands_feet'].rolling(roll_windows[i],min_periods=1).median()
            currentColName = 'Distance_unilateral_left_hands_feet_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_unilateral_left_hands_feet'].rolling(roll_windows[i],min_periods=1).mean()
            currentColName = 'Distance_unilateral_left_hands_feet_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_unilateral_left_hands_feet'].rolling(roll_windows[i],min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_unilateral_right_hands_feet_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_unilateral_right_hands_feet'].rolling(roll_windows[i],min_periods=1).median()
            currentColName = 'Distance_unilateral_right_hands_feet_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_unilateral_right_hands_feet'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Distance_unilateral_right_hands_feet_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_unilateral_right_hands_feet'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_bilateral_left_foot_right_hand_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_bilateral_left_foot_right_hand'].rolling(roll_windows[i],min_periods=1).median()
            currentColName = 'Distance_bilateral_left_foot_right_hand_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_bilateral_left_foot_right_hand'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Distance_bilateral_left_foot_right_hand_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_bilateral_left_foot_right_hand'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_bilateral_right_foot_left_hand_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_bilateral_right_foot_left_hand'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Distance_bilateral_right_foot_left_hand_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_bilateral_right_foot_left_hand'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Distance_bilateral_right_foot_left_hand_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_bilateral_right_foot_left_hand'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_back_tail_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_back_tail'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Distance_back_tail_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_back_tail'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Distance_back_tail_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_back_tail'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Distance_back_nose_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_back_nose'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Distance_back_nose_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_back_nose'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Distance_back_nose_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Distance_back_nose'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_nose_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_nose'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_nose_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_nose'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_nose_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_nose'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_back_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_back'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_back_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_back'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_back_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_back'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_left_ear_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_ear'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_left_ear_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_ear'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_left_ear_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_ear'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_right_ear_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_ear'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_right_ear_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_ear'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_right_ear_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_ear'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_left_foot_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_foot'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_left_foot_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_foot'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_left_foot_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_foot'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_right_foot_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_foot'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_right_foot_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_foot'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_right_foot_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_foot'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_right_foot_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_foot'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_right_foot_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_foot'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_right_foot_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_foot'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_tail_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_tail'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_tail_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_tail'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_tail_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_tail'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_left_hand_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_hand'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_left_hand_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_hand'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_left_hand_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_left_hand'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Movement_right_hand_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_hand'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Movement_right_hand_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_hand'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_right_hand_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Movement_right_hand'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_movement_all_bodyparts_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_all_bodyparts'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Total_movement_all_bodyparts_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_all_bodyparts'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Total_movement_all_bodyparts_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Total_movement_all_bodyparts'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Mean_euclidean_distance_hull_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mean_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Mean_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mean_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Mean_euclidean_distance_hull_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Mean_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Smallest_euclidean_distance_hull_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Smallest_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Smallest_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Smallest_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Smallest_euclidean_distance_hull_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Smallest_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).sum()

        for i in range(len(roll_windows_values)):
            currentColName = 'Largest_euclidean_distance_hull_median_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Largest_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).median()
            currentColName = 'Largest_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Largest_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).mean()
            currentColName = 'Largest_euclidean_distance_hull_sum_' + str(roll_windows_values[i])
            csv_df[currentColName] = csv_df['Largest_euclidean_distance_hull'].rolling(roll_windows[i], min_periods=1).sum()

        ########### ANGLES ###########################################
        print('Calculating angles...')
        csv_df['Mouse_angle'] = csv_df.apply(lambda x: angle3pt(x['Mouse1_nose_x'], x['Mouse1_nose_y'], x['Mouse1_back_x'], x['Mouse1_back_y'], x['Mouse1_tail_x'], x['Mouse1_tail_y']), axis=1)

        ########### DEVIATIONS ###########################################
        print('Calculating deviations...')
        csv_df['Total_movement_all_bodyparts_both_deviation'] = (csv_df['Total_movement_all_bodyparts'].mean() - csv_df['Total_movement_all_bodyparts'])
        csv_df['Smallest_euclid_distances_hull_deviation'] = (csv_df['Smallest_euclidean_distance_hull'].mean() - csv_df['Smallest_euclidean_distance_hull'])
        csv_df['Largest_euclid_distances_hull_deviation'] = (csv_df['Largest_euclidean_distance_hull'].mean() - csv_df['Largest_euclidean_distance_hull'])
        csv_df['Mean_euclid_distances_hull_deviation'] = (csv_df['Mean_euclidean_distance_hull'].mean() - csv_df['Mean_euclidean_distance_hull'])
        csv_df['Movement_deviation_back'] = (csv_df['Movement_back'].mean() - csv_df['Movement_back'])
        csv_df['Polygon_deviation'] = (csv_df['Mouse_poly_area'].mean() - csv_df['Mouse_poly_area'])

        for i in range(len(roll_windows_values)):
            currentColName = 'Smallest_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Largest_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Mean_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_movement_all_bodyparts_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_deviation'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        ########### PERCENTILE RANK ###########################################
        print('Calculating percentile ranks...')
        csv_df['Movement_percentile_rank'] = csv_df['Movement_back'].rank(pct=True)

        for i in range(len(roll_windows_values)):
            currentColName = 'Mean_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Smallest_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Largest_euclidean_distance_hull_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        for i in range(len(roll_windows_values)):
            currentColName = 'Total_movement_all_bodyparts_mean_' + str(roll_windows_values[i])
            currentDev_colName = currentColName + '_percentile_rank'
            csv_df[currentDev_colName] = (csv_df[currentColName].mean() - csv_df[currentColName])

        ########### CALCULATE STRAIGHTNESS OF POLYLINE PATH: tortuosity  ###########################################
        print('Calculating path tortuosities...')
        as_strided = np.lib.stride_tricks.as_strided
        win_size = 3
        centroidList_Mouse1_x = as_strided(csv_df.Mouse1_nose_x, (len(csv_df) - (win_size - 1), win_size),
                                           (csv_df.Mouse1_nose_x.values.strides * 2))
        centroidList_Mouse1_y = as_strided(csv_df.Mouse1_nose_y, (len(csv_df) - (win_size - 1), win_size),
                                           (csv_df.Mouse1_nose_y.values.strides * 2))

        for k in range(len(roll_windows_values)):
            start = 0
            end = start + int(roll_windows_values[k])
            tortuosity_M1 = []
            for y in range(len(csv_df)):
                tortuosity_List_M1 = []
                CurrCentroidList_Mouse1_x = centroidList_Mouse1_x[start:end]
                CurrCentroidList_Mouse1_y = centroidList_Mouse1_y[start:end]
                for i in range(len(CurrCentroidList_Mouse1_x)):
                    currMovementAngle_mouse1 = (
                        angle3pt(CurrCentroidList_Mouse1_x[i][0], CurrCentroidList_Mouse1_y[i][0],
                                 CurrCentroidList_Mouse1_x[i][1], CurrCentroidList_Mouse1_y[i][1],
                                 CurrCentroidList_Mouse1_x[i][2], CurrCentroidList_Mouse1_y[i][2]))
                    tortuosity_List_M1.append(currMovementAngle_mouse1)
                tortuosity_M1.append(sum(tortuosity_List_M1) / (2 * math.pi))
                start += 1
                end += 1
            currentColName1 = str('Tortuosity_Mouse1_') + str(roll_windows_values[k])
            csv_df[currentColName1] = tortuosity_M1

        ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
        print('Calculating pose probability scores...')
        csv_df['Sum_probabilities'] = (csv_df['Mouse1_left_ear_p'] + csv_df['Mouse1_right_ear_p'] + csv_df['Mouse1_left_hand_p'] + csv_df['Mouse1_right_hand_p'] + csv_df['Mouse1_left_foot_p'] + csv_df['Mouse1_tail_p'] + csv_df['Mouse1_right_foot_p'] + csv_df['Mouse1_back_p'] + csv_df['Mouse1_nose_p'])
        csv_df['Sum_probabilities_deviation'] = (csv_df['Sum_probabilities'].mean() - csv_df['Sum_probabilities'])
        csv_df['Sum_probabilities_deviation_percentile_rank'] = csv_df['Sum_probabilities_deviation'].rank(pct=True)
        csv_df['Sum_probabilities_percentile_rank'] = csv_df['Sum_probabilities_deviation_percentile_rank'].rank(pct=True)
        csv_df_probability = csv_df.filter(
            ['Ear_left_p', 'Ear_right_p', 'Nose_p', 'Center_p', 'Lat_left_p', 'Lat_right_p',
             'Tail_base_p', 'Tail_end_p'])
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
        csv_df = csv_df.reset_index(drop=True).fillna(0)
        if 'index' in csv_df.columns:
            csv_df = csv_df.drop(columns=['index'])
        save_path = os.path.join(csv_dir_out, currVidName + '.' + wfileType)
        save_df(csv_df, wfileType, save_path)
        print('Feature extraction complete for ' + '"' + str(currVidName) + '".')
    print('All feature extraction complete.')

#extract_features_wotarget_9(r"Z:\DeepLabCut\DLC_extract\Troubleshooting\9bpagain\project_folder\project_config.ini")