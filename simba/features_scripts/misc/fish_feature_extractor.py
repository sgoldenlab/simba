from __future__ import division
import os
import pandas as pd
import numpy as np
from configparser import ConfigParser, NoOptionError
import glob
import math
from numba import jit
from simba.rw_dfs import *
from simba.drop_bp_cords import *
from simba.features_scripts.unit_tests import read_video_info, check_minimum_roll_windows
from simba.drop_bp_cords import get_fn_ext, getBpNames, getBpHeaders


class FishFeatureExtractor:
    def __init__(self, config_path: str):

        self.TAIL_BP_NAMES = ['Zebrafish_Tail1', 'Zebrafish_Tail2', 'Zebrafish_Tail3','Zebrafish_Tail4']
        self.CENTER_BP_NAMES = ['Zebrafish_SwimBladder']

        self.windows_angular_dispersion_seconds = [10]

        self.compass_brackets = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        self.compass_brackets_long = ["Direction_N", "Direction_NE", "Direction_E", "Direction_SE", "Direction_S", "Direction_SW", "Direction_W", "Direction_NW"]
        self.compass_brackets_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "0"]


        self.config = ConfigParser()
        self.config.read(str(config_path))
        self.project_path = self.config.get('General settings', 'project_path')
        self.input_file_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.save_dir = os.path.join(self.project_path, 'csv', 'features_extracted')
        self.video_info_path = os.path.join(self.project_path, 'logs', 'video_info.csv')
        self.video_info_df = pd.read_csv(self.video_info_path)
        bp_names_path = os.path.join(os.path.join( self.project_path, 'logs'), 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv')
        bp_names_df = pd.read_csv(bp_names_path, header=None)
        self.bp_names_list = list(bp_names_df[0])
        try:
            self.wfileType = self.config.get('General settings', 'workflow_file_type')
        except NoOptionError:
            self.wfileType = 'csv'

        self.roll_windows_values = check_minimum_roll_windows([10, 4, 2, 1, 0.1], self.video_info_df['fps'].min())
        self.files_found = glob.glob(self.input_file_dir + '/*.{}'.format(self.wfileType))
        print('Extracting features from {} {}'.format(str(len(self.files_found)), ' files'))

        for file_path in self.files_found:
            self.roll_windows, self.angular_dispersion_windows = [], []
            dir_name, file_name, ext = get_fn_ext(file_path)
            self.save_path = os.path.join(self.save_dir, os.path.basename(file_path))
            currVideoSettings, self.currPixPerMM, self.fps = read_video_info(self.video_info_df, file_name)

            for i in range(len(self.roll_windows_values)):
                self.roll_windows.append(int(self.fps / self.roll_windows_values[i]))
            for i in range(len(self.windows_angular_dispersion_seconds)):
                self.angular_dispersion_windows.append(int(self.fps * self.windows_angular_dispersion_seconds[i]))

            self.x_cols, self.y_cols, self.p_cols = getBpNames(str(config_path))
            self.x_cols_shifted, self.y_cols_shifted = [], []
            for col in zip(self.x_cols, self.y_cols, self.p_cols):
                self.x_cols_shifted.append(col[0] + '_shifted')
                self.y_cols_shifted.append(col[1] + '_shifted')

            bp_header_list = getBpHeaders(str(config_path))
            self.col_headers_shifted = []
            for bp in self.bp_names_list:
                self.col_headers_shifted.extend((bp + '_x_shifted', bp + '_y_shifted', bp + '_p_shifted'))

            csv_df = read_df(file_path, self.wfileType)
            csv_df.columns = bp_header_list
            csv_df = csv_df.fillna(0).apply(pd.to_numeric)

            csv_df_shifted = csv_df.shift(periods=1)
            csv_df_shifted.columns = self.col_headers_shifted
            self.csv_df_combined = pd.concat([csv_df, csv_df_shifted], axis=1, join='inner').fillna(0)
            self.calc_X_relative_to_Y_movement()
            self.calc_tail_and_center_movement()
            self.calc_X_relative_to_Y_movement_rolling_windows()
            self.calc_center_velocity()
            self.calc_rotation()
            self.calc_direction_switches()
            self.hot_end_encode_compass()
            self.calc_directional_switches_in_rolling_windows()
            self.calc_angular_dispersion()
            self.save_file()

        print('Features extracted for all {} files'.format(str(len(self.files_found))))


    def angle2pt_degrees(self, ax, ay, bx, by):
        angle_degrees = math.degrees(math.atan2(ax - bx, by - ay))
        return angle_degrees + 360 if angle_degrees < 0 else angle_degrees

    def angle2pt_radians(self, degrees):
        angle_radians = degrees * math.pi / 180
        return angle_radians

    def angle2pt_sin(self, angle_radians):
        angle_sin = math.sin(angle_radians)
        return angle_sin

    def angle2pt_cos(self, angle_radians):
        angle_cos = math.cos(angle_radians)
        return angle_cos

    @staticmethod
    @jit(nopython=True)
    def euclidian_distance_calc(bp1xVals, bp1yVals, bp2xVals, bp2yVals):
        return np.sqrt((bp1xVals - bp2xVals) ** 2 + (bp1yVals - bp2yVals) ** 2)

    @staticmethod
    @jit(nopython=True)
    def angular_dispersion(cumsum_cos_np, cumsum_sin_np):
        out_array = np.empty((cumsum_cos_np.shape))
        for index in range(cumsum_cos_np.shape[0]):
            X, Y = cumsum_cos_np[index] / (index + 1), cumsum_sin_np[index] / (index + 1)
            out_array[index] = math.sqrt(X ** 2 + Y ** 2)
        return out_array

    def calc_angular_dispersion(self):
        dispersion_array = self.angular_dispersion(self.csv_df_combined['Angle_cos_cumsum'].values, self.csv_df_combined['Angle_sin_cumsum'].values)
        self.csv_df_combined['Angular_dispersion'] = dispersion_array

        for win in range(len(self.angular_dispersion_windows)):
            col_name = 'Angular_dispersion_window_' + str(self.windows_angular_dispersion_seconds[win])
            self.csv_df_combined[col_name] = self.csv_df_combined['Angular_dispersion'].rolling(self.angular_dispersion_windows[win], min_periods=1).mean()


    def calc_X_relative_to_Y_movement(self):
        temp_df = pd.DataFrame()
        for bp in range(len(self.x_cols)):
            curr_x_col, curr_x_shifted_col, curr_y_col, curr_y_shifted_col = self.x_cols[bp], self.x_cols_shifted[bp], self.y_cols[bp], self.y_cols_shifted[bp]
            temp_df['x'] = (self.csv_df_combined[curr_x_col] - self.csv_df_combined[curr_x_shifted_col])
            temp_df['y'] = (self.csv_df_combined[curr_y_col] - self.csv_df_combined[curr_y_shifted_col])
            temp_df['Movement_{}_X_relative_2_Y'.format(bp)] = (temp_df['x'] - temp_df['y'])
            temp_df.drop(['x', 'y'], axis=1, inplace=True)
        self.csv_df_combined['Movement_X_axis_relative_to_Y_axis'] = temp_df.sum(axis=1)

    def calc_tail_and_center_movement(self):
        tail_movement_col_names = []
        for tail_bp in self.TAIL_BP_NAMES:
            x_name, y_name = tail_bp + '_x', tail_bp + '_y'
            x_name_sh, y_name_sh = tail_bp + '_x_shifted', tail_bp + '_y_shifted'
            print(self.csv_df_combined.columns)
            self.csv_df_combined[tail_bp + '_movement'] = self.euclidian_distance_calc(self.csv_df_combined[x_name].values, self.csv_df_combined[y_name].values, self.csv_df_combined[x_name_sh].values, self.csv_df_combined[y_name_sh].values) / self.currPixPerMM
            tail_movement_col_names.append(tail_bp + '_movement')
        self.csv_df_combined['total_tail_bp_movement'] = self.csv_df_combined[tail_movement_col_names].sum(axis=1)

        for center_bp in self.CENTER_BP_NAMES:
            x_name, y_name = center_bp + '_x', center_bp + '_y'
            x_name_sh, y_name_sh = center_bp + '_x_shifted', center_bp + '_y_shifted'
            self.csv_df_combined[center_bp + '_movement'] = self.euclidian_distance_calc(self.csv_df_combined[x_name].values, self.csv_df_combined[y_name].values, self.csv_df_combined[x_name_sh].values, self.csv_df_combined[y_name_sh].values) / self.currPixPerMM
            self.csv_df_combined[center_bp + '_cum_distance_travelled'] = self.csv_df_combined[center_bp + '_movement'].cumsum()

    def calc_X_relative_to_Y_movement_rolling_windows(self):
        for i in range(len(self.roll_windows_values)):
            currentColName = 'Movement_X_axis_relative_to_Y_axis_mean_' + str(self.roll_windows_values[i])
            self.csv_df_combined[currentColName] = self.csv_df_combined['Movement_X_axis_relative_to_Y_axis'].rolling(self.roll_windows[i], min_periods=1).mean()
            currentColName = 'Movement_X_axis_relative_to_Y_axis_sum_' + str(self.roll_windows_values[i])
            self.csv_df_combined[currentColName] = self.csv_df_combined['Movement_X_axis_relative_to_Y_axis'].rolling(self.roll_windows[i], min_periods=1).sum()

    def calc_directional_switches_in_rolling_windows(self):
        for win in range(len(self.roll_windows_values)):
            currentColName = 'Number_of_direction_switches_' + str(self.roll_windows_values[win])
            self.csv_df_combined[currentColName] = self.csv_df_combined['Direction_switch'].rolling(self.roll_windows[win], min_periods=1).sum()
            currentColName = 'Directionality_of_switches_switches_' + str(self.roll_windows_values[win])
            self.csv_df_combined[currentColName] = self.csv_df_combined['Switch_direction_value'].rolling(self.roll_windows[win], min_periods=1).sum()

    def calc_center_velocity(self):
        for center_bp in self.CENTER_BP_NAMES:
            self.csv_df_combined[center_bp + '_velocity'] = self.csv_df_combined[center_bp + '_movement'].rolling(int(self.fps), min_periods=1).sum()

    def calc_rotation(self):
        self.csv_df_combined['Clockwise_angle_degrees'] = self.csv_df_combined.apply(lambda x: self.angle2pt_degrees(x[self.CENTER_BP_NAMES[0] + '_x'], x[self.CENTER_BP_NAMES[0] + '_y'], x[self.TAIL_BP_NAMES[0] + '_x'], x[self.TAIL_BP_NAMES[0] + '_y']), axis=1)
        self.csv_df_combined['Angle_radians'] = self.angle2pt_radians(self.csv_df_combined['Clockwise_angle_degrees'])
        self.csv_df_combined['Angle_sin'] = self.csv_df_combined.apply(lambda x: self.angle2pt_sin(x['Angle_radians']),  axis=1)
        self.csv_df_combined['Angle_cos'] = self.csv_df_combined.apply(lambda x: self.angle2pt_cos(x['Angle_radians']), axis=1)
        self.csv_df_combined['Angle_sin_cumsum'] = self.csv_df_combined['Angle_sin'].cumsum()
        self.csv_df_combined['Angle_cos_cumsum'] = self.csv_df_combined['Angle_cos'].cumsum()
        compass_lookup = list(round(self.csv_df_combined['Clockwise_angle_degrees'] / 45))
        compass_lookup = [int(i) for i in compass_lookup]
        compasFaceList_bracket, compasFaceList_digit = [], []
        for compasDirection in compass_lookup:
            compasFaceList_bracket.append(self.compass_brackets[compasDirection])
            compasFaceList_digit.append(self.compass_brackets_digits[compasDirection])
        self.csv_df_combined['Compass_direction'] = compasFaceList_bracket
        self.csv_df_combined['Compass_digit'] = compasFaceList_digit

        for i in range(len(self.roll_windows_values)):
            column_name = 'Mean_angle_time_window_{}'.format(str(self.roll_windows_values[i]))
            self.csv_df_combined[column_name] = self.csv_df_combined['Clockwise_angle_degrees'].rolling(self.roll_windows[i], min_periods=1).mean()

    def hot_end_encode_compass(self):
        compass_hot_end = pd.get_dummies(self.csv_df_combined['Compass_direction'], prefix='Direction')
        compass_hot_end = compass_hot_end.T.reindex(self.compass_brackets_long).T.fillna(0)
        self.csv_df_combined = pd.concat([self.csv_df_combined, compass_hot_end], axis=1)

    def calc_direction_switches(self):
        self.grouped_df = pd.DataFrame()
        v = (self.csv_df_combined['Compass_digit'] != self.csv_df_combined['Compass_digit'].shift()).cumsum()
        u = self.csv_df_combined.groupby(v)['Compass_digit'].agg(['all', 'count'])
        m = u['all'] & u['count'].ge(1)
        self.grouped_df['groups'] = self.csv_df_combined.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
        currdirectionList, DirectionSwitchIndexList, currdirectionListValue = [], [], []
        for indexes, row in self.grouped_df.iterrows():
            currdirectionList.append(self.csv_df_combined.loc[row['groups'][0]]['Compass_direction'])
            DirectionSwitchIndexList.append(row['groups'][1])
            currdirectionListValue.append(self.csv_df_combined.loc[row['groups'][0]]['Compass_digit'])
        self.grouped_df['Direction_switch'] = currdirectionList
        self.grouped_df['Direction_value'] = currdirectionListValue
        self.csv_df_combined.loc[DirectionSwitchIndexList, 'Direction_switch'] = 1
        self.csv_df_combined['Compass_digit_shifted'] = self.csv_df_combined['Compass_digit'].shift(-1)
        self.csv_df_combined = self.csv_df_combined.fillna(0)
        self.csv_df_combined['Switch_direction_value'] = self.csv_df_combined.apply(lambda x: self.calc_switch_direction(x['Compass_digit_shifted'], x['Compass_digit']), axis=1)

    def calc_switch_direction(self, compass_digit_shifted, compass_digit):
        if ((compass_digit_shifted == '0') and (compass_digit == '7')):
             return 1
        else:
            return int(compass_digit_shifted) - int(compass_digit)

    def save_file(self):
        self.csv_df_combined = self.csv_df_combined.drop(self.col_headers_shifted, axis=1)
        self.csv_df_combined = self.csv_df_combined.drop(['Compass_digit_shifted', 'Direction_switch', 'Switch_direction_value', 'Compass_digit', 'Compass_direction', 'Angle_sin_cumsum', 'Angle_cos_cumsum'], axis=1)
        save_df(self.csv_df_combined, self.wfileType, self.save_path)
        print('Saved file {}'.format(self.save_path), '...')