__author__ = "Simon Nilsson", "JJ Choong"

import glob, os

import pandas as pd
from simba.rw_dfs import read_df
import numpy as np
from shapely import geometry
from shapely.geometry import Point, Polygon
from simba.features_scripts.unit_tests import *
from datetime import datetime
from simba.misc_tools import (get_fn_ext,
                              detect_bouts,
                              SimbaTimer)
from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          check_that_column_exist,
                                          check_if_filepath_list_is_empty)
from simba.features_scripts.unit_tests import (read_video_info_csv,
                                               read_video_info)
from simba.enums import Keys, Paths, ReadConfig, Dtypes


class clf_within_ROI(object):
    """
    Class for computing aggregate statistics of classification results within user-defined ROIs.
    results are stored in `project_folder/logs` directory of the SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    -----

    Examples
    -----
    >>> clf_ROI_analyzer = clf_within_ROI(config_ini="MyConfigPath")
    >>> clf_ROI_analyzer.perform_ROI_clf_analysis()
    """

    def __init__(self,
                 config_ini: str):

        self.config = read_config_file(ini_path=config_ini)
        self.projectPath = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.PROJECT_PATH.value, data_type=ReadConfig.FOLDER_PATH.value)
        model_nos = read_config_entry(config=self.config, section=ReadConfig.SML_SETTINGS.value, option=ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.log_path_dir = os.path.join(self.projectPath, 'logs')
        self.video_info_df = read_video_info_csv(os.path.join(self.log_path_dir, 'video_info.csv'))
        self.body_parts_path = os.path.join(self.projectPath, Paths.BP_NAMES.value)
        body_parts_df = pd.read_csv(self.body_parts_path, names=['bodyparts'])
        self.body_part_list = list(body_parts_df['bodyparts'])
        self.ROIcoordinatesPath = os.path.join(self.log_path_dir, Paths.ROI_DEFINITIONS.value)
        if not os.path.isfile(self.ROIcoordinatesPath):
            print('No ROI coordinates found. Please use the [ROI] tab to define ROIs')
            raise FileNotFoundError()
        self.rectangles_df = pd.read_hdf(self.ROIcoordinatesPath, key=Keys.ROI_RECTANGLES.value)
        self.circles_df = pd.read_hdf(self.ROIcoordinatesPath, key=Keys.ROI_CIRCLES.value)
        self.polygon_df = pd.read_hdf(self.ROIcoordinatesPath, key=Keys.ROI_POLYGONS.value)
        self.ROI_str_name_list = []
        for index, row in self.rectangles_df.iterrows(): self.ROI_str_name_list.append(row['Shape_type'] + ': ' + row['Name'])
        for index, row in self.circles_df.iterrows(): self.ROI_str_name_list.append(row['Shape_type'] + ': ' + row['Name'])
        for index, row in self.polygon_df.iterrows(): self.ROI_str_name_list.append(row['Shape_type'] + ': ' + row['Name'])
        self.ROI_str_name_list = list(set(self.ROI_str_name_list))
        self.behavior_names = []
        for i in range(model_nos):
            self.behavior_names.append(self.config.get(ReadConfig.SML_SETTINGS.value, 'target_name_' + str(i + 1)))

    def __inside_rectangle(self, bp_x, bp_y, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        """
        Private helper to calculate if body-part is inside a rectangle.
        """
        if (((top_left_x) <= bp_x <= (bottom_right_x)) and ((top_left_y) <= bp_y <= (bottom_right_y))):
            return 1
        else:
            return 0

    def __inside_circle(self, bp_x, bp_y, center_x, center_y, radius):
        """
        Private helper to calculate if body-part is inside a circle.
        """
        px_dist = int(np.sqrt((bp_x - center_x) ** 2 + (bp_y - center_y) ** 2))
        if px_dist <= radius:
            return 1
        else:
            return 0

    def __inside_polygon(self, bp_x, bp_y, polygon):
        """
        Private helper to calculate if body-part is inside a polygon.
        """
        if polygon.contains(Point(int(bp_x), int(bp_y))):
            return 1
        else:
            return 0

    def compute_agg_statistics(self, data: pd.DataFrame):
        """

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe with boolean columns representing behaviors (behavior present: 1, behavior absent: 0)
            and ROI data (inside ROI: 1, outside ROI: 0).

        """
        self.results_dict[self.video_name] = {}
        for clf in self.behavior_list:
            self.results_dict[self.video_name][clf] = {}
            for roi in self.found_rois:
                self.results_dict[self.video_name][clf][roi] = {}
                if 'Total time by ROI (s)' in self.measurements:
                    frame_cnt = len(data.loc[(data[clf] == 1) & (data[roi] == 1)])
                    if frame_cnt > 0:
                        self.results_dict[self.video_name][clf][roi]['Total time by ROI (s)'] = frame_cnt / self.fps
                    else:
                        self.results_dict[self.video_name][clf][roi]['Total time (s)'] = 0
                if 'Started bouts by ROI (count)' in self.measurements:
                    start_frames = list(detect_bouts(data_df=data, target_lst=[clf], fps=int(self.fps))['Start_frame'])
                    self.results_dict[self.video_name][clf][roi]['Started bouts by ROI (count)'] = len(data[(data.index.isin(start_frames)) & (data[roi] == 1)])
                if 'Ended bouts by ROI (count)' in self.measurements:
                    start_frames = list(detect_bouts(data_df=data, target_lst=[clf], fps=int(self.fps))['End_frame'])
                    self.results_dict[self.video_name][clf][roi]['Ended bouts by ROI (count)'] = len(data[(data.index.isin(start_frames)) & (data[roi] == 1)])

    def __print_missing_roi_warning(self,
                                  roi_type: str,
                                  roi_name: str):
        """
        Private helper to print warnings when ROI shapes have been defined in some videos but missing in others.
        """
        print('SIMBA WARNING: ROI named "{}" of shape type "{}" not found for video {}. Skipping shape...'.format(roi_name, roi_type, self.video_name))
        if roi_type.lower() == 'rectangle': names = list(self.rectangles_df['Name'][self.rectangles_df['Video'] == self.video_name])
        elif roi_type.lower() == 'circle': names = list(self.circles_df['Name'][self.circles_df['Video'] == self.video_name])
        elif roi_type.lower() == 'polygon': names = list(self.polygon_df['Name'][self.polygon_df['Video'] == self.video_name])
        print('SIMBA WARNING NOTE: Video {} has the following {} shape names: {}'.format(self.video_name, roi_type, names))

    def perform_ROI_clf_analysis(self,
                                 ROI_dict_lists: dict,
                                 measurements: list,
                                 behavior_list: list,
                                 body_part_list: list):
        """
        Parameters
        ----------
        ROI_dict_lists: dict
            A dictionary with the shape type as keys (i.e., Rectangle, Circle, Polygon) and lists of shape names
            as values.
        measurements: list
            Measurements to calculate aggregate statistics for. E.g., ['Total time by ROI (s)', 'Started bouts', 'Ended bouts']
        behavior_list: list
            Classifier names to calculate ROI statistics. E.g., ['Attack', 'Sniffing']
        body_part_list: list
            Body-part names to use to infer animal location. Eg., ['Nose_1'].
        """
        self.timer = SimbaTimer()
        self.timer.start_timer()
        machine_results_path = os.path.join(self.projectPath, 'csv', 'machine_results')
        self.ROI_dict_lists, self.behavior_list, self.measurements = ROI_dict_lists, behavior_list, measurements
        self.file_type = read_config_entry(config=self.config, section='General settings', option='workflow_file_type', data_type='str')
        files_found = glob.glob(machine_results_path + '/*.' + self.file_type)
        check_if_filepath_list_is_empty(filepaths=files_found,
                                        error_msg='SIMBA ERROR: No machine learning results found in the project_folder/csv/machine_results directory. Create machine classifications before analyzing classifications by ROI')
        if len(behavior_list) == 0:
            print('SIMBA ERROR: Please select at least one classifier')
            raise ValueError('SIMBA ERROR: Please select at least one classifier')
        print('Analyzing {} files...'.format(str(len(files_found))))
        body_part_col_names = []
        body_part_col_names_x, body_part_col_names_y = [], []
        for body_part in body_part_list:
            body_part_col_names.extend((body_part + '_x', body_part + '_y', body_part + '_p'))
            body_part_col_names_x.append(body_part + '_x')
            body_part_col_names_y.append(body_part + '_y')
        all_columns = body_part_col_names + behavior_list
        self.date_time = datetime.now().strftime('%Y%m%d%H%M%S')
        self.results_dict = {}

        self.frame_counter_dict = {}
        for file_cnt, file_path in enumerate(files_found):
            _, self.video_name, ext = get_fn_ext(file_path)
            print('Analyzing {}....'.format(self.video_name))
            data_df = read_df(file_path, self.file_type)
            for column in all_columns:
                check_that_column_exist(file_name=self.video_name, df=data_df, column_name=column)
            data_df = data_df[all_columns]
            self.results = data_df[behavior_list]
            shapes_in_video = len(self.rectangles_df.loc[(self.rectangles_df['Video'] == self.video_name)]) + len(self.circles_df.loc[(self.circles_df['Video'] == self.video_name)]) + len(self.polygon_df.loc[(self.polygon_df['Video'] == self.video_name)])
            if shapes_in_video == 0:
                print('WARNING: Skipping {}: Video {} has 0 user-defined ROI shapes.'.format(self.video_name, self.video_name))
                continue
            _, _, self.fps = read_video_info(self.video_info_df, self.video_name)
            self.found_rois = []
            for roi_type, roi_data in self.ROI_dict_lists.items():
                shape_info = pd.DataFrame()
                for roi_name in roi_data:
                    if roi_type.lower() == 'rectangle':
                        shape_info = self.rectangles_df.loc[(self.rectangles_df['Video'] == self.video_name) & (self.rectangles_df['Shape_type'] == roi_type) & (self.rectangles_df['Name'] == roi_name)]
                    elif roi_type.lower() == 'circle':
                        shape_info = self.circles_df.loc[(self.circles_df['Video'] == self.video_name) & (self.circles_df['Shape_type'] == roi_type) & (self.circles_df['Name'] == roi_name)]
                    elif roi_type.lower() == 'polygon':
                        shape_info = self.polygon_df.loc[(self.polygon_df['Video'] == self.video_name) & (self.polygon_df['Shape_type'] == roi_type) & (self.polygon_df['Name'] == roi_name)]
                    if len(shape_info) == 0:
                        self.__print_missing_roi_warning(roi_type=roi_type, roi_name=roi_name)
                        continue
                    if roi_type.lower() == 'rectangle':
                        data_df['top_left_x'], data_df['top_left_y'] = shape_info['topLeftX'].values[0], shape_info['topLeftY'].values[0]
                        data_df['bottom_right_x'], data_df['bottom_right_y'] = shape_info['Bottom_right_X'].values[0], shape_info['Bottom_right_Y'].values[0]
                        self.results[roi_name] = data_df.apply(lambda x: self.__inside_rectangle(bp_x=x[body_part_col_names_x[0]],
                                                                                            bp_y=x[body_part_col_names_y[0]],
                                                                                            top_left_x=x['top_left_x'],
                                                                                            top_left_y=x['top_left_y'],
                                                                                            bottom_right_x=x['bottom_right_x'],
                                                                                            bottom_right_y=x['bottom_right_y']), axis=1)
                        self.found_rois.append(roi_name)
                    elif (roi_type.lower() == 'circle'):
                        data_df['center_x'], data_df['center_y'], data_df['radius'] = shape_info['centerX'].values[0], shape_info['centerY'].values[0], shape_info['radius'].values[0]
                        self.results[roi_name] = data_df.apply(lambda x: self.__inside_circle(bp_x=x[body_part_col_names_x[0]], bp_y=x[body_part_col_names_y[0]], center_x=x['center_x'], center_y=x['center_y'], radius=x['radius']), axis=1)
                        self.found_rois.append(roi_name)
                    elif roi_type.lower() == 'polygon':
                        polygon_vertices = []
                        for i in shape_info['vertices'].values[0]:
                            polygon_vertices.append(geometry.Point(i))
                        polygon = Polygon([[p.x, p.y] for p in polygon_vertices])
                        self.results[roi_name] = data_df.apply(lambda x: self.__inside_polygon(bp_x=x[body_part_col_names_x[0]], bp_y=x[body_part_col_names_y[0]], polygon=polygon), axis=1)
                        self.found_rois.append(roi_name)
                self.compute_agg_statistics(data=self.results)
        self.__organize_output_data()

    def __organize_output_data(self):
        """
        Helper to organize the results[dict] into a human-readable CSV file.
        """

        if len(self.results_dict.keys()) == 0:
            print('SIMBA ERROR: ZERO ROIs found the videos represented in the project_folder/csv/machine_results directory')
            raise ValueError('SIMBA ERROR: ZERO ROIs found the videos represented in the project_folder/csv/machine_results directory')
        out_df = pd.DataFrame(columns=['VIDEO', 'CLASSIFIER', 'ROI', 'MEASUREMENT', 'VALUE'])
        for video_name, video_data in self.results_dict.items():
            for clf, clf_data in video_data.items():
                for roi_name, roi_data in clf_data.items():
                    for measurement_name, mesurement_value in roi_data.items():
                        out_df.loc[len(out_df)] = [video_name, clf, roi_name, measurement_name, mesurement_value]
        out_path = os.path.join(self.log_path_dir, 'Classification_time_by_ROI_{}.csv'.format(self.date_time))
        out_df.to_csv(out_path)
        self.timer.stop_timer()
        print('SIMBA COMPLETE: Classification data by ROIs saved in {} (selapsed time: {}s)'.format(out_path, self.timer.elapsed_time_str))

# clf_ROI_analyzer = clf_within_ROI(config_ini="/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini")
# clf_ROI_analyzer.perform_ROI_clf_analysis(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['rec'], 'Circle': ['Stimulus 1', 'Stimulus 2', 'Stimulus 3']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])