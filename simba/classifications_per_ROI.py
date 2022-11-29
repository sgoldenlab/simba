__author__ = "Simon Nilsson", "JJ Choong"

import glob, os

import pandas as pd
from simba.rw_dfs import read_df
import numpy as np
from shapely import geometry
from shapely.geometry import Point, Polygon
from simba.features_scripts.unit_tests import *
from datetime import datetime
from simba.misc_tools import get_fn_ext, detect_bouts
from simba.read_config_unit_tests import read_config_file, read_config_entry, check_that_column_exist
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info


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
        self.projectPath = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        model_nos = read_config_entry(config=self.config, section='SML settings', option='No_targets', data_type='int')
        self.log_path_dir = os.path.join(self.projectPath, 'logs')
        self.video_info_df = read_video_info_csv(os.path.join(self.log_path_dir, 'video_info.csv'))
        self.body_parts_path = os.path.join(os.path.join(self.log_path_dir, 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv'))
        body_parts_df = pd.read_csv(self.body_parts_path, names=['bodyparts'])
        self.body_part_list = list(body_parts_df['bodyparts'])
        self.ROIcoordinatesPath = os.path.join(self.log_path_dir, 'measures', 'ROI_definitions.h5')
        if not os.path.isfile(self.ROIcoordinatesPath):
            print('No ROI coordinates found. Please use the [ROI] tab to define ROIs')
            raise FileNotFoundError()
        self.rectangles_df = pd.read_hdf(self.ROIcoordinatesPath, key='rectangles')
        self.circles_df = pd.read_hdf(self.ROIcoordinatesPath, key='circleDf')
        self.polygon_df = pd.read_hdf(self.ROIcoordinatesPath, key='polygons')

        self.ROI_str_name_list = []
        for index, row in self.rectangles_df.iterrows(): self.ROI_str_name_list.append(row['Shape_type'] + ': ' + row['Name'])

        for index, row in self.circles_df.iterrows(): self.ROI_str_name_list.append(row['Shape_type'] + ': ' + row['Name'])
        for index, row in self.polygon_df.iterrows(): self.ROI_str_name_list.append(row['Shape_type'] + ': ' + row['Name'])
        self.ROI_str_name_list = list(set(self.ROI_str_name_list))

        self.behavior_names = []
        for i in range(model_nos):
            self.behavior_names.append(self.config.get('SML settings', 'target_name_' + str(i + 1)))

    def __inside_rectangle(self, bp_x, bp_y, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        if (((top_left_x) <= bp_x <= (bottom_right_x)) and ((top_left_y) <= bp_y <= (bottom_right_y))):
            return 1
        else:
            return 0

    def __inside_circle(self, bp_x, bp_y, center_x, center_y, radius):
        px_dist = int(np.sqrt((bp_x - center_x) ** 2 + (bp_y - center_y) ** 2))
        if px_dist <= radius:
            return 1
        else:
            return 0

    def __inside_polygon(self, bp_x, bp_y, polygon):
        if polygon.contains(Point(int(bp_x), int(bp_y))):
            return 1
        else:
            return 0

    def compute_agg_statistics(self, data: pd.DataFrame):
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

    def print_missing_roi_warning(self,
                                  roi_type: str,
                                  roi_name: str):
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
            List of classifier names to calculate ROI statistics. E.g., ['Attack', 'Sniffing']
        body_part_list: list
            List of body-part names to use to infer animal location. Eg., ['Nose_1'].
        """

        machine_results_path = os.path.join(self.projectPath, 'csv', 'machine_results')
        self.ROI_dict_lists, self.behavior_list, self.measurements = ROI_dict_lists, behavior_list, measurements
        self.file_type = read_config_entry(config=self.config, section='General settings', option='workflow_file_type', data_type='str')
        files_found = glob.glob(machine_results_path + '/*.' + self.file_type)
        if len(files_found) == 0:
            print('SIMBA ERROR: No machine learning results found in the project_folder/csv/machine_results directory. Create machine classifications before analyzing classifications by ROI')
            raise ValueError()
        if len(behavior_list) == 0:
            print('SIMBA ERROR: Please select at least one classifier')
            raise ValueError()
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
                        self.print_missing_roi_warning(roi_type=roi_type, roi_name=roi_name)
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
        print('SIMBA COMPLETE: Classification data by ROIs saved in {}'.format(out_path))

# clf_ROI_analyzer = clf_within_ROI(config_ini="/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini")
# clf_ROI_analyzer.perform_ROI_clf_analysis(behavior_list=['Attack', 'Sniffing'], ROI_dict_lists={'Rectangle': ['rec'], 'Circle': ['Stimulus 1', 'Stimulus 2', 'Stimulus 3']}, body_part_list=['Nose_1'], measurements=['Total time by ROI (s)', 'Started bouts by ROI (count)', 'Ended bouts by ROI (count)'])

        #  =
                        #
                        # #                             current_behavior_df['topLeftY'] = topLeftY
                        #
                        #
                        #
    #                         if
    #                             pass
    #                         else:
    #                             topLeftX, topLeftY = current_rectangle_info['topLeftX'].values[0], current_rectangle_info['topLeftY'].values[0]
    #                             bottomRightX, bottomRightY = topLeftX + current_rectangle_info['width'].values[0], topLeftY + + current_rectangle_info['height'].values[0]
    #                             current_behavior_df['topLeftX'] = topLeftX
    #                             current_behavior_df['topLeftY'] = topLeftY
    #                             current_behavior_df['bottomRightX'] = bottomRightX
    #                             current_behavior_df['bottomRightY'] = bottomRightY
    #
    #
    #                         current_circle_info = self.circleInfo.loc[(self.circleInfo['Video'] == self.video_name) & ( self.circleInfo['Shape_type'] == current_type) & (self.circleInfo['Name'] == self.current_name)]
    #                         if len(current_circle_info) == 0:
    #                             pass
    #                         else:
    #                             centre_x, centre_y = current_circle_info['centerX'].values[0], current_circle_info['centerY'].values[0]
    #                             radius = current_circle_info['radius'].values[0]
    #                             current_behavior_df['centerX'] = centre_x
    #                             current_behavior_df['centerY'] = centre_y
    #                             current_behavior_df['radius'] = radius
    #                             current_behavior_df.apply(lambda row: self.__inside_circle(row[body_part_col_names_x[0]],
    #                                                                                         row[body_part_col_names_y[0]],
    #                                                                                         row['centerX'],
    #                                                                                         row['centerY'],
    #                                                                                         row['radius']), axis=1)
    #                     if current_type.lower() == 'polygon':
    #                         current_polygon_info = self.polygonInfo.loc[(self.polygonInfo['Video'] == self.video_name) & (self.polygonInfo['Shape_type'] == current_type) & (self.polygonInfo['Name'] == self.current_name)]
    #                         if len(current_polygon_info) == 0:
    #                             pass
    #                         else:
    #                             vertices = current_polygon_info['vertices'].values[0]
    #                             polygon_shape = []
    #                             for i in vertices:
    #                                 polygon_shape.append(geometry.Point(i))
    #                             current_behavior_df.apply(lambda row: self.__inside_polygon(row[body_part_col_names_x[0]], row[body_part_col_names_y[0]], polygon_shape), axis=1)
    #
    #     self.__organize_output_data(self.frame_counter_dict)
    #
    # def __inside_rectangle(self, bp_x, bp_y, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    #     self.__populate_dict()
    #     if (((top_left_x - 10) <= bp_x <= (bottom_right_x + 10)) and ((top_left_y - 10) <= bp_y <= (bottom_right_y + 10))):
    #         self.__add_to_counter()
    #     else:
    #         pass
    #
    # def __inside_circle(self, bp_x, bp_y, center_x, center_y, radius):
    #     self.__populate_dict()
    #     px_dist = int(np.sqrt((bp_x - center_x) ** 2 + (bp_y - center_y) ** 2))
    #     if px_dist <= radius:
    #         self.__add_to_counter()
    #     else:
    #         pass
    #
    # def __inside_polygon(self, bp_x, bp_y, vertices):
    #     self.__populate_dict()
    #     current_point = Point(int(bp_x), int(bp_y))
    #     current_polygon = geometry.Polygon([[p.x, p.y] for p in vertices])
    #     polygon_status = current_polygon.contains(current_point)
    #     if polygon_status:
    #         self.__add_to_counter()
    #     else:
    #         pass
    #
    # def __populate_dict(self):
    #     if not self.video_name in self.frame_counter_dict:
    #         self.frame_counter_dict[self.video_name] = {}
    #     if not self.current_name in self.frame_counter_dict[self.video_name]:
    #         self.frame_counter_dict[self.video_name][self.current_name] = {}
    #     if not self.current_behavior in self.frame_counter_dict[self.video_name][self.current_name]:
    #         self.frame_counter_dict[self.video_name][self.current_name][self.current_behavior] = 0
    #
    # def __add_to_counter(self):
    #     self.frame_counter_dict[self.video_name][self.current_name][self.current_behavior] += 1 / self.fps
    #
    # def __organize_output_data(self, frame_counter_dict):
    #     if len(frame_counter_dict.keys()) == 0:
    #         print('SIMBA ERROR: ZERO ROIs found the videos represented in the project_folder/csv/machine_results directory')
    #         raise ValueError('SIMBA ERROR: ZERO ROIs found the videos represented in the project_folder/csv/machine_results directory')
    #     out_df = pd.concat({k: pd.DataFrame(v) for k, v in frame_counter_dict.items()}, axis=1).T.reset_index()
    #     out_df = out_df.rename(columns={'level_0': 'Video', 'level_1': 'ROI'})
    #     out_df = out_df.fillna(0)
    #     time_columns = list(out_df.columns)
    #     time_columns.remove('Video')
    #     time_columns.remove('ROI')
    #     for column in time_columns:
    #         out_df = out_df.round({column: 3})
    #         out_df = out_df.rename(columns={column: str(column) + ' (s)'})
    #     dateTime = datetime.now().strftime('%Y%m%d%H%M%S')
    #     out_path = os.path.join(self.logFolderPath, 'Classification_time_by_ROI_' + str(dateTime) + '.csv')
    #     out_df.to_csv(out_path)
    #     print('All videos analysed.')
    #     print('Data saved @ ' + out_path)
    #

