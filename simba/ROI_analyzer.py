__author__ = "Simon Nilsson", "JJ Choong"


import glob
import itertools
from collections import defaultdict
import os
from datetime import datetime
import numpy as np
from shapely.geometry import Point, Polygon
from simba.drop_bp_cords import getBpHeaders, get_fn_ext
from simba.rw_dfs import *
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import check_multi_animal_status
from simba.read_config_unit_tests import read_config_entry, check_that_column_exist, read_config_file

class ROIAnalyzer(object):
    """

    Class for analyzing movements, entries, exits, and time-spent-in user-defined ROIs. Results are stored in the
    'project_folder/logs' directory of the SimBA project.

    Parameters
    ----------
    ini_path: str
        Path to SimBA project config file in Configparser format
    data_path: str
        Path to folder holding the data used to caluclate ROI aggregate statistics. E.g., `project_folder/
        csv/outlier_corrected_movement_location`
    calculate_distances: bool
        If True, calculate movements aggregate statistics (distances and velocities) inside ROIs


    Notes
    ----------
    `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.


    Examples
    ----------
    >>> roi_analyzer = ROIAnalyzer(ini_path='MyProjectConfig', data_path='outlier_corrected_movement_location', calculate_distances=True)
    >>> roi_analyzer.read_roi_dfs()
    >>> roi_analyzer.analyze_ROIs()
    >>> roi_analyzer.save_data()
    """

    def __init__(self,
                 ini_path: str,
                 data_path: str,
                 calculate_distances: bool=False):

        self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        config = read_config_file(ini_path)
        self.no_animals = read_config_entry(config, 'General settings', 'animal_no', 'int')
        self.project_path = read_config_entry(config, 'General settings', 'project_path', data_type='folder_path')
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.detailed_ROI_data_path = os.path.join(self.project_path, 'logs', 'Detailed_ROI_data')
        self.vid_info_df = read_video_info_csv(os.path.join(self.logs_path, 'video_info.csv'))
        self.file_type = read_config_entry(config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.calculate_distances = calculate_distances
        self.bp_headers = getBpHeaders(ini_path)
        if not os.path.exists(self.detailed_ROI_data_path): os.makedirs(self.detailed_ROI_data_path)
        self.p_thresh = read_config_entry(config, 'ROI settings', 'probability_threshold', 'float', 0.00)
        if data_path != None:
            self.input_folder = os.path.join(self.project_path, 'csv', data_path)
            self.files_found = glob.glob(self.input_folder + '/*.' + self.file_type)
            if len(self.files_found) == 0:
                print('SIMBA ERROR: No data files found in {}'.format(self.input_folder))
                raise ValueError('SIMBA ERROR: No data files found in {}'.format(self.input_folder))

        self.multiAnimalStatus, self.multiAnimalIDList = check_multi_animal_status(config, self.no_animals)

        self.bp_dict = defaultdict(list)
        self.bp_columns = []
        for cnt, animal in enumerate(self.multiAnimalIDList):
            bp_name = config.get('ROI settings', 'animal_' + str(cnt + 1) + '_bp')
            if bp_name == 'None':
                print('SIMBA ERROR: Please analyze ROI data and set body-parts first.')
                raise ValueError
            for c in ['_x', '_y', '_p']:
                self.bp_dict[animal].append(bp_name + c)
                self.bp_columns.append(bp_name + c)

    def read_roi_dfs(self):

        """
        Method to read in ROI definitions from SimBA project

        Returns
        -------
        Attribute: pd.DataFrame
            rectangles_df
        Attribute: pd.DataFrame
            circles_df
        Attribute: pd.DataFrame
            polygon_df
        Attribute: list
            shape_names
        """


        if not os.path.isfile(os.path.join(self.logs_path, 'measures', 'ROI_definitions.h5')):
            raise FileNotFoundError(print('No ROI definitions were found in your SimBA project. Please draw some ROIs before analyzing your ROI data'))
        else:
            self.roi_h5_path = os.path.join(self.logs_path, 'measures', 'ROI_definitions.h5')
            self.rectangles_df = pd.read_hdf(self.roi_h5_path, key='rectangles')
            self.circles_df = pd.read_hdf(self.roi_h5_path, key='circleDf')
            self.polygon_df = pd.read_hdf(self.roi_h5_path, key='polygons')
            self.shape_names = list(itertools.chain(self.rectangles_df['Name'].unique(), self.circles_df['Name'].unique(), self.polygon_df['Name'].unique()))
            self.__reset_dicts()

    def __get_bouts(self, lst=None):
        lst=list(lst)
        return lst[0], lst[-1]

    def __reset_dicts(self):
        self.time_dict, self.entries_dict, self.entries_exit_dict, self.movement_dict = {}, {}, {}, {}
        for animal in self.multiAnimalIDList:
            self.entries_exit_dict[animal] = {}
            self.movement_dict[animal] = {}
            for shape in self.shape_names:
                self.time_dict['{} {} (s)'.format(animal, shape)] = 0
                self.entries_dict['{} {} (entries)'.format(animal, shape)] = 0
                self.entries_exit_dict[animal][shape] = {'Entry_times': [], 'Exit_times': []}
                self.movement_dict[animal][shape] = 0

    def analyze_ROIs(self):
        """
        Method to analyze ROI statistics.

        Returns
        -------
        Attribute: list
            dist_lst, list of pd.DataFrame holding ROI-dependent movement statistics.
        """

        self.time_df_lst, self.entry_df_lst, self.entry_exit_df_lst, self.dist_lst = [], [], [], []
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(file_path)
            print('Analysing {}...'.format(video_name))
            self.video_recs = self.rectangles_df.loc[self.rectangles_df['Video'] == video_name]
            self.video_circs = self.circles_df.loc[self.circles_df['Video'] == video_name]
            self.video_polys = self.polygon_df.loc[self.polygon_df['Video'] == video_name]
            video_shapes = list(itertools.chain(self.video_recs['Name'].unique(), self.video_circs['Name'].unique(), self.video_polys['Name'].unique()))

            if video_shapes == 0:
                print('Skipping video {}: No user-defined ROI data found for video...'.format(video_name))
                continue

            else:
                video_settings, pix_per_mm, self.fps = read_video_info(self.vid_info_df, video_name)
                self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
                self.data_df.columns = self.bp_headers
                data_df_sliced = self.data_df[self.bp_columns]
                video_length_s = data_df_sliced.shape[0] / self.fps
                proportion_time_dict = {}
                for animal_name in self.multiAnimalIDList:
                    animal_df = self.data_df[self.bp_dict[animal_name]]
                    for _, row in self.video_recs.iterrows():
                        top_left_x, top_left_y, shape_name = row['topLeftX'], row['topLeftY'], row['Name']
                        bottom_right_x, bottom_right_y = row['Bottom_right_X'], row['Bottom_right_Y']
                        slice_x = animal_df[animal_df[self.bp_dict[animal_name][0]].between(top_left_x, bottom_right_x, inclusive=True)]
                        slice_y = slice_x[slice_x[self.bp_dict[animal_name][1]].between(top_left_y, bottom_right_y, inclusive=True)]
                        slice = slice_y[slice_y[self.bp_dict[animal_name][2]] >= self.p_thresh].reset_index().rename(columns={'index': 'frame_no'})
                        bouts = [self.__get_bouts(g)for _, g in itertools.groupby(list(slice['frame_no']), key=lambda n, c=itertools.count(): n - next(c))]
                        self.time_dict['{} {} (s)'.format(animal_name, shape_name)] = round(len(slice) / self.fps, 3)
                        self.entries_dict['{} {} (entries)'.format(animal_name, shape_name)] = len(bouts)
                        self.entries_exit_dict[animal_name][shape_name]['Entry_times'] = list(map(lambda x: x[0], bouts))
                        self.entries_exit_dict[animal_name][shape_name]['Exit_times'] = list(map(lambda x: x[1], bouts))

                    for _, row in self.video_circs.iterrows():
                        center_x, center_y, radius, shape_name = row['centerX'], row['centerY'], row['radius'], row['Name']
                        animal_df['distance'] = np.sqrt((animal_df[self.bp_dict[animal_name][0]] - center_x) ** 2 + (animal_df[self.bp_dict[animal_name][1]] - center_y) ** 2)
                        slice = animal_df.loc[(animal_df['distance'] <= radius) & (animal_df[self.bp_dict[animal_name][2]] >= self.p_thresh)].reset_index().rename(columns={'index': 'frame_no'})
                        bouts = [self.__get_bouts(g) for _, g in itertools.groupby(list(slice['frame_no']), key=lambda n, c=itertools.count(): n - next(c))]
                        self.time_dict['{} {} (s)'.format(animal_name, shape_name)] = round(len(slice) / self.fps, 3)
                        self.entries_dict['{} {} (entries)'.format(animal_name, shape_name)] = len(bouts)
                        self.entries_exit_dict[animal_name][shape_name]['Entry_times'] = list(map(lambda x: x[0], bouts))
                        self.entries_exit_dict[animal_name][shape_name]['Exit_times'] = list(map(lambda x: x[1], bouts))

                    for _, row in self.video_polys.iterrows():
                        polygon_shape, shape_name = Polygon(list(zip(row['vertices'][:,0],row['vertices'][ :,1]))), row['Name']
                        points_arr = animal_df[[self.bp_dict[animal_name][0], self.bp_dict[animal_name][1]]].to_numpy()
                        contains_func = np.vectorize(lambda p: polygon_shape.contains(Point(p)), signature='(n)->()')
                        inside_frame_no = [j for sub in np.argwhere(contains_func(points_arr)) for j in sub]
                        slice = animal_df.loc[(animal_df.index.isin(inside_frame_no)) & (animal_df[self.bp_dict[animal_name][2]] >= self.p_thresh)].reset_index().rename(columns={'index': 'frame_no'})
                        bouts = [self.__get_bouts(g) for _, g in itertools.groupby(list(slice['frame_no']), key=lambda n, c=itertools.count(): n - next(c))]
                        self.time_dict['{} {} (s)'.format(animal_name, shape_name)] = round(len(slice) / self.fps, 3)
                        self.entries_dict['{} {} (entries)'.format(animal_name, shape_name)] = len(bouts)
                        self.entries_exit_dict[animal_name][shape_name]['Entry_times'] = list(map(lambda x: x[0], bouts))
                        self.entries_exit_dict[animal_name][shape_name]['Exit_times'] = list(map(lambda x: x[1], bouts))

                for k, v in self.time_dict.items():
                    proportion_time_dict[' '.join(k.split(' ')[:-1]) + ' (% of session)'] = round(v / video_length_s, 3)

                self.time_dict = {**self.time_dict, **proportion_time_dict}
                time_df = pd.DataFrame.from_dict(self.time_dict, orient='index').T
                entries_df = pd.DataFrame.from_dict(self.entries_dict, orient='index').T
                time_df.insert(loc=0, column='Video', value=video_name)
                self.time_df_lst.append(time_df)
                entries_df.insert(loc=0, column='Video', value=video_name)
                self.entry_df_lst.append(entries_df)
                for animal, shape_dicts in self.entries_exit_dict.items():
                    for shape_name, shape_data in shape_dicts.items():
                        d = pd.DataFrame.from_dict(shape_data, orient='index').T
                        d.insert(loc=0, column='Shape', value=shape_name)
                        d.insert(loc=0, column='Animal', value=animal)
                        d.insert(loc=0, column='Video', value=video_name)
                        self.entry_exit_df_lst.append(d)

                if self.calculate_distances:
                    for animal, shape_dicts in self.entries_exit_dict.items():
                        for shape_name, shape_data in shape_dicts.items():
                            d = pd.DataFrame.from_dict(shape_data, orient='index').T.values.tolist()
                            for entry in d:
                                df = self.data_df[self.bp_dict[animal][0:2]][self.data_df.index.isin(list(range(entry[0], entry[1]+1)))]
                                df_shifted = df.shift(1)
                                df_shifted = df_shifted.combine_first(df).add_prefix('shifted_')
                                df = pd.concat([df, df_shifted], axis=1)
                                df['Movement'] = (np.sqrt((df.iloc[:, 0] - df.iloc[:, 2]) ** 2 + (df.iloc[:, 1] - df.iloc[:,3]) ** 2)) / pix_per_mm
                                self.movement_dict[animal][shape_name] = self.movement_dict[animal][shape_name] + df['Movement'].sum() / 10
                    movement_df = pd.melt(pd.DataFrame.from_dict(self.movement_dict, orient='columns').reset_index().rename(columns={'index': 'Shape'}), id_vars=['Shape'])
                    movement_df.rename(columns={'variable': 'Animal', 'value': 'Movement inside shape (cm))'}, inplace=True)
                    movement_df.insert(loc=0, column='Video', value=video_name)
                    self.dist_lst.append(movement_df)

                self.__reset_dicts()

    def compute_framewise_distance_to_roi_centroids(self):
        """
        Method to compute frame-wise distances between ROI centroids and animal body-parts.

        Returns
        -------
        Attribute: dict
            roi_centroid_distance
        """

        self.roi_centroid_distance = {}
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(file_path)
            self.roi_centroid_distance[video_name] = {}
            video_recs = self.rectangles_df.loc[self.rectangles_df['Video'] == video_name]
            video_circs = self.circles_df.loc[self.circles_df['Video'] == video_name]
            video_polys = self.polygon_df.loc[self.polygon_df['Video'] == video_name]
            data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            data_df.columns = self.bp_headers
            for animal_name in self.multiAnimalIDList:
                self.roi_centroid_distance[video_name][animal_name] = {}
                animal_df = data_df[self.bp_dict[animal_name]]
                for _, row in video_recs.iterrows():
                    center_cord = ((int(row['Bottom_right_Y'] - ((row['Bottom_right_Y'] - row['topLeftY']) / 2))),
                                   (int(row['Bottom_right_X'] - ((row['Bottom_right_X'] - row['topLeftX']) / 2))))
                    self.roi_centroid_distance[video_name][animal_name][row['Name']] = np.sqrt((animal_df[self.bp_dict[animal_name][0]] - center_cord[0]) ** 2 + (animal_df[self.bp_dict[animal_name][1]] - center_cord[1]) ** 2)

                for _, row in video_circs.iterrows():
                    center_cord = (row['centerX'], row['centerY'])
                    self.roi_centroid_distance[video_name][animal_name][row['Name']] = np.sqrt((animal_df[self.bp_dict[animal_name][0]] - center_cord[0]) ** 2 + (animal_df[self.bp_dict[animal_name][1]] - center_cord[1]) ** 2)

                for _, row in video_polys.iterrows():
                    polygon_shape = Polygon(list(zip(row['vertices'][:, 0], row['vertices'][:, 1])))
                    center_cord = polygon_shape.centroid.coords[0]
                    self.roi_centroid_distance[video_name][animal_name][row['Name']] = np.sqrt((animal_df[self.bp_dict[animal_name][0]] - center_cord[0]) ** 2 + (animal_df[self.bp_dict[animal_name][1]] - center_cord[1]) ** 2)

    def save_data(self):
        """
        Method to save ROI data to disk. ROI latency and ROI entry data is saved in the "project_folder/logs/" directory.
        If ``calculate_distances`` is True, ROI movement data is saved in the "project_folder/logs/" directory.

        Returns
        -------
        None
        """



        for data_name, data in zip(['Detailed_ROI_data', 'ROI_time_data', 'ROI_entry_data'], [self.entry_exit_df_lst, self.time_df_lst, self.entry_df_lst]):
            save_df = pd.concat(data, axis=0).reset_index(drop=True)
            save_path = os.path.join(self.logs_path, data_name + '_' + self.timestamp + '.csv')
            save_df.to_csv(save_path)
        print('SIMBA COMPLETE: ROI time, ROI entry, and Detailed ROI data, have been saved in the "project_folder/logs/" directory in CSV format.')

        if self.calculate_distances:
            save_df = pd.concat(self.dist_lst, axis=0).reset_index(drop=True)
            save_path = os.path.join(self.logs_path, 'ROI_movement_data_' + self.timestamp + '.csv')
            save_df.to_csv(save_path)
        print('ROI movement data saved in the "project_folder/logs/" directory')
        print('SIMBA COMPLETE: ROI analysis complete.')

# test = ROIAnalyzer(ini_path = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\ROI_2_animals\project_folder\project_config.ini",
#                    data_path = "outlier_corrected_movement_location", calculate_distances=True)
# test.read_roi_dfs()
# test.analyze_ROIs()
# test.save_data()
