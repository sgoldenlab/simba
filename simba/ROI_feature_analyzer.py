__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (check_int, check_str, check_float, read_config_entry, read_config_file)
from simba.ROI_directing_analyzer import DirectingROIAnalyzer
from datetime import datetime
import numpy as np
import os, glob
from simba.rw_dfs import read_df, save_df
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.misc_tools import check_multi_animal_status, check_directionality_viable
from simba.drop_bp_cords import getBpNames, createColorListofList, create_body_part_dictionary, get_fn_ext
from simba.ROI_analyzer import ROIAnalyzer
import pandas as pd
import itertools
from copy import deepcopy


class ROIFeatureCreator(object):
    """
    Class for computing features based on the relationships between the location of the animals and the location of
    user-defined ROIs.

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format

    Notes
    ----------
    `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> roi_featurizer = ROIFeatureCreator(config_path='MyProjectConfig')
    >>> roi_featurizer.analyze_ROI_data()
    >>> roi_featurizer.save_new_features_files()

    """

    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.config_path = config_path
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.out_dir = os.path.join(self.project_path, 'csv', 'features_extracted')
        self.video_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.color_lst_of_lst = createColorListofList(self.no_animals, int(len(self.x_cols) + 1))
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst,
                                                          self.no_animals, self.x_cols, self.y_cols, [],
                                                          self.color_lst_of_lst)
        self.roi_directing_viable = check_directionality_viable(self.animal_bp_dict)[0]
        if self.roi_directing_viable:
            print('Directionality calculations are VIABLE.')
            self.directing_analyzer = DirectingROIAnalyzer(config_path=config_path, data_path=self.in_dir)
        else:
            self.directing_analyzer = None

        self.tracked_animal_bps = []
        for animal in range(self.no_animals):
            bp = self.config.get('ROI settings', 'animal_{}_bp'.format(str(animal + 1)))
            if len(bp) == 0:
                print(
                    'SIMBA ERROR: Please analyze ROI data for all animals before appending ROI features . No body-part setting found in config file [ROI settings][animal_{}_bp]'.format(
                        str(animal + 1)))
                raise ValueError
            else:
                self.tracked_animal_bps.append([bp + '_x', bp + '_y'])
        self.files_found = glob.glob(self.in_dir + '/*.' + self.file_type)
        if len(self.files_found) == 0:
            print('SIMBA ERROR: No data files found in {}'.format(self.in_dir))
            raise ValueError('SIMBA ERROR: No data files found in {}'.format(self.in_dir))
        self.features_files = glob.glob(self.out_dir + '/*.' + self.file_type)
        if len(self.features_files) == 0:
            print('SIMBA ERROR: No data files found in {}'.format(self.out_dir))
            raise ValueError('SIMBA ERROR: No data files found in {}'.format(self.out_dir))
        print('Processing {} videos for ROI features...'.format(str(len(self.files_found))))

    def analyze_ROI_data(self):
        """
        Method to run the ROI feature analysis

        Returns
        -------
        Attribute: dict
            data
        """

        self.roi_analyzer = ROIAnalyzer(ini_path=self.config_path, data_path=self.in_dir, calculate_distances=True)
        self.roi_analyzer.files_found = self.files_found
        self.roi_analyzer.read_roi_dfs()
        self.all_shape_names = self.roi_analyzer.shape_names
        self.roi_analyzer.analyze_ROIs()
        self.roi_analyzer.compute_framewise_distance_to_roi_centroids()
        self.roi_distances_dict = self.roi_analyzer.roi_centroid_distance
        self.roi_entries_df = pd.concat(self.roi_analyzer.entry_exit_df_lst, axis=0)
        if self.roi_directing_viable:
            self.directing_analyzer.calc_directing_to_ROIs()
            self.roi_direction_df = self.directing_analyzer.results_df

        self.data = {}
        for file_cnt, file_path in enumerate(self.features_files):
            _, self.video_name, _ = get_fn_ext(file_path)
            _, _, self.fps = read_video_info(self.video_info_df, self.video_name)
            data_df = read_df(file_path, self.file_type)
            self.out_df = deepcopy(data_df)
            self.__process_within_rois()
            self.__distance_to_roi_centroids()
            if self.roi_directing_viable:
                self.__process_directionality()
            self.data[self.video_name] = self.out_df

    def __process_within_rois(self):
        self.inside_roi_columns = []
        for animal_name, shape_name in itertools.product(self.multi_animal_id_lst, self.all_shape_names):
            column_name = '{} {} {}'.format(shape_name, animal_name, 'in zone')
            self.inside_roi_columns.append(column_name)
            video_animal_shape_df = self.roi_entries_df.loc[(self.roi_entries_df['Video'] == self.video_name) &
                                                            (self.roi_entries_df['Shape'] == shape_name) &
                                                            (self.roi_entries_df['Animal'] == animal_name)]
            if len(video_animal_shape_df) > 0:
                inside_roi_idx = list(video_animal_shape_df.apply(lambda x: list(range(int(x['Entry_times']), int(x['Exit_times']) + 1)), 1))
                inside_roi_idx = [x for xs in inside_roi_idx for x in xs]
                self.out_df.loc[inside_roi_idx, column_name] = 1
            else:
                self.out_df[column_name] = 0
            self.out_df[column_name + '_cumulative_time'] = self.out_df[column_name].cumsum() * float(1 / self.fps)
            self.out_df[column_name + '_cumulative_percent'] = self.out_df[column_name].cumsum() / (self.out_df.index + 1)
            self.out_df.replace([np.inf, -np.inf], 1, inplace=True)

    def __distance_to_roi_centroids(self):
        self.roi_distance_columns = []
        video_distances = self.roi_distances_dict[self.video_name]
        for animal_name, shape_name in itertools.product(self.multi_animal_id_lst, self.all_shape_names):
            column_name = '{} {} {}'.format(shape_name, animal_name, 'distance')
            self.roi_distance_columns.append(column_name)
            video_animal_shape_df = video_distances[animal_name][shape_name]
            self.out_df[column_name] = video_animal_shape_df

    def __process_directionality(self):
        self.roi_directing_columns = []
        video_directionality = self.roi_direction_df[self.roi_direction_df['Video'] == self.video_name]
        for animal_name, shape_name in itertools.product(self.multi_animal_id_lst, self.all_shape_names):
            column_name = '{} {} {}'.format(shape_name, animal_name, 'facing')
            self.roi_directing_columns.append(column_name)
            video_animal_shape_df = video_directionality.loc[(video_directionality['ROI'] == shape_name) &
                                                             (video_directionality['Animal'] == animal_name)]
            if len(video_animal_shape_df) > 0:
                directing_idx = list(video_animal_shape_df['Frame'])
                self.out_df.loc[directing_idx, column_name] = 1
            else:
                self.out_df[column_name] = 0

    def save_new_features_files(self):
        """
        Method to save new featurized files inside the ``project_folder/csv/features_extracted`` directory
        of the SimBA project

        > Note: Method **overwrites** existing files in the project_folder/csv/features_extracted directory.

        Returns
        -------
        None

        """

        for video_name, video_data in self.data.items():
            save_path = os.path.join(self.out_dir, video_name + '.' + self.file_type)
            save_df(video_data.fillna(0), self.file_type, save_path)
            print('Created additional ROI features for {}...'.format(self.video_name))
        print(
            'SIMBA COMPLETE: Created additional ROI features for files within the project_folder/csv/features_extracted directory')





