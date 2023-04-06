__author__ = "Simon Nilsson"

from simba.roi_tools.ROI_directing_analyzer import DirectingROIAnalyzer
import numpy as np
import os, glob
from simba.rw_dfs import read_df, save_df
from simba.feature_extractors.unit_tests import read_video_info
from simba.misc_tools import (check_directionality_viable,
                              find_animal_name_from_body_part_name)
from simba.drop_bp_cords import createColorListofList, create_body_part_dictionary, get_fn_ext
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
import itertools
from copy import deepcopy
from simba.mixins.config_reader import ConfigReader
from simba.utils.errors import NoFilesFoundError, NoROIDataError


class ROIFeatureCreator(ConfigReader):
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
                 config_path: str,
                 settings: dict or None=None):

        super().__init__(config_path=config_path)
        self.color_lst_of_lst = createColorListofList(self.animal_cnt, int(len(self.x_cols) + 1))
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_list,
                                                          self.animal_cnt, self.x_cols, self.y_cols, [],
                                                          self.color_lst_of_lst)
        self.roi_directing_viable = check_directionality_viable(self.animal_bp_dict)[0]
        self.settings = settings
        if self.roi_directing_viable:
            print('Directionality calculations are VIABLE.')
            self.directing_analyzer = DirectingROIAnalyzer(config_path=config_path, data_path=self.outlier_corrected_dir, settings=settings)
        else:
            self.directing_analyzer = None

        self.tracked_animal_bps = []
        for animal in range(self.animal_cnt):
            bp = self.config.get('ROI settings', 'animal_{}_bp'.format(str(animal + 1)))
            if len(bp) == 0:
                print(
                    'SIMBA ERROR: Please analyze ROI data for all animals before appending ROI features . No body-part setting found in config file [ROI settings][animal_{}_bp]'.format(
                        str(animal + 1)))
                raise ValueError
            else:
                self.tracked_animal_bps.append([bp + '_x', bp + '_y'])
        self.files_found = glob.glob(self.outlier_corrected_dir + '/*.' + self.file_type)
        if len(self.files_found) == 0:
            raise NoFilesFoundError(msg=f'SIMBA ERROR: No data files found in {self.outlier_corrected_dir}')
        self.features_files = glob.glob(self.features_dir + '/*.' + self.file_type)
        if len(self.features_files) == 0:
            print('SIMBA WARNING: No data files found in {}'.format(self.features_dir))
        print('Processing {} videos for ROI features...'.format(str(len(self.files_found))))

    def analyze_ROI_data(self):
        """
        Method to run the ROI feature analysis

        Returns
        -------
        Attribute: dict
            data
        """

        self.roi_analyzer = ROIAnalyzer(ini_path=self.config_path,
                                        data_path=self.outlier_corrected_dir,
                                        calculate_distances=True,
                                        settings=self.settings)
        self.roi_analyzer.files_found = self.files_found
        self.roi_analyzer.read_roi_dfs()
        self.all_shape_names = self.roi_analyzer.shape_names
        self.roi_analyzer.analyze_ROIs()
        self.roi_analyzer.compute_framewise_distance_to_roi_centroids()
        self.roi_distances_dict = self.roi_analyzer.roi_centroid_distance
        self.roi_entries_df = self.roi_analyzer.detailed_df
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
        if not self.settings:
            for animal_name, shape_name in itertools.product(self.multi_animal_id_list, self.all_shape_names):
                column_name = '{} {} {}'.format(shape_name, animal_name, 'in zone')
                self.inside_roi_columns.append(column_name)
                video_animal_shape_df = self.roi_entries_df.loc[(self.roi_entries_df['VIDEO'] == self.video_name) &
                                                                (self.roi_entries_df['SHAPE'] == shape_name) &
                                                                (self.roi_entries_df['ANIMAL'] == animal_name)]
                self.out_df[column_name] = 0
                if len(video_animal_shape_df) > 0:
                    inside_roi_idx = list(video_animal_shape_df.apply(lambda x: list(range(int(x['ENTRY FRAMES']), int(x['EXIT FRAMES']) + 1)), 1))
                    inside_roi_idx = [x for xs in inside_roi_idx for x in xs]
                    self.out_df.loc[inside_roi_idx, column_name] = 1
                self.out_df[column_name + '_cumulative_time'] = self.out_df[column_name].cumsum() * float(1 / self.fps)
                self.out_df[column_name + '_cumulative_percent'] = self.out_df[column_name].cumsum() / (self.out_df.index + 1)
                self.out_df.replace([np.inf, -np.inf], 1, inplace=True)
        else:
            for animal_cnt, shape_name in itertools.product(list(self.settings['body_parts'].keys()), self.all_shape_names):
                bp_name = self.settings['body_parts'][animal_cnt]
                animal_name = find_animal_name_from_body_part_name(bp_name=self.settings['body_parts'][animal_cnt], bp_dict=self.animal_bp_dict)
                column_name = '{} {} {} {}'.format(shape_name, animal_name, bp_name, 'zone')
                self.inside_roi_columns.append(column_name)
                video_body_part_shape_df = self.roi_entries_df.loc[(self.roi_entries_df['VIDEO'] == self.video_name) &
                                                                   (self.roi_entries_df['SHAPE'] == shape_name) &
                                                                   (self.roi_entries_df['ANIMAL'] == animal_name) &
                                                                   (self.roi_entries_df['BODY-PART'] == bp_name)]
                self.out_df[column_name] = 0
                if len(video_body_part_shape_df) > 0:
                    inside_roi_idx = list(video_body_part_shape_df.apply(lambda x: list(range(int(x['ENTRY FRAMES']), int(x['EXIT FRAMES']) + 1)), 1))
                    inside_roi_idx = [x for xs in inside_roi_idx for x in xs]
                    self.out_df.loc[inside_roi_idx, column_name] = 1
                self.out_df[column_name + '_cumulative_time'] = self.out_df[column_name].cumsum() * float(1 / self.fps)
                self.out_df[column_name + '_cumulative_percent'] = self.out_df[column_name].cumsum() / (self.out_df.index + 1)
                self.out_df.replace([np.inf, -np.inf], 1, inplace=True)

    def __distance_to_roi_centroids(self):
        self.roi_distance_columns = []
        video_distances = self.roi_distances_dict[self.video_name]
        if not self.settings:
            for animal_name, shape_name in itertools.product(self.multi_animal_id_list, self.all_shape_names):
                column_name = '{} {} {}'.format(shape_name, animal_name, 'distance')
                self.roi_distance_columns.append(column_name)
                try:
                    video_animal_shape_df = video_distances[animal_name][shape_name]
                except KeyError:
                    raise NoROIDataError(msg='The ROI named {} could not be find in video {}'.format(shape_name, self.video_name))
                self.out_df[column_name] = video_animal_shape_df
        else:
            for animal_cnt, shape_name in itertools.product(list(self.settings['body_parts'].keys()), self.all_shape_names):
                bp_name = self.settings['body_parts'][animal_cnt]
                animal_name = find_animal_name_from_body_part_name(bp_name=self.settings['body_parts'][animal_cnt], bp_dict=self.animal_bp_dict)
                column_name = '{} {} {} {}'.format(shape_name, animal_name, bp_name, 'distance')
                self.roi_distance_columns.append(column_name)
                try:
                    video_animal_shape_df = video_distances[animal_cnt][shape_name]
                except KeyError:
                    raise NoROIDataError(msg='The ROI named {} could not be find in video {}'.format(shape_name, self.video_name))
                self.out_df[column_name] = video_animal_shape_df

    def __process_directionality(self):
        self.roi_directing_columns = []
        video_directionality = self.roi_direction_df[self.roi_direction_df['Video'] == self.video_name]
        for animal_name, shape_name in itertools.product(self.multi_animal_id_list, self.all_shape_names):
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
            save_path = os.path.join(self.features_dir, video_name + '.' + self.file_type)
            save_df(video_data.fillna(0), self.file_type, save_path)
            print('Created additional ROI features for {}...'.format(video_name))
        self.timer.stop_timer()
        print('SIMBA COMPLETE: Created additional ROI features for files within the project_folder/csv/features_extracted directory (elapsed time: {}s)'.format(self.timer.elapsed_time_str))

# roi_featurizer = ROIFeatureCreator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
# roi_featurizer.analyze_ROI_data()
#roi_featurizer.save_new_features_files()



