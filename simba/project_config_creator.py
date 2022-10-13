__author__ = "Simon Nilsson", "JJ Choong"

import os
import re
import platform
from configparser import ConfigParser
import simba
import csv
import pandas as pd


class ProjectConfigCreator(object):

    """
    Class for creating SimBA project directory tree and project_config.ini

    Parameters
    ----------
    project_path: str
        Path to directory where to save the SimBA project directory tree
    project_name: str
        Name of the SimBA project
    target_list: list
        List of classifiers in the SimBA project
    pose_estimation_bp_cnt: str
        String representing the number of body-parts in the pose-estimation data used in the simba project. E.g.,
        '16' or 'user-defined.
    body_part_config_idx: int
        The index of the SimBA GUI dropdown pose-estimation selection. E.g., ``1``.
    animal_cnt: int
        Number of animals tracked in the input pose-estimation data.
    file_type: str
        The SimBA project file type. OPTIONS: ``csv`` or ``parquet``.

    Notes
    ----------
    `Create project tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#part-1-create-a-new-project-1>`__.

    Examples
    ----------
    >>> _ = ProjectConfigCreator(project_path = 'project/path', project_name='project_name', target_list=['Attack'], pose_estimation_bp_cnt='16', body_part_config_idx=9, animal_cnt=2, file_type='csv')

    """

    def __init__(self,
                 project_path: str,
                 project_name: str,
                 target_list: list,
                 pose_estimation_bp_cnt: str,
                 body_part_config_idx: int,
                 animal_cnt: int,
                 file_type: str = 'csv'):

        self.simba_dir = os.path.dirname(simba.__file__)
        self.animal_cnt = animal_cnt
        self.os_platform = platform.system()
        self.project_path = project_path
        self.project_name = project_name
        self.target_list = target_list
        self.pose_estimation_bp_cnt = pose_estimation_bp_cnt
        self.body_part_config_idx = body_part_config_idx
        self.file_type = file_type
        self.__create_directories()
        self.__create_configparser_config()

    def __create_directories(self):
        self.project_folder = os.path.join(self.project_path, self.project_name, 'project_folder')
        self.models_folder = os.path.join(self.project_path, self.project_name, 'models')
        self.config_folder = os.path.join(self.project_folder, 'configs')
        self.csv_folder = os.path.join(self.project_folder, 'csv')
        self.frames_folder = os.path.join(self.project_folder, 'frames')
        self.logs_folder = os.path.join(self.project_folder, 'logs')
        self.measures_folder = os.path.join(self.logs_folder, 'measures')
        self.pose_configs_folder = os.path.join(self.measures_folder, 'pose_configs')
        self.bp_names_folder = os.path.join(self.pose_configs_folder, 'bp_names')
        self.videos_folder = os.path.join(self.project_folder, 'videos')
        self.features_extracted_folder = os.path.join(self.csv_folder, 'features_extracted')
        self.input_csv_folder = os.path.join(self.csv_folder, 'input_csv')
        self.machine_results_folder = os.path.join(self.csv_folder, 'machine_results')
        self.outlier_corrected_movement_folder = os.path.join(self.csv_folder, 'outlier_corrected_movement')
        self.outlier_corrected_location_folder = os.path.join(self.csv_folder, 'outlier_corrected_movement_location')
        self.targets_inserted_folder = os.path.join(self.csv_folder, 'targets_inserted')
        self.input_folder = os.path.join(self.frames_folder, 'input')
        self.output_folder = os.path.join(self.frames_folder, 'output')

        folder_lst = [self.project_folder, self.models_folder, self.config_folder, self.csv_folder, self.frames_folder, self.logs_folder,
                       self.videos_folder, self.features_extracted_folder, self.input_csv_folder, self.machine_results_folder,
                       self.outlier_corrected_movement_folder, self.outlier_corrected_location_folder, self.targets_inserted_folder,
                       self.input_folder,self.output_folder, self.measures_folder, self.pose_configs_folder, self.bp_names_folder]

        for folder_path in folder_lst:
            try:
                os.makedirs(folder_path)
            except FileExistsError:
               print('SIMBA ERROR: SimBA tried to create {}, but it already exists. Please create your SimBA project ' 'in a new path, or move/delete your previous SimBA project')
               raise FileExistsError

    def __create_configparser_config(self):
        self.config = ConfigParser(allow_no_value=True)
        self.config.add_section('General settings')
        self.config['General settings']['project_path'] = self.project_folder
        self.config['General settings']['project_name'] = self.project_name
        self.config['General settings']['workflow_file_type'] = self.file_type
        self.config['General settings']['animal_no'] = str(self.animal_cnt)
        self.config['General settings']['OS_system'] = self.os_platform

        self.config.add_section('SML settings')
        self.config['SML settings']['model_dir'] = self.models_folder
        for clf_cnt in range(len(self.target_list)):
            self.config['SML settings']['model_path_{}'.format(str(clf_cnt+1))] = os.path.join(self.models_folder, str(self.target_list[clf_cnt]) + '.sav')

        self.config['SML settings']['No_targets'] = str(len(self.target_list))
        for clf_cnt in range(len(self.target_list)):
            self.config['SML settings']['target_name_{}'.format(str(clf_cnt+1))] = str(self.target_list[clf_cnt])

        self.config.add_section('threshold_settings')
        for clf_cnt in range(len(self.target_list)):
            self.config['threshold_settings']['threshold_{}'.format(str(clf_cnt + 1))] = 'None'

        self.config.add_section('Minimum_bout_lengths')
        for clf_cnt in range(len(self.target_list)):
            self.config['Minimum_bout_lengths']['min_bout_{}'.format(str(clf_cnt+1))] = 'None'

        # frame settings
        self.config.add_section('Frame settings')
        self.config['Frame settings']['mm_per_pixel'] = 'None'
        self.config['Frame settings']['distance_mm'] = 'None'

        self.config.add_section('Line plot settings')
        self.config['Line plot settings']['Bodyparts'] = 'None'

        self.config.add_section('Path plot settings')
        self.config['Path plot settings']['Deque_points'] = 'None'
        self.config['Path plot settings']['Behaviour_points'] = 'None'
        self.config['Path plot settings']['plot_severity'] = 'None'
        self.config['Path plot settings']['severity_brackets'] = 'None'

        # distance plot
        self.config.add_section('Distance plot')
        self.config['Distance plot']['POI_1'] = 'None'
        self.config['Distance plot']['POI_2'] = 'None'

        self.config.add_section('Heatmap settings')
        self.config['Heatmap settings']['bin_size_pixels'] = 'None'
        self.config['Heatmap settings']['Scale_max_seconds'] = 'None'
        self.config['Heatmap settings']['Scale_increments_seconds'] = 'None'
        self.config['Heatmap settings']['Palette'] = 'None'
        self.config['Heatmap settings']['target_behaviour'] = 'None'
        self.config['Heatmap settings']['body_part'] = 'None'

        self.config.add_section('ROI settings')
        self.config['ROI settings']['animal_1_bp'] = 'None'
        self.config['ROI settings']['animal_2_bp'] = 'None'

        self.config.add_section('process movements')
        self.config['process movements']['animal_1_bp'] = 'None'
        self.config['process movements']['animal_2_bp'] = 'None'
        self.config['process movements']['no_of_animals'] = 'None'

        self.config.add_section('create ensemble settings')
        self.config['create ensemble settings']['pose_estimation_body_parts'] = str(self.pose_estimation_bp_cnt)
        self.config['create ensemble settings']['classifier'] = 'None'
        self.config['create ensemble settings']['train_test_size'] = str(0.20)
        self.config['create ensemble settings']['under_sample_setting'] = 'None'
        self.config['create ensemble settings']['under_sample_ratio'] = 'None'
        self.config['create ensemble settings']['over_sample_setting'] = 'None'
        self.config['create ensemble settings']['over_sample_ratio'] = 'None'
        self.config['create ensemble settings']['RF_n_estimators'] = str(2000)
        self.config['create ensemble settings']['RF_min_sample_leaf'] = str(1)
        self.config['create ensemble settings']['RF_max_features'] = 'sqrt'
        self.config['create ensemble settings']['RF_n_jobs'] = str(-1)
        self.config['create ensemble settings']['RF_criterion'] = 'entropy'
        self.config['create ensemble settings']['RF_meta_data'] = 'None'
        self.config['create ensemble settings']['generate_example_decision_tree'] = 'None'
        self.config['create ensemble settings']['generate_example_decision_tree_fancy'] = 'None'
        self.config['create ensemble settings']['generate_features_importance_log'] = 'None'
        self.config['create ensemble settings']['generate_features_importance_bar_graph'] = 'None'
        self.config['create ensemble settings']['compute_permutation_importance'] = 'None'
        self.config['create ensemble settings']['generate_learning_curve'] = 'None'
        self.config['create ensemble settings']['generate_precision_recall_curve'] = 'None'
        self.config['create ensemble settings']['N_feature_importance_bars'] = 'None'
        self.config['create ensemble settings']['LearningCurve_shuffle_k_splits'] = 'None'

        # validation/run model
        self.config.add_section('validation/run model')
        self.config['validation/run model']['generate_validation_video'] = 'None'
        self.config['validation/run model']['sample_feature_file'] = 'None'
        self.config['validation/run model']['save_individual_frames'] = 'None'
        self.config['validation/run model']['classifier_path'] = 'None'
        self.config['validation/run model']['classifier_name'] = 'None'
        self.config['validation/run model']['frames_dir_out_validation'] = 'None'
        self.config['validation/run model']['save_frames'] = 'None'
        self.config['validation/run model']['save_gantt'] = 'None'
        self.config['validation/run model']['discrimination_threshold'] = 'None'

        self.config.add_section('Multi animal IDs')
        self.config['Multi animal IDs']['ID_list'] = 'None'

        self.config.add_section('Outlier settings')
        self.config['Outlier settings']['movement_criterion'] = 'None'
        self.config['Outlier settings']['location_criterion'] = 'None'

        self.config_path = os.path.join(self.project_folder, 'project_config.ini')
        with open(self.config_path, 'w') as file:
            self.config.write(file)

        bp_dir_path = os.path.join(self.simba_dir, 'pose_configurations', 'bp_names', 'bp_names.csv')
        with open(bp_dir_path, "r", encoding='utf8') as f:
            cr = csv.reader(f, delimiter=",")  # , is default
            rows = list(cr)  # create a list of rows for instance

        chosen_bps = rows[self.body_part_config_idx]
        chosen_bps = list(filter(None, chosen_bps))

        project_bp_file_path = os.path.join(self.bp_names_folder, 'project_bp_names.csv')
        f = open(project_bp_file_path, 'w+')
        for i in chosen_bps:
            f.write(i + '\n')
        f.close()
        print('SIMBA COMPLETE: Project directory tree and project_config.ini created in {}'.format(self.project_folder))