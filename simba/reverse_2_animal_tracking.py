__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
import shutil
from simba.read_config_unit_tests import read_config_entry, check_that_column_exist, read_config_file, check_file_exist_and_readable
from datetime import datetime
from simba.misc_tools import check_multi_animal_status, get_fn_ext
import os, glob
from simba.rw_dfs import read_df, save_df
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from simba.features_scripts.feature_extractor_16bp import ExtractFeaturesFrom16bps
from simba.features_scripts.feature_extractor_14bp import ExtractFeaturesFrom14bps
from simba.features_scripts.extract_features_14bp_from_16bp import extract_features_wotarget_14_from_16
from simba.features_scripts.extract_features_9bp import extract_features_wotarget_9
from simba.features_scripts.feature_extractor_8bp import ExtractFeaturesFrom8bps
from simba.features_scripts.feature_extractor_7bp import ExtractFeaturesFrom7bps
from simba.features_scripts.feature_extractor_4bp import ExtractFeaturesFrom4bps
from simba.features_scripts.extract_features_user_defined_new import UserDefinedFeatureExtractor
from simba.train_model_functions import get_all_clf_names
from _legacy.test import check_that_two_dfs_are_equal_len

class Reverse2AnimalTracking(object):
    def __init__(self,
                 config_path: str=None):

        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.config, self.config_path = read_config_file(config_path), config_path
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.features_path = os.path.join(self.project_path, 'csv', 'features_extracted')
        self.targets_path = os.path.join(self.project_path, 'csv', 'targets_inserted')
        self.model_cnt = read_config_entry(self.config, 'SML settings', 'No_targets', data_type='int')
        self.store_path_features = os.path.join(self.features_path, 'Non_reversed_files_at_' + str(self.datetime))
        self.store_path_targets = os.path.join(self.targets_path, 'Non_reversed_files_at_' + str(datetime))
        self.store_path_outliers = os.path.join(self.in_dir, 'Non_reversed_files_at_' + str(datetime))
        if not os.path.exists(self.store_path_features): os.makedirs(self.store_path_features)
        if not os.path.exists(self.store_path_targets): os.makedirs(self.store_path_targets)
        if not os.path.exists(self.store_path_outliers): os.makedirs(self.store_path_outliers)
        self.animal_cnt = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.animal_cnt)
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.animal_cnt, self.x_cols, self.y_cols, [], [])
        self.pose_estimation_setting = read_config_entry(self.config, 'create ensemble settings', 'pose_estimation_body_parts', 'str')

        self.files_found = glob.glob(self.in_dir + '/*.' + self.file_type)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def reverse_tracking(self):
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.data_df = read_df(file_path, self.file_type)
            self.reversed_df = pd.DataFrame()
            animal_dict, col_counter = {}, 0
            for animal_cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
                animal_dict[animal_cnt] = self.data_df.iloc[:, col_counter:col_counter + len(animal_bps['X_bps'] * 3)]
                col_counter += len(animal_bps['X_bps'] * 3)
            for k in reversed(list(animal_dict.keys())):
                self.reversed_df = pd.concat([self.reversed_df, animal_dict[k]], axis=1)
            shutil.move(file_path, os.path.join(self.store_path_outliers, os.path.basename(file_path)))
            save_df(self.reversed_df, self.file_type, file_path)

    def create_features(self):
        old_feature_files = glob.glob(self.features_path + '/*.' + self.file_type)
        for file_path in old_feature_files:
            shutil.move(file_path, os.path.join(self.store_path_features, os.path.basename(file_path)))
        if self.pose_estimation_setting == '16':
            ExtractFeaturesFrom16bps(self.config_path)
        elif self.pose_estimation_setting == '14':
            ExtractFeaturesFrom14bps(self.config_path)
        elif self.pose_estimation_setting == '987':
            extract_features_wotarget_14_from_16(self.config_path)
        elif self.pose_estimation_setting == '9':
            extract_features_wotarget_9(self.config_path)
        elif self.pose_estimation_setting == '8':
            ExtractFeaturesFrom8bps(self.config_path)
        elif self.pose_estimation_setting == '7':
            ExtractFeaturesFrom7bps(self.config_path)
        elif self.pose_estimation_setting == '4':
            ExtractFeaturesFrom4bps(self.config_path)
        elif self.pose_estimation_setting == 'user_defined':
            UserDefinedFeatureExtractor(self.config_path)

    def reappend_targets(self):
        clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        feature_files = glob.glob(self.features_path + '/*.' + self.file_type)
        for file_path in feature_files:
            _, video_name, _ = get_fn_ext(file_path)
            data_df = read_df(file_path, self.file_type)
            target_df_path = os.path.join(self.targets_path, video_name) + '.' + self.file_type
            check_file_exist_and_readable(target_df_path)
            target_df = read_df(target_df_path, self.file_type)
            for clf_name in clf_names:
                check_that_column_exist(target_df,column_name=clf_name, file_name=file_path)
                check_that_two_dfs_are_equal_len(df_1=data_df[clf_name], df_2=target_df[clf_name], file_path_1=file_path, file_path_2=target_df_path, col_name=clf_name)
                data_df[clf_name] = target_df[clf_name]
            shutil.move(file_path, os.path.join(self.store_path_targets, os.path.basename(file_path)))



















test = Reverse2AnimalTracking(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
test.reverse_tracking()



