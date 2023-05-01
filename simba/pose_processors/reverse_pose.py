__author__ = "Simon Nilsson"

import pandas as pd
import shutil
import os, glob
from datetime import datetime

from simba.feature_extractors.feature_extractor_16bp import ExtractFeaturesFrom16bps
from simba.feature_extractors.feature_extractor_14bp import ExtractFeaturesFrom14bps
from simba.feature_extractors.extract_features_9bp import extract_features_wotarget_9
from simba.feature_extractors.feature_extractor_8bp import ExtractFeaturesFrom8bps
from simba.feature_extractors.feature_extractor_7bp import ExtractFeaturesFrom7bps
from simba.feature_extractors.feature_extractor_4bp import ExtractFeaturesFrom4bps
from simba.feature_extractors.feature_extractor_user_defined import UserDefinedFeatureExtractor
from simba.mixins.config_reader import ConfigReader
from simba.utils.read_write import read_config_entry, read_df, write_df, get_fn_ext, get_all_clf_names
from simba.utils.checks import check_that_column_exist, check_file_exist_and_readable

class Reverse2AnimalTracking(ConfigReader):
    def __init__(self,
                 config_path: str):

        ConfigReader.__init__(self, config_path=config_path)
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
            write_df(self.reversed_df, self.file_type, file_path)

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



