__author__ = "Simon Nilsson", "JJ Choong"

from configparser import ConfigParser, MissingSectionHeaderError
import os, glob
from simba.train_model_functions import (read_all_files_in_folder,
                                            read_in_all_model_names_to_remove,
                                            delete_other_annotation_columns,
                                            split_df_to_x_y,
                                            random_undersampler,
                                            smoteen_oversampler,
                                            smote_oversampler,
                                            calc_permutation_importance,
                                            calc_learning_curve,
                                            calc_pr_curve,
                                            create_example_dt,
                                            create_clf_report,
                                            create_x_importance_bar_chart,
                                            create_shap_log,
                                            dviz_classification_visualization,
                                            create_x_importance_log,
                                            create_meta_data_csv_training_multiple_models,
                                            print_machine_model_information,
                                            save_rf_model)
from simba.read_config_unit_tests import (check_int, check_str, check_float, read_config_entry, read_simba_meta_files, read_meta_file)
from simba.drop_bp_cords import drop_bp_cords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class TrainMultipleModelsFromMeta(object):
    """
    Class for grid-searching random forest models from hyperparameter setting and sampling methods
    stored within the `project_folder/configs` directory of a SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Example
    ----------
    >>> model_trainer = TrainMultipleModelsFromMeta(config_path='MyConfigPath')
    >>> model_trainer.train_models_from_meta()
    """

    def __init__(self,
                 config_path: str):

        self.config = ConfigParser()
        try:
            self.config.read(config_path)
        except MissingSectionHeaderError:
            raise AssertionError(('ERROR:  Not a valid project_config file. Please check the project_config.ini path.'))
        self.ini_path = config_path
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='str')
        self.data_in_path = os.path.join(self.project_path, 'csv', 'targets_inserted')
        self.model_dir_out = os.path.join(read_config_entry(self.config, 'SML settings', 'model_dir', data_type='str'), 'validations')
        if not os.path.exists(self.model_dir_out): os.makedirs(self.model_dir_out)
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.model_cnt = read_config_entry(self.config, 'SML settings', 'No_targets', data_type='int')
        self.annotation_file_paths = glob.glob(self.data_in_path + '/*.' + self.file_type)
        self.meta_files_folder = os.path.join(self.project_path, 'configs')
        if not os.path.exists(self.meta_files_folder): os.makedirs(self.meta_files_folder)
        self.meta_file_lst = sorted(read_simba_meta_files(self.meta_files_folder))
        self.annotation_file_paths = glob.glob(self.data_in_path + '/*.' + self.file_type)
        print('Reading in {} annotated files...'.format(str(len(self.annotation_file_paths))))
        self.in_df = read_all_files_in_folder(self.annotation_file_paths, self.file_type)
        self.data_df_wo_cords = drop_bp_cords(self.in_df, config_path)

    def perform_sampling(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_df, self.y_df, test_size=self.meta_dict['train_test_size'])
        if self.meta_dict['under_sample_setting'].lower() == 'random undersample':
            self.x_train, self.y_train = random_undersampler(self.x_train, self.y_train, self.meta_dict['under_sample_ratio'])
        if self.meta_dict['over_sample_setting'] == 'SMOTEENN':
            self.x_train, self.y_train = smoteen_oversampler(self.x_train, self.y_train, self.meta_dict['over_sample_ratio'])
        elif self.meta_dict['over_sample_setting'] == 'SMOTE':
            self.x_train, self.y_train = smote_oversampler(self.x_train, self.y_train, self.meta_dict['over_sample_ratio'])

    def train_models_from_meta(self):
        print('Training {} models...'.format(str(len(self.meta_file_lst))))
        for config_cnt, file_path in enumerate(self.meta_file_lst):
            print('Training model {}/{}...'.format(str(config_cnt+1), str(len(self.meta_file_lst))))
            self.meta_dict = read_meta_file(file_path)
            check_str(name='Classifier_name', value=self.meta_dict['Classifier_name'])
            check_int(name='RF_n_estimators', value=self.meta_dict['RF_n_estimators'])
            check_str(name='RF_criterion', value=self.meta_dict['RF_criterion'], options=['gini', 'entropy'])
            check_float(name='train_test_size', value=self.meta_dict['train_test_size'])
            check_int(name='RF_min_sample_leaf', value=self.meta_dict['RF_min_sample_leaf'])
            check_str(name='generate_rf_model_meta_data_file', value=self.meta_dict['generate_rf_model_meta_data_file'])
            check_str(name='generate_example_decision_tree', value=self.meta_dict['generate_example_decision_tree'])
            check_str(name='generate_classification_report', value=self.meta_dict['generate_classification_report'])
            check_str(name='generate_features_importance_log', value=self.meta_dict['generate_features_importance_log'])
            check_str(name='generate_features_importance_bar_graph', value=self.meta_dict['generate_features_importance_bar_graph'])
            check_str(name='compute_feature_permutation_importance', value=self.meta_dict['compute_feature_permutation_importance'])
            check_str(name='generate_sklearn_learning_curves', value=self.meta_dict['generate_sklearn_learning_curves'])
            check_str(name='generate_precision_recall_curves', value=self.meta_dict['generate_precision_recall_curves'])
            check_str(name='under_sample_setting', value=self.meta_dict['under_sample_setting'], allow_blank=True)
            check_str(name='over_sample_setting', value=self.meta_dict['over_sample_setting'], allow_blank=True)


            if self.meta_dict['generate_sklearn_learning_curves'] == 'yes':
                check_int(name='learning_curve_k_splits', value=self.meta_dict['learning_curve_k_splits'])
                check_int(name='learning_curve_data_splits', value=self.meta_dict['learning_curve_data_splits'])
            if self.meta_dict['generate_features_importance_bar_graph'] == 'yes':
                check_int(name='n_feature_importance_bars', value=self.meta_dict['n_feature_importance_bars'])
            if 'generate_shap_scores' in self.meta_dict.keys():
                if self.meta_dict['generate_shap_scores'] == 'yes':
                    check_int(name='shap_target_present_no', value=self.meta_dict['shap_target_present_no'])
                    check_int(name='shap_target_absetn_no', value=self.meta_dict['shap_target_absetn_no'])
            if self.meta_dict['under_sample_setting'].lower() == 'random undersample':
                check_float(name='under_sample_ratio', value=self.meta_dict['under_sample_ratio'])
            if (self.meta_dict['over_sample_setting'].lower() == 'smoteenn') or (self.meta_dict['over_sample_setting'].lower() == 'smote'):
                check_float(name='over_sample_ratio', value=self.meta_dict['over_sample_ratio'])

            self.clf_name = self.meta_dict['Classifier_name']
            self.class_names = ['Not_' + self.clf_name, self.clf_name]
            annotation_cols_to_remove = read_in_all_model_names_to_remove(self.config, self.model_cnt, self.clf_name)
            self.x_y_df = delete_other_annotation_columns(self.data_df_wo_cords, annotation_cols_to_remove)
            self.x_df, self.y_df = split_df_to_x_y(self.x_y_df, self.clf_name)
            self.feature_names = self.x_df.columns
            self.perform_sampling()
            print('MODEL {} settings'.format(str(config_cnt)))
            print_machine_model_information(self.meta_dict)
            print('# {} features.'.format(len(self.feature_names)))

            self.rf_clf = RandomForestClassifier(n_estimators=self.meta_dict['RF_n_estimators'], max_features=self.meta_dict['RF_max_features'], n_jobs=-1,criterion=self.meta_dict['RF_criterion'], min_samples_leaf=self.meta_dict['RF_min_sample_leaf'], bootstrap=True, verbose=1)
            try:
                self.rf_clf.fit(self.x_train, self.y_train)
            except Exception as e:
                raise ValueError(e, 'ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')

            if self.meta_dict['compute_feature_permutation_importance'] == 'yes':
                calc_permutation_importance(self.x_test, self.y_test, self.rf_clf, self.feature_names, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict['generate_sklearn_learning_curves'] == 'yes':
                calc_learning_curve(self.x_y_df, self.clf_name, self.meta_dict['learning_curve_k_splits'], self.meta_dict['learning_curve_data_splits'], self.meta_dict['train_test_size'], self.rf_clf, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict['generate_precision_recall_curves'] == 'yes':
                calc_pr_curve(self.rf_clf, self.x_test, self.y_test, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict['generate_example_decision_tree'] == 'yes':
                create_example_dt(self.rf_clf, self.clf_name, self.feature_names, self.class_names, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict['generate_classification_report'] == 'yes':
                create_clf_report(self.rf_clf, self.x_test, self.y_test, self.class_names, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict['generate_features_importance_log'] == 'yes':
                create_x_importance_log(self.rf_clf, self.feature_names, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict['generate_features_importance_bar_graph'] == 'yes':
                create_x_importance_bar_chart(self.rf_clf, self.feature_names, self.clf_name, self.model_dir_out, self.meta_dict['n_feature_importance_bars'], save_file_no=config_cnt)
            if 'generate_shap_scores' in self.meta_dict.keys():
                if self.meta_dict['generate_shap_scores'] == 'yes':
                    create_shap_log(self.ini_path, self.rf_clf, self.x_train, self.y_train, self.feature_names, self.clf_name, self.meta_dict['shap_target_present_no'], self.meta_dict['shap_target_absetn_no'], self.model_dir_out, save_file_no=config_cnt)
            create_meta_data_csv_training_multiple_models(self.meta_dict, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            save_rf_model(self.rf_clf, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            print('Classifier {} saved in models/validations/model_files directory ...'.format(str(self.clf_name + '_' + str(config_cnt))))
        print('All models and evaluations complete. The models/evaluation files are in models/validations folders')

# test = TrainMultipleModelsFromMeta(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
# test.train_models_from_meta()






