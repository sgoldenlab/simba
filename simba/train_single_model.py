__author__ = "Simon Nilsson", "JJ Choong"

from configparser import ConfigParser, MissingSectionHeaderError, NoSectionError,NoOptionError
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
                                            create_meta_data_csv_training_one_model,
                                            save_rf_model)
from simba.read_config_unit_tests import (check_int, check_str, check_float, read_config_entry, read_config_file)
from simba.drop_bp_cords import drop_bp_cords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class TrainSingleModel(object):
    """
    Class for training a single random forest model from hyper-parameter setting and sampling methods
    stored within the SimBA project config .ini file (`global environment`).

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Example
    ----------
    >>> model_trainer = TrainSingleModel(config_path='MyConfigPath')
    >>> model_trainer.perform_sampling()
    >>> model_trainer.train_model()
    >>> model_trainer.save_model()

    """


    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.ini_path = config_path
        self.model_dir_out = os.path.join(read_config_entry(self.config, 'SML settings', 'model_dir', data_type='str'), 'generated_models')
        if not os.path.exists(self.model_dir_out): os.makedirs(self.model_dir_out)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='str')
        self.data_in_path = os.path.join(self.project_path, 'csv', 'targets_inserted')
        self.eval_out_path = os.path.join(self.model_dir_out, 'model_evaluations')
        if not os.path.exists(self.eval_out_path): os.makedirs(self.eval_out_path)
        self.clf_name = read_config_entry(self.config, 'create ensemble settings', 'classifier', data_type='str')
        self.tt_size = read_config_entry(self.config, 'create ensemble settings', 'train_test_size', data_type='float')
        self.model_cnt = read_config_entry(self.config, 'SML settings', 'No_targets', data_type='int')
        self.algo = read_config_entry(self.config, 'create ensemble settings', 'model_to_run', data_type='str')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.under_sample_setting = read_config_entry(self.config, 'create ensemble settings', 'under_sample_setting', data_type='str').lower().strip()
        self.over_sample_setting = read_config_entry(self.config, 'create ensemble settings', 'over_sample_setting', data_type='str').lower().strip()
        if self.under_sample_setting == 'random undersample':
            self.under_sample_ratio = read_config_entry(self.config, 'create ensemble settings', 'under_sample_ratio', data_type='float', default_value='NaN')
            check_float(name='under_sample_ratio', value=self.under_sample_ratio)
        else:
            self.under_sample_ratio = 'NaN'
        if (self.over_sample_setting == 'smoteenn') or (self.over_sample_setting == 'smote'):
            self.over_sample_ratio = read_config_entry(self.config, 'create ensemble settings', 'over_sample_ratio', data_type='float', default_value='NaN')
            check_float(name='over_sample_ratio', value=self.over_sample_ratio)
        else:
            self.over_sample_ratio = 'NaN'

        self.annotation_file_paths = glob.glob(self.data_in_path + '/*.' + self.file_type)
        print('Reading in {} annotated files...'.format(str(len(self.annotation_file_paths))))
        self.data_df = read_all_files_in_folder(self.annotation_file_paths, self.file_type, [self.clf_name])
        self.data_df_wo_cords = drop_bp_cords(self.data_df, config_path)
        annotation_cols_to_remove = read_in_all_model_names_to_remove(self.config, self.model_cnt, self.clf_name)
        self.x_y_df = delete_other_annotation_columns(self.data_df_wo_cords, list(annotation_cols_to_remove))
        self.class_names = ['Not_' + self.clf_name, self.clf_name]
        self.x_df, self.y_df = split_df_to_x_y(self.x_y_df, self.clf_name)
        self.feature_names = self.x_df.columns
        print('# of features in dataset: ' + str(len(self.x_df.columns)))
        print('# of {} frames in dataset: {}'.format(self.clf_name, str(self.y_df.sum())))

    def perform_sampling(self):
        """
        Method for sampling data for training and testing, and perform over and under-sampling of the training sets
        as indicated within the SimBA project config.

        Returns
        ----------
        Attribute: np.array
            x_train
        Attribute: np.array
            y_train
        Attribute: np.array
            x_test
        Attribute: np.array
            y_test
        """

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_df, self.y_df,test_size=self.tt_size)
        if self.under_sample_setting == 'random undersample':
            self.x_train, self.y_train = random_undersampler(self.x_train, self.y_train, float(self.under_sample_ratio))
        if self.over_sample_setting == 'smoteenn':
            self.x_train, self.y_train = smoteen_oversampler(self.x_train, self.y_train, float(self.over_sample_ratio))
        elif self.over_sample_setting == 'smote':
            self.x_train, self.y_train = smote_oversampler(self.x_train, self.y_train, float(self.over_sample_ratio))

    def train_model(self):
        """
        Method for training single random forest model.

        Returns
        ----------
        Attribute: object
            rf_clf
        """

        if self.algo == 'RF':
            n_estimators = read_config_entry(self.config, 'create ensemble settings', 'RF_n_estimators', data_type='int')
            max_features = read_config_entry(self.config, 'create ensemble settings', 'RF_max_features', data_type='str')
            criterion = read_config_entry(self.config, 'create ensemble settings', 'RF_criterion', data_type='str', options=['gini', 'entropy'])
            min_sample_leaf = read_config_entry(self.config, 'create ensemble settings', 'RF_min_sample_leaf', data_type='int')
            compute_permutation_importance = read_config_entry(self.config, 'create ensemble settings', 'compute_permutation_importance', data_type='str', default_value='no')
            generate_learning_curve = read_config_entry(self.config, 'create ensemble settings', 'generate_learning_curve', data_type='str', default_value='no')
            generate_precision_recall_curve = read_config_entry(self.config, 'create ensemble settings', 'generate_precision_recall_curve', data_type='str', default_value='no')
            generate_example_decision_tree = read_config_entry(self.config, 'create ensemble settings', 'generate_example_decision_tree', data_type='str', default_value='no')
            generate_classification_report = read_config_entry(self.config, 'create ensemble settings', 'generate_classification_report', data_type='str', default_value='no')
            generate_features_importance_log = read_config_entry(self.config, 'create ensemble settings', 'generate_features_importance_log', data_type='str', default_value='no')
            generate_features_importance_bar_graph = read_config_entry(self.config, 'create ensemble settings', 'generate_features_importance_bar_graph', data_type='str', default_value='no')
            generate_example_decision_tree_fancy = read_config_entry(self.config, 'create ensemble settings', 'generate_example_decision_tree_fancy', data_type='str', default_value='no')
            generate_shap_scores = read_config_entry(self.config, 'create ensemble settings', 'generate_shap_scores', data_type='str', default_value='no')
            save_meta_data = read_config_entry(self.config, 'create ensemble settings', 'RF_meta_data' ,data_type='str', default_value='no')

            if generate_learning_curve == 'yes':
                shuffle_splits = read_config_entry(self.config, 'create ensemble settings', 'LearningCurve_shuffle_k_splits', data_type='int', default_value='NaN')
                dataset_splits = read_config_entry(self.config, 'create ensemble settings','LearningCurve_shuffle_data_splits', data_type='int', default_value='NaN')
                check_int(name='shuffle_splits', value=shuffle_splits)
                check_int(name='dataset_splits', value=dataset_splits)
            else:
                shuffle_splits, dataset_splits = 'NaN', 'NaN'
            if generate_features_importance_bar_graph == 'yes':
                feature_importance_bars = read_config_entry(self.config, 'create ensemble settings', 'N_feature_importance_bars', 'int', 'NaN')
                check_int(name='n_feature_importance_bars', value=feature_importance_bars)
            else:
                feature_importance_bars = 'NaN'
            if generate_shap_scores == 'yes':
                shap_target_present_cnt = read_config_entry(self.config, 'create ensemble settings', 'shap_target_present_no', data_type='int', default_value=0)
                shap_target_absent_cnt = read_config_entry(self.config, 'create ensemble settings', 'shap_target_absent_no', data_type='int', default_value=0)
                check_int(name='shap_target_present_cnt', value=shap_target_present_cnt)
                check_int(name='shap_target_absent_cnt', value=shap_target_absent_cnt)

            self.rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=-1, criterion=criterion, min_samples_leaf=min_sample_leaf, bootstrap=True, verbose=1)
            try:
                self.rf_clf.fit(self.x_train, self.y_train)
            except Exception as e:
                print('ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')
                raise ValueError(e, 'SIMBA ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')
            if compute_permutation_importance == 'yes':
                calc_permutation_importance(self.x_test, self.y_test, self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path)
            if generate_learning_curve == 'yes':
                calc_learning_curve(x_y_df=self.x_y_df,
                                    clf_name=self.clf_name,
                                    shuffle_splits=shuffle_splits,
                                    dataset_splits=dataset_splits,
                                    tt_size=self.tt_size,
                                    rf_clf=self.rf_clf,
                                    save_dir=self.eval_out_path)

            if generate_precision_recall_curve == 'yes':
                calc_pr_curve(self.rf_clf, self.x_test, self.y_test, self.clf_name, self.eval_out_path)
            if generate_example_decision_tree == 'yes':
                create_example_dt(self.rf_clf, self.clf_name, self.feature_names, self.class_names, self.eval_out_path)
            if generate_classification_report == 'yes':
                create_clf_report(self.rf_clf, self.x_test, self.y_test, self.class_names, self.eval_out_path)
            if generate_features_importance_log == 'yes':
                create_x_importance_log(self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path)
            if generate_features_importance_bar_graph == 'yes':
                create_x_importance_bar_chart(self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path, feature_importance_bars)
            if generate_example_decision_tree_fancy == 'yes':
                dviz_classification_visualization(self.x_train, self.y_train, self.clf_name, self.class_names, self.eval_out_path)
            if generate_shap_scores == 'yes':
                create_shap_log(self.ini_path, self.rf_clf, self.x_train, self.y_train, self.feature_names, self.clf_name, shap_target_present_cnt, shap_target_absent_cnt, self.eval_out_path)

            if save_meta_data == 'yes':
                meta_data_lst = [self.clf_name, criterion, max_features, min_sample_leaf,
                                 n_estimators, compute_permutation_importance, generate_classification_report,
                                 generate_example_decision_tree, generate_features_importance_bar_graph,
                                 generate_features_importance_log, generate_precision_recall_curve, save_meta_data,
                                 generate_learning_curve, dataset_splits, shuffle_splits, feature_importance_bars,
                                 self.over_sample_ratio, self.over_sample_setting, self.tt_size,
                                 self.under_sample_ratio, self.under_sample_ratio]
                create_meta_data_csv_training_one_model(meta_data_lst, self.clf_name, self.eval_out_path)

    def save_model(self):
        """
        Method for saving pickled RF model. The model is saved in the `models/generated_models` directory
        of the SimBA project tree.

        Returns
        ----------
        None
        """


        save_rf_model(self.rf_clf, self.clf_name, self.model_dir_out)
        print('Classifier ' + self.clf_name + ' saved @ ' + str('models/generated_models ') + 'folder')
        print('Evaluation files are in models/generated_models/model_evaluations folders')

# test = TrainSingleModel(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
# test.perform_sampling()
# test.train_model()
# test.save_model()


