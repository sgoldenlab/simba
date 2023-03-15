__author__ = "Simon Nilsson", "JJ Choong"

import os, glob, ast
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
                                         save_rf_model,
                                         bout_train_test_splitter)

from simba.read_config_unit_tests import (check_int,
                                          check_float,
                                          read_config_entry,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.drop_bp_cords import drop_bp_cords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from simba.misc_tools import SimbaTimer
from simba.enums import (Options,
                         ReadConfig,
                         Dtypes,
                         Paths,
                         Methods)

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
        self.model_dir_out = os.path.join(read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.MODEL_DIR.value, data_type=Dtypes.STR.value), 'generated_models')
        if not os.path.exists(self.model_dir_out): os.makedirs(self.model_dir_out)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_path = os.path.join(self.project_path, Paths.TARGETS_INSERTED_DIR.value)
        self.eval_out_path = os.path.join(self.model_dir_out, 'model_evaluations')
        if not os.path.exists(self.eval_out_path): os.makedirs(self.eval_out_path)
        self.clf_name = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.CLASSIFIER.value, data_type=Dtypes.STR.value)
        self.tt_size = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.TT_SIZE.value, data_type=Dtypes.FLOAT.value)
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.algo = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.MODEL_TO_RUN.value, data_type=Dtypes.STR.value)
        self.split_type = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.SPLIT_TYPE.value, data_type=Dtypes.STR.value, options=Options.TRAIN_TEST_SPLIT.value, default_value=Methods.SPLIT_TYPE_FRAMES.value)
        self.under_sample_setting = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.UNDERSAMPLE_SETTING.value, data_type=Dtypes.STR.value).lower().strip()
        self.over_sample_setting = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.OVERSAMPLE_SETTING.value, data_type=Dtypes.STR.value).lower().strip()
        if self.under_sample_setting == Methods.RANDOM_UNDERSAMPLE.value:
            self.under_sample_ratio = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.UNDERSAMPLE_RATIO.value, data_type=Dtypes.FLOAT.value, default_value=Dtypes.NAN.value)
            check_float(name=ReadConfig.UNDERSAMPLE_RATIO.value, value=self.under_sample_ratio)
        else:
            self.under_sample_ratio = Dtypes.NAN.value
        if (self.over_sample_setting == Methods.SMOTEENN.value.lower()) or (self.over_sample_setting == Methods.SMOTE.value.lower()):
            self.over_sample_ratio = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.OVERSAMPLE_RATIO.value, data_type=Dtypes.FLOAT.value, default_value=Dtypes.NAN.value)
            check_float(name=ReadConfig.OVERSAMPLE_RATIO.value, value=self.over_sample_ratio)
        else:
            self.over_sample_ratio = Dtypes.NAN.value

        self.annotation_file_paths = glob.glob(self.data_in_path + '/*.' + self.file_type)
        check_if_filepath_list_is_empty(filepaths=self.annotation_file_paths, error_msg='SIMBA ERROR: Zero annotation files found in project_folder/csv/targets_inserted, cannot create model.')
        print('Reading in {} annotated files...'.format(str(len(self.annotation_file_paths))))
        self.data_df = read_all_files_in_folder(self.annotation_file_paths, self.file_type, [self.clf_name])
        self.data_df_wo_cords = drop_bp_cords(self.data_df, config_path)
        annotation_cols_to_remove = read_in_all_model_names_to_remove(self.config, self.model_cnt, self.clf_name)
        self.x_y_df = delete_other_annotation_columns(self.data_df_wo_cords, list(annotation_cols_to_remove))
        self.class_names = ['Not_' + self.clf_name, self.clf_name]
        self.x_df, self.y_df = split_df_to_x_y(self.x_y_df, self.clf_name)
        self.feature_names = self.x_df.columns
        print('Number of features in dataset: ' + str(len(self.x_df.columns)))
        print('Number of {} frames in dataset: {} ({}%)'.format(self.clf_name, str(self.y_df.sum()), str(round(self.y_df.sum() / len(self.y_df), 4) * 100)))
        print('Training and evaluating model...')
        self.timer = SimbaTimer()
        self.timer.start_timer()

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

        if self.split_type == Methods.SPLIT_TYPE_FRAMES.value:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_df, self.y_df,test_size=self.tt_size)
        elif self.split_type == Methods.SPLIT_TYPE_BOUTS.value:
            self.x_train, self.x_test, self.y_train, self.y_test = bout_train_test_splitter(x_df=self.x_df,y_df=self.y_df, test_size=self.tt_size)

        if self.under_sample_setting == Methods.RANDOM_UNDERSAMPLE.value.lower():
            self.x_train, self.y_train = random_undersampler(self.x_train, self.y_train, float(self.under_sample_ratio))
        if self.over_sample_setting == Methods.SMOTEENN.value.lower():
            self.x_train, self.y_train = smoteen_oversampler(self.x_train, self.y_train, float(self.over_sample_ratio))
        elif self.over_sample_setting == Methods.SMOTE.value.lower():
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
            n_estimators = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.RF_ESTIMATORS.value, data_type=Dtypes.INT.value)
            max_features = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.RF_MAX_FEATURES.value, data_type=Dtypes.STR.value)
            if max_features == 'None':
                max_features = None
            criterion = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.RF_CRITERION.value, data_type=Dtypes.STR.value, options=Options.CLF_CRITERION.value)
            min_sample_leaf = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.MIN_LEAF.value, data_type=Dtypes.INT.value)
            compute_permutation_importance = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.PERMUTATION_IMPORTANCE.value, data_type=Dtypes.STR.value, default_value=False)
            generate_learning_curve = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.LEARNING_CURVE.value, data_type=Dtypes.STR.value, default_value=False)
            generate_precision_recall_curve = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.PRECISION_RECALL.value, data_type=Dtypes.STR.value, default_value=False)
            generate_example_decision_tree = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.EX_DECISION_TREE.value, data_type=Dtypes.STR.value, default_value=False)
            generate_classification_report = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.CLF_REPORT.value, data_type=Dtypes.STR.value, default_value=False)
            generate_features_importance_log = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.IMPORTANCE_LOG.value, data_type=Dtypes.STR.value, default_value=False)
            generate_features_importance_bar_graph = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.IMPORTANCE_LOG.value, data_type=Dtypes.STR.value, default_value=False)
            generate_example_decision_tree_fancy = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.EX_DECISION_TREE_FANCY.value, data_type=Dtypes.STR.value, default_value=False)
            generate_shap_scores = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.SHAP_SCORES.value, data_type=Dtypes.STR.value, default_value=False)
            save_meta_data = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.RF_METADATA.value ,data_type=Dtypes.STR.value, default_value=False)

            if self.config.has_option(ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.CLASS_WEIGHTS.value):
                class_weights = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.CLASS_WEIGHTS.value, data_type=Dtypes.STR.value, default_value=Dtypes.NONE.value)
                if class_weights == 'custom':
                    class_weights = ast.literal_eval(read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.CUSTOM_WEIGHTS.value, data_type=Dtypes.STR.value))
                    for k, v in class_weights.items():
                        class_weights[k] = int(v)
                if class_weights == Dtypes.NONE.value:
                    class_weights = None
            else:
                class_weights = None

            if generate_learning_curve in Options.PERFORM_FLAGS.value:
                shuffle_splits = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.LEARNING_CURVE_K_SPLITS.value, data_type=Dtypes.INT.value, default_value=Dtypes.NAN.value)
                dataset_splits = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.LEARNING_DATA_SPLITS.value, data_type=Dtypes.INT.value, default_value=Dtypes.NAN.value)
                check_int(name=ReadConfig.LEARNING_CURVE_K_SPLITS.value, value=shuffle_splits)
                check_int(name=ReadConfig.LEARNING_DATA_SPLITS.value, value=dataset_splits)
            else:
                shuffle_splits, dataset_splits = Dtypes.NAN.value, Dtypes.NAN.value
            if generate_features_importance_bar_graph in Options.PERFORM_FLAGS.value:
                feature_importance_bars = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.IMPORTANCE_BARS_N.value, Dtypes.INT.value, Dtypes.NAN.value)
                check_int(name=ReadConfig.IMPORTANCE_BARS_N.value, value=feature_importance_bars)
            else:
                feature_importance_bars = Dtypes.NAN.value
            if generate_shap_scores in Options.PERFORM_FLAGS.value:
                shap_target_present_cnt = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.SHAP_PRESENT.value, data_type=Dtypes.INT.value, default_value=0)
                shap_target_absent_cnt = read_config_entry(self.config, ReadConfig.CREATE_ENSEMBLE_SETTINGS.value, ReadConfig.SHAP_ABSENT.value, data_type=Dtypes.INT.value, default_value=0)
                check_int(name=ReadConfig.SHAP_PRESENT.value, value=shap_target_present_cnt)
                check_int(name=ReadConfig.SHAP_ABSENT.value, value=shap_target_absent_cnt)

            self.rf_clf = RandomForestClassifier(n_estimators=n_estimators,
                                                 max_features=max_features,
                                                 n_jobs=-1, criterion=criterion,
                                                 min_samples_leaf=min_sample_leaf,
                                                 bootstrap=True,
                                                 verbose=1,
                                                 class_weight=class_weights)
            try:
                self.rf_clf.fit(self.x_train, self.y_train)
            except Exception as e:
                print('ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')
                raise ValueError(e, 'SIMBA ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')

            if compute_permutation_importance in Options.PERFORM_FLAGS.value:
                calc_permutation_importance(self.x_test, self.y_test, self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path)
            if generate_learning_curve in Options.PERFORM_FLAGS.value:
                calc_learning_curve(x_y_df=self.x_y_df,
                                    clf_name=self.clf_name,
                                    shuffle_splits=shuffle_splits,
                                    dataset_splits=dataset_splits,
                                    tt_size=self.tt_size,
                                    rf_clf=self.rf_clf,
                                    save_dir=self.eval_out_path)

            if generate_precision_recall_curve in Options.PERFORM_FLAGS.value:
                calc_pr_curve(self.rf_clf, self.x_test, self.y_test, self.clf_name, self.eval_out_path)
            if generate_example_decision_tree in Options.PERFORM_FLAGS.value:
                create_example_dt(self.rf_clf, self.clf_name, self.feature_names, self.class_names, self.eval_out_path)
            if generate_classification_report in Options.PERFORM_FLAGS.value:
                create_clf_report(self.rf_clf, self.x_test, self.y_test, self.class_names, self.eval_out_path)
            if generate_features_importance_log in Options.PERFORM_FLAGS.value:
                create_x_importance_log(self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path)
            if generate_features_importance_bar_graph in Options.PERFORM_FLAGS.value:
                create_x_importance_bar_chart(self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path, feature_importance_bars)
            if generate_example_decision_tree_fancy in Options.PERFORM_FLAGS.value:
                dviz_classification_visualization(self.x_train, self.y_train, self.clf_name, self.class_names, self.eval_out_path)
            if generate_shap_scores in Options.PERFORM_FLAGS.value:
                create_shap_log(self.ini_path, self.rf_clf, self.x_train, self.y_train, self.feature_names, self.clf_name, shap_target_present_cnt, shap_target_absent_cnt, self.eval_out_path)

            if save_meta_data in Options.PERFORM_FLAGS.value:
                meta_data_lst = [self.clf_name, criterion, max_features, min_sample_leaf,
                                 n_estimators, compute_permutation_importance, generate_classification_report,
                                 generate_example_decision_tree, generate_features_importance_bar_graph,
                                 generate_features_importance_log, generate_precision_recall_curve, save_meta_data,
                                 generate_learning_curve, dataset_splits, shuffle_splits, feature_importance_bars,
                                 self.over_sample_ratio, self.over_sample_setting, self.tt_size, self.split_type,
                                 self.under_sample_ratio, self.under_sample_setting, str(class_weights)]

                create_meta_data_csv_training_one_model(meta_data_lst, self.clf_name, self.eval_out_path)

    def save_model(self):
        """
        Method for saving pickled RF model. The model is saved in the `models/generated_models` directory
        of the SimBA project tree.

        Returns
        ----------
        None
        """

        self.timer.stop_timer()
        save_rf_model(self.rf_clf, self.clf_name, self.model_dir_out)
        print('Classifier {} saved in models/generated_models directory (elapsed time: {}s).'.format(self.clf_name, self.timer.elapsed_time_str))
        print('Evaluation files are in models/generated_models/model_evaluations folders')

# test = TrainSingleModel(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
# test.perform_sampling()
# test.train_model()
# test.save_model()


