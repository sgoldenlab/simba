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
                                        create_meta_data_csv_training_multiple_models,
                                        print_machine_model_information,
                                        bout_train_test_splitter,
                                        save_rf_model)
from simba.read_config_unit_tests import (check_int,
                                          check_str,
                                          check_float,
                                          read_config_entry,
                                          read_simba_meta_files,
                                          read_meta_file,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          check_if_valid_input,
                                          read_project_path_and_file_type)
from simba.drop_bp_cords import drop_bp_cords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from simba.enums import (Options,
                         ReadConfig,
                         Paths,
                         Dtypes,
                         Methods,
                         MetaKeys)


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

        self.config = read_config_file(ini_path=config_path)
        self.ini_path = config_path
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_path = os.path.join(self.project_path, Paths.TARGETS_INSERTED_DIR.value)
        self.model_dir_out = os.path.join(read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.MODEL_DIR.value, data_type=Dtypes.STR.value), 'validations')
        if not os.path.exists(self.model_dir_out): os.makedirs(self.model_dir_out)
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, data_type=Dtypes.INT.value)
        self.annotation_file_paths = glob.glob(self.data_in_path + '/*.' + self.file_type)
        self.meta_files_folder = os.path.join(self.project_path, 'configs')
        if not os.path.exists(self.meta_files_folder): os.makedirs(self.meta_files_folder)
        self.meta_file_lst = sorted(read_simba_meta_files(self.meta_files_folder))
        self.annotation_file_paths = glob.glob(self.data_in_path + '/*.' + self.file_type)
        check_if_filepath_list_is_empty(filepaths=self.annotation_file_paths,
                                        error_msg='SIMBA ERROR: Zero annotation files found in project_folder/csv/targets_inserted, cannot create models.')
        print('Reading in {} annotated files...'.format(str(len(self.annotation_file_paths))))
        self.in_df = read_all_files_in_folder(self.annotation_file_paths, self.file_type)
        self.data_df_wo_cords = drop_bp_cords(self.in_df, config_path)

    def perform_sampling(self):
        if self.tt_split_type == Methods.SPLIT_TYPE_FRAMES.value:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_df, self.y_df, test_size=self.meta_dict['train_test_size'])
        if self.tt_split_type == Methods.SPLIT_TYPE_BOUTS.value:
            self.x_train, self.x_test, self.y_train, self.y_test = bout_train_test_splitter(x_df=self.x_df, y_df=self.y_df, test_size=self.meta_dict['train_test_size'])
        if self.meta_dict[ReadConfig.UNDERSAMPLE_SETTING.value].lower() == Methods.RANDOM_UNDERSAMPLE.value:
            self.x_train, self.y_train = random_undersampler(self.x_train, self.y_train, self.meta_dict[ReadConfig.UNDERSAMPLE_RATIO.value])
        if self.meta_dict[ReadConfig.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTEENN.value:
            self.x_train, self.y_train = smoteen_oversampler(self.x_train, self.y_train, self.meta_dict[ReadConfig.OVERSAMPLE_RATIO.value])
        elif self.meta_dict[ReadConfig.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTE.value:
            self.x_train, self.y_train = smote_oversampler(self.x_train, self.y_train, self.meta_dict[ReadConfig.OVERSAMPLE_RATIO.value])

    def train_models_from_meta(self):
        print(f'Training {str(len(self.meta_file_lst))} models...')
        for config_cnt, file_path in enumerate(self.meta_file_lst):
            print('Training model {}/{}...'.format(str(config_cnt+1), str(len(self.meta_file_lst))))
            self.meta_dict = read_meta_file(file_path)
            self.meta_dict = {k.lower(): v for k, v in self.meta_dict.items()}
            check_str(name=self.meta_dict[MetaKeys.CLF_NAME.value], value=self.meta_dict[MetaKeys.CLF_NAME.value])
            check_int(name=MetaKeys.RF_ESTIMATORS.value, value=self.meta_dict[MetaKeys.RF_ESTIMATORS.value], min_value=1)
            check_str(name=MetaKeys.CRITERION.value, value=self.meta_dict[MetaKeys.CRITERION.value], options=Options.CLF_CRITERION.value)
            check_str(name=MetaKeys.RF_MAX_FEATURES.value, value=self.meta_dict[MetaKeys.RF_MAX_FEATURES.value], options=Options.CLF_MAX_FEATURES.value)
            check_float(name=MetaKeys.TT_SIZE.value, value=self.meta_dict[MetaKeys.TT_SIZE.value])
            check_int(name=MetaKeys.MIN_LEAF.value, value=self.meta_dict[MetaKeys.MIN_LEAF.value])
            check_if_valid_input(name=MetaKeys.META_FILE.value, input=self.meta_dict[MetaKeys.META_FILE.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_if_valid_input(MetaKeys.EX_DECISION_TREE.value, input=self.meta_dict[MetaKeys.EX_DECISION_TREE.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_if_valid_input(MetaKeys.CLF_REPORT.value, input=self.meta_dict[MetaKeys.CLF_REPORT.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_if_valid_input(MetaKeys.IMPORTANCE_LOG.value, input=self.meta_dict[MetaKeys.IMPORTANCE_LOG.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_if_valid_input(MetaKeys.IMPORTANCE_BAR_CHART.value, input=self.meta_dict[MetaKeys.IMPORTANCE_BAR_CHART.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_if_valid_input(MetaKeys.PERMUTATION_IMPORTANCE.value, input=self.meta_dict[MetaKeys.PERMUTATION_IMPORTANCE.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_if_valid_input(MetaKeys.LEARNING_CURVE.value, input=self.meta_dict[MetaKeys.LEARNING_CURVE.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_if_valid_input(MetaKeys.PRECISION_RECALL.value, input=self.meta_dict[MetaKeys.PRECISION_RECALL.value], options=Options.RUN_OPTIONS_FLAGS.value)
            check_str(ReadConfig.UNDERSAMPLE_SETTING.value, self.meta_dict[ReadConfig.UNDERSAMPLE_SETTING.value].lower(), options=[x.lower() for x in Options.UNDERSAMPLE_OPTIONS.value])
            check_str(ReadConfig.OVERSAMPLE_SETTING.value, self.meta_dict[ReadConfig.OVERSAMPLE_SETTING.value].lower(), options=[x.lower() for x in Options.OVERSAMPLE_OPTIONS.value])
            if self.meta_dict[MetaKeys.RF_MAX_FEATURES.value] == Dtypes.NONE.value:
                self.meta_dict[MetaKeys.RF_MAX_FEATURES.value] = None
            if self.meta_dict[MetaKeys.LEARNING_CURVE.value] in Options.PERFORM_FLAGS.value:
                check_int(name=MetaKeys.LEARNING_CURVE_K_SPLITS.value, value=self.meta_dict[MetaKeys.LEARNING_CURVE_K_SPLITS.value])
                check_int(name=MetaKeys.LEARNING_CURVE_DATA_SPLITS.value, value=self.meta_dict[MetaKeys.LEARNING_CURVE_DATA_SPLITS.value])
            if self.meta_dict[MetaKeys.IMPORTANCE_BAR_CHART.value] in Options.PERFORM_FLAGS.value:
                check_int(name=MetaKeys.N_FEATURE_IMPORTANCE_BARS.value, value=self.meta_dict[MetaKeys.N_FEATURE_IMPORTANCE_BARS.value])
            if MetaKeys.SHAP_SCORES.value in self.meta_dict.keys():
                if self.meta_dict[MetaKeys.SHAP_SCORES.value] in Options.PERFORM_FLAGS.value:
                    check_int(name=MetaKeys.SHAP_PRESENT.value, value=self.meta_dict[MetaKeys.SHAP_PRESENT.value])
                    check_int(name=MetaKeys.SHAP_ABSENT.value, value=self.meta_dict[MetaKeys.SHAP_ABSENT.value])
            if self.meta_dict[ReadConfig.UNDERSAMPLE_SETTING.value].lower() == Methods.RANDOM_UNDERSAMPLE.value:
                check_float(name=ReadConfig.UNDERSAMPLE_RATIO.value, value=self.meta_dict[ReadConfig.UNDERSAMPLE_RATIO.value])
            if (self.meta_dict[ReadConfig.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTEENN.value.lower()) or (self.meta_dict[ReadConfig.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTE.value.lower()):
                check_float(name=ReadConfig.OVERSAMPLE_RATIO.value, value=self.meta_dict[ReadConfig.OVERSAMPLE_RATIO.value])

            if ReadConfig.CLASS_WEIGHTS.value in self.meta_dict.keys():
                class_weights = self.meta_dict[ReadConfig.CLASS_WEIGHTS.value]
                if class_weights == 'custom':
                    class_weights = self.meta_dict['custom_weights']
                    for k, v in class_weights.items():
                        class_weights[k] = int(v)
                if class_weights == Dtypes.NONE.value:
                    class_weights = None
            else:
                class_weights = None

            if MetaKeys.TRAIN_TEST_SPLIT_TYPE.value in self.meta_dict.keys():
                check_str(name=self.meta_dict[MetaKeys.TRAIN_TEST_SPLIT_TYPE.value], value=self.meta_dict[MetaKeys.TRAIN_TEST_SPLIT_TYPE.value], options=Options.TRAIN_TEST_SPLIT.value)
                self.tt_split_type = self.meta_dict[MetaKeys.TRAIN_TEST_SPLIT_TYPE.value]
            else:
                self.tt_split_type = Methods.SPLIT_TYPE_FRAMES.value

            self.clf_name = self.meta_dict[MetaKeys.CLF_NAME.value]
            self.class_names = ['Not_' + self.clf_name, self.clf_name]
            annotation_cols_to_remove = read_in_all_model_names_to_remove(self.config, self.model_cnt, self.clf_name)
            self.x_y_df = delete_other_annotation_columns(self.data_df_wo_cords, annotation_cols_to_remove)
            self.x_df, self.y_df = split_df_to_x_y(self.x_y_df, self.clf_name)
            self.feature_names = self.x_df.columns
            self.perform_sampling()
            print('MODEL {} settings'.format(str(config_cnt)))
            print_machine_model_information(self.meta_dict)
            print('# {} features.'.format(len(self.feature_names)))

            self.rf_clf = RandomForestClassifier(n_estimators=self.meta_dict[MetaKeys.RF_ESTIMATORS.value],
                                                 max_features=self.meta_dict[MetaKeys.RF_MAX_FEATURES.value],
                                                 n_jobs=-1,
                                                 criterion=self.meta_dict[MetaKeys.CRITERION.value],
                                                 min_samples_leaf=self.meta_dict[MetaKeys.MIN_LEAF.value],
                                                 bootstrap=True,
                                                 verbose=1,
                                                 class_weight=class_weights)
            try:
                self.rf_clf.fit(self.x_train, self.y_train)
            except Exception as e:
                raise ValueError(e, 'ERROR: The model contains a faulty array. This may happen when trying to train a model with 0 examples of the behavior of interest')

            if self.meta_dict[MetaKeys.PERMUTATION_IMPORTANCE.value] in Options.PERFORM_FLAGS.value:
                calc_permutation_importance(self.x_test, self.y_test, self.rf_clf, self.feature_names, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict[MetaKeys.LEARNING_CURVE.value] in Options.PERFORM_FLAGS.value:
                calc_learning_curve(self.x_y_df, self.clf_name, self.meta_dict[MetaKeys.LEARNING_CURVE_K_SPLITS.value], self.meta_dict[MetaKeys.LEARNING_CURVE_DATA_SPLITS.value], self.meta_dict[MetaKeys.TT_SIZE.value], self.rf_clf, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict[MetaKeys.PRECISION_RECALL.value] in Options.PERFORM_FLAGS.value:
                calc_pr_curve(self.rf_clf, self.x_test, self.y_test, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict[MetaKeys.EX_DECISION_TREE.value] in Options.PERFORM_FLAGS.value:
                create_example_dt(self.rf_clf, self.clf_name, self.feature_names, self.class_names, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict[MetaKeys.CLF_REPORT.value] in Options.PERFORM_FLAGS.value:
                create_clf_report(self.rf_clf, self.x_test, self.y_test, self.class_names, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict[MetaKeys.IMPORTANCE_LOG.value] in Options.PERFORM_FLAGS.value:
                create_x_importance_log(self.rf_clf, self.feature_names, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            if self.meta_dict[MetaKeys.IMPORTANCE_BAR_CHART.value] in Options.PERFORM_FLAGS.value:
                create_x_importance_bar_chart(self.rf_clf, self.feature_names, self.clf_name, self.model_dir_out, self.meta_dict[MetaKeys.N_FEATURE_IMPORTANCE_BARS.value], save_file_no=config_cnt)
            if MetaKeys.SHAP_SCORES.value in self.meta_dict.keys():
                if self.meta_dict[MetaKeys.SHAP_SCORES.value] in Options.PERFORM_FLAGS.value:
                    create_shap_log(self.ini_path, self.rf_clf, self.x_train, self.y_train, self.feature_names, self.clf_name, self.meta_dict[MetaKeys.SHAP_PRESENT.value], self.meta_dict[MetaKeys.SHAP_ABSENT.value], self.model_dir_out, save_file_no=config_cnt)
            create_meta_data_csv_training_multiple_models(self.meta_dict, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            save_rf_model(self.rf_clf, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            print('Classifier {} saved in models/validations/model_files directory ...'.format(str(self.clf_name + '_' + str(config_cnt))))
        print('All models and evaluations complete. The models/evaluation files are in models/validations folders')

# test = TrainMultipleModelsFromMeta(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini')
# test.train_models_from_meta()






