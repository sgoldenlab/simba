__author__ = "Simon Nilsson"

import ast
import os
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import check_if_filepath_list_is_empty, check_int
from simba.utils.enums import (ConfigKey, Dtypes, Formats, Methods,
                               MLParamKeys, Options)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import read_config_entry, write_df


class TrainRandomForestClassifier(ConfigReader, TrainModelMixin):
    """
    Train a single random forest model using hyperparameter setting and evaluation methods
    stored within the SimBA project config .ini file (``global environment``).

    :param str config_path: path to SimBA project config file in Configparser format

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model>`_

    :example:
    >>> model_trainer = TrainRandomForestClassifier(config_path='MyConfigPath')
    >>> model_trainer.run()
    >>> model_trainer.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        TrainModelMixin.__init__(self)
        self.read_model_settings_from_config(config=self.config)
        check_if_filepath_list_is_empty(filepaths=self.target_file_paths, error_msg=f"Zero annotation files found in directory {self.targets_folder}, cannot create model.")
        print(f"Reading in {len(self.target_file_paths)} annotated files...")
        self.data_df, self.frm_idx = self.read_all_files_in_folder_mp_futures(self.target_file_paths, self.file_type, [self.clf_name])
        self.frm_idx = pd.DataFrame({"VIDEO": list(self.data_df.index), "FRAME_IDX": self.frm_idx})
        self.data_df = self.check_raw_dataset_integrity(df=self.data_df, logs_path=self.logs_path)
        self.data_df_wo_cords = self.drop_bp_cords(df=self.data_df)
        annotation_cols_to_remove = self.read_in_all_model_names_to_remove(self.config, self.clf_cnt, self.clf_name)
        self.x_y_df = self.delete_other_annotation_columns(self.data_df_wo_cords, list(annotation_cols_to_remove))
        self.class_names = ["Not_" + self.clf_name, self.clf_name]
        self.x_df, self.y_df = self.split_df_to_x_y(self.x_y_df, self.clf_name)
        self.feature_names = self.x_df.columns
        self.check_sampled_dataset_integrity(x_df=self.x_df, y_df=self.y_df)
        print(f"Number of features in dataset: {len(self.x_df.columns)}")
        print(f"Number of {self.clf_name} frames in dataset: {self.y_df.sum()} ({str(round(self.y_df.sum() / len(self.y_df), 4) * 100)}%)")

    def perform_sampling(self):
        """
        Method for sampling data for training and testing, and perform over and under-sampling of the training sets
        as indicated within the SimBA project config.
        """

        if self.split_type == Methods.SPLIT_TYPE_FRAMES.value:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_df, self.y_df, test_size=self.tt_size
            )
        elif self.split_type == Methods.SPLIT_TYPE_BOUTS.value:
            self.x_train, self.x_test, self.y_train, self.y_test = (
                self.bout_train_test_splitter(
                    x_df=self.x_df, y_df=self.y_df, test_size=self.tt_size
                )
            )
        if self.under_sample_setting == Methods.RANDOM_UNDERSAMPLE.value.lower():
            self.x_train, self.y_train = self.random_undersampler(
                self.x_train, self.y_train, float(self.under_sample_ratio)
            )
        if self.over_sample_setting == Methods.SMOTEENN.value.lower():
            self.x_train, self.y_train = self.smoteen_oversampler(
                self.x_train, self.y_train, float(self.over_sample_ratio)
            )
        elif self.over_sample_setting == Methods.SMOTE.value.lower():
            self.x_train, self.y_train = self.smote_oversampler(
                self.x_train, self.y_train, float(self.over_sample_ratio)
            )

        if self.save_train_test_frm_info:
            train_data = self.frm_idx[
                self.frm_idx.index.isin(self.x_train.index)
            ].set_index("VIDEO")
            test_data = self.frm_idx[
                self.frm_idx.index.isin(self.x_test.index)
            ].set_index("VIDEO")
            write_df(
                df=train_data,
                file_type=Formats.CSV.value,
                save_path=os.path.join(self.eval_out_path, "train_idx.csv"),
            )
            write_df(
                df=test_data,
                file_type=Formats.CSV.value,
                save_path=os.path.join(self.eval_out_path, "test_idx.csv"),
            )
            print(
                f"Frame index for the train and test sets saved in {self.eval_out_path} directory..."
            )

    def run(self):
        """
        Method for training single random forest model.
        """
        print("Training and evaluating model...")
        self.timer = SimbaTimer(start=True)
        self.perform_sampling()
        if self.algo == "RF":
            n_estimators = read_config_entry(self.config,ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,MLParamKeys.RF_ESTIMATORS.value,data_type=Dtypes.INT.value)
            max_features = read_config_entry(self.config,ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,MLParamKeys.RF_MAX_FEATURES.value,data_type=Dtypes.STR.value)
            if max_features == "None":
                max_features = None
            criterion = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.RF_CRITERION.value, data_type=Dtypes.STR.value, options=Options.CLF_CRITERION.value)
            min_sample_leaf = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.MIN_LEAF.value, data_type=Dtypes.INT.value)
            compute_permutation_importance = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.PERMUTATION_IMPORTANCE.value, data_type=Dtypes.STR.value, default_value=False)
            generate_learning_curve = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.LEARNING_CURVE.value, data_type=Dtypes.STR.value, default_value=False)
            generate_precision_recall_curve = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.PRECISION_RECALL.value, data_type=Dtypes.STR.value, default_value=False)
            generate_example_decision_tree = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.EX_DECISION_TREE.value, data_type=Dtypes.STR.value, default_value=False)
            generate_classification_report = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLF_REPORT.value, data_type=Dtypes.STR.value, default_value=False)
            generate_features_importance_log = read_config_entry(self.config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.IMPORTANCE_LOG.value, data_type=Dtypes.STR.value, default_value=False)
            generate_features_importance_bar_graph = read_config_entry(self.config,ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,MLParamKeys.IMPORTANCE_LOG.value,data_type=Dtypes.STR.value,default_value=False)
            generate_example_decision_tree_fancy = read_config_entry(self.config,ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,MLParamKeys.EX_DECISION_TREE_FANCY.value,data_type=Dtypes.STR.value,default_value=False)
            generate_shap_scores = read_config_entry(self.config,ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,MLParamKeys.SHAP_SCORES.value,data_type=Dtypes.STR.value,default_value=False)
            save_meta_data = read_config_entry(self.config,ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,MLParamKeys.RF_METADATA.value,data_type=Dtypes.STR.value,default_value=False)
            compute_partial_dependency = read_config_entry(self.config,ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,MLParamKeys.PARTIAL_DEPENDENCY.value,data_type=Dtypes.STR.value,default_value=False)
            if self.config.has_option(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLASS_WEIGHTS.value):
                class_weights = read_config_entry(self.config,
                                                  ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                  MLParamKeys.CLASS_WEIGHTS.value,
                                                  data_type=Dtypes.STR.value,
                                                  default_value=Dtypes.NONE.value)
                if class_weights == "custom":
                    class_weights = ast.literal_eval(
                        read_config_entry(
                            self.config,
                            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                            MLParamKeys.CLASS_CUSTOM_WEIGHTS.value,
                            data_type=Dtypes.STR.value,
                        )
                    )
                    for k, v in class_weights.items():
                        class_weights[k] = int(v)
                if class_weights == Dtypes.NONE.value:
                    class_weights = None
            else:
                class_weights = None

            if generate_learning_curve in Options.PERFORM_FLAGS.value:
                shuffle_splits = read_config_entry(
                    self.config,
                    ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                    MLParamKeys.LEARNING_CURVE_K_SPLITS.value,
                    data_type=Dtypes.INT.value,
                    default_value=Dtypes.NAN.value,
                )
                dataset_splits = read_config_entry(
                    self.config,
                    ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                    MLParamKeys.LEARNING_DATA_SPLITS.value,
                    data_type=Dtypes.INT.value,
                    default_value=Dtypes.NAN.value,
                )
                check_int(
                    name=MLParamKeys.LEARNING_CURVE_K_SPLITS.value, value=shuffle_splits
                )
                check_int(
                    name=MLParamKeys.LEARNING_DATA_SPLITS.value, value=dataset_splits
                )
            else:
                shuffle_splits, dataset_splits = Dtypes.NAN.value, Dtypes.NAN.value
            if generate_features_importance_bar_graph in Options.PERFORM_FLAGS.value:
                feature_importance_bars = read_config_entry(
                    self.config,
                    ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                    MLParamKeys.IMPORTANCE_BARS_N.value,
                    Dtypes.INT.value,
                    Dtypes.NAN.value,
                )
                check_int(
                    name=MLParamKeys.IMPORTANCE_BARS_N.value,
                    value=feature_importance_bars,
                    min_value=1,
                )
            else:
                feature_importance_bars = Dtypes.NAN.value
            (
                shap_target_present_cnt,
                shap_target_absent_cnt,
                shap_save_n,
                shap_multiprocess,
            ) = (None, None, None, None)
            if generate_shap_scores in Options.PERFORM_FLAGS.value:
                shap_target_present_cnt = read_config_entry(
                    self.config,
                    ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                    MLParamKeys.SHAP_PRESENT.value,
                    data_type=Dtypes.INT.value,
                    default_value=0,
                )
                shap_target_absent_cnt = read_config_entry(
                    self.config,
                    ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                    MLParamKeys.SHAP_ABSENT.value,
                    data_type=Dtypes.INT.value,
                    default_value=0,
                )
                shap_save_n = read_config_entry(
                    self.config,
                    ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                    MLParamKeys.SHAP_SAVE_ITERATION.value,
                    data_type=Dtypes.STR.value,
                    default_value=Dtypes.NONE.value,
                )
                shap_multiprocess = read_config_entry(
                    self.config,
                    ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                    MLParamKeys.SHAP_MULTIPROCESS.value,
                    data_type=Dtypes.STR.value,
                    default_value="False",
                )
                try:
                    shap_save_n = int(shap_save_n)
                except ValueError:
                    shap_save_n = shap_target_present_cnt + shap_target_absent_cnt
                check_int(
                    name=MLParamKeys.SHAP_PRESENT.value, value=shap_target_present_cnt
                )
                check_int(
                    name=MLParamKeys.SHAP_ABSENT.value, value=shap_target_absent_cnt
                )



            self.rf_clf = self.clf_define(n_estimators=n_estimators, max_depth=self.rf_max_depth, max_features=max_features, n_jobs=-1, criterion=criterion, min_samples_leaf=min_sample_leaf, verbose=1, class_weight=class_weights)

            print(f"Fitting {self.clf_name} model...")
            self.rf_clf = self.clf_fit(
                clf=self.rf_clf, x_df=self.x_train, y_df=self.y_train
            )

            if compute_permutation_importance in Options.PERFORM_FLAGS.value:
                self.calc_permutation_importance(
                    self.x_test,
                    self.y_test,
                    self.rf_clf,
                    self.feature_names,
                    self.clf_name,
                    self.eval_out_path,
                )
            if generate_learning_curve in Options.PERFORM_FLAGS.value:
                self.calc_learning_curve(
                    x_y_df=self.x_y_df,
                    clf_name=self.clf_name,
                    shuffle_splits=shuffle_splits,
                    dataset_splits=dataset_splits,
                    tt_size=self.tt_size,
                    rf_clf=self.rf_clf,
                    save_dir=self.eval_out_path,
                )

            if generate_precision_recall_curve in Options.PERFORM_FLAGS.value:
                self.calc_pr_curve(
                    self.rf_clf,
                    self.x_test,
                    self.y_test,
                    self.clf_name,
                    self.eval_out_path,
                )
            if generate_example_decision_tree in Options.PERFORM_FLAGS.value:
                self.create_example_dt(
                    self.rf_clf,
                    self.clf_name,
                    self.feature_names,
                    self.class_names,
                    self.eval_out_path,
                )
            if generate_classification_report in Options.PERFORM_FLAGS.value:
                self.create_clf_report(
                    self.rf_clf,
                    self.x_test,
                    self.y_test,
                    self.class_names,
                    self.eval_out_path,
                )
            if generate_features_importance_log in Options.PERFORM_FLAGS.value:
                self.create_x_importance_log(
                    self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path
                )
            if generate_features_importance_bar_graph in Options.PERFORM_FLAGS.value:
                self.create_x_importance_bar_chart(
                    self.rf_clf,
                    self.feature_names,
                    self.clf_name,
                    self.eval_out_path,
                    feature_importance_bars,
                )
            if generate_example_decision_tree_fancy in Options.PERFORM_FLAGS.value:
                self.dviz_classification_visualization(
                    self.x_train,
                    self.y_train,
                    self.clf_name,
                    self.class_names,
                    self.eval_out_path,
                )
            if generate_shap_scores in Options.PERFORM_FLAGS.value:
                if not shap_multiprocess in Options.PERFORM_FLAGS.value:
                    self.create_shap_log(
                        ini_file_path=self.config_path,
                        rf_clf=self.rf_clf,
                        x_df=self.x_train,
                        y_df=self.y_train,
                        x_names=self.feature_names,
                        clf_name=self.clf_name,
                        cnt_present=shap_target_present_cnt,
                        cnt_absent=shap_target_absent_cnt,
                        save_it=shap_save_n,
                        save_path=self.eval_out_path,
                    )
                else:
                    self.create_shap_log_mp(
                        ini_file_path=self.config_path,
                        rf_clf=self.rf_clf,
                        x_df=self.x_train,
                        y_df=self.y_train,
                        x_names=self.feature_names,
                        clf_name=self.clf_name,
                        cnt_present=shap_target_present_cnt,
                        cnt_absent=shap_target_absent_cnt,
                        save_path=self.eval_out_path,
                    )

            if compute_partial_dependency in Options.PERFORM_FLAGS.value:
                self.partial_dependence_calculator(
                    clf=self.rf_clf,
                    x_df=self.x_train,
                    clf_name=self.clf_name,
                    save_dir=self.eval_out_path,
                )

            if save_meta_data in Options.PERFORM_FLAGS.value:
                meta_data_lst = [
                    self.clf_name,
                    criterion,
                    max_features,
                    min_sample_leaf,
                    n_estimators,
                    compute_permutation_importance,
                    generate_classification_report,
                    generate_example_decision_tree,
                    generate_features_importance_bar_graph,
                    generate_features_importance_log,
                    generate_precision_recall_curve,
                    save_meta_data,
                    generate_learning_curve,
                    dataset_splits,
                    shuffle_splits,
                    feature_importance_bars,
                    self.over_sample_ratio,
                    self.over_sample_setting,
                    self.tt_size,
                    self.split_type,
                    self.under_sample_ratio,
                    self.under_sample_setting,
                    str(class_weights),
                    self.rf_max_depth,
                ]

                self.create_meta_data_csv_training_one_model(
                    meta_data_lst, self.clf_name, self.eval_out_path
                )

    def save(self) -> None:
        """
        Method for saving pickled RF model. The model is saved in the `models/generated_models` directory
        of the SimBA project tree.
        """

        self.timer.stop_timer()
        if not os.listdir(self.model_dir_out):
            os.makedirs(self.model_dir_out)
        self.save_rf_model(self.rf_clf, self.clf_name, self.model_dir_out)
        stdout_success(msg=f"Classifier {self.clf_name} saved in models/generated_models directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)
        stdout_success(msg=f"Evaluation files are in models/generated_models/model_evaluations folders", source=self.__class__.__name__)

#
# test = TrainRandomForestClassifier(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# test.run()
# test.save()


# test = TrainRandomForestClassifier(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini')
# test.run()
# test.save()


# test = TrainRandomForestClassifier(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.perform_sampling()
# test.train_model()
# test.save_model()

# test = TrainSingleModel(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini')
# test.perform_sampling()
# test.train_model()
# test.save_model()


# test = TrainSingleModel(config_path='/Users/simon/Desktop/envs/troubleshooting/prueba/project_folder/project_config.ini')
# test.perform_sampling()
# test.train_model()
# test.save_model()

# test = TrainSingleModel(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini')
# test.perform_sampling()
# test.train_model()
# test.save_model()

# test = TrainSingleModel(config_path='/Users/simon/Desktop/envs/troubleshooting/Lucas/project_folder/project_config.ini')
# test.perform_sampling()
# test.train_model()
# test.save`_model()
