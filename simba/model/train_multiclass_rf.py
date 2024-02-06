import ast
import json
import os
from itertools import product
from typing import Dict, Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_if_list_contains_values,
                                check_if_valid_input)
from simba.utils.enums import (ConfigKey, Dtypes, Formats, Methods,
                               MLParamKeys, Options, TagNames)
from simba.utils.errors import SamplingError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import read_config_entry, write_df


class TrainMultiClassRandomForestClassifier(ConfigReader, TrainModelMixin):
    def __init__(self, config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        self.read_model_settings_from_config(config=self.config)

        check_if_filepath_list_is_empty(
            filepaths=self.target_file_paths,
            error_msg="Zero annotation files found in project_folder/csv/targets_inserted, cannot create model.",
        )
        print(f"Reading in {len(self.target_file_paths)} annotated files...")
        self.data_df, self.frm_idx = self.read_all_files_in_folder_mp_futures(
            annotations_file_paths=self.target_file_paths,
            file_type=self.file_type,
            classifier_names=self.clf_names,
            raise_bool_clf_error=False,
        )
        self.frm_idx = pd.DataFrame(
            {"VIDEO": list(self.data_df.index), "FRAME_IDX": self.frm_idx}
        )

        self.check_raw_dataset_integrity(df=self.data_df, logs_path=self.logs_path)
        self.data_df_wo_cords = self.drop_bp_cords(df=self.data_df)
        annotation_cols_to_remove = self.read_in_all_model_names_to_remove(
            self.config, self.clf_cnt, self.clf_name
        )
        self.x_y_df = self.delete_other_annotation_columns(
            df=self.data_df_wo_cords,
            annotations_lst=list(annotation_cols_to_remove),
            raise_error=False,
        )
        self.classifier_map = ast.literal_eval(
            read_config_entry(
                self.config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                "classifier_map",
                data_type="str",
            )
        )
        self.under_sample_setting = (
            read_config_entry(
                self.config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.UNDERSAMPLE_SETTING.value,
                data_type=Dtypes.STR.value,
                default_value=Dtypes.NONE.value,
            )
            .lower()
            .strip()
        )
        self.under_sample_ratio = ast.literal_eval(
            read_config_entry(
                self.config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.UNDERSAMPLE_RATIO.value,
                data_type=Dtypes.STR.value,
            )
        )
        self.x_df, self.y_df = self.split_df_to_x_y(self.x_y_df, self.clf_name)
        check_if_list_contains_values(
            data=list(self.y_df),
            values=list(self.classifier_map.keys()),
            name=self.clf_name,
        )

        self.feature_names = self.x_df.columns
        self.check_sampled_dataset_integrity(x_df=self.x_df, y_df=self.y_df)
        print(f"Number of features in dataset: {len(self.feature_names)}")
        for k, v in self.classifier_map.items():
            print(
                f"Number of {v} frames in dataset: {len(self.y_df[self.y_df == k])} ({(len(self.y_df[self.y_df == k]) / len(self.x_df)) * 100}%)"
            )

    def _check_presence_of_classes_post_sampling(self):
        for set, clf_code in product(
            [self.y_train, self.y_test], self.classifier_map.keys()
        ):
            obs_cnt = len(self.y_df[self.y_df == clf_code])
            if obs_cnt == 0:
                raise SamplingError(
                    msg=f"Zero observations of {self.classifier_map[clf_code]} found in the training and/or test set."
                )

    def _perform_sampling(self):
        if self.split_type == Methods.SPLIT_TYPE_FRAMES.value:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_df, self.y_df, test_size=self.tt_size
            )
        if self.under_sample_setting.lower() != Dtypes.NONE.value.lower():
            if self.under_sample_setting == "random undersample multiclass frames":
                self.x_train, self.y_train = self.random_multiclass_frm_sampler(
                    x_df=self.x_train,
                    y_df=self.y_train,
                    target_field=self.clf_name,
                    target_var=self.under_sample_ratio["target_var"],
                    sampling_ratio=self.under_sample_ratio["sampling_ratio"],
                )

            elif self.under_sample_setting == "random undersample multiclass bouts":
                self.x_train, self.y_train = self.random_multiclass_bout_sampler(
                    x_df=self.x_train,
                    y_df=self.y_train,
                    target_field=self.clf_name,
                    target_var=self.under_sample_ratio["target_var"],
                    sampling_ratio=self.under_sample_ratio["sampling_ratio"],
                )
            else:
                raise SamplingError(
                    msg=f"Under sample setting {self.under_sample_setting} not recognized. Options: [None, random undersample multiclass frames, random undersample multiclass bouts]",
                    source=self.__class__.__name__,
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
                f"Frame index of train and test set saved in {self.eval_out_path} directory..."
            )

    def run(self):
        print("Training and evaluating model...")
        self._perform_sampling()
        self.timer = SimbaTimer(start=True)
        self.rf_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.rf_max_depth,
            max_features=self.max_features,
            n_jobs=-1,
            criterion=self.criterion,
            min_samples_leaf=self.min_sample_leaf,
            bootstrap=True,
            verbose=1,
            class_weight=self.class_weights,
        )
        self.rf_clf = self.clf_fit(clf=self.rf_clf, x_df=self.x_df, y_df=self.y_df)
        if self.compute_permutation_importance in Options.PERFORM_FLAGS.value:
            self.calc_permutation_importance(
                self.x_test,
                self.y_test,
                self.rf_clf,
                self.feature_names,
                self.clf_name,
                self.eval_out_path,
            )
        if self.generate_learning_curve in Options.PERFORM_FLAGS.value:
            self.calc_learning_curve(
                x_y_df=self.x_y_df,
                clf_name=self.clf_name,
                shuffle_splits=self.shuffle_splits,
                dataset_splits=self.dataset_splits,
                tt_size=self.tt_size,
                rf_clf=self.rf_clf,
                save_dir=self.eval_out_path,
                multiclass=True,
            )
        if self.generate_precision_recall_curve in Options.PERFORM_FLAGS.value:
            self.calc_pr_curve(
                rf_clf=self.rf_clf,
                x_df=self.x_test,
                y_df=self.y_test,
                clf_name=self.clf_name,
                save_dir=self.eval_out_path,
                multiclass=True,
                classifier_map=self.classifier_map,
            )

        if self.generate_example_decision_tree in Options.PERFORM_FLAGS.value:
            self.create_example_dt(
                self.rf_clf,
                self.clf_name,
                self.feature_names,
                list(self.classifier_map.values()),
                self.eval_out_path,
            )
        if self.generate_classification_report in Options.PERFORM_FLAGS.value:
            self.create_clf_report(
                self.rf_clf,
                self.x_test,
                self.y_test,
                list(self.classifier_map.values()),
                self.eval_out_path,
                clf_name=self.clf_name,
            )
        if self.generate_features_importance_log in Options.PERFORM_FLAGS.value:
            self.create_x_importance_log(
                self.rf_clf, self.feature_names, self.clf_name, self.eval_out_path
            )
        if self.generate_features_importance_bar_graph in Options.PERFORM_FLAGS.value:
            self.create_x_importance_bar_chart(
                self.rf_clf,
                self.feature_names,
                self.clf_name,
                self.eval_out_path,
                self.feature_importance_bars,
            )
        if self.generate_example_decision_tree_fancy in Options.PERFORM_FLAGS.value:
            self.dviz_classification_visualization(
                self.x_train,
                self.y_train,
                self.clf_name,
                list(self.classifier_map.values()),
                self.eval_out_path,
            )

        if self.generate_shap_scores in Options.PERFORM_FLAGS.value:
            if not self.shap_multiprocess in Options.PERFORM_FLAGS.value:
                self.create_shap_log(
                    ini_file_path=self.config_path,
                    rf_clf=self.rf_clf,
                    x_df=self.x_train,
                    y_df=self.y_train,
                    x_names=self.feature_names,
                    clf_name=self.clf_name,
                    cnt_present=self.shap_target_present_cnt,
                    cnt_absent=self.shap_target_absent_cnt,
                    save_it=self.shap_save_n,
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
                    cnt_present=self.shap_target_present_cnt,
                    cnt_absent=self.shap_target_absent_cnt,
                    save_path=self.eval_out_path,
                )

            if self.compute_partial_dependency in Options.PERFORM_FLAGS.value:
                self.partial_dependence_calculator(
                    clf=self.rf_clf,
                    x_df=self.x_train,
                    clf_name=self.clf_name,
                    save_dir=self.eval_out_path,
                )

            if self.save_meta_data in Options.PERFORM_FLAGS.value:
                meta_data_lst = [
                    self.clf_name,
                    self.criterion,
                    self.max_features,
                    self.min_sample_leaf,
                    self.n_estimators,
                    self.compute_permutation_importance,
                    self.generate_classification_report,
                    self.generate_example_decision_tree,
                    self.generate_features_importance_bar_graph,
                    self.generate_features_importance_log,
                    self.generate_precision_recall_curve,
                    self.save_meta_data,
                    self.generate_learning_curve,
                    self.dataset_splits,
                    self.shuffle_splits,
                    self.feature_importance_bars,
                    self.over_sample_ratio,
                    self.over_sample_setting,
                    self.tt_size,
                    self.split_type,
                    self.under_sample_ratio,
                    self.under_sample_setting,
                    str(self.class_weights),
                    self.rf_max_depth,
                ]

                self.create_meta_data_csv_training_one_model(
                    meta_data_lst, self.clf_name, self.eval_out_path
                )

    def save_model(self) -> None:
        """
        Method for saving pickled RF model. The model is saved in the `models/generated_models` directory of the SimBA project tree.
        """
        self.timer.stop_timer()
        self.save_rf_model(self.rf_clf, self.clf_name, self.model_dir_out)
        stdout_success(
            msg=f"Classifier {self.clf_name} saved in models/generated_models directory",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )
        stdout_success(
            msg=f"Evaluation files are in models/generated_models/model_evaluations folders",
            source=self.__class__.__name__,
        )


# model_trainer = TrainMultiClassRandomForestClassifier(config_path='/Users/simon/Desktop/envs/troubleshooting/multilabel/project_folder/project_config.ini')
# model_trainer.run()
# model_trainer.save_model()
