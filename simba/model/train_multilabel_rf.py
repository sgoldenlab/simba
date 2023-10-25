import ast
import os
from itertools import product
from typing import Dict, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.enums import ConfigKey, Methods, Options, TagNames
from simba.utils.errors import SamplingError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import read_config_entry


class TrainMultiLabelRandomForestClassifier(ConfigReader, TrainModelMixin):
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
        self.data_df = self.read_all_files_in_folder_mp_futures(
            file_paths=self.target_file_paths,
            file_type=self.file_type,
            classifier_names=self.clf_names,
            raise_bool_clf_error=False,
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
        self.x_df, self.y_df = self.split_df_to_x_y(self.x_y_df, self.clf_name)
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

    def perform_sampling(self):
        if self.split_type == Methods.SPLIT_TYPE_FRAMES.value:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_df, self.y_df, test_size=self.tt_size
            )
        self._check_presence_of_classes_post_sampling()

    def run(self):
        print("Training and evaluating model...")
        self.timer = SimbaTimer(start=True)
        self.rf_clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
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
            )
        if self.generate_precision_recall_curve in Options.PERFORM_FLAGS.value:
            self.calc_pr_curve(
                self.rf_clf, self.x_test, self.y_test, self.clf_name, self.eval_out_path
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


model_trainer = TrainMultiLabelRandomForestClassifier(
    config_path="/Users/simon/Desktop/envs/troubleshooting/multilabel/project_folder/project_config.ini"
)
model_trainer.perform_sampling()
model_trainer.run()
