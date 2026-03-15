__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from typing import Optional, Union
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import train_test_split

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import check_if_filepath_list_is_empty, check_str, check_if_dir_exists
from simba.utils.enums import (ConfigKey, Dtypes, Formats, Methods,
                               MLParamKeys, Options)
from simba.utils.errors import InvalidInputError, NoDataError
from simba.utils.printing import stdout_information, stdout_success
from simba.utils.read_write import (read_config_entry, read_simba_meta_files,
                                    write_df, find_files_of_filetypes_in_directory)


class GridSearchRandomForestClassifier(ConfigReader, TrainModelMixin):
    """
    Train one or more random-forest classifiers from SimBA meta-config files.

    Reads model hyperparameters and sampling settings from meta files in ``project_folder/configs`` and trains one model per valid meta file. Training data is loaded from annotated target files and saved models plus evaluation artifacts are written to the configured output directory.

    .. note::
       Searches the SimBA project ``project_folder/configs`` directory for meta files and builds one model per valid config file. `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#step-7-train-machine-model>`__.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in ConfigParser format.
    :param Optional[str] feature_subset_suffix: Optional suffix used to subset feature columns before training. If set, only feature columns ending with this suffix are retained.
    :param Optional[Union[str, os.PathLike]] target_dir: Optional directory with annotated target files (CSV or parquet, matching project file type). If None, project default targets directory is used.
    :param Optional[Union[str, os.PathLike]] save_dir: Optional directory where trained models and evaluation artifacts are saved. If None, defaults to ``<model_dir>/validations`` from project configuration.

    :example:
    >>> _ = GridSearchRandomForestClassifier(config_path='MyConfigPath').run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 feature_subset_suffix: Optional[str] = None,
                 target_dir: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None):

        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        TrainModelMixin.__init__(self)
        if target_dir is not None:
            self.target_file_paths = find_files_of_filetypes_in_directory(directory=target_dir, extensions=self.file_type, raise_error=True, sort_alphabetically=True)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', raise_error=True)
            self.model_dir_out = deepcopy(save_dir)
        else:
            self.model_dir_out = os.path.join(read_config_entry(self.config,ConfigKey.SML_SETTINGS.value,ConfigKey.MODEL_DIR.value,data_type=Dtypes.STR.value), "validations")
        self.bp_config = read_config_entry(config=self.config, section=ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, option=ConfigKey.POSE_SETTING.value, default_value='user_defined', data_type=Dtypes.STR.value)
        if not os.path.exists(self.model_dir_out): os.makedirs(self.model_dir_out)
        check_if_filepath_list_is_empty(filepaths=self.target_file_paths, error_msg=f"Zero data files found in {self.targets_folder}, cannot create models.")
        if not os.path.exists(self.configs_meta_dir): os.makedirs(self.configs_meta_dir)
        if feature_subset_suffix is not None: check_str(name=f'{self.__class__.__name__} feature_subset_suffix', value=feature_subset_suffix, allow_blank=False, raise_error=True)
        self.meta_file_lst = sorted(read_simba_meta_files(self.configs_meta_dir))
        stdout_information(msg=f"Reading in {len(self.target_file_paths)} annotated files found in the {self.targets_folder} directory...")
        self.data_df, self.frm_idx = self.read_all_files_in_folder_mp_futures(self.target_file_paths, self.file_type)
        self.frm_idx = pd.DataFrame({"VIDEO": list(self.data_df.index), "FRAME_IDX": self.frm_idx})
        _ = self.check_raw_dataset_integrity(self.data_df, logs_path=self.logs_path)
        self.data_df, self.feature_subset_suffix = self.drop_bp_cords(df=self.data_df), feature_subset_suffix

    def perform_sampling(self, meta_dict: dict):
        if (meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value] == Methods.SPLIT_TYPE_FRAMES.value):
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_df, self.y_df, test_size=meta_dict["train_test_size"])
        elif (meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value] == Methods.SPLIT_TYPE_BOUTS.value):
            self.x_train, self.x_test, self.y_train, self.y_test = (self.bout_train_test_splitter(x_df=self.x_df.reset_index(drop=True), y_df=self.y_df.reset_index(drop=True), test_size=meta_dict["train_test_size"]))
        else:
            raise InvalidInputError(msg=f"{meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value]} is not recognized as a valid SPLIT TYPE (OPTIONS: FRAMES, BOUTS).", source=self.__class__.__name__)
        if (meta_dict[MLParamKeys.UNDERSAMPLE_SETTING.value].lower() == Methods.RANDOM_UNDERSAMPLE.value):
            self.x_train, self.y_train = self.random_undersampler(self.x_train,self.y_train,meta_dict[MLParamKeys.UNDERSAMPLE_RATIO.value])
        if (meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTEENN.value.lower()):
            self.x_train, self.y_train = self.smoteen_oversampler(self.x_train,self.y_train,meta_dict[MLParamKeys.OVERSAMPLE_RATIO.value])
        elif (meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTE.value.lower()):
            self.x_train, self.y_train = self.smote_oversampler(self.x_train,self.y_train,meta_dict[MLParamKeys.OVERSAMPLE_RATIO.value])

        if meta_dict[MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value]:
            train_data = self.frm_idx[self.frm_idx.index.isin(self.x_train.index)].set_index("VIDEO")
            test_data = self.frm_idx[self.frm_idx.index.isin(self.x_test.index)].set_index("VIDEO")
            write_df(df=train_data, file_type=Formats.CSV.value, save_path=os.path.join(self.model_dir_out, f"train_idx_model_{self.config_cnt}.csv"))
            write_df(df=test_data, file_type=Formats.CSV.value, save_path=os.path.join(self.model_dir_out, f"test_idx_{self.config_cnt}.csv"))
            stdout_information(msg=f"Frame index for train and test set in model {self.config_cnt} saved in {self.model_dir_out} directory...")

    def run(self):
        self.meta_dicts = self.check_validity_of_meta_files(data_df=self.data_df, meta_file_paths=self.meta_file_lst)
        if len(self.meta_dicts.keys()) == 0:
            raise NoDataError(msg=f"No valid hyper-parameter config files found in {self.configs_meta_dir}", source=self.__class__.__name__)
        for config_cnt, meta_dict in self.meta_dicts.items():
            self.config_cnt = config_cnt
            self.clf_name = meta_dict[MLParamKeys.CLASSIFIER_NAME.value]
            stdout_information(msg=f"Training model {config_cnt+1}/{len(self.meta_dicts.keys())} ({meta_dict[MLParamKeys.CLASSIFIER_NAME.value]})...")
            self.class_names = [f"Not_{meta_dict[MLParamKeys.CLASSIFIER_NAME.value]}", meta_dict[MLParamKeys.CLASSIFIER_NAME.value]]
            annotation_cols_to_remove = self.read_in_all_model_names_to_remove(self.config, self.clf_cnt, meta_dict[MLParamKeys.CLASSIFIER_NAME.value])
            self.x_y_df = self.delete_other_annotation_columns(self.data_df, annotation_cols_to_remove, raise_error=False)
            self.x_df, self.y_df = self.split_df_to_x_y(self.x_y_df, meta_dict[MLParamKeys.CLASSIFIER_NAME.value])
            if self.feature_subset_suffix is not None:
                self.x_df = self.x_df[[x for x in self.x_df.columns if x.endswith(self.feature_subset_suffix)]]
                if len(list(self.x_df.columns)) < 1:
                    raise InvalidInputError(msg=f'Cannot use feature_subset_suffix {self.feature_subset_suffix}: It produces zero columns for training.', source=self.__class__.__name__)
            self.feature_names = self.x_df.columns
            self.check_sampled_dataset_integrity(x_df=self.x_df, y_df=self.y_df)
            self.perform_sampling(meta_dict=meta_dict)
            stdout_information(msg=f"MODEL {config_cnt+1} settings")
            self.print_machine_model_information(meta_dict)
            stdout_information(msg=f"# {len(self.feature_names)} features.")
            self.rf_clf = self.clf_define(n_estimators=meta_dict[MLParamKeys.RF_ESTIMATORS.value],
                                          max_depth=meta_dict[MLParamKeys.RF_MAX_DEPTH.value],
                                          max_features=meta_dict[MLParamKeys.RF_MAX_FEATURES.value],
                                          n_jobs=-1, criterion=meta_dict[MLParamKeys.RF_CRITERION.value],
                                          min_samples_leaf=meta_dict[MLParamKeys.MIN_LEAF.value],
                                          bootstrap=True,
                                          verbose=1,
                                          class_weight=meta_dict[MLParamKeys.CLASS_WEIGHTS.value],
                                          cuda=meta_dict[MLParamKeys.CUDA.value])
            stdout_information(msg=f"Fitting {self.clf_name} model...(follow progress in OS terminal)")
            self.rf_clf = self.clf_fit(clf=self.rf_clf, x_df=self.x_train, y_df=self.y_train, selected_feature_names=None)
            if (meta_dict[MLParamKeys.PERMUTATION_IMPORTANCE.value] in Options.PERFORM_FLAGS.value):
                self.calc_permutation_importance(x_test=self.x_test,
                                                 y_test=self.y_test,
                                                 clf=self.rf_clf,
                                                 feature_names=self.feature_names,
                                                 clf_name=self.clf_name,
                                                 save_dir=self.model_dir_out,
                                                 save_file_no=config_cnt,
                                                 plot=True)
            if (meta_dict[MLParamKeys.LEARNING_CURVE.value] in Options.PERFORM_FLAGS.value):
                self.calc_learning_curve(self.x_y_df, self.clf_name, meta_dict[MLParamKeys.LEARNING_CURVE_K_SPLITS.value], meta_dict[MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value], meta_dict[MLParamKeys.TT_SIZE.value], self.rf_clf, self.model_dir_out, save_file_no=config_cnt)
            if (meta_dict[MLParamKeys.PRECISION_RECALL.value] in Options.PERFORM_FLAGS.value):
                self.calc_pr_curve(rf_clf=self.rf_clf, x_df=self.x_test, y_df=self.y_test, clf_name=self.clf_name, save_dir=self.model_dir_out, save_file_no=config_cnt)
            if (meta_dict[MLParamKeys.EX_DECISION_TREE.value] in Options.PERFORM_FLAGS.value):
                self.create_example_dt(self.rf_clf, self.clf_name, self.feature_names, self.class_names, self.model_dir_out, save_file_no=config_cnt)
            if meta_dict[MLParamKeys.CLF_REPORT.value] in Options.PERFORM_FLAGS.value:
                self.create_clf_report(self.rf_clf, self.x_test, self.y_test, self.class_names, self.model_dir_out, save_file_no=config_cnt)
            if (meta_dict[MLParamKeys.IMPORTANCE_LOG.value] in Options.PERFORM_FLAGS.value):
                self.create_x_importance_log(rf_clf=self.rf_clf, x_names=self.feature_names,clf_name=self.clf_name, save_dir=self.model_dir_out, save_file_no=config_cnt)
            if (meta_dict[MLParamKeys.IMPORTANCE_BAR_CHART.value] in Options.PERFORM_FLAGS.value):
                self.create_x_importance_bar_chart(rf_clf=self.rf_clf, x_names=self.feature_names, clf_name=self.clf_name, save_dir=self.model_dir_out, n_bars=meta_dict[MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value], save_file_no=config_cnt)

            if MLParamKeys.SHAP_SCORES.value in meta_dict.keys():
                save_n = (meta_dict[MLParamKeys.SHAP_PRESENT.value] + meta_dict[MLParamKeys.SHAP_ABSENT.value])
                shap_plot = self.bp_config in {'14', '16'}
                shap_multiprocess = False
                if MLParamKeys.SHAP_SAVE_ITERATION.value in meta_dict.keys():
                    try:
                        save_n = int(meta_dict[MLParamKeys.SHAP_SAVE_ITERATION.value])
                    except ValueError:
                        save_n = (meta_dict[MLParamKeys.SHAP_PRESENT.value] + meta_dict[MLParamKeys.SHAP_ABSENT.value])
                if MLParamKeys.SHAP_MULTIPROCESS.value in meta_dict.keys():
                    shap_multiprocess = meta_dict[MLParamKeys.SHAP_MULTIPROCESS.value]
                if (meta_dict[MLParamKeys.SHAP_SCORES.value] in Options.PERFORM_FLAGS.value):
                    if not shap_multiprocess in Options.PERFORM_FLAGS.value:
                        self.create_shap_log(rf_clf=self.rf_clf,
                                             x=self.x_train,
                                             y=self.y_train,
                                             x_names=list(self.feature_names),
                                             clf_name=self.clf_name,
                                             cnt_present=meta_dict[MLParamKeys.SHAP_PRESENT.value],
                                             cnt_absent=meta_dict[MLParamKeys.SHAP_ABSENT.value],
                                             verbose=True,
                                             plot=shap_plot,
                                             save_it=save_n,
                                             save_dir=self.model_dir_out)
                    else:
                        self.create_shap_log_concurrent_mp(rf_clf=self.rf_clf,
                                                           x=self.x_train,
                                                           y=self.y_train,
                                                           x_names=list(self.feature_names),
                                                           clf_name=self.clf_name,
                                                           cnt_present=meta_dict[MLParamKeys.SHAP_PRESENT.value],
                                                           cnt_absent=meta_dict[MLParamKeys.SHAP_ABSENT.value],
                                                           save_dir=self.model_dir_out,
                                                           save_file_suffix=config_cnt,
                                                           plot=shap_plot)

            if MLParamKeys.PARTIAL_DEPENDENCY.value in meta_dict.keys():
                if (meta_dict[MLParamKeys.PARTIAL_DEPENDENCY.value] in Options.PERFORM_FLAGS.value):
                    self.partial_dependence_calculator(clf=self.rf_clf, x_df=self.x_train, clf_name=self.clf_name, save_dir=self.model_dir_out)
            self.create_meta_data_csv_training_multiple_models(meta_dict, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            self.save_rf_model(self.rf_clf, self.clf_name, self.model_dir_out, save_file_no=config_cnt)
            stdout_information(msg=f"Classifier {self.clf_name}_{config_cnt} saved in directory {self.model_dir_out} ...")
        stdout_success(msg=f"All {len(list(self.meta_dicts.keys()))} model(s) and their evaluations complete. The models/evaluation files are saved in {self.model_dir_out} directory", source=self.__class__.__name__)


#
# test = GridSearchRandomForestClassifier(config_path=r"F:\troubleshooting\sophiaa\project_folder\project_config.ini", feature_subset_suffix='_animal_1', save_dir=r'F:\troubleshooting\sophiaa\models\test')
# test.run()


#
# test = GridSearchRandomForestClassifier(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# test.run()


# test = GridSearchRandomForestClassifier(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# test.run()


# test = GridSearchRandomForestClassifier(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# test.run()


#
# test = GridSearchRandomForestClassifier(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# test.run()

#
# test = GridSearchRandomForestClassifier(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# test.run()

# test = GridSearchRandomForestClassifier(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/open_field_below/project_folder/project_config.ini')
# test.run()

# test = GridSearchRandomForestClassifier(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run()

# test = GridSearchRandomForestClassifier(config_path='/Users/simon/Desktop/envs/troubleshooting/testing_enum/project_folder/project_config.ini')
# test.run()


# test = GridSearchRandomForestClassifier(config_path='/Users/simon/Desktop/envs/troubleshooting/sophie/project_folder/project_config.ini')
# test.run()

# test = TrainMultipleModelsFromMeta(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini')
# test.run()

# test = GridSearchRandomForestClassifier(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run()

# test = TrainMultipleModelsFromMeta(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run()

# file_paths = glob.glob('/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/targets_inserted' + '/*.csv')
# df = read_all_files_in_folder_mp(file_paths=file_paths, file_type='csv')
# check_dataset_integrity(df=df)

#
# test = GridSearchRandomForestClassifier(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini')
# test.run()
