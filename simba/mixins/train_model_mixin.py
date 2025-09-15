__author__ = "Simon Nilsson"

import warnings

warnings.filterwarnings("ignore")
import ast
import concurrent
import configparser
import os
import pickle
import platform
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from json import loads
from subprocess import call

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from numba import njit, typed, types
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   StandardScaler)
from sklearn.tree import export_graphviz
from sklearn.utils import parallel_backend

from simba.mixins import cuRF

try:
    from dtreeviz.trees import dtreeviz, tree
except:
    from dtreeviz import dtreeviz
    from dtreeviz.trees import tree

import functools
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.plotting_mixin import PlottingMixin
from simba.plotting.shap_agg_stats_visualizer import \
    ShapAggregateStatisticsCalculator
from simba.ui.tkinter_functions import TwoOptionQuestionPopUp
from simba.utils.checks import (check_all_dfs_in_list_has_same_cols,
                                check_file_exist_and_readable,
                                check_filepaths_in_iterable_exist, check_float,
                                check_if_dir_exists, check_if_valid_input,
                                check_instance, check_int, check_str,
                                check_that_column_exist, check_valid_array,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst, is_lxc_container)
from simba.utils.data import (detect_bouts, detect_bouts_multiclass,
                              get_library_version)
from simba.utils.enums import (OS, ConfigKey, Defaults, Dtypes, Formats, Links,
                               Methods, MLParamKeys, Options)
from simba.utils.errors import (ClassifierInferenceError, CorruptedFileError,
                                DataHeaderError, FaultyTrainingSetError,
                                FeatureNumberMismatchError, InvalidInputError,
                                MissingColumnsError, NoDataError,
                                SamplingError, SimBAModuleNotFoundError)
from simba.utils.lookups import get_meta_data_file_headers, get_table
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt, get_fn_ext,
                                    get_memory_usage_of_df, get_pkg_version,
                                    read_config_entry, read_df, read_meta_file,
                                    str_2_bool)
from simba.utils.warnings import (GPUToolsWarning, MissingUserInputWarning,
                                  MultiProcessingFailedWarning,
                                  NoModuleWarning, NotEnoughDataWarning,
                                  SamplingWarning, ShapWarning)

plt.switch_backend("agg")

CUML = 'cuml'
SKLEARN = 'sklearn'

class TrainModelMixin(object):
    """Train model methods"""

    def __init__(self):
        pass

    def read_all_files_in_folder(self,
                                 file_paths: List[str],
                                 file_type: str,
                                 classifier_names: Optional[List[str]] = None,
                                 raise_bool_clf_error: bool = True) -> Tuple[pd.DataFrame, List[int]]:

        """
        Read in all data files in a folder into a single pd.DataFrame.

        .. note::
           For improved runtime using multiprocessing and pyarrow, use :func:`~simba.mixins.train_model_mixin.read_all_files_in_folder_mp`
           For improved runtime using ``concurrent` library, use :func:`simba.mixins.train_model_mixin.TrainModelMixin.read_all_files_in_folder_mp_futures`.

        :param List[str] file_paths: List of file paths representing files to be read in.
        :param str file_type: The type of files to be read in (e.g., `csv`)
        :param Optional[List[str]] classifier_names: Optional list of classifier names representing fields of human annotations. If not None, then assert that classifier names are present in each data file.
        :returns: concatenated DataFrame if all data represented in ``file_paths``, and a aligned list of frame numbers associated with the rows in the DataFrame.
        :rtype: Tuple[pd.DataFrame, List[int]]

        :examples:
        >>> self.read_all_files_in_folder(file_paths=['targets_inserted/Video_1.csv', 'targets_inserted/Video_2.csv'], file_type='csv', classifier_names=['Attack'])
        """
        check_filepaths_in_iterable_exist(file_paths=file_paths, name=self.__class__.__name__)
        timer = SimbaTimer(start=True)
        frm_number_lst, dfs = [], []
        if len(file_paths) == 0:
            raise NoDataError(msg="SimBA found 0 annotated frames in the project_folder/csv/targets_inserted directory", source=self.__class__.__name__)
        for file_cnt, file in enumerate(file_paths):
            _, vid_name, _ = get_fn_ext(file)
            print(f"Reading in {vid_name} (file {str(file_cnt + 1)}/{str(len(file_paths))})...")
            df = (read_df(file, file_type).dropna(axis=0, how="all").fillna(0).astype(np.float32))
            frm_number_lst.extend(list(df.index))
            df.index = [vid_name] * len(df)
            if classifier_names != None:
                for clf_name in classifier_names:
                    if not clf_name in df.columns:
                        raise MissingColumnsError(msg=f"Data for video {vid_name} does not contain any annotations for behavior {clf_name}. Delete classifier {clf_name} from the SimBA project, or add annotations for behavior {clf_name} to the video {vid_name}", source=self.__class__.__name__,)
                    elif (len(set(df[clf_name].unique()) - {0, 1}) > 0 and raise_bool_clf_error):
                        raise InvalidInputError(msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {file} the {clf_name} field contains additional value(s): {list(set(df[clf_name].unique()) - {0, 1})}.", source=self.__class__.__name__)
            dfs.append(df)

        check_all_dfs_in_list_has_same_cols(dfs=dfs, source='/project_folder/csv/targets_inserted', raise_error=True)
        col_headers = [list(x.columns) for x in dfs]
        dfs = [x[col_headers[0]] for x in dfs]
        dfs = pd.concat(dfs, axis=0)
        if 'scorer' in dfs.columns:
            dfs = dfs.set_index("scorer")
        dfs = dfs.loc[:, ~dfs.columns.str.contains("^Unnamed")].fillna(0)
        timer.stop_timer()
        memory_size = get_memory_usage_of_df(df=dfs)
        print(f'Dataset size: {memory_size["megabytes"]}MB / {memory_size["gigabytes"]}GB')
        print(f"{len(file_paths)} file(s) read (elapsed time: {timer.elapsed_time_str}s) ...")

        return dfs.astype(np.float32), frm_number_lst

    def read_in_all_model_names_to_remove(self, config: configparser.ConfigParser, model_cnt: int, clf_name: str) -> List[str]:
        """
        Helper to find all field names that are annotations but are not the target.

        :param configparser.ConfigParser config: Configparser object holding data from the project_config.ini
        :param int model_cnt: Number of classifiers in the SimBA project
        :param str clf_name: Name of the classifier.
        :return: List of non-target annotation column names.
        :rtype: List[str]

        :examples:
        >>> self.read_in_all_model_names_to_remove(config=config, model_cnt=2, clf_name=['Attack'])
        """

        annotation_cols_to_remove = []
        for model_no in range(model_cnt):
            model_name = config.get(
                ConfigKey.SML_SETTINGS.value, "target_name_" + str(model_no + 1)
            )
            if model_name != clf_name:
                annotation_cols_to_remove.append(model_name)
        return annotation_cols_to_remove

    def delete_other_annotation_columns(self, df: pd.DataFrame, annotations_lst: List[str], raise_error: bool = True) -> pd.DataFrame:
        """
        Helper to drop fields that contain annotations which are not the target.

        :param pd.DataFrame df: Dataframe holding features and annotations.
        :param List[str] annotations_lst: column fields to be removed from df
        :raise_error bool raise_error: If True, throw error if annotation column doesn't exist. Else, skip. Default: True.
        :return: Dataframe without non-target annotation columns
        :rtype: pd.DataFrame

        :examples:
        >>> self.delete_other_annotation_columns(df=df, annotations_lst=['Sniffing'])
        """

        for col_name in annotations_lst:
            if col_name not in df.columns and raise_error:
                raise NoDataError(
                    msg=f"Could not find expected column {col_name} in the data"
                )
            elif col_name not in df.columns and not raise_error:
                continue
            else:
                df = df.drop([col_name], axis=1)
        return df

    def split_df_to_x_y(self, df: pd.DataFrame, clf_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper to split dataframe into features and target.

        :param pd.DataFrame df: Dataframe holding features and annotations.
        :param str clf_name: Name of target.
        :returns: Size-2 tuple containing two dataframes - the features, and the target.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]

        :examples:
        >>> self.split_df_to_x_y(df=df, clf_name='Attack')
        """

        df = deepcopy(df)
        y = df.pop(clf_name)
        return df, y

    def random_undersampler(self, x_train: np.ndarray, y_train: np.ndarray, sample_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform random under-sampling of behavior-absent frames in a dataframe.

        :param np.ndarray x_train: 2-dimensional array representing the features in train set
        :param np.ndarray y_train: Array representing the target in the training set.
        :param float sample_ratio: Ratio of behavior-absent frames to keep relative to the behavior-present frames. E.g., ``1.0`` returns an equal count of behavior-absent and behavior-present frames. ``2.0`` returns twice as many behavior-absent frames as  and behavior-present frames.
        :returns: Size-2 tuple with DataFrames representing the under-sampled feature set and under-sampled target set.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]

        :examples:
        >>> self.random_undersampler(x_train=x_train, y_train=y_train, sample_ratio=1.0)
        """

        print(f"Performing under-sampling at sample ratio {str(sample_ratio)}...")
        data_df = pd.concat([x_train, y_train], axis=1)
        present_df, absent_df = (
            data_df[data_df[y_train.name] == 1],
            data_df[data_df[y_train.name] == 0],
        )
        ratio_n = int(len(present_df) * sample_ratio)
        if len(absent_df) < ratio_n:
            raise SamplingError(
                msg=f"SIMBA UNDER SAMPLING ERROR: The under-sample ratio of {str(sample_ratio)} in classifier {y_train.name} demands {str(ratio_n)} behavior-absent annotations. This is more than the number of behavior-absent annotations in the entire dataset {str(len(absent_df))}. Please annotate more images or decrease the under-sample ratio.",
                source=self.__class__.__name__,
            )
        data_df = pd.concat(
            [present_df, absent_df.sample(n=ratio_n, replace=False)], axis=0
        )
        return self.split_df_to_x_y(data_df, y_train.name)

    def smoteen_oversampler(self, x_train: pd.DataFrame, y_train: pd.DataFrame, sample_ratio: float ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper to perform SMOTEEN oversampling of behavior-present annotations.

        :param np.ndarray x_train: Features in train set
        :param np.ndarray y_train: Target in train set
        :param float sample_ratio: Over-sampling ratio
        :returns: Size-2 tuple arrays representing the over-sampled feature set and over-sampled target set.
        :rtype: Tuple[np.ndarray, np.ndarray]

        :examples:
        >>> self.smoteen_oversampler(x_train=x_train, y_train=y_train, sample_ratio=1.0)
        """

        print("Performing SMOTEENN oversampling...")
        smt = SMOTEENN(sampling_strategy=sample_ratio)
        if hasattr(smt, "fit_sample"):
            return smt.fit_sample(x_train, y_train)
        else:
            return smt.fit_resample(x_train, y_train)

    def smote_oversampler(self, x_train: pd.DataFrame or np.array, y_train: pd.DataFrame or np.array, sample_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper to perform SMOTE oversampling of behavior-present annotations.

        :param np.ndarray x_train: Features in train set
        :param np.ndarray y_train: Target in train set
        :param float sample_ratio: Over-sampling ratio
        :returns: Size-2 tuple arrays representing the over-sampled feature set and over-sampled target set.
        :rtype: Tuple[np.ndarray, np.ndarray]

        :examples:
        >>> self.smote_oversampler(x_train=x_train, y_train=y_train, sample_ratio=1.0)
        """
        print("Performing SMOTE oversampling...")
        smt = SMOTE(sampling_strategy=sample_ratio)
        if hasattr(smt, "fit_sample"):
            return smt.fit_sample(x_train, y_train)
        else:
            return smt.fit_resample(x_train, y_train)

    def calc_permutation_importance(self,
                                    x_test: np.ndarray,
                                    y_test: np.ndarray,
                                    clf: RandomForestClassifier,
                                    feature_names: List[str],
                                    clf_name: str,
                                    save_dir: Optional[Union[str, os.PathLike]] = None,
                                    save_file_no: Optional[int] = None,
                                    plot: Optional[bool] = True,
                                    n_repeats: Optional[int] = 10) -> Union[None, Tuple[pd.DataFrame, Union[None, np.ndarray]]]:
        """
        Computes feature permutation importance scores.

        :param np.ndarray x_test: 2d feature test data of shape len(frames) x len(features)
        :param np.ndarray y_test: 2d feature target test data of shape len(frames) x 1
        :param RandomForestClassifier clf: random forest classifier object
        :param List[str] feature_names: Names of features in x_test
        :param str clf_name: Name of classifier in y_test.
        :param str save_dir: Directory where to save results in CSV format. If None, then returns the dataframe and the plot (if plot
        :param Optional[bool] plot: If True, creates bar plot chart and saves in same directory as the CSV file.
        :param Optional[int] save_file_no: If permutation importance calculation is part of a grid search, provide integer identifier representing the model in the grid serach sequence. This will be used as suffix in output filename.
        :returns:  Either non or a Tuple with the dataframe and the plot. A CSV file representing the permutation importances is stored in ``save_dir`` if save_dir is passed.
        """

        print("Calculating feature permutation importances...")
        timer = SimbaTimer(start=True)
        p_importances = permutation_importance(clf, x_test, y_test, n_repeats=n_repeats, random_state=0)
        df = pd.DataFrame(np.column_stack([feature_names, p_importances.importances_mean, p_importances.importances_std]), columns=["FEATURE_NAME", "FEATURE_IMPORTANCE_MEAN", "FEATURE_IMPORTANCE_STDEV"])
        df = df.sort_values(by=["FEATURE_IMPORTANCE_MEAN"], ascending=False)
        df["FEATURE_IMPORTANCE_MEAN"] = df["FEATURE_IMPORTANCE_MEAN"].astype(np.float64)
        df["FEATURE_IMPORTANCE_STDEV"] = df["FEATURE_IMPORTANCE_STDEV"].astype(np.float64)
        save_file_path_plot, save_file_path, bar_chart = None, None, None
        if save_dir is not None:
            if save_file_no != None:
                save_file_path = os.path.join(save_dir, f'{clf_name}_{save_file_no}_permutations_importances.csv')
                save_file_path_plot = os.path.join(save_dir, f'{clf_name}_{save_file_no}_permutations_importances.png')
            else:
                save_file_path = os.path.join(save_dir, f"{clf_name}_permutations_importances.csv")
                save_file_path_plot = os.path.join(save_dir, f"{clf_name}_permutations_importances.png")

        if plot:
            bar_chart = PlottingMixin.plot_bar_chart(df=df,
                                             x='FEATURE_NAME',
                                             y="FEATURE_IMPORTANCE_MEAN",
                                             error='FEATURE_IMPORTANCE_STDEV',
                                             x_label='FEATURE',
                                             y_label='IMPORTANCE',
                                             title=f'SimBA feature importances {clf_name} (permutation)',
                                             save_path=save_file_path_plot)
        if save_file_path is not None:
            df.to_csv(save_file_path, index=False)
            timer.stop_timer()
            print(f"Permutation importance calculation complete (elapsed time: {timer.elapsed_time_str}s) ...")
        else:
            return df.reset_index(drop=True), bar_chart

    def calc_learning_curve(self,
                            x_y_df: pd.DataFrame,
                            clf_name: str,
                            shuffle_splits: int,
                            dataset_splits: int,
                            tt_size: float,
                            rf_clf: RandomForestClassifier,
                            save_dir: Union[str, os.PathLike],
                            save_file_no: Optional[int] = None,
                            multiclass: Optional[bool] = False,
                            scoring: Optional[str] = 'f1',
                            plot: Optional[bool] = True) -> None:

        """
        Helper to compute random forest learning curves with cross-validation.

        .. image:: _static/img/learning_curves.png
           :width: 600
           :align: center

        :param pd.DataFrame x_y_df: Dataframe holding features and target.
        :param str clf_name: Name of the classifier
        :param int shuffle_splits: Number of cross-validation datasets at each data split.
        :param int dataset_splits: Number of data splits.
        :param float tt_size: The size of the test set as a ratio of the dataset. E.g., 0.2.
        :param RandomForestClassifier rf_clf: A sklearn RandomForestClassifier object.
        :param str save_dir: Directory where to save output in csv file format.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        :param bool multiclass: If True, then target consist of several categories [0, 1, 2 ...] and scoring becomes ``None``. If False, then scoring ``f1``.
        :param Optional[str] scoring: The score of the models to present. Default: 'f1'.
        :param Optional[bool] plot: If True, creates plot with the train fraction size on x and ``scoring`` on y.
        :returns: None. Results are stored in ``save_dir``.
        """

        print("Calculating learning curves...")
        timer = SimbaTimer(start=True)
        x_df, y_df = self.split_df_to_x_y(x_y_df, clf_name)
        if save_file_no != None:
            self.learning_curve_save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_learning_curve.csv")
            self.plot_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_learning_curve_plot.png")
        else:
            self.learning_curve_save_path = os.path.join(save_dir, f"{clf_name}_learning_curve.csv")
            self.plot_path = os.path.join(save_dir, f"{clf_name}_learning_curve.png")
        check_int(name=f'calc_learning_curve shuffle_splits', value=shuffle_splits, min_value=2)
        check_int(name=f'calc_learning_curve dataset_splits', value=dataset_splits, min_value=2)
        cv = ShuffleSplit(n_splits=shuffle_splits, test_size=tt_size)
        if multiclass:
            scoring = None

        if (platform.system() == "Darwin" or platform.system() == "Linux") and not is_lxc_container():
            with parallel_backend("threading", n_jobs=-2):
                train_sizes, train_scores, test_scores = learning_curve(estimator=rf_clf, X=x_df.values, y=y_df, cv=cv, scoring=scoring, shuffle=False, verbose=0, train_sizes=np.linspace(0.01, 1.0, dataset_splits), error_score="raise")
        else:
            n_jobs = 32 if find_core_cnt()[0] > 32 else find_core_cnt()[0]
            train_sizes, train_scores, test_scores = learning_curve(estimator=rf_clf,
                                                                    X=x_df,
                                                                    y=y_df,
                                                                    cv=cv,
                                                                    scoring=scoring,
                                                                    shuffle=False,
                                                                    n_jobs=n_jobs,
                                                                    verbose=0,
                                                                    train_sizes=np.linspace(0.01, 1.0, dataset_splits),
                                                                    error_score="raise")
        results_df = pd.DataFrame()
        results_df["FRACTION TRAIN SIZE"] = np.linspace(0.01, 1.0, dataset_splits)
        results_df[f"TRAIN_MEAN_{scoring.upper()}"] = np.mean(train_scores, axis=1)
        results_df[f"TEST_MEAN_{scoring.upper()}"] = np.mean(test_scores, axis=1)
        results_df[f"TRAIN_STDEV_{scoring.upper()}"] = np.std(train_scores, axis=1)
        results_df[f"TEST_STDEV_{scoring.upper()}"] = np.std(test_scores, axis=1)

        if plot:
            _ = PlottingMixin.line_plot(df=results_df,
                                        x='FRACTION TRAIN SIZE',
                                        y=[f"TRAIN_MEAN_{scoring.upper()}", f"TEST_MEAN_{scoring.upper()}"],
                                        error=[f"TRAIN_STDEV_{scoring.upper()}", f"TEST_STDEV_{scoring.upper()}"],
                                        save_path=self.plot_path, y_label=scoring.upper(),
                                        title=f'SimBA learning curve {clf_name}')

        results_df.to_csv(self.learning_curve_save_path, index=False)
        timer.stop_timer()
        print(f"Learning curve calculation complete (elapsed time: {timer.elapsed_time_str}s) ...")

    def calc_pr_curve(self,
                      rf_clf: RandomForestClassifier,
                      x_df: pd.DataFrame,
                      y_df: pd.DataFrame,
                      clf_name: str,
                      save_dir: Union[str, os.PathLike],
                      multiclass: bool = False,
                      plot: Optional[bool] = True,
                      classifier_map: Dict[int, str] = None,
                      save_file_no: Optional[int] = None) -> None:
        """
        Compute random forest precision-recall curve.

        .. image:: _static/img/pr_curves.png
           :width: 800
           :align: center

        :param RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :param pd.DataFrame x_df: Pandas dataframe holding test features.
        :param pd.DataFrame y_df: Pandas dataframe holding test target.
        :param str clf_name: Classifier name.
        :param str save_dir: Directory where to save output in csv file format.
        :param bool multiclass: If the classifier is a multi-classifier. Default: False.
        :param Optional[bool] plot: If True, creates and saves line plot PR curve in the same lication as the output CSV file.
        :param Dict[int, str] classifier_map: If multiclass, dictionary mapping integers to classifier names.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        :returns: None. Results are stored in `save_dir``.
        """

        if multiclass and classifier_map is None:
            raise InvalidInputError(
                msg="Creating PR curve for multi-classifier but classifier_map not defined. Pass classifier_map argument")
        print("Calculating PR curves...")
        timer = SimbaTimer(start=True)
        if not multiclass:
            p = self.clf_predict_proba(clf=rf_clf,x_df=x_df, multiclass=False, model_name=clf_name, data_path=None)
            precision, recall, thresholds = precision_recall_curve(y_df, p, pos_label=1)
            pr_df = pd.DataFrame()
            pr_df["PRECISION"] = precision
            pr_df["RECALL"] = recall
            pr_df["F1"] = (2 * (pr_df["RECALL"] * pr_df["PRECISION"]) / (pr_df["RECALL"] + pr_df["PRECISION"]))
            thresholds = list(thresholds)
            thresholds.insert(0, 0.00)
            pr_df["DISCRIMINATION THRESHOLDS"] = thresholds
        else:
            pr_df_lst = []
            p = self.clf_predict_proba(clf=rf_clf, x_df=x_df, multiclass=True, model_name=clf_name)
            for i in range(p.shape[1]):
                precision, recall, thresholds = precision_recall_curve(y_df, p[:, i], pos_label=i)
                df = pd.DataFrame()
                df["PRECISION"] = precision
                df["RECALL"] = recall
                df["F1"] = (2 * (df["RECALL"] * df["PRECISION"]) / (df["RECALL"] + df["PRECISION"]))
                thresholds = list(thresholds)
                thresholds.insert(0, 0.00)
                df["DISCRIMINATION THRESHOLDS"] = thresholds
                df.insert(0, "BEHAVIOR CLASS", classifier_map[i])
                pr_df_lst.append(df)
            pr_df = pd.concat(pr_df_lst, axis=0).reset_index(drop=True)
        if save_file_no != None:
            self.pr_save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_pr_curve.csv")
            self.pr_save_path_plot = os.path.join(save_dir, f"{clf_name}_{save_file_no}_pr_curve.png")
        else:
            self.pr_save_path = os.path.join(save_dir, f"{clf_name}_pr_curve.csv")
            self.pr_save_path_plot = os.path.join(save_dir, f"{clf_name}_pr_curve.png")
        pr_df.to_csv(self.pr_save_path, index=False)
        if plot:
            _ = PlottingMixin.line_plot(df=pr_df, x="DISCRIMINATION THRESHOLDS", y=['PRECISION', 'RECALL', 'F1'], x_label='discrimination threshold', y_label='PERFORMANCE', title=f'SimBA {clf_name} precision-recall curve', save_path=self.pr_save_path_plot)
        timer.stop_timer()
        print(f"Precision-recall curve calculation complete (elapsed time: {timer.elapsed_time_str}s) ...")

    def create_example_dt(self,
                          rf_clf: RandomForestClassifier,
                          clf_name: str,
                          feature_names: List[str],
                          class_names: List[str],
                          save_dir: str,
                          tree_id: Optional[int] = 3,
                          save_file_no: Optional[int] = None) -> None:
        """
        Helper to produce visualization of random forest decision tree using graphviz.

        .. image:: _static/img/create_example_dt.png
           :width: 700
           :align: center

        .. note::
           `Example expected output  <https://github.com/sgoldenlab/simba/blob/master/misc/create_example_dt.pdf>`__.

        :param RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :param str clf_name: Classifier name.
        :param List[str] feature_names: List of feature names.
        :param List[str] class_names: List of classes. E.g., ['Attack absent', 'Attack present']
        :param str save_dir: Directory where to save output in csv file format.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        """

        print("Visualizing example decision tree using graphviz...")
        timer = SimbaTimer(start=True)
        if CUML in str(rf_clf.__module__).lower():
            GPUToolsWarning(msg="Can't visualize trees using CUML")
        else:
            estimator = rf_clf.estimators_[tree_id]
            if save_file_no != None:
                dot_name = os.path.join(save_dir, f"{clf_name}_{save_file_no}_tree.dot")
                file_name = os.path.join(save_dir, f"{clf_name}_{save_file_no}_tree.pdf")
            else:
                dot_name = os.path.join(save_dir, f"{clf_name}_tree.dot")
                file_name = os.path.join(save_dir, f"{clf_name}_tree.pdf")
            export_graphviz(estimator,
                            out_file=dot_name,
                            filled=True,
                            rounded=True,
                            special_characters=False,
                            impurity=False,
                            class_names=class_names,
                            feature_names=feature_names)

            command = f"dot {dot_name} -T pdf -o {file_name} -Gdpi=600"
            call(command, shell=True)
            timer.stop_timer()
            print(f'Example tree saved at {file_name} (elapsed time: {timer.elapsed_time_str}s)')

    def cuml_rf_x_importances(self, nodes: dict, n_features: int) -> np.ndarray:
        """
        Method for computing feature importance's from cuml RF object.

        From `szchixy <https://github.com/szchixy/cuml-sklearn/blob/main/cuml-sklearn.ipynb>`__.
        """

        importances = np.zeros((len(nodes), n_features))
        feature_gains = np.zeros(n_features)

        def calculate_node_importances(node, i_root):
            if "gain" not in node:
                return

            samples = node["instance_count"]
            gain = node["gain"]
            feature = node["split_feature"]
            feature_gains[feature] += gain * samples

            for child in node["children"]:
                calculate_node_importances(child, i_root)

        for i, root in enumerate(nodes):
            calculate_node_importances(root, i)
            importances[i] = feature_gains / feature_gains.sum()

        return np.mean(importances, axis=0)

    def create_clf_report(self,
                          rf_clf: Union[RandomForestClassifier, cuRF],
                          x_df: pd.DataFrame,
                          y_df: pd.DataFrame,
                          class_names: List[str],
                          save_dir: Union[str, os.PathLike],
                          digits: Optional[int] = 4,
                          clf_name: Optional[str] = None,
                          img_size: Optional[tuple] = (2500, 4500), #width by height
                          cmap: Optional[str] = "coolwarm",
                          threshold: Optional[int] = 0.5,
                          save_file_no: Optional[int] = None,
                          dpi: Optional[int] = 300) -> None:

        """
        Create classifier truth table report.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_

        .. image:: _static/img/clf_report.png
           :width: 500
           :align: center

        .. image:: _static/img/clf_report.webp
           :width: 500
           :align: center

        :param RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :param pd.DataFrame x_df: dataframe holding test features
        :param pd.DataFrame y_df: dataframe holding test target
        :param int digits: Number of floats in the classification report
        :param str cmap: The palette to plot the heatmap in. Default blue to red ("coolwarm").
        :param Tuple img_size: The size of the image in inches.
        :param float threshold: The presence classification threshold. Default: 0.5.
        :param List[str] class_names: List of classes. E.g., ['Attack absent', 'Attack present']
        :param Optional[str] clf_name: Name of the classifier. If not None, then used in the output file name.
        :param str save_dir: Directory where to save output in csv file format.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        :returns: None. Results are stored in `save_dir``.
        """

        print("Creating classification report visualization...")
        timer = SimbaTimer(start=True)
        if save_file_no != None:
            if not clf_name:
                save_path = os.path.join(save_dir, f'{class_names[1]}_{save_file_no}_classification_report.png')
            else:
                save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_classification_report.png")
        else:
            if not clf_name:
                save_path = os.path.join(save_dir, f"{class_names[1]}_classification_report.png")
            else:
                save_path = os.path.join(save_dir, f"{clf_name}_classification_report.png")

        y_pred = self.clf_predict_proba(clf=rf_clf, x_df=x_df)
        y_pred = np.where(y_pred > threshold, 1, 0)

        plt.figure()
        clf_report = classification_report(y_true=y_df.values, y_pred=y_pred, target_names=class_names, digits=digits, output_dict=True, zero_division=0)
        clf_report = pd.DataFrame.from_dict({key: clf_report[key] for key in class_names})
        plt.figure(figsize=(round((img_size[1] / dpi), 2), round((img_size[0] / dpi), 2)), dpi=dpi)
        img = sns.heatmap(pd.DataFrame(clf_report).T, annot=True, cmap=cmap, vmin=0.0, vmax=1.0, linewidth=2.0, linecolor='black', fmt='g', annot_kws={"size": 20, "weight": "bold", "color": "white", "family": "sans-serif"})
        img.set_xticklabels(img.get_xticklabels(), size=16)
        img.set_yticklabels(img.get_yticklabels(), size=16)
        plt.savefig(save_path, dpi=dpi)
        plt.close("all")
        timer.stop_timer()
        print(f'Classification report saved at {save_path} (elapsed time: {timer.elapsed_time_str}s)')

    def create_x_importance_log(self,
                                rf_clf: Union[RandomForestClassifier, cuRF],
                                x_names: List[str],
                                clf_name: str,
                                save_dir: Optional[str] = None,
                                save_file_no: Optional[int] = None) -> Union[None, pd.DataFrame]:
        """
        Compute gini / entropy based feature importance scores.

        .. note::
           `Example expected output  <https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_importance_log.csv>`__.

        .. seealso::
           To plot gini / entropy based feature importance scores, see :func:`~simba.mixins.train_model_mixin.TrainModelMixin.create_x_importance_bar_chart`

        :param RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :param List[str] x_names: Names of features.
        :param str clf_name: Name of classifier
        :param str save_dir: Directory where to save output in csv file format. If None, then returns the dataframe.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        :returns: None. Results are stored in `save_dir``.
        """

        print("Creating feature importance log...")
        timer = SimbaTimer(start=True)
        if save_dir is not None:
            if save_file_no != None:
                self.f_importance_save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_feature_importance_log.csv")
            else:
                self.f_importance_save_path = os.path.join(save_dir, f"{clf_name}_feature_importance_log.csv")
        if cuRF is not None and isinstance(rf_clf, cuRF) and hasattr(rf_clf, 'get_json'):
            cuml_tree_nodes = loads(rf_clf.get_json())
            importances = list(self.cuml_rf_x_importances(nodes=cuml_tree_nodes, n_features=len(x_names)))
            std_importances = [np.nan] * len(importances)
        else:
            importances_per_tree = np.array([tree.feature_importances_ for tree in rf_clf.estimators_])
            importances = list(np.mean(importances_per_tree, axis=0))
            std_importances = list(np.std(importances_per_tree, axis=0))
        importances = [round(importance, 25) for importance in importances]
        df = pd.DataFrame({'FEATURE': x_names,'FEATURE_IMPORTANCE_MEAN': importances, 'FEATURE_IMPORTANCE_STDEV': std_importances}).sort_values(by=["FEATURE_IMPORTANCE_MEAN"], ascending=False)
        if save_dir is not None:
            df.to_csv(self.f_importance_save_path, index=False)
            timer.stop_timer()
            stdout_success(msg=f'Feature importance log saved at {self.f_importance_save_path}!', elapsed_time=timer.elapsed_time_str)
        else:
            return df.reset_index(drop=True)

    def create_x_importance_bar_chart(self,
                                      rf_clf: RandomForestClassifier,
                                      x_names: list,
                                      clf_name: str,
                                      save_dir: str,
                                      n_bars: int,
                                      palette: Optional[str] = 'hot',
                                      save_file_no: Optional[int] = None) -> None:
        """
        Helper to create a bar chart displaying the top N gini or entropy feature importance scores.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_

        .. image:: _static/img/gini_bar_chart.png
           :width: 600
           :align: center

        :param RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :param List[str] x_names: Names of features.
        :param str clf_name: Name of classifier.
        :param str save_dir: Directory where to save output in csv file format.
        :param int n_bars: Number of bars in the plot.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search
        :returns: None. Results are stored in `save_dir``.
        """

        check_int(name="FEATURE IMPORTANCE BAR COUNT", value=n_bars, min_value=1)
        print("Creating feature importance bar chart...")
        timer = SimbaTimer(start=True)
        if save_file_no != None:
            save_file_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_feature_importance_bar_graph.png")
        else:
            save_file_path = os.path.join(save_dir, f"{clf_name}_feature_importance_bar_graph.png")
        self.create_x_importance_log(rf_clf, x_names, clf_name, save_dir)
        importances_df = pd.read_csv(os.path.join(save_dir, f"{clf_name}_feature_importance_log.csv"))
        importances_head = importances_df.head(n_bars)
        _ = PlottingMixin.plot_bar_chart(df=importances_head,
                                         x='FEATURE',
                                         y="FEATURE_IMPORTANCE_MEAN",
                                         error='FEATURE_IMPORTANCE_STDEV',
                                         x_label='FEATURE',
                                         y_label='IMPORTANCE',
                                         title=f'SimBA feature importances {clf_name}',
                                         save_path=save_file_path)
        timer.stop_timer()
        print(f'Feature importance bar chart complete, saved at {save_file_path} (elapsed time: {timer.elapsed_time_str}s)')

    def dviz_classification_visualization(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            clf_name: str,
            class_names: List[str],
            save_dir: str,
    ) -> None:
        """
        Helper to create visualization of example decision tree using dtreeviz.

        :parameter np.ndarray x_train: training features
        :parameter np.ndarray y_train: training targets
        :parameter str clf_name: Name of classifier
        :parameter List[str] class_names: List of class names. E.g., ['Attack absent', 'Attack present']
        :parameter str save_dir: Directory where to save output in csv file format.
        """

        print('Creating example decision tree using dtreeviz ....')
        try:
            clf = tree.DecisionTreeClassifier(max_depth=5, random_state=666)
            clf.fit(x_train, y_train)
            svg_tree = dtreeviz(
                clf,
                x_train,
                y_train,
                target_name=clf_name,
                feature_names=x_train.columns,
                orientation="TD",
                class_names=class_names,
                fancy=True,
                histtype="strip",
                X=None,
                label_fontsize=12,
                ticks_fontsize=8,
                fontname="Arial",
            )
            save_path = os.path.join(save_dir, f"{clf_name}_fancy_decision_tree_example.svg")
            svg_tree.save(save_path)
        except:
            NoModuleWarning(
                msg='Skipping dtreeviz example decision tree visualization. Make sure "graphviz" is installed.',
                source=self.__class__.__name__,
            )

    @staticmethod
    def split_and_group_df(df: pd.DataFrame, splits: int, include_split_order: bool = True) -> (List[pd.DataFrame], int):
        """
        Helper to split a dataframe for multiprocessing. If include_split_order, then include the group number
        in split data as a column. Returns split data and approximations of number of observations per split.
        """
        data_arr = np.array_split(df, splits)
        if include_split_order:
            for df_cnt in range(len(data_arr)):
                data_arr[df_cnt]["group"] = df_cnt
        obs_per_split = len(data_arr[0])
        return data_arr, obs_per_split

    def create_shap_log(self,
                        rf_clf: Union[RandomForestClassifier, cuRF],
                        x: Union[pd.DataFrame, np.ndarray],
                        y: Union[pd.DataFrame, pd.Series, np.ndarray],
                        x_names: List[str],
                        clf_name: str,
                        cnt_present: int,
                        cnt_absent: int,
                        verbose: bool = True,
                        plot: bool = True,
                        save_it: Optional[int] = 100,
                        save_dir: Optional[Union[str, os.PathLike]] = None,
                        save_file_suffix: Optional[int] = None) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray]]:

        """
        Compute SHAP values for a random forest classifier.
        This method computes SHAP (SHapley Additive exPlanations) values for a given random forest classifier.
        The SHAP value for feature 'i' in the context of a prediction 'f' and input 'x' is calculated using the following formula:

        .. math::
           \phi_i(f, x) = \\sum_{S \\subseteq F \\setminus {i}} \\frac{|S|!(|F| - |S| - 1)!}{|F|!} (f_{S \cup {i}}(x_{S \\cup {i}}) - f_S(x_S))

        .. note::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_
           Uses TreeSHAP `Documentation <https://shap.readthedocs.io/en/latest/index.html>`_

        .. image:: _static/img/shap.png
           :width: 600
           :align: center

        .. seealso::
           For multicore solution, see :func:`~simba.mixins.train_model_mixins.TrainModelMixin.create_shap_log_mp`
           For GPU method, see :func:`~simba.data_processors.cuda.create_shap_log.create_shap_log`

        :param RandomForestClassifier rf_clf: sklearn random forest classifier
        :param Union[pd.DataFrame, np.ndarray] x: Test features.
        :param Union[pd.DataFrame, pd.Series, np.ndarray] y: Test target.
        :param List[str] x_names: Feature names.
        :param str clf_name: Classifier name.
        :param int cnt_present: Number of behavior-present frames to calculate SHAP values for.
        :param int cnt_absent: Number of behavior-absent frames to calculate SHAP values for.
        :param int save_it: Save iteration cadence. If None, then only saves at completion.
        :param str save_dir: Optional directory where to save output in csv file format. If None, the data is returned.
        :param Optional[int] save_file_suffix: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.

        :example:
        >>> from simba.mixins.train_model_mixin import TrainModelMixin
        >>> x_cols = list(pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0).columns)
        >>> x = pd.DataFrame(np.random.randint(0, 500, (9000, len(x_cols))), columns=x_cols)
        >>> y = pd.Series(np.random.randint(0, 2, (9000,)))
        >>> rf_clf = TrainModelMixin().clf_define(n_estimators=100)
        >>> rf_clf = TrainModelMixin().clf_fit(clf=rf_clf, x_df=x, y_df=y)
        >>> feature_names = [str(x) for x in list(x.columns)]
        >>> TrainModelMixin.create_shap_log(rf_clf=rf_clf, x=x, y=y, x_names=feature_names, clf_name='test', save_it=10, cnt_present=50, cnt_absent=50, plot=True, save_dir=r'/Users/simon/Desktop/feltz')
        """

        if SKLEARN in str(rf_clf.__module__).lower():
            print("Calculating SHAP values (SINGLE CORE)...")
            timer = SimbaTimer(start=True)
            check_instance(source='create_shap_log', instance=rf_clf, accepted_types=(RandomForestClassifier,))
            check_instance(source=f'{TrainModelMixin.create_shap_log.__name__} x', instance=x, accepted_types=(np.ndarray, pd.DataFrame))
            check_instance(source=f'{TrainModelMixin.create_shap_log.__name__} y', instance=y, accepted_types=(np.ndarray, pd.Series, pd.DataFrame))
            if isinstance(x, pd.DataFrame):
                check_valid_dataframe(df=x, source=f'{TrainModelMixin.create_shap_log.__name__} x', valid_dtypes=Formats.NUMERIC_DTYPES.value)
                x = x.values
            else:
                check_valid_array(data=x, source=f'{TrainModelMixin.create_shap_log.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            if isinstance(y, pd.DataFrame):
                check_valid_dataframe(df=y, source=f'{TrainModelMixin.create_shap_log.__name__} y', valid_dtypes=Formats.NUMERIC_DTYPES.value, max_axis_1=1)
                y = y.values
            else:
                if isinstance(y, pd.Series):
                    y = y.values
                check_valid_array(data=y, source=f'{TrainModelMixin.create_shap_log.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            check_valid_lst(data=x_names, source=f'{TrainModelMixin.create_shap_log.__name__} x_names', valid_dtypes=(str,), exact_len=x.shape[1])
            check_str(name=f'{TrainModelMixin.create_shap_log.__name__} clf_name', value=clf_name)
            check_int(name=f'{TrainModelMixin.create_shap_log.__name__} cnt_present', value=cnt_present, min_value=1)
            check_int(name=f'{TrainModelMixin.create_shap_log.__name__} cnt_absent', value=cnt_absent, min_value=1)
            check_instance(source=f'{TrainModelMixin.create_shap_log.__name__} save_it', instance=save_it, accepted_types=(type(None), int))
            if save_it is not None and save_dir is None:
                ShapWarning(msg='Omitting save_it as save_dir is None')
            if save_it is not None:
                check_int(name=f'{TrainModelMixin.create_shap_log.__name__} save_it', value=save_it, min_value=1)
            if save_it is None or save_it > x.shape[0]:
                save_it = x.shape[0]
            if save_file_suffix is not None:
                check_int(name=f'{TrainModelMixin.create_shap_log.__name__} save_it', value=save_it, min_value=0)
            check_valid_lst(data=list(x_names), valid_dtypes=(str,), exact_len=x.shape[1])
            check_valid_boolean(value=[verbose, plot], source=f'{TrainModelMixin.create_shap_log.__name__} verbose, plot')
            df = pd.DataFrame(np.hstack((x, y.reshape(-1, 1))), columns=x_names + [clf_name])
            del x; del y
            present_df, absent_df = df[df[clf_name] == 1], df[df[clf_name] == 0]
            if len(present_df) == 0:
                raise NoDataError(msg=f'Cannot calculate SHAP values: no target PRESENT annotations detected.', source=TrainModelMixin.create_shap_log.__name__)
            elif len(absent_df) == 0:
                raise NoDataError(msg=f'Cannot calculate SHAP values: no target ABSENT annotations detected.', source=TrainModelMixin.create_shap_log.__name__)
            if len(present_df) < cnt_present:
                NotEnoughDataWarning(msg=f"Train data contains {len(present_df)} behavior-present annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_present)}). SimBA will calculate shap scores for the {len(present_df)} behavior-present frames available", source=TrainModelMixin.create_shap_log.__name__)
                cnt_present = len(present_df)
            if len(absent_df) < cnt_absent:
                NotEnoughDataWarning(msg=f"Train data contains {len(absent_df)} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_absent)}). SimBA will calculate shap scores for the {len(absent_df)} behavior-absent frames available", source=TrainModelMixin.create_shap_log.__name__, )
                cnt_absent = len(absent_df)
            out_shap_path, out_raw_path, img_save_path, df_save_paths, summary_dfs, img = None, None, None, None, None, None
            if save_dir is not None:
                check_if_dir_exists(in_dir=save_dir)
                if save_file_suffix is not None:
                    check_int(name=f'{TrainModelMixin.create_shap_log.__name__} save_file_no', value=save_file_suffix, min_value=0)
                    out_shap_path = os.path.join(save_dir, f"SHAP_values_{save_file_suffix}_{clf_name}.csv")
                    out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{save_file_suffix}_{clf_name}.csv")
                    df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT_{save_file_suffix}.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT_{save_file_suffix}.csv")}
                    img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}_{save_file_suffix}.png")
                else:
                    out_shap_path = os.path.join(save_dir, f"SHAP_values_{clf_name}.csv")
                    out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{clf_name}.csv")
                    df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT.csv")}
                    img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}.png")
            shap_x = pd.concat([present_df.sample(cnt_present, replace=False), absent_df.sample(cnt_absent, replace=False)], axis=0).reset_index(drop=True)
            shap_y = shap_x[clf_name].values.flatten()
            shap_x = shap_x.drop([clf_name], axis=1)
            explainer = TrainModelMixin().define_tree_explainer(clf=rf_clf)
            expected_value = explainer.expected_value[1]
            raw_df = pd.DataFrame(columns=x_names)
            shap_headers = list(x_names) + ["Expected_value", "Sum", "Prediction_probability", clf_name]
            shap_df = pd.DataFrame(columns=shap_headers)
            for cnt, frame in enumerate(range(len(shap_x))):
                shap_frm_timer = SimbaTimer(start=True)
                frame_data = shap_x.iloc[[frame]]
                frame_shap = explainer.shap_values(frame_data, check_additivity=False)[1][0].tolist()
                frame_shap.extend((expected_value, sum(frame_shap) + expected_value, rf_clf.predict_proba(frame_data)[0][1], shap_y[cnt]))
                raw_df.loc[len(raw_df)] = list(shap_x.iloc[frame])
                shap_df.loc[len(shap_df)] = frame_shap
                if ((cnt % save_it == 0) or (cnt == len(shap_x) - 1)) and (cnt != 0) and (save_dir is not None):
                    print(f"Saving SHAP data after {cnt} iterations...")
                    shap_df.to_csv(out_shap_path)
                    raw_df.to_csv(out_raw_path)
                shap_frm_timer.stop_timer()
                print(f"SHAP frame: {cnt + 1} / {len(shap_x)}, elapsed time: {shap_frm_timer.elapsed_time_str}...")
            if plot:
                shap_computer = ShapAggregateStatisticsCalculator(classifier_name=clf_name,
                                                                  shap_df=shap_df,
                                                                  shap_baseline_value=int(expected_value * 100),
                                                                  save_dir=None)
                summary_dfs, img = shap_computer.run()
                if save_dir is not None:
                    summary_dfs['PRESENT'].to_csv(df_save_paths['PRESENT'])
                    summary_dfs['ABSENT'].to_csv(df_save_paths['ABSENT'])
                    cv2.imwrite(img_save_path, img)

            timer.stop_timer()
            if save_dir is not None and verbose:
                shap_df.to_csv(out_shap_path)
                raw_df.to_csv(out_raw_path)
                stdout_success(msg=f"SHAP calculations complete! Results saved at {out_shap_path} and {out_raw_path}", elapsed_time=timer.elapsed_time_str, source=TrainModelMixin.create_shap_log.__name__)

            if not save_dir:
                return shap_df, raw_df, summary_dfs, img
        else:
            GPUToolsWarning(msg=f'Cannot compute SHAP scores using cuml random forest model. To compute SHAP scores, turn off cuda. Alternatively, for GPU solution, see simba/data_processors/cuda/create_shap_log.create_shap_log')

    def print_machine_model_information(self, model_dict: dict) -> None:
        """
        Helper to print model information in tabular form.

        :param dict model_dict: dictionary holding model meta data in SimBA meta-config format.
        """

        table = get_table(data=model_dict, headers=('SETTINGS', 'VALUE',), tablefmt="grid")
        print(f"{table} {Defaults.STR_SPLIT_DELIMITER.value}TABLE")

    def create_meta_data_csv_training_one_model(self, meta_data_lst: list, clf_name: str,
                                                save_dir: Union[str, os.PathLike]) -> None:

        """
        Helper to save single model meta data (hyperparameters, sampling settings etc.) from list format into SimBA
        compatible CSV config file.

        :param list meta_data_lst: Meta data in list format
        :param str clf_name: Name of classifier
        :param str clf_name: Name of classifier
        :param str save_dir: Directory where to save output in csv file format.
        """
        print("Saving model meta data file...")
        save_path = os.path.join(save_dir, clf_name + "_meta.csv")
        out_df = pd.DataFrame(columns=get_meta_data_file_headers())
        out_df.loc[len(out_df)] = meta_data_lst
        out_df.to_csv(save_path)

    def create_meta_data_csv_training_multiple_models(self, meta_data, clf_name, save_dir,
                                                      save_file_no: Optional[int] = None) -> None:
        print("Saving model meta data file...")
        save_path = os.path.join(save_dir, f"{clf_name}_{str(save_file_no)}_meta.csv")
        out_df = pd.DataFrame.from_dict(meta_data, orient="index").T
        out_df.to_csv(save_path)

    def save_rf_model(self,
                      rf_clf: RandomForestClassifier,
                      clf_name: str,
                      save_dir: Union[str, os.PathLike],
                      save_file_no: Optional[int] = None) -> None:
        """
        Helper to save pickled classifier object to disk.

        .. seealso::
           To write pickle, can also use :func:`~simba.utils.read_write.write_pickle`
           To read pickle, see :func:`~simba.utils.read_write.read_pickle` or :func:`~simba.mixins.train_model_mixin.TrainModelMixin.read_pickle`.

        :param RandomForestClassifier rf_clf: sklearn random forest classifier
        :param str clf_name: Classifier name
        :param str save_dir: Directory where to save output as pickle.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        :returns: None. Results are saved in ``save_dir``.

        """
        if save_file_no != None:
            save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}.sav")
        else:
            save_path = os.path.join(save_dir, f"{clf_name}.sav")
        pickle.dump(rf_clf, open(save_path, "wb"))

    def get_model_info(self, config: configparser.ConfigParser, model_cnt: int) -> Dict[int, Any]:
        """
        Helper to read in N SimBA random forest config meta files to python dict memory.

        :parameter configparser.ConfigParser config: Parsed SimBA project_config.ini
        :parameter int model_cnt: Count of models
        :return dict: Dictionary with integers as keys and hyperparameter dictionaries as keys.
        """

        model_dict = {}
        for n in range(model_cnt):
            try:
                if config.get("SML settings", "model_path_" + str(n + 1)) == "":
                    MissingUserInputWarning(msg=f'Skipping {str(config.get("SML settings", "target_name_" + str(n + 1)))} classifier analysis: no path set to model file', source=self.__class__.__name__)
                    continue
                if (config.get("SML settings", "model_path_" + str(n + 1)) == "No file selected"):
                    MissingUserInputWarning(msg=f'Skipping {str(config.get("SML settings", "target_name_" + str(n + 1)))} classifier analysis: The classifier path is set to "No file selected', source=self.__class__.__name__,)
                    continue
                model_dict[n] = {}
                model_dict[n]["model_path"] = config.get(ConfigKey.SML_SETTINGS.value, "model_path_" + str(n + 1))
                if not os.path.isfile(model_dict[n]["model_path"]):
                    MissingUserInputWarning(msg=f'Skipping {str(config.get("SML settings", "target_name_" + str(n + 1)))} classifier analysis: The classifier file path does not exist: {model_dict[n]["model_path"]}', source=self.__class__.__name__,)
                    del model_dict[n]
                    continue
                model_dict[n]["model_name"] = config.get(ConfigKey.SML_SETTINGS.value, "target_name_" + str(n + 1))
                check_str("model_name", model_dict[n]["model_name"])
                model_dict[n]["threshold"] = config.getfloat(ConfigKey.THRESHOLD_SETTINGS.value, "threshold_" + str(n + 1))
                check_float( "threshold", model_dict[n]["threshold"], min_value=0.0, max_value=1.0)
                model_dict[n]["minimum_bout_length"] = config.getfloat(ConfigKey.MIN_BOUT_LENGTH.value, "min_bout_" + str(n + 1)
                )
                check_int("minimum_bout_length", model_dict[n]["minimum_bout_length"])
                if config.has_option(ConfigKey.SML_SETTINGS.value, f"classifier_map_{n + 1}"):
                    model_dict[n]["classifier_map"] = config.get(
                        ConfigKey.SML_SETTINGS.value, f"classifier_map_{n + 1}"
                    )
                    model_dict[n]["classifier_map"] = ast.literal_eval(
                        model_dict[n]["classifier_map"]
                    )
                    if type(model_dict[n]["classifier_map"]) != dict:
                        raise InvalidInputError(
                            msg=f"SimBA found a classifier map for classifier {n + 1} that could not be interpreted as a dictionary",
                            source=self.__class__.__name__,
                        )

            except ValueError:
                model_dict.pop(n, None)
                MissingUserInputWarning(msg=f'Skipping {str(config.get("SML settings", "target_name_" + str(n + 1)))} classifier analysis: missing information (e.g., no discrimination threshold and/or minimum bout set in the project_config.ini',source=self.__class__.__name__)

        if len(model_dict.keys()) == 0:
            raise NoDataError(
                msg=f"There are no models with accurate data specified in the RUN MODELS menu. Specify the model information to SimBA RUN MODELS menu to use them to analyze videos",
                source=self.get_model_info.__name__,
            )
        else:
            return model_dict

    def get_all_clf_names(self, config: configparser.ConfigParser, target_cnt: int) -> List[str]:
        """
        Helper to get all classifier names in a SimBA project.

        :parameter configparser.ConfigParser config: Parsed SimBA project_config.ini
        :parameter int.ConfigParser target_cnt: Parsed SimBA project_config.ini
        :return: All classifier names in project
        :rtype: List[str]

        :example:
        >>> self.get_all_clf_names(config=config, target_cnt=2)
        >>> ['Attack', 'Sniffing']
        """

        model_names = []
        for i in range(target_cnt):
            entry_name = "target_name_{}".format(str(i + 1))
            model_names.append(
                read_config_entry(
                    config,
                    ConfigKey.SML_SETTINGS.value,
                    entry_name,
                    data_type=Dtypes.STR.value,
                )
            )
        return model_names

    def insert_column_headers_for_outlier_correction( self, data_df: pd.DataFrame, new_headers: List[str], filepath: Union[str, os.PathLike]) -> pd.DataFrame:
        """
        Helper to insert new column headers onto a dataframe following outlier correction.

        :param pd.DataFrame data_df: Dataframe with headers to-be replaced.
        :param str filepath: Path to where ``data_df`` is stored on disk.
        :param: DataFRame with the corrected headers following outlier correction.
        """

        if len(new_headers) != len(data_df.columns):
            difference = int(len(data_df.columns) - len(new_headers))
            bp_missing = int(abs(difference) / 3)
            if difference < 0:
                raise DataHeaderError(
                    msg=f"SimBA expects {len(new_headers)} columns of data inside the files within project_folder/csv/input_csv directory. However, within file {filepath} file, SimBA found {len(data_df.columns)} columns. Thus, there is {abs(difference)} missing data columns in the imported data, which may represent {int(bp_missing)} bodyparts if each body-part has an x, y and p value. Either revise the SimBA project pose-configuration with {str(bp_missing)} less body-part, or include {bp_missing} more body-part in the imported data",
                    source=self.__class__.__name__,
                )
            else:
                raise DataHeaderError(
                    msg=f"SimBA expects {len(new_headers)} columns of data inside the files within project_folder/csv/input_csv directory. However, within file {filepath} file, SimBA found {len(data_df.columns)} columns. Thus, there is {abs(difference)} more data columns in the imported data than anticipated, which may represent {int(bp_missing)} bodyparts if each body-part has an x, y and p value. Either revise the SimBA project pose-configuration with {str(bp_missing)} more body-part, or include {bp_missing} less body-part in the imported data",
                    source=self.__class__.__name__,
                )
        else:
            data_df.columns = new_headers
            return data_df

    def read_pickle(self, file_path: Union[str, os.PathLike]) -> RandomForestClassifier:
        """
        Read pickled RandomForestClassifier object.

        :param Union[str, os.PathLike] file_path: Path to pickle file on disk.
        :returns: A scikitRandomForestClassifier object.
        :rtype: RandomForestClassifier
        """

        check_file_exist_and_readable(file_path=file_path)
        try:
            clf = pickle.load(open(file_path, "rb"))
        except (pickle.UnpicklingError, EOFError, ImportError) as e:
            raise CorruptedFileError(msg=f"Can not read {file_path} as a classifier file (pickle).", source=self.__class__.__name__)
        except ValueError as e:
            scikit_version = get_pkg_version(pkg='scikit-learn')
            python_version = OS.PYTHON_VER.value
            raise CorruptedFileError(msg=f"Cannot read {file_path}. The file is created in a different SimBA environment with a different scikit-learn version and/or different python version than the current scikit-learn version ({scikit_version}) and the current python version ({python_version}).", source=self.__class__.__name__)

        return clf

    def bout_train_test_splitter(self,
                                 x_df: pd.DataFrame,
                                 y_df: pd.DataFrame,
                                 test_size: float ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Helper to split train and test based on annotated `bouts`.

        .. image:: _static/img/bout_vs_frames_split.png
           :width: 600
           :align: center

        :param pd.DataFrame x_df: Features
        :param pd.Series y_df: Target
        :param float test_size: Size of test as ratio of all annotated bouts (e.g., ``0.2``).
        :returns: Size-4 tuple with DataFrames of Series representing, (i) Features for training, (ii) Features for testing, (iii) Target for training, (iv) Target for testing.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

        :examples:
        >>> x = pd.DataFrame(data=[[11, 23, 12], [87, 65, 76], [23, 73, 27], [10, 29, 2], [12, 32, 42], [32, 73, 2], [21, 83, 98], [98, 1, 1]])
        >>> y =  pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        >>> x_train, x_test, y_train, y_test = TrainModelMixin().bout_train_test_splitter(x_df=x, y_df=y, test_size=0.5)
        """

        print("Using bout sampling...")
        def find_bouts(s: pd.Series, type: str):
            test_bouts_frames, train_bouts_frames = [], []
            bouts = detect_bouts(pd.DataFrame(s), target_lst=pd.DataFrame(s).columns, fps=1)
            print(f"{str(len(bouts))} {type} bouts found...")
            bouts = list(bouts.apply(lambda x: list(range(int(x["Start_frame"]), int(x["End_frame"]) + 1)),1).values)
            test_bouts_idx = np.random.choice(np.arange(0, len(bouts)), int(len(bouts) * test_size))
            train_bouts_idx = np.array([x for x in list(range(len(bouts))) if x not in test_bouts_idx])
            for i in range(0, len(bouts)):
                if i in test_bouts_idx:
                    test_bouts_frames.append(bouts[i])
                if i in train_bouts_idx:
                    train_bouts_frames.append(bouts[i])
            return [i for s in test_bouts_frames for i in s], [i for s in train_bouts_frames for i in s]

        test_bouts_frames, train_bouts_frames = find_bouts(s=y_df, type="behavior present")
        test_nonbouts_frames, train_nonbouts_frames = find_bouts(s=np.logical_xor(y_df, 1).astype(int), type="behavior absent")
        x_train = x_df[x_df.index.isin(train_bouts_frames + train_nonbouts_frames)]
        x_test = x_df[x_df.index.isin(test_bouts_frames + test_nonbouts_frames)]
        y_train = y_df[y_df.index.isin(train_bouts_frames + train_nonbouts_frames)]
        y_test = y_df[y_df.index.isin(test_bouts_frames + test_nonbouts_frames)]

        return x_train, x_test, y_train, y_test

    @staticmethod
    @njit("(float32[:, :], float64, types.ListType(types.unicode_type))")
    def find_highly_correlated_fields(data: np.ndarray,threshold: float, field_names: types.ListType(types.unicode_type)) -> List[str]:

        """
        Find highly correlated fields in a dataset using Pearson product-moment correlation coefficient.

        Calculates the absolute correlation coefficients between columns in a given dataset and identifies
        pairs of columns that have a correlation coefficient greater than the specified threshold. For every pair of correlated
        features identified, the function returns the field name of one feature. These field names can later be dropped from the input data to reduce memory requirements and collinearity.

        .. seealso::
           For non-numba method, see :func:`simba.mixins.statistics_mixin.Statistics.find_collinear_features`.
           For validation wrapper, see :func:`simba.mixins.train_model_mixin.TrainModelMixin.find_collinear_features`

        :param np.ndarray data: Two dimension numpy array with features represented as columns and frames represented as rows.
        :param float threshold: Threshold value for significant collinearity.
        :param List[str] field_names: List mapping the column names in data to a field name. Use types.ListType(types.unicode_type) to take advantage of JIT compilation
        :return: Unique field names that correlates with at least one other field above the threshold value.
        :rtype: List[str]


        :example:
        >>> data = np.random.randint(0, 1000, (1000, 5000)).astype(np.float32)
        >>> field_names = []
        >>> for i in range(data.shape[1]): field_names.append(f'Feature_{i+1}')
        >>> highly_correlated_fields = TrainModelMixin().find_highly_correlated_fields(data=data, field_names=typed.List(field_names), threshold=0.10)
        """

        column_corr = np.abs(np.corrcoef(data.T))
        remove_set = {(-1, -1)}
        for i in range(column_corr.shape[0]):
            field_corr = column_corr[i]
            idxs = np.argwhere((field_corr > threshold)).flatten()
            idxs = idxs[idxs != i]
            for j in idxs:
                remove_set.add((np.min(np.array([i, j])), np.max(np.array([i, j]))))

        remove_col_idx = {-1}
        for i in remove_set:
            remove_col_idx.add(i[0])
        remove_col_idx.remove(-1)
        remove_col_idx = list(remove_col_idx)

        return [field_names[x] for x in remove_col_idx]

    @staticmethod
    def find_collinear_features(data: pd.DataFrame,
                                threshold: float) -> List[str]:

        """
        Identify collinear features in a pandas DataFrame for removal.

        Finds pairs of features with Pearson correlation coefficients above the specified threshold and returns the names of features that should be removed to reduce multicollinearity.

        Serves as a validation wrapper around numba implementation.

        .. seealso::
           For the underlying numba-accelerated implementation, see :func:`simba.mixins.train_model_mixin.TrainModelMixin.find_highly_correlated_fields`
           For non-numba statistical methods, see :func:`simba.mixins.statistics_mixin.Statistics.find_collinear_features`


        .. csv-table::
           :header: EXPECTED RUNTIMES
           :file: ../../docs/tables/find_collinear_features.csv
           :widths: 10, 90
           :align: center
           :header-rows: 1

        :param pd.DataFrame data: Input DataFrame containing numeric features. Each column represents a feature and each row represents an observation. Must contain only numeric data types.
        :param float threshold: Correlation threshold for identifying collinear features. Must be between 0.0 and 1.0. Higher values (e.g., 0.9) identify only very highly correlated features, while lower values  (e.g., 0.1) identify more loosely correlated features.
        :return: List of column names that are highly correlated with other features and should be considered for removal to reduce multicollinearity.
        :rtype: List[str]

        :example:
        >>> a = np.random.randint(0, 5, (1_000_000, 100))
        >>> df = pd.DataFrame(a)
        >>> c = find_collinear_features(data=df, threshold=0.0025)
        """

        check_valid_dataframe(df=data, source=f'{TrainModelMixin.find_collinear_features.__name__} data', valid_dtypes=Formats.NUMERIC_DTYPES.value, allow_duplicate_col_names=False)
        check_float(name=f'{TrainModelMixin.find_collinear_features.__name__} threshold', value=threshold, min_value=0, max_value=1, raise_error=True)

        field_names = typed.List([str(x) for x in data.columns])

        x = TrainModelMixin.find_highly_correlated_fields(data=data.values.astype(np.float32), threshold=np.float64(threshold), field_names=field_names)
        return list(x)

    def check_sampled_dataset_integrity(self, x_df: pd.DataFrame, y_df: pd.DataFrame) -> None:
        """
        Helper to check for non-numerical entries post data sampling

        :parameter pd.DataFrame x_df: Features
        :parameter pd.DataFrame y_df: Target

        :raise FaultyTrainingSetError: Training or testing data sets contain non-numerical values
        """

        x_df = x_df.replace([np.inf, -np.inf, None], np.nan)
        x_nan_cnt = x_df.isna().sum()
        x_nan_cnt = x_nan_cnt[x_nan_cnt > 0]

        if len(x_nan_cnt) > 0:
            if len(x_nan_cnt) < 10:
                raise FaultyTrainingSetError(
                    msg=f"{str(len(x_nan_cnt))} feature column(s) exist in some files within the project_folder/csv/targets_inserted directory, but missing in others. "
                        f"SimBA expects all files within the project_folder/csv/targets_inserted directory to have the same number of features: the "
                        f"column names with mismatches are: {list(x_nan_cnt.index)}",
                    source=self.__class__.__name__,
                )
            else:
                raise FaultyTrainingSetError(
                    msg=f"{str(len(x_nan_cnt))} feature columns exist in some files, but missing in others. The feature files are found in the project_folder/csv/targets_inserted directory. "
                        f"SimBA expects all files within the project_folder/csv/targets_inserted directory to have the same number of features: the first 10 "
                        f"column names with mismatches are: {list(x_nan_cnt.index)[0:9]}",
                    source=self.__class__.__name__,
                )

        if len(y_df.unique()) == 1:
            if y_df.unique()[0] == 0:
                raise FaultyTrainingSetError(msg=f"All training annotations for classifier {str(y_df.name)} is labelled as ABSENT. A classifier has be be trained with both behavior PRESENT and ABSENT ANNOTATIONS.", source=self.__class__.__name__)
            if y_df.unique()[0] == 1:
                raise FaultyTrainingSetError(msg=f"All training annotations for classifier {str(y_df.name)} is labelled as PRESENT. A classifier has be be trained with both behavior PRESENT and ABSENT ANNOTATIONS.", source=self.__class__.__name__)

    def partial_dependence_calculator(self,
                                      clf: RandomForestClassifier,
                                      x_df: pd.DataFrame,
                                      clf_name: str,
                                      save_dir: Union[str, os.PathLike],
                                      clf_cnt: Optional[int] = None,
                                      grid_resolution: Optional[int] = 50,
                                      plot: Optional[bool] = True) -> None:

        """
        Compute feature partial dependencies for every feature in training set.

        :parameter RandomForestClassifier clf: Random forest classifier
        :parameter pd.DataFrame x_df: Features training set
        :parameter str clf_name: Name of classifier
        :parameter str save_dir: Directory where to save the data
        :parameter Optional[int] clf_cnt: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        """

        timer = SimbaTimer(start=True)
        print(f"Calculating partial dependencies for {len(x_df.columns)} features...")
        clf.verbose = 0
        check_if_dir_exists(save_dir)
        scikit_version = get_library_version(library_name='sklearn')
        if clf_cnt:
            save_dir = os.path.join(save_dir, f"partial_dependencies_{clf_name}_{clf_cnt}")
        else:
            save_dir = os.path.join(save_dir, f"partial_dependencies_{clf_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for feature_cnt, feature_name in enumerate(x_df.columns):
            feature_timer = SimbaTimer(start=True)
            save_path = os.path.join(save_dir, f"{feature_name}.csv")
            plot_path = os.path.join(save_dir, f"{feature_name}.png")
            if scikit_version == '0.22.2':
                pdp, axes = partial_dependence(clf, features=[feature_name], X=x_df, percentiles=(0, 1), grid_resolution=grid_resolution)
                df = pd.DataFrame({feature_name: axes[0], clf_name: pdp[0]})
            else:
                pdp = partial_dependence(clf, features=[feature_name], X=x_df, percentiles=(0, 1), grid_resolution=grid_resolution, kind='both')
                pdp, axes = pdp['values'], pdp['average']
                df = pd.DataFrame({feature_name: pdp[0], clf_name: axes[0]})
            if plot:
                _ = PlottingMixin.line_plot(df=df,
                                            x=feature_name,
                                            y=clf_name,
                                            save_path=plot_path,
                                            y_label='PARTIAL DEPENDENCE',
                                            x_label=feature_name,
                                            title=f'SimBA partial dependence {clf_name}')
            df.to_csv(save_path)
            feature_timer.stop_timer()
            print(f"Partial dependencies for {feature_name} complete ({feature_cnt + 1}/{len(x_df.columns)}) (elapsed time: {feature_timer.elapsed_time_str}s)...")

        timer.stop_timer()
        stdout_success(msg=f'Partial dependencies for {len(x_df.columns)} features saved in {save_dir}', elapsed_time=timer.elapsed_time_str)

    def clf_predict_proba(self,
                          clf: Union[RandomForestClassifier, cuRF],
                          x_df: Union[pd.DataFrame, np.ndarray],
                          multiclass: bool = False,
                          model_name: Optional[str] = None,
                          data_path: Optional[Union[str, os.PathLike]] = None) -> np.ndarray:

        """
        :param RandomForestClassifier clf: Random forest classifier object
        :param Union[pd.DataFrame, np.ndarray] x_df: Features for data to predict as a dataframe or array of size (M,N).
        :param bool multiclass: If True, the classifier predicts more than 2 targets. Else, boolean classifier.
        :param Optional[str] model_name: Name of model
        :param Optional[str] data_path: Path to model on disk
        :return np.ndarray: 2D array with frame represented by rows and present/absent probabilities as columns
        :raises FeatureNumberMismatchError: If shape of x_df and clf.n_features_ or n_features_in_ show mismatch
        """

        if hasattr(clf, "n_features_"):
            clf_n_features = clf.n_features_
        elif hasattr(clf, "n_features_in_"):
            clf_n_features = clf.n_features_in_
        else:
            raise InvalidInputError(msg=f"Could not determine the number of features in the classifier {model_name}", source=self.__class__.__name__)
        if hasattr(clf, "n_classes_"):
            clf_n_classes = clf.n_classes_
        elif hasattr(clf, "classes_"):
            clf_n_classes = len(clf.classes_)
        else:
            raise InvalidInputError(msg=f"Could not determine the number of classes in the classifier {model_name}", source=self.__class__.__name__)

        if not multiclass and clf_n_classes != 2:
            raise ClassifierInferenceError(msg=f"The classifier {model_name} (data path {data_path}) has not been created properly. See The SimBA GitHub FAQ page or Gitter for more information and suggested fixes. The classifier is not a binary classifier and does not predict two targets (absence and presence of behavior). One or more files inside the project_folder/csv/targets_inserted directory has an annotation column with a value other than 0 or 1", source=self.__class__.__name__,)
        if isinstance(x_df, pd.DataFrame):
            x_df = x_df.values
        if x_df.shape[1] != clf_n_features:
            if model_name and data_path:
                raise FeatureNumberMismatchError(f"Mismatch in the number of features in input file {data_path}, and what is expected by the model {model_name}. The model expects {clf_n_features} features. The data contains {x_df.shape[1]} features.", source=self.__class__.__name__)
            else:
                raise FeatureNumberMismatchError(f"The model expects {clf_n_features} features. The data contains {x_df.shape[1]} features.", source=self.__class__.__name__)
        p_vals = clf.predict_proba(x_df)
        if multiclass and (clf.n_classes_ != p_vals.shape[1]):
            raise ClassifierInferenceError(msg=f"The classifier {model_name} (data path: {data_path}) is a multiclassifier expected to create {clf.n_classes_} behavior probabilities. However, it produced probabilities for {p_vals.shape[1]} behaviors. See The SimBA GitHub FAQ page or Gitter for more information and suggested fixes.", source=self.__class__.__name__)
        if not multiclass:
            if isinstance(p_vals, pd.DataFrame):
                return p_vals[1].values
            else:
                return p_vals[:, 1]
        else:
            return p_vals

    def define_tree_explainer(self,
                              clf: RandomForestClassifier,
                              data: Optional[np.ndarray] = None,
                              model_output: str = 'raw',
                              feature_perturbation: str = "tree_path_dependent") -> shap.TreeExplainer:

        check_instance(source=f'{TrainModelMixin.define_tree_explainer.__name__} rf_clf', instance=clf, accepted_types=(RandomForestClassifier,))
        return shap.TreeExplainer(clf, data=data, model_output=model_output, feature_perturbation=feature_perturbation)

    def clf_define(self,
                   n_estimators: Optional[int] = 2000,
                   max_depth: Optional[int] = None,
                   max_features: Optional[Union[str, int]] = 'sqrt',
                   n_jobs: Optional[int] = -1,
                   criterion: Optional[str] = 'gini',
                   min_samples_leaf: Optional[int] = 1,
                   bootstrap: Optional[bool] = True,
                   verbose: Optional[int] = 1,
                   class_weight: Optional[dict] = None,
                   cuda: Optional[bool] = False) -> RandomForestClassifier:

        if not cuda:
            # NOTE: LOKY ISSUES ON WINDOWS WITH SCIKIT IF THE CORE COUNT EXCEEDS 61.
            # if n_jobs == -1:
            #     n_jobs = find_core_cnt()[0]
            # if (n_jobs > Defaults.THREADSAFE_CORE_COUNT.value) and (platform.system() == OS.WINDOWS.value):
            #     n_jobs = Defaults.THREADSAFE_CORE_COUNT.value
            return RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          max_features=max_features,
                                          n_jobs=n_jobs,
                                          criterion=criterion,
                                          min_samples_leaf=min_samples_leaf,
                                          bootstrap=bootstrap,
                                          verbose=verbose,
                                          class_weight=class_weight)

        else:
            if CUML in str(cuRF.__module__).lower():
                if max_depth is None or max_depth > 32:
                    max_depth = 32
                return cuRF(n_estimators=n_estimators,
                            bootstrap=bootstrap,
                            max_depth=max_depth,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf,
                            verbose=6)
            else:
                raise SimBAModuleNotFoundError(msg='SimBA could not find the cuml library for GPU machine learning algorithms. Set CUML to False in the SimBA model parameters, or import CUML environment variable using `export CUML=True`', source=self.__class__.__name__)

    def clf_fit(self,
                clf: Union[RandomForestClassifier, cuRF],
                x_df: pd.DataFrame,
                y_df: pd.DataFrame,
                ) -> RandomForestClassifier:

        """
        Helper to fit clf model

        :param clf: Un-fitted random forest classifier object
        :param pd.DataFrame x_df: Pandas dataframe with features.
        :param pd.DataFrame y_df: Pandas dataframe/Series with target
        :return: Fitted random forest classifier object
        :rtype: RandomForestClassifier
        """

        nan_features = x_df[~x_df.applymap(np.isreal).all(1)]
        nan_target = y_df.loc[pd.to_numeric(y_df).isna()]
        if len(nan_features) > 0:
            raise FaultyTrainingSetError(
                msg=f"{len(nan_features)} frame(s) in your project_folder/csv/targets_inserted directory contains FEATURES with non-numerical values",
                source=self.__class__.__name__)
        if len(nan_target) > 0:
            raise FaultyTrainingSetError(
                msg=f"{len(nan_target)} frame(s) in your project_folder/csv/targets_inserted directory contains ANNOTATIONS with non-numerical values",
                source=self.__class__.__name__)

        clf.fit(x_df, y_df)

        return clf

    @staticmethod
    def _read_data_file_helper(file_path: str,
                               file_type: str,
                               clf_names: Optional[List[str]] = None,
                               raise_bool_clf_error: bool = True):

        """
        Private function called by :meth:`simba.train_model_functions.read_all_files_in_folder_mp`
        """

        timer = SimbaTimer(start=True)
        _, vid_name, _ = get_fn_ext(file_path)
        df = read_df(file_path, file_type).dropna(axis=0, how="all").fillna(0)
        frame_numbers = df.index
        df.index = [vid_name] * len(df)
        if clf_names != None:
            for clf_cnt, clf_name in enumerate(clf_names):
                if not clf_name in df.columns:
                    if clf_name.lower() == Dtypes.NONE.value.lower():
                        raise InvalidInputError(msg=f'The classifier {clf_cnt+1} name is set to {clf_name}. Make sure you have correctly set the model hyperparameters as documented at {Links.TRAIN_ML_MODEL.value}', source=TrainModelMixin._read_data_file_helper.__name__)
                    else:
                        raise MissingColumnsError(msg=f'The SimBA project specifies a classifier named "{clf_name}" that could not be found in your dataset for file {file_path}. Make sure that your project_config.ini is created correctly as documented here: {Links.TRAIN_ML_MODEL.value}.', source=TrainModelMixin._read_data_file_helper.__name__)
                elif (len(set(df[clf_name].unique()) - {0, 1}) > 0 and raise_bool_clf_error):
                    raise InvalidInputError(msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {file_path} the {clf_name} field column contains additional value(s): {list(set(df[clf_name].unique()) - {0, 1})}.", source=TrainModelMixin._read_data_file_helper.__name__)
        timer.stop_timer()
        print(f"Reading complete {vid_name} (elapsed time: {timer.elapsed_time_str}s)...")

        return df, frame_numbers

    @staticmethod
    def read_all_files_in_folder_mp(file_paths: List[str],
                                    file_type: Literal["csv", "parquet", "pickle"],
                                    classifier_names: Optional[List[str]] = None,
                                    raise_bool_clf_error: bool = True) -> Tuple[pd.DataFrame, List[int]]:
        """

        Multiprocessing helper function to read in all data files in a folder to a single
        pd.DataFrame for downstream ML. Defaults to ceil(CPU COUNT / 2) cores. Asserts that all classifiers
        have annotation fields present in each dataframe.

        .. note::
          If multiprocess fail, reverts to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder`

        .. seealso::
           For single process method, use :func:`~simba.mixins.train_model_mixin.TrainModelMixin.read_all_files_in_folder`
           For `concurrent` library, use :func:`simba.mixins.train_model_mixin.TrainModelMixin.read_all_files_in_folder_mp_futures`.

        :param List[str] file_paths: List of file-paths
        :param List[str] file_paths: The filetype of ``file_paths`` OPTIONS: csv or parquet.
        :param Optional[List[str]] classifier_names: List of classifier names representing fields of human annotations. If not None, then assert that classifier names are present in each data file.
        :returns: concatenated DataFrame if all data represented in ``file_paths``, and an aligned list of frame numbers associated with the rows in the DataFrame.
        :rtype: Tuple[pd.DataFrame, List[int]]

        """
        if (platform.system() == "Darwin") and (
                multiprocessing.get_start_method() != "spawn"
        ):
            multiprocessing.set_start_method("spawn", force=True)
        cpu_cnt, _ = find_core_cnt()
        df_lst, frame_numbers_lst = [], []
        try:
            with ProcessPoolExecutor(int(np.ceil(cpu_cnt / 2))) as pool:
                for res in pool.map(
                        TrainModelMixin._read_data_file_helper,
                        file_paths,
                        repeat(file_type),
                        repeat(classifier_names),
                        repeat(raise_bool_clf_error),
                ):
                    df_lst.append(res[0])
                    frame_numbers_lst.extend((res[1]))
            df_concat = pd.concat(df_lst, axis=0).round(4)
            if "scorer" in df_concat.columns:
                df_concat = df_concat.drop(["scorer"], axis=1)
            if len(df_concat) == 0:
                raise NoDataError(
                    msg="SimBA found 0 observations (frames) in the project_folder/csv/targets_inserted directory",
                    source=TrainModelMixin.read_all_files_in_folder_mp.__name__,
                )
            df_concat = df_concat.loc[
                        :, ~df_concat.columns.str.contains("^Unnamed")
                        ].astype(np.float32)
            memory_size = get_memory_usage_of_df(df=df_concat)
            print(
                f'Dataset size: {memory_size["megabytes"]}MB / {memory_size["gigabytes"]}GB'
            )

            return df_concat, frame_numbers_lst

        except BrokenProcessPool or AttributeError:
            MultiProcessingFailedWarning(msg="Multi-processing file read failed, reverting to single core (increased run-time).")
            return TrainModelMixin().read_all_files_in_folder(
                file_paths=file_paths,
                file_type=file_type,
                classifier_names=classifier_names,
                raise_bool_clf_error=raise_bool_clf_error,
            )

    @staticmethod
    def _read_data_file_helper_futures(file_path: str,
                                       file_type: str,
                                       clf_names: Optional[List[str]] = None,
                                       raise_bool_clf_error: bool = True):
        """
        Private function called by :meth:`simba.train_model_functions.read_all_files_in_folder_mp_futures`
        """

        timer = SimbaTimer(start=True)
        _, vid_name, _ = get_fn_ext(file_path)
        df = read_df(file_path, file_type).dropna(axis=0, how="all").fillna(0)
        frm_numbers = list(df.index)
        df.index = [vid_name] * len(df)
        if clf_names != None:
            for clf_name in clf_names:
                if not clf_name in df.columns:
                    raise MissingColumnsError(msg=f'The SimBA project specifies a classifier named "{clf_name}" that could not be found in your dataset for file {file_path}. Make sure that your project_config.ini is created correctly.')
                elif (len(set(df[clf_name].unique()) - {0, 1}) > 0 and raise_bool_clf_error):
                    raise InvalidInputError(msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {file_path} the {clf_name} field contains additional value(s): {list(set(df[clf_name].unique()) - {0, 1})}.")
        timer.stop_timer()
        return df, vid_name, timer.elapsed_time_str, frm_numbers

    def read_all_files_in_folder_mp_futures(self,
                                            annotations_file_paths: List[str],
                                            file_type: Literal["csv", "parquet", "pickle"],
                                            classifier_names: Optional[List[str]] = None,
                                            raise_bool_clf_error: bool = True) -> Tuple[pd.DataFrame, List[int]]:

        """
        Multiprocessing helper function to read in all data files in a folder to a single
        pd.DataFrame for downstream ML through ``concurrent.Futures``. Asserts that all classifiers
        have annotation fields present in each dataframe.

        .. note::
           A ``concurrent.Futures`` alternative to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder_mp` which
           has uses ``multiprocessing.ProcessPoolExecutor`` and reported unstable on Linux machines.

           If multiprocess failure, reverts to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder`

        .. seealso::
           For single process method, use :func:`~simba.mixins.train_model_mixin.TrainModelMixin.read_all_files_in_folder`
           For improved runtime using multiprocessing and pyarrow, use :func:`~simba.mixins.train_model_mixin.read_all_files_in_folder_mp`

        :param List[str] file_paths: List of file-paths
        :param List[str] file_paths: The filetype of ``file_paths`` OPTIONS: csv or parquet.
        :param Optional[List[str]] classifier_names: List of classifier names representing fields of human annotations. If not None, then assert that classifier names are present in each data file.
        :param bool raise_bool_clf_error: If True, raises an error if a classifier column contains values outside 0 and 1.
        :returns: concatenated DataFrame if all data represented in ``file_paths``, and an aligned list of frame numbers associated with the rows in the DataFrame.
        :rtype: Tuple[pd.DataFrame, List[int]]

        """

        THREADSAFE_CORE_COUNT = 16
        check_filepaths_in_iterable_exist(file_paths=annotations_file_paths, name=self.__class__.__name__)
        try:
            if (platform.system() == "Darwin") and (multiprocessing.get_start_method() != "spawn"):
                multiprocessing.set_start_method("spawn", force=True)
            cpu_cnt, _ = find_core_cnt()
            if (cpu_cnt > THREADSAFE_CORE_COUNT) and (platform.system() == OS.WINDOWS.value):
                cpu_cnt = THREADSAFE_CORE_COUNT
            dfs, frm_number_list = [], []
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_cnt) as executor:
                results = [executor.submit(self._read_data_file_helper_futures, data, file_type, classifier_names, raise_bool_clf_error) for data in annotations_file_paths]
                for result in concurrent.futures.as_completed(results):
                    dfs.append(result.result()[0])
                    frm_number_list.extend((result.result()[-1]))
                    print(f"Reading complete {result.result()[1]} (elapsed time: {result.result()[2]}s)...")

            check_all_dfs_in_list_has_same_cols(dfs=dfs, source='/project_folder/csv/targets_inserted', raise_error=True)
            col_headers = [list(x.columns) for x in dfs]
            dfs = [x[col_headers[0]] for x in dfs]
            dfs = pd.concat(dfs, axis=0).round(4)
            if "scorer" in dfs.columns: dfs = dfs.drop(["scorer"], axis=1)
            memory_size = get_memory_usage_of_df(df=dfs)
            print(f'Dataset size: {memory_size["megabytes"]}MB / {memory_size["gigabytes"]}GB')
            return dfs, frm_number_list

        except Exception as e:
            MultiProcessingFailedWarning(msg=f"Multi-processing file read failed, reverting to single core (increased run-time on large datasets). Exception: {e.args}")
            return self.read_all_files_in_folder(
                file_paths=annotations_file_paths,
                file_type=file_type,
                classifier_names=classifier_names,
                raise_bool_clf_error=raise_bool_clf_error,
            )

    def check_raw_dataset_integrity(self, df: pd.DataFrame, logs_path: Optional[Union[str, os.PathLike]]) -> None:
        """
        Helper to check column-wise NaNs in raw input data for fitting model.

        :param pd.DataFrame df
        :param str logs_path: The logs directory of the SimBA project
        :raise FaultyTrainingSetError: When the dataset contains NaNs
        """

        nan_cols = (
            df.reset_index(drop=True)
            .replace([np.inf, -np.inf, None], np.nan)
            .columns[df.isna().any()]
            .tolist()
        )
        if len(nan_cols) == 0:
            return df.reset_index(drop=True)
        else:
            save_log_path = os.path.join(
                logs_path,
                f'missing_columns_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv',
            )
            results = {}
            for video in list(df.index.unique()):
                results[video] = {}
            for nan_col in nan_cols:
                nan_videos = list(df[df[nan_col].isna()].index.unique())
                non_nan_video = [x for x in list(results.keys()) if x not in nan_videos]
                for video in nan_videos:
                    results[video][nan_col] = False
                for video in non_nan_video:
                    results[video][nan_col] = True
            results = pd.DataFrame.from_dict(data=results, orient="index")
            results.to_csv(save_log_path)
            raise FaultyTrainingSetError(
                msg=f"{len(nan_cols)} feature columns exist in some files, but missing in others. The feature files are found in the project_folder/csv/targets_inserted directory. "
                    f"SimBA expects all files within the project_folder/csv/targets_inserted directory to have the same number of features: the first 10 "
                    f"column names with mismatches are: {nan_cols[0:9]}. For a log of the files that contain, and not contain, the mis-matched columns, see {save_log_path}",
                source=self.__class__.__name__,
            )


    @staticmethod
    def _create_shap_mp_helper(data: Tuple[int, pd.DataFrame],
                               explainer: shap.TreeExplainer,
                               clf_name: str,
                               verbose: bool) -> Tuple[np.ndarray, int]:

        if verbose:
            print(f'Processing SHAP core batch {data[0] + 1}... ({len(data[1])} observations)')
        _ = data[1].pop(clf_name).values.reshape(-1, 1)
        shap_batch_results = np.full(shape=(len(data[1]), len(data[1].columns)), fill_value=np.nan, dtype=np.float32)
        for idx in range(len(data[1])):
            timer = SimbaTimer(start=True)
            obs = data[1].iloc[idx, :].values
            shap_batch_results[idx] = explainer.shap_values(obs, check_additivity=False)[1]
            timer.stop_timer()
            if verbose:
                print(f'SHAP frame complete (core batch: {data[0] + 1}, core batch frame: {idx+1}/{len(data[1])}, frame processing time: {timer.elapsed_time_str}s)')

        return shap_batch_results, data[0]

    def create_shap_log_mp(self,
                           rf_clf: RandomForestClassifier,
                           x: Union[pd.DataFrame, np.ndarray],
                           y: Union[pd.DataFrame, pd.Series, np.ndarray],
                           x_names: List[str],
                           clf_name: str,
                           cnt_present: int,
                           cnt_absent: int,
                           core_cnt: int = -1,
                           chunk_size: int = 100,
                           verbose: bool = True,
                           save_dir: Optional[Union[str, os.PathLike]] = None,
                           save_file_suffix: Optional[int] = None,
                           plot: bool = False) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray]]:
        """
        Compute SHAP values using multiprocessing.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_
            For single-core solution, see :func:`~simba.mixins.train_model_mixins.TrainModelMixin.create_shap_log`
            For GPU method, see :func:`~simba.data_processors.cuda.create_shap_log.create_shap_log`
            For multiprocassing concurrent futures method (should be more reliable on Linux distros), see :func:`~simba.mixins.train_model_mixin.TrainModelMixin.create_shap_log_concurrent_mp`

        .. image:: _static/img/shap.png
           :width: 400
           :align: center

        :param RandomForestClassifier rf_clf: Fitted sklearn random forest classifier
        :param Union[pd.DataFrame, np.ndarray] x: Test features.
        :param Union[pd.DataFrame, pd.Series, np.ndarray] y_df: Test target.
        :param List[str] x_names: Feature names.
        :param str clf_name: Classifier name.
        :param int cnt_present: Number of behavior-present frames to calculate SHAP values for.
        :param int cnt_absent: Number of behavior-absent frames to calculate SHAP values for.
        :param int chunk_size: How many observations to process in each chunk. Increase value for faster processing if your memory allows.
        :param bool verbose: If True, prints progress.
        :param Optional[Union[str, os.PathLike]] save_dir: Optional directory where to store the results. If None, then the results are returned.
        :param Optional[int] save_file_suffix: Optional suffix to add to the shap output filenames. Useful for gridsearches and multiple shap data output files are to-be stored in the same `save_dir`.
        :param bool plot: If True, create SHAP aggregation and plots.

        :example:
        >>> from simba.mixins.train_model_mixin import TrainModelMixin
        >>> x_cols = list(pd.read_csv('/Users/simon/Desktop/envs/simba/simba/tests/data/sample_data/shap_test.csv', index_col=0).columns)
        >>> x = pd.DataFrame(np.random.randint(0, 500, (9000, len(x_cols))), columns=x_cols)
        >>> y = pd.Series(np.random.randint(0, 2, (9000,)))
        """

        if SKLEARN in str(rf_clf.__module__).lower():
            timer = SimbaTimer(start=True)
            check_instance(source=f'{TrainModelMixin.create_shap_log_mp.__name__} rf_clf', instance=rf_clf, accepted_types=(RandomForestClassifier,))
            check_instance(source=f'{TrainModelMixin.create_shap_log_mp.__name__} x', instance=x, accepted_types=(np.ndarray, pd.DataFrame))
            check_instance(source=f'{TrainModelMixin.create_shap_log_mp.__name__} y', instance=y, accepted_types=(np.ndarray, pd.Series, pd.DataFrame))
            if isinstance(x, pd.DataFrame):
                check_valid_dataframe(df=x, source=f'{TrainModelMixin.create_shap_log_mp.__name__} x', valid_dtypes=Formats.NUMERIC_DTYPES.value)
                x = x.values
            else:
                check_valid_array(data=x, source=f'{TrainModelMixin.create_shap_log_mp.__name__} x', accepted_ndims=(2,),  accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            if isinstance(y, pd.DataFrame):
                check_valid_dataframe(df=y, source=f'{TrainModelMixin.create_shap_log_mp.__name__} y', valid_dtypes=Formats.NUMERIC_DTYPES.value, max_axis_1=1)
                y = y.values
            else:
                if isinstance(y, pd.Series):
                    y = y.values
                check_valid_array(data=y, source=f'{TrainModelMixin.create_shap_log_mp.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            check_valid_lst(data=x_names, source=f'{TrainModelMixin.create_shap_log_mp.__name__} x_names', valid_dtypes=(str,), exact_len=x.shape[1])
            check_str(name=f'{TrainModelMixin.create_shap_log_mp.__name__} clf_name', value=clf_name)
            check_int(name=f'{TrainModelMixin.create_shap_log_mp.__name__} cnt_present', value=cnt_present, min_value=1)
            check_int(name=f'{TrainModelMixin.create_shap_log_mp.__name__} cnt_absent', value=cnt_absent, min_value=1)
            check_int(name=f'{TrainModelMixin.create_shap_log_mp.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
            check_int(name=f'{TrainModelMixin.create_shap_log_mp.__name__} chunk_size', value=chunk_size, min_value=1)
            check_valid_boolean(value=[verbose, plot], source=f'{TrainModelMixin.create_shap_log_mp.__name__} verbose, plot')
            core_cnt = [find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt][0]
            df = pd.DataFrame(np.hstack((x, y.reshape(-1, 1))), columns=x_names + [clf_name])
            del x; del y
            present_df, absent_df = df[df[clf_name] == 1], df[df[clf_name] == 0]
            if len(present_df) == 0:
                raise NoDataError(msg=f'Cannot calculate SHAP values: no target PRESENT annotations detected.', source=TrainModelMixin.create_shap_log_mp.__name__)
            elif len(absent_df) == 0:
                raise NoDataError(msg=f'Cannot calculate SHAP values: no target ABSENT annotations detected.', source=TrainModelMixin.create_shap_log_mp.__name__)
            if len(present_df) < cnt_present:
                NotEnoughDataWarning(msg=f"Train data contains {len(present_df)} behavior-present annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_present)}). SimBA will calculate shap scores for the {len(present_df)} behavior-present frames available", source=TrainModelMixin.create_shap_log_mp.__name__)
                cnt_present = len(present_df)
            if len(absent_df) < cnt_absent:
                NotEnoughDataWarning(msg=f"Train data contains {len(absent_df)} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_absent)}). SimBA will calculate shap scores for the {len(absent_df)} behavior-absent frames available", source=TrainModelMixin.create_shap_log_mp.__name__)
                cnt_absent = len(absent_df)
            shap_data = pd.concat([present_df.sample(cnt_present, replace=False), absent_df.sample(cnt_absent, replace=False)], axis=0).reset_index(drop=True)
            batch_cnt = max(1, int(np.ceil(len(shap_data) / chunk_size)))
            shap_data = np.array_split(shap_data, batch_cnt)
            shap_data = [(x, y) for x, y in enumerate(shap_data)]
            explainer = TrainModelMixin().define_tree_explainer(clf=rf_clf)
            expected_value = explainer.expected_value[1]
            shap_results, shap_raw = [], []
            print(f"Computing {cnt_present + cnt_absent} SHAP values. Follow progress in OS terminal... (CORES: {core_cnt}, CHUNK SIZE: {chunk_size})")
            with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
                constants = functools.partial(TrainModelMixin._create_shap_mp_helper, explainer=explainer, clf_name=clf_name, verbose=verbose)
                for cnt, result in enumerate(pool.imap(constants, shap_data, chunksize=1)):
                    proba = TrainModelMixin().clf_predict_proba(clf=rf_clf, x_df=shap_data[result[1]][1].drop(clf_name, axis=1), model_name=clf_name).reshape(-1, 1)
                    shap_sum = np.sum(result[0], axis=1).reshape(-1, 1)
                    batch_shap_results = np.hstack((result[0], np.full((result[0].shape[0]), expected_value).reshape(-1, 1), shap_sum + expected_value, proba, shap_data[result[1]][1][clf_name].values.reshape(-1, 1))).astype(np.float32)
                    shap_results.append(batch_shap_results)
                    shap_raw.append(shap_data[result[1]][1].drop(clf_name, axis=1))
                    if verbose: print(f"Completed SHAP care batch (Batch {result[1] + 1}/{len(shap_data)}).")

            pool.terminate(); pool.join()
            shap_df = pd.DataFrame(data=np.row_stack(shap_results), columns=list(x_names) + ["Expected_value", "Sum", "Prediction_probability", clf_name])
            raw_df = pd.DataFrame(data=np.row_stack(shap_raw), columns=list(x_names))
            out_shap_path, out_raw_path, img_save_path, df_save_paths, summary_dfs, img = None, None, None, None, None, None
            if save_dir is not None:
                check_if_dir_exists(in_dir=save_dir)
                if save_file_suffix is not None:
                    check_int(name=f'{TrainModelMixin.create_shap_log_mp.__name__} save_file_no', value=save_file_suffix, min_value=0)
                    out_shap_path = os.path.join(save_dir, f"SHAP_values_{save_file_suffix}_{clf_name}.csv")
                    out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{save_file_suffix}_{clf_name}.csv")
                    df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT_{save_file_suffix}.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT_{save_file_suffix}.csv")}
                    img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}_{save_file_suffix}.png")
                else:
                    out_shap_path = os.path.join(save_dir, f"SHAP_values_{clf_name}.csv")
                    out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{clf_name}.csv")
                    df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT.csv")}
                    img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}.png")

                shap_df.to_csv(out_shap_path); raw_df.to_csv(out_raw_path)
            if plot:
                shap_computer = ShapAggregateStatisticsCalculator(classifier_name=clf_name, shap_df=shap_df, shap_baseline_value=int(expected_value * 100), save_dir=None)
                summary_dfs, img = shap_computer.run()
                if save_dir is not None:
                    summary_dfs['PRESENT'].to_csv(df_save_paths['PRESENT'])
                    summary_dfs['ABSENT'].to_csv(df_save_paths['ABSENT'])
                    cv2.imwrite(img_save_path, img)

            timer.stop_timer()
            if save_dir and verbose:
                stdout_success(msg=f'SHAP data saved in {save_dir}', source=TrainModelMixin.create_shap_log_mp.__name__, elapsed_time=timer.elapsed_time_str)
            if not save_dir:
                return shap_df, raw_df, summary_dfs, img
        else:
            GPUToolsWarning(msg=f'Cannot compute SHAP scores using cuml random forest model. To compute SHAP scores, turn off cuda. Alternatively, for GPU solution, see simba/data_processors/cuda/create_shap_log.create_shap_log')

    def check_df_dataset_integrity(self, df: pd.DataFrame, file_name: str, logs_path: Union[str, os.PathLike]) -> None:
        """
        Helper to check for non-numerical np.inf, -np.inf, NaN, None in a single dataframe.
        :parameter pd.DataFrame x_df: Features
        :raise NoDataError: If data contains np.inf, -np.inf, None.
        """

        df = df.replace([np.inf, -np.inf, None], np.nan)
        x_nan_cnt = df.isna().sum()
        x_nan_cnt = x_nan_cnt[x_nan_cnt > 0]
        if 0 < len(x_nan_cnt) <= 10:
            msg = f"{len(x_nan_cnt)} feature column(s) exist in {file_name} with non-numerical values. The columns with non-numerical values are: {list(x_nan_cnt.index)}"
            raise NoDataError(msg=msg, source=self.check_df_dataset_integrity.__name__)
        elif len(x_nan_cnt) > 10:
            logs_path = os.path.join(logs_path, f"corrupt_columns_{file_name}.csv")
            msg = f"{len(x_nan_cnt)} feature column(s) exist in {file_name} with non-numerical values. The first 10 columns with non-numerical values are: {list(x_nan_cnt.index)[0:10]}. For a full list of columns with missing values, see {logs_path}"
            x_nan_cnt.to_csv(logs_path)
            raise NoDataError(msg=msg, source=self.check_df_dataset_integrity.__name__)
        else:
            pass

    def read_model_settings_from_config(self, config: configparser.ConfigParser):
        self.model_dir_out = os.path.join(read_config_entry(config, ConfigKey.SML_SETTINGS.value, ConfigKey.MODEL_DIR.value, data_type=Dtypes.STR.value), "generated_models")
        if not os.path.exists(self.model_dir_out):
            os.makedirs(self.model_dir_out)
        self.eval_out_path = os.path.join(self.model_dir_out, "model_evaluations")
        if not os.path.exists(self.eval_out_path):
            os.makedirs(self.eval_out_path)
        self.clf_name = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLASSIFIER.value, data_type=Dtypes.STR.value)
        self.tt_size = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.TT_SIZE.value,
                                         data_type=Dtypes.FLOAT.value)
        self.algo = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.MODEL_TO_RUN.value, data_type=Dtypes.STR.value, default_value="RF")
        self.split_type = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                            MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value, data_type=Dtypes.STR.value,
                                            options=Options.TRAIN_TEST_SPLIT.value,
                                            default_value=Methods.SPLIT_TYPE_FRAMES.value)
        self.under_sample_setting = (
            read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.UNDERSAMPLE_SETTING.value,
                              data_type=Dtypes.STR.value).lower().strip())
        self.over_sample_setting = (
            read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.OVERSAMPLE_SETTING.value,
                              data_type=Dtypes.STR.value).lower().strip())
        self.n_estimators = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                              MLParamKeys.RF_ESTIMATORS.value, data_type=Dtypes.INT.value)
        self.rf_max_depth = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                              MLParamKeys.RF_MAX_DEPTH.value, data_type=Dtypes.INT.value,
                                              default_value=Dtypes.NONE.value)
        if self.rf_max_depth == "None":
            self.rf_max_depth = None
        self.max_features = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                              MLParamKeys.RF_MAX_FEATURES.value, data_type=Dtypes.STR.value)
        self.criterion = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                           MLParamKeys.RF_CRITERION.value, data_type=Dtypes.STR.value,
                                           options=Options.CLF_CRITERION.value)
        self.min_sample_leaf = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                 MLParamKeys.MIN_LEAF.value, data_type=Dtypes.INT.value)
        self.compute_permutation_importance = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                MLParamKeys.PERMUTATION_IMPORTANCE.value,
                                                                data_type=Dtypes.STR.value, default_value=False)
        self.generate_learning_curve = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                         MLParamKeys.LEARNING_CURVE.value, data_type=Dtypes.STR.value,
                                                         default_value=False)
        self.generate_precision_recall_curve = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                 MLParamKeys.PRECISION_RECALL.value,
                                                                 data_type=Dtypes.STR.value, default_value=False)
        self.generate_example_decision_tree = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                MLParamKeys.EX_DECISION_TREE.value,
                                                                data_type=Dtypes.STR.value, default_value=False)
        self.generate_classification_report = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                MLParamKeys.CLF_REPORT.value,
                                                                data_type=Dtypes.STR.value, default_value=False)
        self.generate_features_importance_log = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                  MLParamKeys.IMPORTANCE_LOG.value,
                                                                  data_type=Dtypes.STR.value, default_value=False)
        self.generate_features_importance_bar_graph = read_config_entry(config,
                                                                        ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                        MLParamKeys.IMPORTANCE_LOG.value,
                                                                        data_type=Dtypes.STR.value, default_value=False)
        self.generate_example_decision_tree_fancy = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                      MLParamKeys.EX_DECISION_TREE_FANCY.value,
                                                                      data_type=Dtypes.STR.value, default_value=False)
        self.generate_shap_scores = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                      MLParamKeys.SHAP_SCORES.value, data_type=Dtypes.STR.value,
                                                      default_value=False)
        self.save_meta_data = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                MLParamKeys.RF_METADATA.value, data_type=Dtypes.STR.value,
                                                default_value=False)
        self.compute_partial_dependency = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                            MLParamKeys.PARTIAL_DEPENDENCY.value,
                                                            data_type=Dtypes.STR.value, default_value=False)
        self.save_train_test_frm_info = str_2_bool(read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                                     MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value,
                                                                     data_type=Dtypes.STR.value, default_value="False"))

        if self.under_sample_setting == Methods.RANDOM_UNDERSAMPLE.value:
            self.under_sample_ratio = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                        MLParamKeys.UNDERSAMPLE_RATIO.value,
                                                        data_type=Dtypes.FLOAT.value, default_value=Dtypes.NAN.value)
            check_float(name=MLParamKeys.UNDERSAMPLE_RATIO.value, value=self.under_sample_ratio)
        else:
            self.under_sample_ratio = Dtypes.NAN.value
        if (self.over_sample_setting == Methods.SMOTEENN.value.lower()) or (
                self.over_sample_setting == Methods.SMOTE.value.lower()):
            self.over_sample_ratio = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                       MLParamKeys.OVERSAMPLE_RATIO.value, data_type=Dtypes.FLOAT.value,
                                                       default_value=Dtypes.NAN.value)
            check_float(name=MLParamKeys.OVERSAMPLE_RATIO.value, value=self.over_sample_ratio)
        else:
            self.over_sample_ratio = Dtypes.NAN.value

        if config.has_option(ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLASS_WEIGHTS.value):
            self.class_weights = read_config_entry(config, ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                                                   MLParamKeys.CLASS_WEIGHTS.value, data_type=Dtypes.STR.value,
                                                   default_value=Dtypes.NONE.value)
            if self.class_weights == "custom":
                self.class_weights = ast.literal_eval(
                    read_config_entry(
                        config,
                        ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                        MLParamKeys.CLASS_CUSTOM_WEIGHTS.value,
                        data_type=Dtypes.STR.value,
                    )
                )
                for k, v in self.class_weights.items():
                    self.class_weights[k] = int(v)
            if self.class_weights == Dtypes.NONE.value:
                self.class_weights = None
        else:
            self.class_weights = None

        if self.generate_learning_curve in Options.PERFORM_FLAGS.value:
            self.shuffle_splits = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.LEARNING_CURVE_K_SPLITS.value,
                data_type=Dtypes.INT.value,
                default_value=Dtypes.NAN.value,
            )
            self.dataset_splits = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.LEARNING_DATA_SPLITS.value,
                data_type=Dtypes.INT.value,
                default_value=Dtypes.NAN.value,
            )
            check_int(
                name=MLParamKeys.LEARNING_CURVE_K_SPLITS.value,
                value=self.shuffle_splits,
            )
            check_int(
                name=MLParamKeys.LEARNING_DATA_SPLITS.value, value=self.dataset_splits
            )
        else:
            self.shuffle_splits, self.dataset_splits = (
                Dtypes.NAN.value,
                Dtypes.NAN.value,
            )
        if self.generate_features_importance_bar_graph in Options.PERFORM_FLAGS.value:
            self.feature_importance_bars = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.IMPORTANCE_BARS_N.value,
                Dtypes.INT.value,
                Dtypes.NAN.value,
            )
            check_int(
                name=MLParamKeys.IMPORTANCE_BARS_N.value,
                value=self.feature_importance_bars,
                min_value=1,
            )
        else:
            self.feature_importance_bars = Dtypes.NAN.value
        (
            self.shap_target_present_cnt,
            self.shap_target_absent_cnt,
            self.shap_save_n,
            self.shap_multiprocess,
        ) = (None, None, None, None)
        if self.generate_shap_scores in Options.PERFORM_FLAGS.value:
            self.shap_target_present_cnt = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.SHAP_PRESENT.value,
                data_type=Dtypes.INT.value,
                default_value=0,
            )
            self.shap_target_absent_cnt = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.SHAP_ABSENT.value,
                data_type=Dtypes.INT.value,
                default_value=0,
            )
            self.shap_save_n = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.SHAP_SAVE_ITERATION.value,
                data_type=Dtypes.STR.value,
                default_value=Dtypes.NONE.value,
            )
            self.shap_multiprocess = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.SHAP_MULTIPROCESS.value,
                data_type=Dtypes.STR.value,
                default_value="False",
            )
            try:
                self.shap_save_n = int(self.shap_save_n)
            except ValueError or TypeError:
                self.shap_save_n = (
                        self.shap_target_present_cnt + self.shap_target_absent_cnt
                )
            check_int(
                name=MLParamKeys.SHAP_PRESENT.value, value=self.shap_target_present_cnt
            )
            check_int(
                name=MLParamKeys.SHAP_ABSENT.value, value=self.shap_target_absent_cnt
            )
        if self.rf_max_depth != None:
            check_int(name="RF MAX DEPTH", value=self.rf_max_depth, min_value=1)

    def check_validity_of_meta_files(self,
                                     data_df: pd.DataFrame,
                                     meta_file_paths: List[Union[str, os.PathLike]]):


        meta_dicts, errors = {}, []
        for config_cnt, path in enumerate(meta_file_paths):
            _, meta_file_name, _ = get_fn_ext(path)
            meta_dict = read_meta_file(path)
            meta_dict = {k.lower(): v for k, v in meta_dict.items()}
            errors.append(check_str(name=meta_dict[MLParamKeys.CLASSIFIER_NAME.value], value=meta_dict[MLParamKeys.CLASSIFIER_NAME.value], raise_error=False)[1])
            errors.append(check_str(name=MLParamKeys.RF_CRITERION.value, value=meta_dict[MLParamKeys.RF_CRITERION.value], options=Options.CLF_CRITERION.value, raise_error=False)[1])
            errors.append(check_str(name=MLParamKeys.RF_MAX_FEATURES.value, value=meta_dict[MLParamKeys.RF_MAX_FEATURES.value],options=Options.CLF_MAX_FEATURES.value, raise_error=False)[1])

            if not isinstance(meta_dict[MLParamKeys.UNDERSAMPLE_SETTING.value], str): meta_dict[MLParamKeys.UNDERSAMPLE_SETTING.value] = Dtypes.NONE.value
            if not isinstance(meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value], str): meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value] = Dtypes.NONE.value
            errors.append(check_str(MLParamKeys.UNDERSAMPLE_SETTING.value, meta_dict[MLParamKeys.UNDERSAMPLE_SETTING.value].lower(), options=[x.lower() for x in Options.UNDERSAMPLE_OPTIONS.value], raise_error=False)[1])
            errors.append(check_str(MLParamKeys.OVERSAMPLE_SETTING.value, meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower(), options=[x.lower() for x in Options.OVERSAMPLE_OPTIONS.value], raise_error=False)[1])
            if MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value in meta_dict.keys():
                errors.append(check_str(name=meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value], value=meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value], options=Options.TRAIN_TEST_SPLIT.value, raise_error=False)[1])

            errors.append(check_int(name=MLParamKeys.RF_ESTIMATORS.value, value=meta_dict[MLParamKeys.RF_ESTIMATORS.value],  min_value=1, raise_error=False)[1])
            errors.append(check_int(name=MLParamKeys.MIN_LEAF.value, value=meta_dict[MLParamKeys.MIN_LEAF.value], raise_error=False)[1])
            if (meta_dict[MLParamKeys.LEARNING_CURVE.value] in Options.PERFORM_FLAGS.value):
                errors.append(check_int(name=MLParamKeys.LEARNING_CURVE_K_SPLITS.value, value=meta_dict[MLParamKeys.LEARNING_CURVE_K_SPLITS.value], raise_error=False)[1])
                errors.append(check_int(name=MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value, value=meta_dict[MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value],raise_error=False)[1])
            if (meta_dict[MLParamKeys.IMPORTANCE_BAR_CHART.value] in Options.PERFORM_FLAGS.value):
                errors.append(check_int(name=MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value, value=meta_dict[MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value], raise_error=False)[1])
            if MLParamKeys.SHAP_SCORES.value in meta_dict.keys():
                if (meta_dict[MLParamKeys.SHAP_SCORES.value] in Options.PERFORM_FLAGS.value):
                    errors.append(check_int(name=MLParamKeys.SHAP_PRESENT.value, value=meta_dict[MLParamKeys.SHAP_PRESENT.value], raise_error=False)[1])
                    errors.append(check_int(name=MLParamKeys.SHAP_ABSENT.value, value=meta_dict[MLParamKeys.SHAP_ABSENT.value], raise_error=False)[1])
            if MLParamKeys.RF_MAX_DEPTH.value in meta_dict.keys():
                if meta_dict[MLParamKeys.RF_MAX_DEPTH.value] != Dtypes.NONE.value:
                    errors.append(check_int(name=MLParamKeys.RF_MAX_DEPTH.value, value=meta_dict[MLParamKeys.RF_MAX_DEPTH.value], min_value=1, raise_error=False)[1])
                else:
                    meta_dict[MLParamKeys.RF_MAX_DEPTH.value] = None
            else:
                meta_dict[MLParamKeys.RF_MAX_DEPTH.value] = None

            errors.append(check_float(name=MLParamKeys.TT_SIZE.value, value=meta_dict[MLParamKeys.TT_SIZE.value], raise_error=False)[1])
            if (meta_dict[MLParamKeys.UNDERSAMPLE_SETTING.value].lower() == Methods.RANDOM_UNDERSAMPLE.value):
                errors.append(check_float(name=MLParamKeys.UNDERSAMPLE_RATIO.value, value=meta_dict[MLParamKeys.UNDERSAMPLE_RATIO.value], raise_error=False)[1])
                try:
                    present_len, absent_len = len(data_df[data_df[meta_dict[MLParamKeys.CLASSIFIER_NAME.value]] == 1]), len(data_df[data_df[meta_dict[MLParamKeys.CLASSIFIER_NAME.value]] == 0])
                    ratio_n = int(present_len * meta_dict[MLParamKeys.UNDERSAMPLE_RATIO.value])
                    if absent_len < ratio_n:
                        errors.append(f"The under-sample ratio of {meta_dict[MLParamKeys.UNDERSAMPLE_RATIO.value]} in \n classifier {meta_dict[MLParamKeys.CLASSIFIER_NAME.value]} demands {ratio_n} behavior-absent annotations.")
                except:
                    pass

            if (meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTEENN.value.lower()) or (meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower() == Methods.SMOTE.value.lower()):
                errors.append(check_float(name=MLParamKeys.OVERSAMPLE_RATIO.value, value=meta_dict[MLParamKeys.OVERSAMPLE_RATIO.value], raise_error=False)[1])

            errors.append(check_if_valid_input(name=MLParamKeys.RF_METADATA.value, input=meta_dict[MLParamKeys.RF_METADATA.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            errors.append(check_if_valid_input(MLParamKeys.EX_DECISION_TREE.value, input=meta_dict[MLParamKeys.EX_DECISION_TREE.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            errors.append(check_if_valid_input(MLParamKeys.CLF_REPORT.value, input=meta_dict[MLParamKeys.CLF_REPORT.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            errors.append(check_if_valid_input(MLParamKeys.IMPORTANCE_LOG.value, input=meta_dict[MLParamKeys.IMPORTANCE_LOG.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            errors.append(check_if_valid_input( MLParamKeys.IMPORTANCE_BAR_CHART.value, input=meta_dict[MLParamKeys.IMPORTANCE_BAR_CHART.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            errors.append(check_if_valid_input(MLParamKeys.PERMUTATION_IMPORTANCE.value, input=meta_dict[MLParamKeys.PERMUTATION_IMPORTANCE.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            errors.append(check_if_valid_input(MLParamKeys.LEARNING_CURVE.value, input=meta_dict[MLParamKeys.LEARNING_CURVE.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            errors.append(check_if_valid_input(MLParamKeys.PRECISION_RECALL.value, input=meta_dict[MLParamKeys.PRECISION_RECALL.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])

            if MLParamKeys.PARTIAL_DEPENDENCY.value in meta_dict.keys():
                errors.append(check_if_valid_input(MLParamKeys.PARTIAL_DEPENDENCY.value, input=meta_dict[MLParamKeys.PARTIAL_DEPENDENCY.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            if MLParamKeys.SHAP_MULTIPROCESS.value in meta_dict.keys():
                errors.append(check_if_valid_input(MLParamKeys.SHAP_MULTIPROCESS.value, input=meta_dict[MLParamKeys.SHAP_MULTIPROCESS.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            if meta_dict[MLParamKeys.RF_MAX_FEATURES.value] == Dtypes.NONE.value:
                meta_dict[MLParamKeys.RF_MAX_FEATURES.value] = None
            if MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value not in meta_dict.keys():
                meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value] = (Methods.SPLIT_TYPE_FRAMES.value)

            if MLParamKeys.CLASS_WEIGHTS.value in meta_dict.keys():
                if (meta_dict[MLParamKeys.CLASS_WEIGHTS.value] not in Options.CLASS_WEIGHT_OPTIONS.value):
                    meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = None
                if meta_dict[MLParamKeys.CLASS_WEIGHTS.value] == "custom":
                    meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = ast.literal_eval(meta_dict[MLParamKeys.CLASS_CUSTOM_WEIGHTS.value])
                    for k, v in meta_dict[MLParamKeys.CLASS_WEIGHTS.value].items():
                        meta_dict[MLParamKeys.CLASS_WEIGHTS.value][k] = int(v)
                if meta_dict[MLParamKeys.CLASS_WEIGHTS.value] == Dtypes.NONE.value:
                    meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = None
            else:
                meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = None

            if MLParamKeys.CLASSIFIER_MAP.value in meta_dict.keys():
                meta_dict[MLParamKeys.CLASSIFIER_MAP.value] = ast.literal_eval(meta_dict[MLParamKeys.CLASSIFIER_MAP.value])
                for k, v in meta_dict[MLParamKeys.CLASSIFIER_MAP.value].items():
                    errors.append(check_int(name="MULTICLASS KEYS", value=k, raise_error=False)[1])
                    errors.append(check_str(name="MULTICLASS VALUES", value=v, raise_error=False)[1])

            else:
                meta_dict[MLParamKeys.CLASSIFIER_MAP.value] = None

            if MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value in meta_dict.keys():
                errors.append(check_if_valid_input(name="SAVE TRAIN AND TEST FRAME INDEXES", input=meta_dict[MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            else:
                meta_dict[MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value] = False

            if MLParamKeys.CUDA.value in meta_dict.keys():
                errors.append(check_if_valid_input(MLParamKeys.CUDA.value, input=meta_dict[MLParamKeys.CUDA.value], options=Options.RUN_OPTIONS_FLAGS.value, raise_error=False)[1])
            else:
                meta_dict[MLParamKeys.CUDA.value] = False

            errors = [x for x in errors if x != ""]
            if errors:
                option = TwoOptionQuestionPopUp(question=f"{errors[0]} \n ({meta_file_name}) \n  Do you want to skip this meta file or terminate training ?",
                                                title="META CONFIG FILE ERROR",
                                                option_one="SKIP",
                                                option_two="TERMINATE")

                if option.selected_option == "SKIP":
                    continue
                else:
                    raise InvalidInputError(msg=errors[0], source=self.__class__.__name__)
            else:
                meta_dicts[config_cnt] = meta_dict

        return meta_dicts

    def random_multiclass_frm_sampler(
            self,
            x_df: pd.DataFrame,
            y_df: pd.DataFrame,
            target_field: str,
            target_var: int,
            sampling_ratio: Union[float, Dict[int, float]],
            raise_error: bool = False,
    ):
        """
        Random multiclass undersampler.

        This function performs random under-sampling on a multiclass dataset to balance the class distribution.
        From each class, the function selects a number of frames computed as a ratio relative to a user-specified class variable.

        All the observations in the user-specified class is selected.

        :param pd.DataFrame x_df: A dataframe holding features.
        :param pd.DataFrame y_df: A dataframe holding target.
        :param str target_field: The name of the target column.
        :param int target_var: The variable in the target that should serve as baseline. E.g., ``0`` if ``0`` represents no behavior.
        :param Union[float, dict] sampling_ratio: The ratio of target_var observations that should be sampled of non-target_var observations.
                                                  E.g., if float ``1.0``, and there are `10`` target_var observations in the dataset, then 10 of each non-target_var observations will be sampled.
                                                  If different under-sampling ratios for different class variables are needed, use dict with the class variable name as key and ratio raletive to target_var as the value.
        :param bool raise_error: If True, then raises error if there are not enough observations of the non-target_var fullfilling the sampling_ratio. Else, takes all observations even though not enough to reach criterion.
        :return (pd.DataFrame, pd.DataFrame): resampled features, and resampled associated target.


        :examples:
        >>> df = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/multilabel/project_folder/csv/targets_inserted/01.YC015YC016phase45-sample_sampler.csv', index_col=0)
        >>> TrainModelMixin().random_multiclass_frm_sampler(data_df=df, target_field='syllable_class', target_var=0, sampling_ratio=0.20)
        >>> TrainModelMixin().random_multiclass_frm_sampler(data_df=df, target_field='syllable_class', target_var=0, sampling_ratio={1: 0.1, 2: 0.2, 3: 0.3})
        """

        data_df = pd.concat([x_df, y_df], axis=1)
        results_df_lst = [data_df[data_df[target_field] == target_var]]
        sampling_n = None
        if len(results_df_lst[0]) == 0:
            raise SamplingError(
                msg=f"No observations found in field {target_field} with value {target_var}.",
                source=self.__class__.__name__,
            )
        if type(sampling_ratio) == float:
            sampling_n = int(len(results_df_lst[0]) * sampling_ratio)
        if type(sampling_ratio) == dict:
            if target_var in sampling_ratio.keys():
                raise SamplingError(
                    msg=f"The target variable {target_var} cannot feature in the sampling ratio dictionary",
                    source=self.__class__.__name__,
                )
            for k, v in sampling_ratio.items():
                check_int(name="Sampling ratio key", value=k)
                check_float(name="Sampling ratio value", value=v, min_value=0.0)

        target_vars = list(data_df[target_field].unique())
        target_vars.remove(target_var)
        for var in target_vars:
            var_df = data_df[data_df[target_field] == var]
            if type(sampling_ratio) == dict:
                if var not in sampling_ratio.keys():
                    raise SamplingError(
                        msg=f"The variable {var} cannot be found in the sampling ratio dictionary"
                    )
                else:
                    sampling_n = int(len(results_df_lst[0]) * sampling_ratio[var])

            if sampling_n <= 0:
                raise SamplingError(
                    msg=f"The variable {var} has a sampling ratio of {sampling_ratio[var]} which gives a sample of zero or less sampled observations"
                )

            if (len(var_df) < sampling_n) and raise_error:
                raise SamplingError(
                    msg=f"SimBA wanted to sample {sampling_n} examples of behavior {var} but found {len(var_df)}. Change the sampling_ratio or set raise_error to False to sample the maximum number of present observations.",
                    source=self.__class__.__name__,
                )
            if (len(var_df) < sampling_n) and not raise_error:
                SamplingWarning(
                    msg=f"SimBA wanted to sample {sampling_n} examples of behavior {var} but found {len(var_df)}. Sampling {len(var_df)} observations.",
                    source=self.__class__.__name__,
                )
                sample = var_df
            else:
                sample = var_df.sample(n=sampling_n)
            results_df_lst.append(sample)

        results = pd.concat(results_df_lst, axis=0)
        return results.drop([target_field], axis=1), results[target_field]

    def random_multiclass_bout_sampler(
            self,
            x_df: pd.DataFrame,
            y_df: pd.DataFrame,
            target_field: str,
            target_var: int,
            sampling_ratio: Union[float, Dict[int, float]],
            raise_error: bool = False,
    ) -> pd.DataFrame:
        """
        Randomly sample multiclass behavioral bouts.

        This function performs random sampling on a multiclass dataset to balance the class distribution. From each class, the function selects a count of "bouts" where the count is computed as a ratio of a user-specified class variable count.
        All bout observations in the user-specified class is selected.

        :param pd.DataFrame x_df: A dataframe holding features.
        :param pd.DataFrame y_df: A dataframe holding target.
        :param str target_field: The name of the target column.
        :param int target_var: The variable in the target that should serve as baseline. E.g., ``0`` if ``0`` represents no behavior.
        :param Union[float, dict] sampling_ratio: The ratio of target_var bout observations that should be sampled of non-target_var observations.
                                                  E.g., if float ``1.0``, and there are `10`` bouts of target_var observations in the dataset, then 10 bouts of each non-target_var observations will be sampled.
                                                  If different under-sampling ratios for different class variables are needed, use dict with the class variable name as key and ratio relative to target_var as the value.
        :param bool raise_error: If True, then raises error if there are not enough observations of the non-target_var fullfilling the sampling_ratio. Else, takes all observations even though not enough to reach criterion.
        :raises SamplingError: If any of the following conditions are met:
                              - No bouts of the target class are detected in the data.
                              - The target variable is present in the sampling ratio dictionary.
                              - The sampling ratio dictionary contains non-integer keys or non-float values less than 0.0.
                              - The variable specified in the sampling ratio is not present in the DataFrame.
                              - The sampling ratio results in a sample size of zero or less.
                              - The requested sample size exceeds the available data and raise_error is True.
        :return (pd.DataFrame, pd.DataFrame): resampled features, and resampled associated target.

        :examples:
        >>> df = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/multilabel/project_folder/csv/targets_inserted/01.YC015YC016phase45-sample_sampler.csv', index_col=0)
        >>> undersampled_df = TrainModelMixin().random_multiclass_bout_sampler(data=df, target_field='syllable_class', target_var=0, sampling_ratio={1: 1.0, 2: 1, 3: 1}, raise_error=True)
        """

        data = pd.concat([x_df, y_df], axis=1).reset_index(drop=True)
        sampling_n, results_df_lst = None, []
        original_data = deepcopy(data)
        check_that_column_exist(df=data, column_name=target_field, file_name=None)
        bouts = detect_bouts_multiclass(data=data, target=target_field)
        target_bouts = bouts[bouts["Event"] == target_var]

        if len(target_bouts) == 0:
            raise SamplingError(
                msg=f"No bouts of target class {target_var} detected. Cannot perform bout sampling."
            )
        if type(sampling_ratio) == float:
            sampling_n = int(len(target_bouts) * sampling_ratio)
        if type(sampling_ratio) == dict:
            if target_var in sampling_ratio.keys():
                raise SamplingError(
                    msg=f"The target variable {target_var} cannot feature in the sampling ratio dictionary"
                )
            for k, v in sampling_ratio.items():
                check_int(name="Sampling ratio key", value=k)
                check_float(name="Sampling ratio value", value=v, min_value=0.0)

        target_annot_idx = list(
            target_bouts.apply(
                lambda x: list(range(int(x["Start_frame"]), int(x["End_frame"]) + 1)), 1
            )
        )
        target_annot_idx = [item for sublist in target_annot_idx for item in sublist]
        results_df_lst.append(original_data.loc[target_annot_idx, :])

        target_vars = list(data[target_field].unique())
        target_vars.remove(target_var)
        for var in target_vars:
            var_bout_df = bouts[bouts["Event"] == var]
            if type(sampling_ratio) == dict:
                if var not in sampling_ratio.keys():
                    raise SamplingError(
                        msg=f"The variable {var} cannot be found in the sampling ratio dictionary"
                    )
                else:
                    sampling_n = int(len(target_bouts) * sampling_ratio[var])

            if sampling_n <= 0:
                raise SamplingError(
                    msg=f"The variable {var} has a sampling ratio of {sampling_ratio[var]} of observed class {target_var} bouts which gives a sample of zero or less sampled observations"
                )

            if (len(var_bout_df) < sampling_n) and raise_error:
                raise SamplingError(
                    msg=f"SimBA wanted to sample {sampling_n} bouts of behavior {var} but found {len(var_bout_df)} bouts. Change the sampling_ratio or set raise_error to False to sample the maximum number of present observations.",
                    source="",
                )

            if (len(var_bout_df) < sampling_n) and not raise_error:
                SamplingWarning(
                    msg=f"SimBA wanted to sample {sampling_n} examples of behavior {var} bouts but found {len(var_bout_df)}. Sampling {len(var_bout_df)} observations.",
                    source=self.__class__.__name__,
                )
                sample = var_bout_df
            else:
                sample = var_bout_df.sample(n=sampling_n)
            annot_idx = list(
                sample.apply(
                    lambda x: list(
                        range(int(x["Start_frame"]), int(x["End_frame"]) + 1)
                    ),
                    1,
                )
            )
            annot_idx = [item for sublist in annot_idx for item in sublist]
            results_df_lst.append(original_data.loc[annot_idx, :])

        results = pd.concat(results_df_lst, axis=0)
        return results.drop([target_field], axis=1), results[target_field]

    @staticmethod
    def scaler_inverse_transform(
            data: pd.DataFrame,
            scaler: Union[MinMaxScaler, StandardScaler, QuantileTransformer],
            name: Optional[str] = "",
    ) -> pd.DataFrame:
        check_instance(
            source=f"{TrainModelMixin.scaler_inverse_transform.__name__} data",
            instance=data,
            accepted_types=(pd.DataFrame,),
        )
        check_instance(
            source=f"{TrainModelMixin.scaler_inverse_transform.__name__} scaler",
            instance=scaler,
            accepted_types=(
                MinMaxScaler,
                StandardScaler,
                QuantileTransformer,
            ),
        )

        if hasattr(scaler, "n_features_in_"):
            if len(data.columns) != scaler.n_features_in_:
                raise FeatureNumberMismatchError(msg=f"The scaler {name} expects {scaler.n_features_in_} features. Got {len(data.columns)}.", source=TrainModelMixin.scaler_transform.__name__)

        return pd.DataFrame(scaler.inverse_transform(data), columns=data.columns).set_index(data.index)

    @staticmethod
    def define_scaler(scaler_name: Literal["min-max", "standard", "quantile"]) -> Union[MinMaxScaler, StandardScaler, QuantileTransformer]:
        """
        Defines a sklearn scaler object. See ``UMLOptions.SCALER_OPTIONS.value`` for accepted scalers.

        :example:
        >>> TrainModelMixin.define_scaler(scaler_name='min-max')
        """

        if scaler_name.upper() not in Options.SCALER_OPTIONS.value:
            raise InvalidInputError(msg=f"Scaler {scaler_name} not supported. Options: {Options.SCALER_OPTIONS.value}", source=TrainModelMixin.define_scaler.__name__)
        if scaler_name.upper() == Options.MIN_MAX_SCALER.value:
            return MinMaxScaler()
        elif scaler_name.upper() == Options.STANDARD_SCALER.value:
            return StandardScaler()
        elif scaler_name.upper() == Options.QUANTILE_SCALER.value:
            return QuantileTransformer()

    @staticmethod
    def fit_scaler(scaler: Union[MinMaxScaler, QuantileTransformer, StandardScaler],
                   data: Union[pd.DataFrame, np.ndarray]) -> Union[
        MinMaxScaler, QuantileTransformer, StandardScaler, object]:

        check_instance(source=f'{TrainModelMixin.fit_scaler} data', instance=data, accepted_types=(pd.DataFrame, np.ndarray))
        check_instance(source=f'{TrainModelMixin.fit_scaler} scaler', instance=scaler, accepted_types=(MinMaxScaler, QuantileTransformer, StandardScaler))
        if isinstance(data, pd.DataFrame): data = data.values
        check_valid_array(data=data, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        return scaler.fit(data)

    @staticmethod
    def scaler_transform(
            data: pd.DataFrame,
            scaler: Union[MinMaxScaler, StandardScaler, QuantileTransformer],
            name: Optional[str] = "",
    ) -> pd.DataFrame:
        """
        Helper to run transform dataframe using previously fitted scaler.

        :param pd.DataFrame data: Data to transform.
        :param scaler: fitted scaler.
        """

        check_instance(
            source=TrainModelMixin.scaler_transform.__name__,
            instance=data,
            accepted_types=(pd.DataFrame,),
        )
        check_instance(
            source=f"{TrainModelMixin.scaler_transform.__name__} scaler",
            instance=scaler,
            accepted_types=(MinMaxScaler, StandardScaler, QuantileTransformer),
        )

        if hasattr(scaler, "n_features_in_"):
            if len(data.columns) != scaler.n_features_in_:
                raise FeatureNumberMismatchError(
                    msg=f"The scaler {name} expects {scaler.n_features_in_} features. Got {len(data.columns)}.",
                    source=TrainModelMixin.scaler_transform.__name__,
                )

        return pd.DataFrame(scaler.transform(data), columns=data.columns).set_index(
            data.index
        )

    @staticmethod
    def find_low_variance_fields(data: pd.DataFrame, variance_threshold: float) -> List[str]:
        """
        Finds fields with variance below provided threshold.

        :param pd.DataFrame data: Dataframe with continoues numerical features.
        :param float variance: Variance threshold (0.0-1.0).
        :return List[str]:
        """

        check_valid_dataframe(df=data, source=TrainModelMixin.find_low_variance_fields.__name__, valid_dtypes=(Formats.NUMERIC_DTYPES.value))
        check_float(name=TrainModelMixin.find_low_variance_fields.__name__, value=variance_threshold, min_value=0.0, max_value=1.0)
        feature_selector = VarianceThreshold(threshold=variance_threshold)
        feature_selector.fit(data)
        low_variance_fields = [c for c in data.columns if c not in data.columns[feature_selector.get_support()]]
        if len(low_variance_fields) == len(data.columns):
            raise NoDataError(msg=f"All feature columns show a variance below the {variance_threshold} threshold. Thus, no data remain for analysis.", source=TrainModelMixin.find_low_variance_fields.__name__)
        return low_variance_fields

    @staticmethod
    def _create_shap_mp_helper_concurrent(data: Tuple[int, pd.DataFrame],
                                          explainer: shap.TreeExplainer,
                                          clf_name: str,
                                          verbose: bool) -> Tuple[np.ndarray, int]:
        if verbose:
            print(f'Processing SHAP core batch {data[0] + 1}... ({len(data[1])} observations)')
        _ = data[1].pop(clf_name).values.reshape(-1, 1)
        shap_batch_results = np.full(shape=(len(data[1]), len(data[1].columns)), fill_value=np.nan, dtype=np.float32)
        for idx in range(len(data[1])):
            timer = SimbaTimer(start=True)
            obs = data[1].iloc[idx, :].values
            shap_batch_results[idx] = explainer.shap_values(obs, check_additivity=False)[1]
            timer.stop_timer()
            if verbose:
                print(f'SHAP frame complete (core batch: {data[0] + 1}, core batch frame: {idx + 1}/{len(data[1])}, frame processing time: {timer.elapsed_time_str}s)')
        return shap_batch_results, data[0]


    def create_shap_log_concurrent_mp(self,
                                      rf_clf: Union[RandomForestClassifier, str, os.PathLike],
                                      x: Union[pd.DataFrame, np.ndarray],
                                      y: Union[pd.DataFrame, pd.Series, np.ndarray],
                                      x_names: List[str],
                                      clf_name: str,
                                      cnt_present: int,
                                      cnt_absent: int,
                                      core_cnt: int = -1,
                                      chunk_size: int = 100,
                                      verbose: bool = True,
                                      save_dir: Optional[Union[str, os.PathLike]] = None,
                                      save_file_suffix: Optional[int] = None,
                                      plot: bool = False) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray]]:
        """
        Compute SHAP values using multiprocessing.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_
            For single-core solution, see :func:`~simba.mixins.train_model_mixins.TrainModelMixin.create_shap_log`
            For GPU method, see :func:`~simba.data_processors.cuda.create_shap_log.create_shap_log`
            For multiprocassing imap method (reliably runs on Windows and Mac), see :func:`~simba.mixins.train_model_mixin.TrainModelMixin.create_shap_log_mp`


        .. image:: _static/img/shap.png
           :width: 400
           :align: center

        :param Union[RandomForestClassifier, str, os.PathLike] rf_clf: Fitted sklearn random forest classifier, or pat to fitted, pickled sklearn random forest classifier.
        :param Union[pd.DataFrame, np.ndarray] x: Test features.
        :param Union[pd.DataFrame, pd.Series, np.ndarray] y_df: Test target.
        :param List[str] x_names: Feature names.
        :param str clf_name: Classifier name.
        :param int cnt_present: Number of behavior-present frames to calculate SHAP values for.
        :param int cnt_absent: Number of behavior-absent frames to calculate SHAP values for.
        :param int chunk_size: How many observations to process in each chunk. Increase value for faster processing if your memory allows.
        :param bool verbose: If True, prints progress.
        :param Optional[Union[str, os.PathLike]] save_dir: Optional directory where to store the results. If None, then the results are returned.
        :param Optional[int] save_file_suffix: Optional suffix to add to the shap output filenames. Useful for gridsearches and multiple shap data output files are to-be stored in the same `save_dir`.
        :param bool plot: If True, create SHAP aggregation and plots.

        :example:
        >>> CONFIG_PATH = r"C:\troubleshooting\mitra\project_folder\project_config.ini"
        >>> RF_PATH = r"C:\troubleshooting\mitra\models\validations\straub_tail_5_new\straub_tail_5.sav"
        >>> DATA_PATH = r"C:\troubleshooting\mitra\project_folder\csv\targets_inserted\new_straub\appended\501_MA142_Gi_CNO_0514.csv"
        >>> config = ConfigReader(config_path=CONFIG_PATH)
        >>> df = read_df(file_path=DATA_PATH, file_type='csv')
        >>> y = df['straub_tail']
        >>> x = df.drop(['immobility', 'rearing', 'grooming', 'circling', 'shaking', 'lay-on-belly', 'straub_tail'], axis=1)
        >>> x = x.drop(config.bp_col_names, axis=1)
        >>> TrainModelMixin.create_shap_log_concurrent_mp(rf_clf=RF_PATH, x=x, y=y, x_names=list(x.columns), clf_name='straub_tail', cnt_absent=100, cnt_present=10, core_cnt=10)
        """

        check_instance(source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} rf_clf', instance=rf_clf, accepted_types=(RandomForestClassifier, str, os.PathLike))
        if isinstance(rf_clf, str):
            rf_clf = TrainModelMixin().read_pickle(file_path=rf_clf)
        if SKLEARN in str(rf_clf.__module__).lower():
            timer = SimbaTimer(start=True)
            check_instance(source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} x', instance=x, accepted_types=(np.ndarray, pd.DataFrame))
            if isinstance(x, pd.DataFrame):
                check_valid_dataframe(df=x, source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} x', valid_dtypes=Formats.NUMERIC_DTYPES.value)
                x = x.values
            else:
                check_valid_array(data=x, source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            check_instance(source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} y', instance=y, accepted_types=(np.ndarray, pd.Series, pd.DataFrame))
            if isinstance(y, pd.DataFrame):
                check_valid_dataframe(df=y, source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} y', valid_dtypes=Formats.NUMERIC_DTYPES.value, max_axis_1=1)
                y = y.values
            else:
                if isinstance(y, pd.Series):
                    y = y.values
            if save_dir is not None: check_if_dir_exists(in_dir=save_dir)
            if save_file_suffix is not None: check_int(name=f'{TrainModelMixin.create_shap_log_mp.__name__} save_file_no', value=save_file_suffix, min_value=0)
            check_valid_array(data=y, source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            check_valid_lst(data=x_names, source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} x_names', valid_dtypes=(str,), exact_len=x.shape[1])
            check_str(name=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} clf_name', value=clf_name)
            check_int(name=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} cnt_present', value=cnt_present, min_value=1)
            check_int(name=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} cnt_absent', value=cnt_absent, min_value=1)
            check_int(name=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
            check_int(name=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} chunk_size', value=chunk_size, min_value=1)
            check_int(name=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} core_cnt', value=chunk_size, min_value=-1, unaccepted_vals=[0])
            check_valid_boolean(value=[verbose], source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} verbose')
            check_valid_boolean(value=[plot], source=f'{TrainModelMixin.create_shap_log_concurrent_mp.__name__} plot')
            core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
            df = pd.DataFrame(np.hstack((x, y.reshape(-1, 1))), columns=x_names + [clf_name])
            del x; del y
            present_df, absent_df = df[df[clf_name] == 1], df[df[clf_name] == 0]
            if len(present_df) == 0:
                raise NoDataError(msg=f'Cannot calculate SHAP values: no target PRESENT annotations detected for classifier {clf_name}.', source=TrainModelMixin.create_shap_log_mp.__name__)
            elif len(absent_df) == 0:
                raise NoDataError(msg=f'Cannot calculate SHAP values: no target ABSENT annotations detected for classifier {clf_name}.', source=TrainModelMixin.create_shap_log_mp.__name__)
            if len(present_df) < cnt_present:
                NotEnoughDataWarning( msg=f"Train data contains {len(present_df)} behavior-present annotations for classifier {clf_name}. This is less the number of frames you specified to calculate shap values for ({cnt_present}). SimBA will calculate shap scores for the {len(present_df)} behavior-present frames available", source=TrainModelMixin.create_shap_log_mp.__name__)
                cnt_present = len(present_df)
            if len(absent_df) < cnt_absent:
                NotEnoughDataWarning(msg=f"Train data contains {len(absent_df)} behavior-absent annotations for classifier {clf_name}. This is less the number of frames you specified to calculate shap values for ({cnt_absent}). SimBA will calculate shap scores for the {len(absent_df)} behavior-absent frames available", source=TrainModelMixin.create_shap_log_mp.__name__)
                cnt_absent = len(absent_df)
            shap_data = pd.concat([present_df.sample(cnt_present, replace=False), absent_df.sample(cnt_absent, replace=False)], axis=0).reset_index(drop=True)
            del df
            shap_data = np.array_split(shap_data, max(1, int(np.ceil(len(shap_data) / chunk_size))))
            shap_data = [(x, y) for x, y in enumerate(shap_data)]
            explainer = TrainModelMixin().define_tree_explainer(clf=rf_clf)
            expected_value = explainer.expected_value[1]
            shap_results, shap_raw = [], []
            out_shap_path, out_raw_path, img_save_path, df_save_paths, summary_dfs, img = None, None, None, None, None, None
            print(f"Computing {cnt_present + cnt_absent} SHAP values. Follow progress in OS terminal... (CORES: {core_cnt}, CHUNK SIZE: {chunk_size})")
            with concurrent.futures.ProcessPoolExecutor(max_workers=core_cnt) as executor:
                results = [executor.submit(self._create_shap_mp_helper_concurrent, data, explainer, clf_name, verbose) for data in shap_data]
                for result in concurrent.futures.as_completed(results):
                    batch_shap, batch_id = result.result()
                    batch_x, batch_y = shap_data[batch_id][1].drop(clf_name, axis=1), shap_data[batch_id][1][clf_name].values.reshape(-1, 1)
                    batch_shap_sum = np.sum(batch_shap, axis=1).reshape(-1, 1)
                    expected_arr = np.full((batch_shap.shape[0]), expected_value).reshape(-1, 1)
                    batch_proba = TrainModelMixin().clf_predict_proba(clf=rf_clf, x_df=batch_x, model_name=clf_name).reshape(-1, 1)
                    batch_shap_results = np.hstack((batch_shap, expected_arr, batch_shap_sum + expected_value, batch_proba, batch_y)).astype(np.float32)
                    shap_results.append(batch_shap_results)
                    shap_raw.append(batch_x)
                    if verbose: print(f"Completed SHAP care batch (Batch {batch_id + 1 + 1}/{len(shap_data)})...")
            shap_df = pd.DataFrame(data=np.row_stack(shap_results), columns=list(x_names) + ["Expected_value", "Sum", "Prediction_probability", clf_name])
            raw_df = pd.DataFrame(data=np.row_stack(shap_raw), columns=list(x_names))
            if save_dir is not None:
                if save_file_suffix is not None:
                    check_int(name=f'{TrainModelMixin.create_shap_log_mp.__name__} save_file_no', value=save_file_suffix,min_value=0)
                    out_shap_path = os.path.join(save_dir, f"SHAP_values_{save_file_suffix}_{clf_name}.csv")
                    out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{save_file_suffix}_{clf_name}.csv")
                    df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT_{save_file_suffix}.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT_{save_file_suffix}.csv")}
                    img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}_{save_file_suffix}.png")
                else:
                    out_shap_path = os.path.join(save_dir, f"SHAP_values_{clf_name}.csv")
                    out_raw_path = os.path.join(save_dir, f"RAW_SHAP_feature_values_{clf_name}.csv")
                    df_save_paths = {'PRESENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_PRESENT.csv"), 'ABSENT': os.path.join(save_dir, f"SHAP_summary_{clf_name}_ABSENT.csv")}
                    img_save_path = os.path.join(save_dir, f"SHAP_summary_line_graph_{clf_name}.png")
                shap_df.to_csv(out_shap_path)
                raw_df.to_csv(out_raw_path)
            if plot:
                shap_computer = ShapAggregateStatisticsCalculator(classifier_name=clf_name, shap_df=shap_df, shap_baseline_value=int(expected_value * 100), save_dir=None)
                summary_dfs, img = shap_computer.run()
                if save_dir is not None:
                    summary_dfs['PRESENT'].to_csv(df_save_paths['PRESENT'])
                    summary_dfs['ABSENT'].to_csv(df_save_paths['ABSENT'])
                    cv2.imwrite(img_save_path, img)
            timer.stop_timer()
            if save_dir and verbose:
                stdout_success(msg=f'SHAP data saved in {save_dir}', source=TrainModelMixin.create_shap_log_mp.__name__, elapsed_time=timer.elapsed_time_str)
            if not save_dir:
                return shap_df, raw_df, summary_dfs, img
        else:
            GPUToolsWarning(msg=f'Cannot compute SHAP scores using cuml random forest model. To compute SHAP scores, turn off cuda. Alternatively, for GPU solution, see simba.data_processors.cuda.create_shap_log.create_shap_log')


# if __name__ == "__main__":
#     #from simba.mixins.train_model_mixin import TrainModelMixin
#     x_cols = list(pd.read_csv(r"C:\projects\simba\simba\tests\data\sample_data\shap_test.csv", index_col=0).columns)
#     x = pd.DataFrame(np.random.randint(0, 500, (1000000, len(x_cols))), columns=x_cols)
#     y = pd.Series(np.random.randint(0, 2, (1000000,)))
#     rf_clf = TrainModelMixin().clf_define(n_estimators=1000)
#     rf_clf = TrainModelMixin().clf_fit(clf=rf_clf, x_df=x, y_df=y)
#     feature_names = [str(x) for x in list(x.columns)]
#     TrainModelMixin().create_shap_log(rf_clf=rf_clf, x=x, y=y, x_names=feature_names, clf_name='test', save_it=None, cnt_present=400, cnt_absent=400, plot=True, save_dir=r'C:\Users\sroni\OneDrive\Desktop\shap_test')
#     #TrainModelMixin().create_shap_log_concurrent_mp(rf_clf=rf_clf, x=x, y=y, x_names=feature_names, clf_name='test', cnt_present=100, cnt_absent=100, plot=True, save_dir=r'C:\Users\sroni\OneDrive\Desktop\shap_test', core_cnt=5)
#     #TrainModelMixin().create_shap_log_mp(rf_clf=rf_clf, x=x, y=y, x_names=feature_names, clf_name='test', cnt_present=100, cnt_absent=100, plot=True, save_dir=r'C:\Users\sroni\OneDrive\Desktop\shap_test', core_cnt=5)
#

# trainer = TrainModelMixin()
# trainer.clf_define(cuda=True)

# from simba.utils.read_write import read_simba_meta_files
# test = TrainModelMixin()
# data_df, _ = test.read_all_files_in_folder(file_paths=[r"C:\troubleshooting\two_black_animals_14bp\project_folder\csv\targets_inserted\Together_1.csv"], file_type='csv', classifier_names=['Attack'])
# meta_file_lst = read_simba_meta_files(folder_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\configs")
#
# meta_dicts = test.check_validity_of_meta_files(data_df=data_df, meta_file_paths=meta_file_lst)
#


# test = TrainModelMixin()
# test.read_all_files_in_folder_mp(file_paths=['/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437C_c3_2022-11-01_13-16-23_color.csv', '/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437D_c4_2022-11-01_13-16-39_color.csv'],
#                               file_type='csv', classifier_names=['attack', 'non-agresive parallel swimming'])
#
