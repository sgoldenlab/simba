__author__ = "Simon Nilsson"


import warnings

warnings.filterwarnings("ignore")

import ast
import concurrent
import configparser
import os
import pickle
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from subprocess import call

import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from numba import njit, typed, types
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   StandardScaler)
from sklearn.tree import export_graphviz
from sklearn.utils import parallel_backend
from tabulate import tabulate
from yellowbrick.classifier import ClassificationReport

try:
    from dtreeviz.trees import dtreeviz, tree
except:
    import dtreeviz

import functools
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.plotting.shap_agg_stats_visualizer import \
    ShapAggregateStatisticsVisualizer
from simba.ui.tkinter_functions import TwoOptionQuestionPopUp
from simba.utils.checks import (check_float, check_if_dir_exists,
                                check_if_valid_input, check_instance,
                                check_int, check_str, check_that_column_exist)
from simba.utils.data import (create_color_palette, detect_bouts,
                              detect_bouts_multiclass)
from simba.utils.enums import (ConfigKey, Defaults, Dtypes, Methods,
                               MLParamKeys, Options)
from simba.utils.errors import (ClassifierInferenceError, ColumnNotFoundError,
                                CorruptedFileError, DataHeaderError,
                                FaultyTrainingSetError,
                                FeatureNumberMismatchError, InvalidInputError,
                                MissingColumnsError, NoDataError,
                                SamplingError)
from simba.utils.lookups import get_meta_data_file_headers
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt, get_fn_ext,
                                    get_memory_usage_of_df, read_config_entry,
                                    read_df, read_meta_file, str_2_bool)
from simba.utils.warnings import (MissingUserInputWarning,
                                  MultiProcessingFailedWarning,
                                  NoModuleWarning, NotEnoughDataWarning,
                                  SamplingWarning, ShapWarning)

plt.switch_backend("agg")


class TrainModelMixin(object):
    """Train model methods"""

    def __init__(self):
        pass

    def read_all_files_in_folder(
        self,
        file_paths: List[str],
        file_type: str,
        classifier_names: Optional[List[str]] = None,
        raise_bool_clf_error: bool = True,
    ) -> (pd.DataFrame, List[int]):
        """
        Read in all data files in a folder to a single pd.DataFrame for downstream ML algo.
        Asserts that all classifiers have annotation fields present in concatenated dataframe.

        .. note::
           For improved runtime through pyarrow, use :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder_mp`

        :parameter List[str] file_paths: List of file paths representing files to be read in.
        :parameter str file_type: List of file paths representing files to be read in.
        :parameter str or None classifier_names: List of classifier names representing fields of human annotations. If not None, then assert that classifier names
            are present in each data file.
        :return pd.DataFrame: concatenated dataframe if all data represented in ``file_paths``.
        :return List[int]: The frame numbers (index) of the sampled data.

        :examples:
        >>> self.read_all_files_in_folder(file_paths=['targets_inserted/Video_1.csv', 'targets_inserted/Video_2.csv'], file_type='csv', classifier_names=['Attack'])
        """

        timer = SimbaTimer(start=True)
        frm_number_lst = []
        df_concat = pd.DataFrame()
        for file_cnt, file in enumerate(file_paths):
            print(f"Reading in file {str(file_cnt + 1)}/{str(len(file_paths))}...")
            _, vid_name, _ = get_fn_ext(file)
            df = (
                read_df(file, file_type)
                .dropna(axis=0, how="all")
                .fillna(0)
                .astype(np.float32)
            )
            frm_number_lst.extend((df.index))
            df.index = [vid_name] * len(df)
            if classifier_names != None:
                for clf_name in classifier_names:
                    if not clf_name in df.columns:
                        raise MissingColumnsError(
                            msg=f"Data for video {vid_name} does not contain any annotations for behavior {clf_name}. Delete classifier {clf_name} from the SimBA project, or add annotations for behavior {clf_name} to the video {vid_name}",
                            source=self.__class__.__name__,
                        )
                    elif (
                        len(set(df[clf_name].unique()) - {0, 1}) > 0
                        and raise_bool_clf_error
                    ):
                        raise InvalidInputError(
                            msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {file} the {clf_name} field contains additional value(s): {list(set(df[clf_name].unique()) - {0, 1})}.",
                            source=self.__class__.__name__,
                        )
                    else:
                        df_concat = pd.concat([df_concat, df], axis=0)
            else:
                df_concat = pd.concat([df_concat, df], axis=0)
        try:
            df_concat = df_concat.set_index("scorer")
        except KeyError:
            pass
        if len(df_concat) == 0:
            raise NoDataError(
                msg="SimBA found 0 annotated frames in the project_folder/csv/targets_inserted directory",
                source=self.__class__.__name__,
            )
        df_concat = df_concat.loc[
            :, ~df_concat.columns.str.contains("^Unnamed")
        ].fillna(0)
        timer.stop_timer()
        memory_size = get_memory_usage_of_df(df=df_concat)
        print(
            f'Dataset size: {memory_size["megabytes"]}MB / {memory_size["gigabytes"]}GB'
        )
        print(
            "{} file(s) read (elapsed time: {}s) ...".format(
                str(len(file_paths)), timer.elapsed_time_str
            )
        )

        return df_concat.astype(np.float32), frm_number_lst

    def read_in_all_model_names_to_remove(
        self, config: configparser.ConfigParser, model_cnt: int, clf_name: str
    ) -> List[str]:
        """
        Helper to find all field names that are annotations but are not the target.

        :parameter configparser.ConfigParser config: Configparser object holding data from the project_config.ini
        :parameter int model_cnt: Number of classifiers in the SimBA project
        :parameter str clf_name: Name of the classifier.
        :return List[str]: List of non-target annotation column names.

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

    def delete_other_annotation_columns(
        self, df: pd.DataFrame, annotations_lst: List[str], raise_error: bool = True
    ) -> pd.DataFrame:
        """
        Helper to drop fields that contain annotations which are not the target.

        :parameter pd.DataFrame df: Dataframe holding features and annotations.
        :parameter List[str] annotations_lst: column fields to be removed from df
        :raise_error bool raise_error: If True, throw error if annotation column doesn't exist. Else, skip. Default: True.
        :return pd.DataFrame: Dataframe without non-target annotation columns

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

    def split_df_to_x_y(
        self, df: pd.DataFrame, clf_name: str
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Helper to split dataframe into features and target.

        :parameter pd.DataFrame df: Dataframe holding features and annotations.
        :parameter str clf_name: Name of target.

        :return pd.DataFrame: features
        :return pd.DataFrame: target

        :examples:
        >>> self.split_df_to_x_y(df=df, clf_name='Attack')
        """

        df = deepcopy(df)
        y = df.pop(clf_name)
        return df, y

    def random_undersampler(
        self, x_train: np.ndarray, y_train: np.ndarray, sample_ratio: float
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Helper to perform random under-sampling of behavior-absent frames in a dataframe.

        :parameter np.ndarray x_train: Features in train set
        :parameter np.ndarray y_train: Target in train set
        :parameter float sample_ratio: Ratio of behavior-absent frames to keep relative to the behavior-present frames. E.g., ``1.0`` returns an equal
            count of behavior-absent and behavior-present frames. ``2.0`` returns twice as many behavior-absent frames as
            and behavior-present frames.

        :return pd.DataFrame: Under-sampled feature-set
        :return pd.DataFrame: Under-sampled target-set

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

    def smoteen_oversampler(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame, sample_ratio: float
    ) -> (np.ndarray, np.ndarray):
        """
        Helper to perform SMOTEEN oversampling of behavior-present annotations.

        :parameter np.ndarray x_train: Features in train set
        :parameter np.ndarray y_train: Target in train set
        :parameter float sample_ratio: Over-sampling ratio
        :return np.ndarray: Oversampled features.
        :return np.ndarray: Oversampled target.

        :examples:
        >>> self.smoteen_oversampler(x_train=x_train, y_train=y_train, sample_ratio=1.0)
        """

        print("Performing SMOTEENN oversampling...")
        smt = SMOTEENN(sampling_strategy=sample_ratio)
        if hasattr(smt, "fit_sample"):
            return smt.fit_sample(x_train, y_train)
        else:
            return smt.fit_resample(x_train, y_train)

    def smote_oversampler(
        self,
        x_train: pd.DataFrame or np.array,
        y_train: pd.DataFrame or np.array,
        sample_ratio: float,
    ) -> (np.ndarray, np.ndarray):
        """
        Helper to perform SMOTE oversampling of behavior-present annotations.

        :parameter np.ndarray x_train: Features in train set
        :parameter np.ndarray y_train: Target in train set
        :parameter float sample_ratio: Over-sampling ratio
        :return np.ndarray: Oversampled features.
        :return np.ndarray: Oversampled target.

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
                                    save_dir: Union[str, os.PathLike],
                                    save_file_no: Optional[int] = None) -> None:
        """
        Computes feature permutation importance scores.

        :parameter np.ndarray x_test: 2d feature test data of shape len(frames) x len(features)
        :parameter np.ndarray y_test: 2d feature target test data of shape len(frames) x 1
        :parameter RandomForestClassifier clf: random forest classifier
        :parameter List[str] feature_names: Names of features in x_test
        :parameter str clf_name: Name of classifier in y_test
        :parameter str save_dir: Directory where to save results in CSV format
        :parameter Optional[int] save_file_no: If permutation importance calculation is part of a grid search, provide integer identifier representing the model in the grid serach sequence. Will be used as suffix in output filename.
        """

        print("Calculating feature permutation importances...")
        timer = SimbaTimer(start=True)
        p_importances = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=0)
        df = pd.DataFrame(np.column_stack([feature_names, p_importances.importances_mean, p_importances.importances_std]), columns=["FEATURE_NAME","FEATURE_IMPORTANCE_MEAN","FEATURE_IMPORTANCE_STDEV"])
        df = df.sort_values(by=["FEATURE_IMPORTANCE_MEAN"], ascending=False)
        if save_file_no != None:
            save_file_path = os.path.join(save_dir, f'{clf_name}_{save_file_no}_permutations_importances.csv')
        else:
            save_file_path = os.path.join(save_dir, f"{clf_name}_permutations_importances.csv")
        df.to_csv(save_file_path, index=False)
        timer.stop_timer()
        print(f"Permutation importance calculation complete (elapsed time: {timer.elapsed_time_str}s) ...")

    def calc_learning_curve(self,
                            x_y_df: pd.DataFrame,
                            clf_name: str,
                            shuffle_splits: int,
                            dataset_splits: int,
                            tt_size: float,
                            rf_clf: RandomForestClassifier,
                            save_dir: str,
                            save_file_no: Optional[int] = None,
                            multiclass: bool = False) -> None:
        """
        Helper to compute random forest learning curves with cross-validation.

        .. image:: _static/img/learning_curves.png
           :width: 600
           :align: center

        :parameter pd.DataFrame x_y_df: Dataframe holding features and target.
        :parameter str clf_name: Name of the classifier
        :parameter int shuffle_splits: Number of cross-validation datasets at each data split.
        :parameter int dataset_splits: Number of data splits.
        :parameter float tt_size: test size
        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        :parameter bool multiclass: If True, then target consist of several categories [0, 1, 2 ...] and scoring becomes ``None``. If False, then coring ``f1``.
        """

        print("Calculating learning curves...")
        timer = SimbaTimer(start=True)
        x_df, y_df = self.split_df_to_x_y(x_y_df, clf_name)
        check_int(name=f'calc_learning_curve shuffle_splits', value=shuffle_splits, min_value=2)
        check_int(name=f'calc_learning_curve dataset_splits', value=dataset_splits, min_value=2)
        cv = ShuffleSplit(n_splits=shuffle_splits, test_size=tt_size)
        scoring = "f1"
        if multiclass:
            scoring = None
        if platform.system() == "Darwin":
            with parallel_backend("threading", n_jobs=-2):
                train_sizes, train_scores, test_scores = learning_curve(estimator=rf_clf, X=x_df, y=y_df, cv=cv, scoring=scoring, shuffle=False, verbose=0, train_sizes=np.linspace(0.01, 1.0, dataset_splits), error_score="raise")
        else:
            train_sizes, train_scores, test_scores = learning_curve(estimator=rf_clf, X=x_df, y=y_df, cv=cv, scoring=scoring, shuffle=False, n_jobs=-1, verbose=0, train_sizes=np.linspace(0.01, 1.0, dataset_splits), error_score="raise")
        results_df = pd.DataFrame()
        results_df["FRACTION TRAIN SIZE"] = np.linspace(0.01, 1.0, dataset_splits)
        results_df["TRAIN_MEAN_F1"] = np.mean(train_scores, axis=1)
        results_df["TEST_MEAN_F1"] = np.mean(test_scores, axis=1)
        results_df["TRAIN_STDEV_F1"] = np.std(train_scores, axis=1)
        results_df["TEST_STDEV_F1"] = np.std(test_scores, axis=1)
        if save_file_no != None:
            self.learning_curve_save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_learning_curve.csv")
        else:
            self.learning_curve_save_path = os.path.join(save_dir, f"{clf_name}_learning_curve.csv")
        results_df.to_csv(self.learning_curve_save_path, index=False)
        timer.stop_timer()
        print(f"Learning curve calculation complete (elapsed time: {timer.elapsed_time_str}s) ...")

    def calc_pr_curve(self,
                      rf_clf: RandomForestClassifier,
                      x_df: pd.DataFrame,
                      y_df: pd.DataFrame,
                      clf_name: str,
                      save_dir: str,
                      multiclass: bool = False,
                      classifier_map: Dict[int, str] = None,
                      save_file_no: Optional[int] = None) -> None:
        """
        Helper to compute random forest precision-recall curve.

        .. image:: _static/img/pr_curves.png
           :width: 800
           :align: center

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter pd.DataFrame x_df: Pandas dataframe holding test features.
        :parameter pd.DataFrame y_df: Pandas dataframe holding test target.
        :parameter str clf_name: Classifier name.
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter bool multiclass: If the classifier is a multi-classifier. Default: False.
        :parameter Dict[int, str] classifier_map: If multiclass, dictionary mapping integers to classifier names.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        """

        if multiclass and classifier_map is None:
            raise InvalidInputError(msg="Creating PR curve for multi-classifier but classifier_map not defined. Pass classifier_map argument")
        print("Calculating PR curves...")
        timer = SimbaTimer(start=True)
        if not multiclass:
            p = rf_clf.predict_proba(x_df)[:, 1]
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
                df["F1"] = (2* (df["RECALL"] * df["PRECISION"]) / (df["RECALL"] + df["PRECISION"]))
                thresholds = list(thresholds)
                thresholds.insert(0, 0.00)
                df["DISCRIMINATION THRESHOLDS"] = thresholds
                df.insert(0, "BEHAVIOR CLASS", classifier_map[i])
                pr_df_lst.append(df)
            pr_df = pd.concat(pr_df_lst, axis=0).reset_index(drop=True)
        if save_file_no != None:
            self.pr_save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_pr_curve.csv")
        else:
            self.pr_save_path = os.path.join(save_dir, f"{clf_name}_pr_curve.csv")
        pr_df.to_csv(self.pr_save_path, index=False)
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

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter str clf_name: Classifier name.
        :parameter List[str] feature_names: List of feature names.
        :parameter List[str] class_names: List of classes. E.g., ['Attack absent', 'Attack present']
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        """

        print("Visualizing example decision tree using graphviz...")
        timer = SimbaTimer(start=True)
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

    def create_clf_report(self,
                          rf_clf: RandomForestClassifier,
                          x_df: pd.DataFrame,
                          y_df: pd.DataFrame,
                          class_names: List[str],
                          save_dir: str,
                          clf_name: Optional[str] = None,
                          save_file_no: Optional[int] = None) -> None:


        """
        Helper to create classifier truth table report.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_

        .. image:: _static/img/clf_report.png
           :width: 500
           :align: center

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter pd.DataFrame x_df: dataframe holding test features
        :parameter pd.DataFrame y_df: dataframe holding test target
        :parameter List[str] class_names: List of classes. E.g., ['Attack absent', 'Attack present']
        :parameter Optional[str] clf_name: Name of the classifier. If not None, then used in the output file name.
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """

        print("Creating classification report visualization...")
        timer = SimbaTimer(start=True)
        try:
            visualizer = ClassificationReport(rf_clf, classes=class_names, support=True)
            visualizer.score(x_df, y_df)
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
            visualizer.poof(outpath=save_path, clear_figure=True)
            timer.stop_timer()
            print(f'Classification report saved at {save_path} (elapsed time: {timer.elapsed_time_str}s)')
        except KeyError as e:
            print(e.args)
            if not clf_name:
                NotEnoughDataWarning(msg=f"Not enough data to create classification report, consider changing sampling settings or create more annotations: {class_names[1]}",source=self.__class__.__name__)
            else:
                NotEnoughDataWarning(msg=f"Not enough data to create classification report, consider changing sampling settings or create more annotations: {clf_name}", source=self.__class__.__name__)

    def create_x_importance_log(self,
                                rf_clf: RandomForestClassifier,
                                x_names: List[str],
                                clf_name: str,
                                save_dir: str,
                                save_file_no: Optional[int] = None) -> None:
        """
        Helper to save gini or entropy based feature importance scores.

        .. note::
           `Example expected output  <https://github.com/sgoldenlab/simba/blob/master/images/BtWGaNP_feature_importance_log.csv>`__.

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter List[str] x_names: Names of features.
        :parameter str clf_name: Name of classifier
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search.
        """

        print("Creating feature importance log...")
        timer = SimbaTimer(start=True)
        importances = list(rf_clf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x_names, importances)]
        df = pd.DataFrame(feature_importances, columns=["FEATURE", "FEATURE_IMPORTANCE"]).sort_values(by=["FEATURE_IMPORTANCE"], ascending=False)
        if save_file_no != None:
            self.f_importance_save_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_feature_importance_log.csv")
        else:
            self.f_importance_save_path = os.path.join(save_dir, f"{clf_name}_feature_importance_log.csv")
        df.to_csv(self.f_importance_save_path, index=False)
        timer.stop_timer()
        print(f'Feature importance log saved at {self.f_importance_save_path} (elapsed time: {timer.elapsed_time_str}s)')

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

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter List[str] x_names: Names of features.
        :parameter str clf_name: Name of classifier.
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter int n_bars: Number of bars in the plot.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not part of a grid search
        """

        check_int(name="FEATURE IMPORTANCE BAR COUNT", value=n_bars, min_value=1)
        print("Creating feature importance bar chart...")
        timer = SimbaTimer(start=True)
        self.create_x_importance_log(rf_clf, x_names, clf_name, save_dir)
        importances_df = pd.read_csv(os.path.join(save_dir, f"{clf_name}_feature_importance_log.csv"))
        importances_head = importances_df.head(n_bars)
        colors = create_color_palette(pallete_name=palette, increments=n_bars, as_rgb_ratio=True)
        colors = [x[::-1] for x in colors]
        ax = importances_head.plot.bar(x="FEATURE", y="FEATURE_IMPORTANCE", legend=False, rot=90, fontsize=6, color=colors)
        plt.ylabel("Feature importances' (mean decrease impurity)", fontsize=6)
        plt.tight_layout()
        if save_file_no != None:
            save_file_path = os.path.join(save_dir, f"{clf_name}_{save_file_no}_feature_importance_bar_graph.png")
        else:
            save_file_path = os.path.join(save_dir, f"{clf_name}_feature_importance_bar_graph.png")
        plt.savefig(save_file_path, dpi=600)
        plt.close("all")
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

        clf = tree.DecisionTreeClassifier(max_depth=5, random_state=666)
        clf.fit(x_train, y_train)
        try:
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
            save_path = os.path.join(
                save_dir, clf_name + "_fancy_decision_tree_example.svg"
            )
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
                        ini_file_path: str,
                        rf_clf: RandomForestClassifier,
                        x_df: pd.DataFrame,
                        y_df: pd.Series,
                        x_names: List[str],
                        clf_name: str,
                        cnt_present: int,
                        cnt_absent: int,
                        save_it: int = 100,
                        save_path: Optional[Union[str, os.PathLike]] = None,
                        save_file_no: Optional[int] = None) -> Union[None, Tuple[pd.DataFrame]]:
        """
        Compute SHAP values for a random forest classifier.

        This method computes SHAP (SHapley Additive exPlanations) values for a given random forest classifier.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_

        .. image:: _static/img/shap.png
           :width: 600
           :align: center

        .. note::
           For improved run-times, use multiprocessing through :meth:`simba.mixins.train_model_mixins.TrainModelMixin.create_shap_log_mp`
           Uses TreeSHAP `Documentation <https://shap.readthedocs.io/en/latest/index.html>`_

        The SHAP value for feature 'i' in the context of a prediction 'f' and input 'x' is calculated using the following formula:

        .. math::

           \phi_i(f, x) = \\sum_{S \\subseteq F \\setminus {i}} \\frac{|S|!(|F| - |S| - 1)!}{|F|!} (f_{S \cup {i}}(x_{S \\cup {i}}) - f_S(x_S))

        :param str ini_file_path: Path to the SimBA project_config.ini
        :param RandomForestClassifier rf_clf: sklearn random forest classifier
        :param pd.DataFrame x_df: Test features.
        :param pd.DataFrame y_df: Test target.
        :param List[str] x_names: Feature names.
        :param str clf_name: Classifier name.
        :param int cnt_present: Number of behavior-present frames to calculate SHAP values for.
        :param int cnt_absent: Number of behavior-absent frames to calculate SHAP values for.
        :param str save_path: Directory where to save output in csv file format.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.

        """

        print("Calculating SHAP values (SINGLE CORE)...")
        shap_timer = SimbaTimer(start=True)
        data_df = pd.concat([x_df, y_df], axis=1)
        if (save_file_no == None) and (save_path is not None):
            self.out_df_shap_path = os.path.join(
                save_path, f"SHAP_values_{clf_name}.csv"
            )
            self.out_df_raw_path = os.path.join(
                save_path, f"RAW_SHAP_feature_values_{clf_name}.csv"
            )
        elif (save_file_no is not None) and (save_path is not None):
            self.out_df_shap_path = os.path.join(
                save_path, f"SHAP_values_{str(save_file_no)}_{clf_name}.csv"
            )
            self.out_df_raw_path = os.path.join(
                save_path, f"RAW_SHAP_feature_values_{str(save_file_no)}_{clf_name}.csv"
            )

        target_df, nontarget_df = (
            data_df[data_df[y_df.name] == 1],
            data_df[data_df[y_df.name] == 0],
        )
        if len(target_df) < cnt_present:
            NotEnoughDataWarning(
                msg=f"Train data contains {len(target_df)} behavior-present annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_present)}). SimBA will calculate shap scores for the {len(target_df)} behavior-present frames available",
                source=self.__class__.__name__,
            )
            cnt_present = len(target_df)
        if len(nontarget_df) < cnt_absent:
            NotEnoughDataWarning(
                msg=f"Train data contains {len(nontarget_df)} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_absent)}). SimBA will calculate shap scores for the {len(nontarget_df)} behavior-absent frames available",
                source=self.__class__.__name__,
            )
            cnt_absent = len(nontarget_df)
        non_target_for_shap = nontarget_df.sample(cnt_absent, replace=False)
        targets_for_shap = target_df.sample(cnt_present, replace=False)
        shap_df = pd.concat([targets_for_shap, non_target_for_shap], axis=0)
        y_df = shap_df.pop(clf_name).values
        explainer = shap.TreeExplainer(
            rf_clf,
            data=None,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )
        expected_value = explainer.expected_value[1]
        out_df_raw = pd.DataFrame(columns=x_names)
        shap_headers = list(x_names)
        shap_headers.extend(
            ("Expected_value", "Sum", "Prediction_probability", clf_name)
        )
        out_df_shap = pd.DataFrame(columns=shap_headers)
        for cnt, frame in enumerate(range(len(shap_df))):
            shap_frm_timer = SimbaTimer(start=True)
            frame_data = shap_df.iloc[[frame]]
            frame_shap = explainer.shap_values(frame_data, check_additivity=False)[1][
                0
            ].tolist()
            frame_shap.extend(
                (
                    expected_value,
                    sum(frame_shap),
                    rf_clf.predict_proba(frame_data)[0][1],
                    y_df[cnt],
                )
            )
            out_df_raw.loc[len(out_df_raw)] = list(shap_df.iloc[frame])
            out_df_shap.loc[len(out_df_shap)] = frame_shap
            if (
                (cnt % save_it == 0)
                or (cnt == len(shap_df) - 1)
                and (cnt != 0)
                and (save_path is not None)
            ):
                print(f"Saving SHAP data after {cnt} iterations...")
                out_df_shap.to_csv(self.out_df_shap_path)
                out_df_raw.to_csv(self.out_df_raw_path)
            shap_frm_timer.stop_timer()
            print(
                f"SHAP frame: {cnt + 1} / {len(shap_df)}, elapsed time: {shap_frm_timer.elapsed_time_str}..."
            )

        shap_timer.stop_timer()
        stdout_success(
            msg="SHAP calculations complete",
            elapsed_time=shap_timer.elapsed_time_str,
            source=self.__class__.__name__,
        )
        if save_path is not None:
            _ = ShapAggregateStatisticsVisualizer(
                config_path=ini_file_path,
                classifier_name=clf_name,
                shap_df=out_df_shap,
                shap_baseline_value=int(expected_value * 100),
                save_path=save_path,
            )
        else:
            return (out_df_shap, out_df_raw, int(expected_value * 100))

    def print_machine_model_information(self, model_dict: dict) -> None:
        """
        Helper to print model information in tabular form.

        :parameter dict model_dict: dictionary holding model meta data in SimBA meta-config format.

        """

        table_view = [
            ["Model name", model_dict[MLParamKeys.CLASSIFIER_NAME.value]],
            ["Ensemble method", "RF"],
            ["Estimators (trees)", model_dict[MLParamKeys.RF_ESTIMATORS.value]],
            ["Max features", model_dict[MLParamKeys.RF_MAX_FEATURES.value]],
            [
                "Under sampling setting",
                model_dict[MLParamKeys.UNDERSAMPLE_SETTING.value],
            ],
            ["Under sampling ratio", model_dict[MLParamKeys.UNDERSAMPLE_RATIO.value]],
            ["Over sampling setting", model_dict[MLParamKeys.OVERSAMPLE_SETTING.value]],
            ["Over sampling ratio", model_dict[MLParamKeys.OVERSAMPLE_RATIO.value]],
            ["criterion", model_dict[MLParamKeys.RF_CRITERION.value]],
            ["Min sample leaf", model_dict[MLParamKeys.MIN_LEAF.value]],
        ]
        table = tabulate(table_view, ["Setting", "value"], tablefmt="grid")
        print(f"{table} {Defaults.STR_SPLIT_DELIMITER.value}TABLE")

    def create_meta_data_csv_training_one_model(self, meta_data_lst: list, clf_name: str, save_dir: Union[str, os.PathLike]) -> None:

        """
        Helper to save single model meta data (hyperparameters, sampling settings etc.) from list format into SimBA
        compatible CSV config file.

        :parameter list meta_data_lst: Meta data in list format
        :parameter str clf_name: Name of classifier
        :parameter str clf_name: Name of classifier
        :parameter str save_dir: Directory where to save output in csv file format.
        """
        print("Saving model meta data file...")
        save_path = os.path.join(save_dir, clf_name + "_meta.csv")
        out_df = pd.DataFrame(columns=get_meta_data_file_headers())
        out_df.loc[len(out_df)] = meta_data_lst
        out_df.to_csv(save_path)

    def create_meta_data_csv_training_multiple_models(self, meta_data, clf_name, save_dir, save_file_no: Optional[int] = None) -> None:
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

        :parameter RandomForestClassifier rf_clf: sklearn random forest classifier
        :parameter str clf_name: Classifier name
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
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
                model_dict[n] = {}
                if config.get("SML settings", "model_path_" + str(n + 1)) == "":
                    MissingUserInputWarning(
                        msg=f'Skipping {str(config.get("SML settings", "target_name_" + str(n + 1)))} classifier analysis: no path set to model file',
                        source=self.__class__.__name__,
                    )
                    continue
                if (
                    config.get("SML settings", "model_path_" + str(n + 1))
                    == "No file selected"
                ):
                    MissingUserInputWarning(
                        msg=f'Skipping {str(config.get("SML settings", "target_name_" + str(n + 1)))} classifier analysis: The classifier path is set to "No file selected',
                        source=self.__class__.__name__,
                    )
                    continue
                model_dict[n]["model_path"] = config.get(
                    ConfigKey.SML_SETTINGS.value, "model_path_" + str(n + 1)
                )
                model_dict[n]["model_name"] = config.get(
                    ConfigKey.SML_SETTINGS.value, "target_name_" + str(n + 1)
                )
                check_str("model_name", model_dict[n]["model_name"])
                model_dict[n]["threshold"] = config.getfloat(
                    ConfigKey.THRESHOLD_SETTINGS.value, "threshold_" + str(n + 1)
                )
                check_float(
                    "threshold",
                    model_dict[n]["threshold"],
                    min_value=0.0,
                    max_value=1.0,
                )
                model_dict[n]["minimum_bout_length"] = config.getfloat(
                    ConfigKey.MIN_BOUT_LENGTH.value, "min_bout_" + str(n + 1)
                )
                check_int("minimum_bout_length", model_dict[n]["minimum_bout_length"])
                if config.has_option(
                    ConfigKey.SML_SETTINGS.value, f"classifier_map_{n+1}"
                ):
                    model_dict[n]["classifier_map"] = config.get(
                        ConfigKey.SML_SETTINGS.value, f"classifier_map_{n+1}"
                    )
                    model_dict[n]["classifier_map"] = ast.literal_eval(
                        model_dict[n]["classifier_map"]
                    )
                    if type(model_dict[n]["classifier_map"]) != dict:
                        raise InvalidInputError(
                            msg=f"SimBA found a classifier map for classifier {n+1} that could not be interpreted as a dictionary",
                            source=self.__class__.__name__,
                        )

            except ValueError:
                MissingUserInputWarning(
                    msg=f'Skipping {str(config.get("SML settings", "target_name_" + str(n + 1)))} classifier analysis: missing information (e.g., no discrimination threshold and/or minimum bout set in the project_config.ini',
                    source=self.__class__.__name__,
                )

        if len(model_dict.keys()) == 0:
            raise NoDataError(
                msg=f"There are no models with accurate data specified in the RUN MODELS menu. Specify the model information to SimBA RUN MODELS menu to use them to analyze videos",
                source=self.get_model_info.__name__,
            )
        else:
            return model_dict

    def get_all_clf_names(
        self, config: configparser.ConfigParser, target_cnt: int
    ) -> List[str]:
        """
        Helper to get all classifier names in a SimBA project.

        :parameter configparser.ConfigParser config: Parsed SimBA project_config.ini
        :parameter int.ConfigParser target_cnt: Parsed SimBA project_config.ini
        :return List[str]: All classifier names in project

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

    def insert_column_headers_for_outlier_correction(
        self,
        data_df: pd.DataFrame,
        new_headers: List[str],
        filepath: Union[str, os.PathLike],
    ) -> pd.DataFrame:
        """
        Helper to insert new column headers onto a dataframe following outlier correction.

        :parameter pd.DataFrame data_df: Dataframe with headers to-be replaced.
        :parameter str filepath: Path to where ``data_df`` is stored on disk.
        :parameter List[str] new_headers: New headers.
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

    def read_pickle(self, file_path: Union[str, os.PathLike]) -> object:
        """
        Read pickle file

        :parameter str file_path: Path to pickle file on disk.
        :return dict

        """

        try:
            clf = pickle.load(open(file_path, "rb"))
        except pickle.UnpicklingError:
            raise CorruptedFileError(
                msg=f"Can not read {file_path} as a classifier file (pickle).",
                source=self.__class__.__name__,
            )
        return clf

    def bout_train_test_splitter(
        self, x_df: pd.DataFrame, y_df: pd.Series, test_size: float
    ) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
        """
        Helper to split train and test based on annotated `bouts`.

        .. image:: _static/img/bout_vs_frames_split.png
           :width: 600
           :align: center

        :parameter pd.DataFrame x_df: Features
        :parameter pd.Series y_df: Target
        :parameter float test_size: Size of test as ratio of all annotated bouts (e.g., ``0.2``).
        :return np.ndarray x_train: Features for training
        :return np.ndarray x_test: Features for testing
        :return np.ndarray y_train: Target for training
        :return np.ndarray y_test: Target for testing

        :examples:
        >>> x = pd.DataFrame(data=[[11, 23, 12], [87, 65, 76], [23, 73, 27], [10, 29, 2], [12, 32, 42], [32, 73, 2], [21, 83, 98], [98, 1, 1]])
        >>> y =  pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
        >>> x_train, x_test, y_train, y_test = TrainModelMixin().bout_train_test_splitter(x_df=x, y_df=y, test_size=0.5)
        """

        print("Using bout sampling...")

        def find_bouts(s: pd.Series, type: str):
            test_bouts_frames, train_bouts_frames = [], []
            bouts = detect_bouts(
                pd.DataFrame(s), target_lst=pd.DataFrame(s).columns, fps=-1
            )
            print(f"{str(len(bouts))} {type} bouts found...")
            bouts = list(
                bouts.apply(
                    lambda x: list(
                        range(int(x["Start_frame"]), int(x["End_frame"]) + 1)
                    ),
                    1,
                ).values
            )
            test_bouts_idx = np.random.choice(
                np.arange(0, len(bouts)), int(len(bouts) * test_size)
            )
            train_bouts_idx = np.array(
                [x for x in list(range(len(bouts))) if x not in test_bouts_idx]
            )
            for i in range(0, len(bouts)):
                if i in test_bouts_idx:
                    test_bouts_frames.append(bouts[i])
                if i in train_bouts_idx:
                    train_bouts_frames.append(bouts[i])
            return [i for s in test_bouts_frames for i in s], [
                i for s in train_bouts_frames for i in s
            ]

        test_bouts_frames, train_bouts_frames = find_bouts(
            s=y_df, type="behavior present"
        )
        test_nonbouts_frames, train_nonbouts_frames = find_bouts(
            s=np.logical_xor(y_df, 1).astype(int), type="behavior absent"
        )
        x_train = x_df[x_df.index.isin(train_bouts_frames + train_nonbouts_frames)]
        x_test = x_df[x_df.index.isin(test_bouts_frames + test_nonbouts_frames)]
        y_train = y_df[y_df.index.isin(train_bouts_frames + train_nonbouts_frames)]
        y_test = y_df[y_df.index.isin(test_bouts_frames + test_nonbouts_frames)]

        return x_train, x_test, y_train, y_test

    @staticmethod
    @njit("(float32[:, :], float64, types.ListType(types.unicode_type))")
    def find_highly_correlated_fields(
        data: np.ndarray,
        threshold: float,
        field_names: types.ListType(types.unicode_type),
    ) -> List[str]:
        """
        Find highly correlated fields in a dataset.

        Calculates the absolute correlation coefficients between columns in a given dataset and identifies
        pairs of columns that have a correlation coefficient greater than the specified threshold. For every pair of correlated
        features identified, the function returns the field name of one feature. These field names can later be dropped from the input data to reduce memory requirements and collinearity.

        :param np.ndarray data: Two dimensional numpy array with features represented as columns and frames represented as rows.
        :param float threshold: Threshold value for significant collinearity.
        :param List[str] field_names: List mapping the column names in data to a field name. Use types.ListType(types.unicode_type) to take advantage of JIT compilation
        :return List[str]: Unique field names that correlates with at least one other field above the threshold value.

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

    def check_sampled_dataset_integrity(
        self, x_df: pd.DataFrame, y_df: pd.DataFrame
    ) -> None:
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
                raise FaultyTrainingSetError(
                    msg=f"All training annotations for classifier {str(y_df.name)} is labelled as ABSENT. A classifier has be be trained with both behavior PRESENT and ABSENT ANNOTATIONS.",
                    source=self.__class__.__name__,
                )
            if y_df.unique()[0] == 1:
                raise FaultyTrainingSetError(
                    msg=f"All training annotations for classifier {str(y_df.name)} is labelled as PRESENT. A classifier has be be trained with both behavior PRESENT and ABSENT ANNOTATIONS.",
                    source=self.__class__.__name__,
                )

    def partial_dependence_calculator(
        self,
        clf: RandomForestClassifier,
        x_df: pd.DataFrame,
        clf_name: str,
        save_dir: Union[str, os.PathLike],
        clf_cnt: Optional[int] = None,
    ) -> None:
        """
        Compute feature partial dependencies for every feature in training set.

        :parameter RandomForestClassifier clf: Random forest classifier
        :parameter pd.DataFrame x_df: Features training set
        :parameter str clf_name: Name of classifier
        :parameter str save_dir: Directory where to save the data
        :parameter Optional[int] clf_cnt: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """
        print(f"Calculating partial dependencies for {len(x_df.columns)} features...")
        clf.verbose = 0
        check_if_dir_exists(save_dir)
        if clf_cnt:
            save_dir = os.path.join(
                save_dir, f"partial_dependencies_{clf_name}_{clf_cnt}"
            )
        else:
            save_dir = os.path.join(save_dir, f"partial_dependencies_{clf_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for feature_cnt, feature_name in enumerate(x_df.columns):
            save_path = os.path.join(save_dir, f"{feature_name}.csv")
            pdp, axes = partial_dependence(
                clf,
                features=[feature_name],
                X=x_df,
                percentiles=(0, 1),
                grid_resolution=30,
            )
            df = pd.DataFrame({"partial dependence": pdp[0], "feature value": axes[0]})
            df.to_csv(save_path)
            print(
                f"Partial dependencies for {feature_name} complete ({feature_cnt+1}/{len(x_df.columns)})..."
            )

    def clf_predict_proba(
        self,
        clf: RandomForestClassifier,
        x_df: pd.DataFrame,
        multiclass: bool = False,
        model_name: Optional[str] = None,
        data_path: Optional[Union[str, os.PathLike]] = None,
    ) -> np.ndarray:
        """

        :param RandomForestClassifier clf: Random forest classifier object
        :param pd.DataFrame x_df: Features df
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
            raise InvalidInputError(
                msg=f"Could not determine the number of features in the classifier {model_name}",
                source=self.__class__.__name__,
            )
        if not multiclass and clf.n_classes_ != 2:
            raise ClassifierInferenceError(
                msg=f"The classifier {model_name} (data path {data_path}) has not been created properly. See The SimBA GitHub FAQ page or Gitter for more information and suggested fixes. The classifier is not a binary classifier and does not predict two targets (absence and presence of behavior). One or more files inside the project_folder/csv/targets_inserted directory has an annotation column with a value other than 0 or 1",
                source=self.__class__.__name__,
            )
        if len(x_df.columns) != clf_n_features:
            if model_name and data_path:
                raise FeatureNumberMismatchError(
                    f"Mismatch in the number of features in input file {data_path}, and what is expected by the model {model_name}. The model expects {clf_n_features} features. The data contains {len(x_df.columns)} features.",
                    source=self.__class__.__name__,
                )
            else:
                raise FeatureNumberMismatchError(
                    f"The model expects {clf_n_features} features. The data contains {len(x_df.columns)} features.",
                    source=self.__class__.__name__,
                )
        p_vals = clf.predict_proba(x_df)
        if multiclass and (clf.n_classes_ != p_vals.shape[1]):
            raise ClassifierInferenceError(
                msg=f"The classifier {model_name} (data path: {data_path}) is a multiclassifier expected to create {clf.n_classes_} behavior probabilities. However, it produced probabilities for {p_vals.shape[1]} behaviors. See The SimBA GitHub FAQ page or Gitter for more information and suggested fixes.",
                source=self.__class__.__name__,
            )
        if not multiclass:
            return p_vals[:, 1]
        else:
            return p_vals

    def clf_fit(
        self, clf: RandomForestClassifier, x_df: pd.DataFrame, y_df: pd.DataFrame
    ) -> RandomForestClassifier:
        """
        Helper to fit clf model

        :param clf: Un-fitted random forest classifier object
        :param pd.DataFrame x_df: Pandas dataframe with features.
        :param pd.DataFrame y_df: Pandas dataframe/Series with target
        :return RandomForestClassifier: Fitted random forest classifier object
        """
        nan_features = x_df[~x_df.applymap(np.isreal).all(1)]
        nan_target = y_df.loc[pd.to_numeric(y_df).isna()]
        if len(nan_features) > 0:
            raise FaultyTrainingSetError(
                msg=f"{len(nan_features)} frame(s) in your project_folder/csv/targets_inserted directory contains FEATURES with non-numerical values",
                source=self.__class__.__name__,
            )
        if len(nan_target) > 0:
            raise FaultyTrainingSetError(
                msg=f"{len(nan_target)} frame(s) in your project_folder/csv/targets_inserted directory contains ANNOTATIONS with non-numerical values",
                source=self.__class__.__name__,
            )
        return clf.fit(x_df, y_df)

    @staticmethod
    def _read_data_file_helper(
        file_path: str,
        file_type: str,
        clf_names: Optional[List[str]] = None,
        raise_bool_clf_error: bool = True,
    ):
        """
        Private function called by :meth:`simba.train_model_functions.read_all_files_in_folder_mp`
        """

        timer = SimbaTimer(start=True)
        _, vid_name, _ = get_fn_ext(file_path)
        df = read_df(file_path, file_type).dropna(axis=0, how="all").fillna(0)
        frame_numbers = df.index
        df.index = [vid_name] * len(df)
        if clf_names != None:
            for clf_name in clf_names:
                if not clf_name in df.columns:
                    raise ColumnNotFoundError(
                        column_name=clf_name,
                        file_name=file_path,
                        source=TrainModelMixin._read_data_file_helper.__name__,
                    )
                elif (
                    len(set(df[clf_name].unique()) - {0, 1}) > 0
                    and raise_bool_clf_error
                ):
                    raise InvalidInputError(
                        msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {file_path} the {clf_name} field contains additional value(s): {list(set(df[clf_name].unique()) - {0, 1})}.",
                        source=TrainModelMixin._read_data_file_helper.__name__,
                    )
        timer.stop_timer()
        print(
            f"Reading complete {vid_name} (elapsed time: {timer.elapsed_time_str}s)..."
        )

        return df, frame_numbers

    @staticmethod
    def read_all_files_in_folder_mp(
        file_paths: List[str],
        file_type: Literal["csv", "parquet", "pickle"],
        classifier_names: Optional[List[str]] = None,
        raise_bool_clf_error: bool = True,
    ) -> (pd.DataFrame, List[int]):
        """

        Multiprocessing helper function to read in all data files in a folder to a single
        pd.DataFrame for downstream ML. Defaults to ceil(CPU COUNT / 2) cores. Asserts that all classifiers
        have annotation fields present in each dataframe.

        .. note::
          If multiprocess failure, reverts to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder`

        :parameter List[str] file_paths: List of file-paths
        :parameter List[str] file_paths: The filetype of ``file_paths`` OPTIONS: csv or parquet.
        :parameter Optional[List[str]] classifier_names: List of classifier names representing fields of human annotations. If not None, then assert that classifier names
            are present in each data file.
        :return pd.DataFrame: Concatenated dataframe of all data in ``file_paths``.
        :return pd.DataFrame: List of frame indexes of all concatenated files.

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
            MultiProcessingFailedWarning(
                msg="Multi-processing file read failed, reverting to single core (increased run-time)."
            )
            return TrainModelMixin.read_all_files_in_folder(
                file_paths=file_paths,
                file_type=file_type,
                classifier_names=classifier_names,
                raise_bool_clf_error=raise_bool_clf_error,
            )

    @staticmethod
    def _read_data_file_helper_futures(
        file_path: str,
        file_type: str,
        clf_names: Optional[List[str]] = None,
        raise_bool_clf_error: bool = True,
    ):
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
                    raise ColumnNotFoundError(column_name=clf_name, file_name=file_path)
                elif (
                    len(set(df[clf_name].unique()) - {0, 1}) > 0
                    and raise_bool_clf_error
                ):
                    raise InvalidInputError(
                        msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {file_path} the {clf_name} field contains additional value(s): {list(set(df[clf_name].unique()) - {0, 1})}."
                    )
        timer.stop_timer()
        return df, vid_name, timer.elapsed_time_str, frm_numbers

    def read_all_files_in_folder_mp_futures(
        self,
        annotations_file_paths: List[str],
        file_type: Literal["csv", "parquet", "pickle"],
        classifier_names: Optional[List[str]] = None,
        raise_bool_clf_error: bool = True,
    ) -> (pd.DataFrame, List[int]):
        """
        Multiprocessing helper function to read in all data files in a folder to a single
        pd.DataFrame for downstream ML through ``concurrent.Futures``. Asserts that all classifiers
        have annotation fields present in each dataframe.

        .. note::
           A ``concurrent.Futures`` alternative to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder_mp` which
           has uses ``multiprocessing.ProcessPoolExecutor`` and reported unstable on Linux machines.

           If multiprocess failure, reverts to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder`

        :parameter List[str] file_paths: List of file-paths
        :parameter List[str] file_paths: The filetype of ``file_paths`` OPTIONS: csv or parquet.
        :parameter Optional[List[str]] classifier_names: List of classifier names representing fields of human annotations. If not None, then assert that classifier names are present in each data file.
        :parameter bool raise_bool_clf_error: If True, raises an error if a classifier column contains values outside 0 and 1.
        :return pd.DataFrame: Concatenated dataframe of all data in ``file_paths``.

        """
        try:
            if (platform.system() == "Darwin") and (
                multiprocessing.get_start_method() != "spawn"
            ):
                multiprocessing.set_start_method("spawn", force=True)
            cpu_cnt, _ = find_core_cnt()
            df_lst, frm_number_list = [], []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cpu_cnt
            ) as executor:
                results = [
                    executor.submit(
                        self._read_data_file_helper_futures,
                        data,
                        file_type,
                        classifier_names,
                        raise_bool_clf_error,
                    )
                    for data in annotations_file_paths
                ]
                for result in concurrent.futures.as_completed(results):
                    df_lst.append(result.result()[0])
                    frm_number_list.extend((result.result()[-1]))
                    print(
                        f"Reading complete {result.result()[1]} (elapsed time: {result.result()[2]}s)..."
                    )
            df_concat = pd.concat(df_lst, axis=0).round(4)
            if "scorer" in df_concat.columns:
                df_concat = df_concat.drop(["scorer"], axis=1)

            return df_concat, frm_number_list

        except Exception as e:
            MultiProcessingFailedWarning(
                msg=f"Multi-processing file read failed, reverting to single core (increased run-time on large datasets). Exception: {e.args}"
            )
            return self.read_all_files_in_folder(
                file_paths=annotations_file_paths,
                file_type=file_type,
                classifier_names=classifier_names,
                raise_bool_clf_error=raise_bool_clf_error,
            )

    def check_raw_dataset_integrity(
        self, df: pd.DataFrame, logs_path: Optional[Union[str, os.PathLike]]
    ) -> None:
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
    def _create_shap_mp_helper(
        data: pd.DataFrame,
        explainer: shap.TreeExplainer,
        clf_name: str,
        rf_clf: RandomForestClassifier,
        expected_value: float,
    ):

        target = data.pop(clf_name).values.reshape(-1, 1)
        frame_batch_shap = explainer.shap_values(data.values, check_additivity=False)[1]
        shap_sum = np.sum(frame_batch_shap, axis=1).reshape(-1, 1)
        proba = rf_clf.predict_proba(data)[:, 1].reshape(-1, 1)
        frame_batch_shap = np.hstack(
            (
                frame_batch_shap,
                np.full((frame_batch_shap.shape[0]), expected_value).reshape(-1, 1),
                shap_sum,
                proba,
                target,
            )
        )
        return frame_batch_shap, data.values

    @staticmethod
    def _create_shap_mp_helper(
        data: pd.DataFrame, explainer: shap.TreeExplainer, clf_name: str
    ):

        target = data.pop(clf_name).values.reshape(-1, 1)
        group_cnt = data.pop("group").values[0]
        shap_vals = np.full((len(data), len(data.columns)), np.nan)
        for cnt, i in enumerate(list(data.index)):
            shap_vals[cnt] = explainer.shap_values(
                data.loc[i].values, check_additivity=False
            )[1]
            print(f"SHAP complete core frame: {i} (CORE BATCH: {group_cnt})")
        return shap_vals, data.values, target

    def create_shap_log_mp(
        self,
        ini_file_path: str,
        rf_clf: RandomForestClassifier,
        x_df: pd.DataFrame,
        y_df: pd.DataFrame,
        x_names: List[str],
        clf_name: str,
        cnt_present: int,
        cnt_absent: int,
        batch_size: int = 10,
        save_path: Optional[Union[str, os.PathLike]] = None,
        save_file_no: Optional[int] = None,
    ) -> Union[None, Tuple[pd.DataFrame]]:
        """
        Helper to compute SHAP values using multiprocessing.
        For single-core alternative, see  meth:`simba.mixins.train_model_mixins.TrainModelMixin.create_shap_log_mp`.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_

           .. image:: _static/img/shap.png
              :width: 400
              :align: center

        :param str ini_file_path: Path to the SimBA project_config.ini
        :param RandomForestClassifier rf_clf: sklearn random forest classifier
        :param pd.DataFrame x_df: Test features.
        :param pd.DataFrame y_df: Test target.
        :param List[str] x_names: Feature names.
        :param str clf_name: Classifier name.
        :param int cnt_present: Number of behavior-present frames to calculate SHAP values for.
        :param int cnt_absent: Number of behavior-absent frames to calculate SHAP values for.
        :param Optional[str, os.PathLike] save_dir: Optional directory where to save output in csv file format. If None, then returns the dataframes.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.

        """
        shap_timer = SimbaTimer(start=True)
        data_df = pd.concat([x_df, y_df], axis=1)
        if (save_file_no == None) and (save_path is not None):
            self.out_df_shap_path = os.path.join(
                save_path, f"SHAP_values_{clf_name}.csv"
            )
            self.out_df_raw_path = os.path.join(
                save_path, f"RAW_SHAP_feature_values_{clf_name}.csv"
            )
        elif (save_file_no is not None) and (save_path is not None):
            self.out_df_shap_path = os.path.join(
                save_path, f"SHAP_values_{str(save_file_no)}_{clf_name}.csv"
            )
            self.out_df_raw_path = os.path.join(
                save_path, f"RAW_SHAP_feature_values_{str(save_file_no)}_{clf_name}.csv"
            )
        target_df, nontarget_df = (
            data_df[data_df[y_df.name] == 1],
            data_df[data_df[y_df.name] == 0],
        )
        if len(target_df) < cnt_present:
            NotEnoughDataWarning(
                msg=f"Train data contains {len(target_df)} behavior-present annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_present)}). SimBA will calculate shap scores for the {len(target_df)} behavior-present frames available",
                source=self.__class__.__name__,
            )
            cnt_present = len(target_df)
        if len(nontarget_df) < cnt_absent:
            NotEnoughDataWarning(
                msg=f"Train data contains {len(nontarget_df)} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for ({str(cnt_absent)}). SimBA will calculate shap scores for the {len(nontarget_df)} behavior-absent frames available",
                source=self.__class__.__name__,
            )
            cnt_absent = len(nontarget_df)
        non_target_for_shap = nontarget_df.sample(cnt_absent, replace=False)
        targets_for_shap = target_df.sample(cnt_present, replace=False)
        explainer = shap.TreeExplainer(
            rf_clf,
            data=None,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )
        expected_value = explainer.expected_value[1]
        cores, _ = find_core_cnt()
        shap_data_df = pd.concat([targets_for_shap, non_target_for_shap], axis=0)
        if (len(shap_data_df) / batch_size) < 1:
            batch_size = 1
        if len(shap_data_df) > 100:
            batch_size = 100
        print(f"Computing {len(shap_data_df)} SHAP values (MULTI-CORE BATCH SIZE: {batch_size}, FOLLOW PROGRESS IN OS TERMINAL)...")
        shap_data, _ = self.split_and_group_df(df=shap_data_df, splits=int(len(shap_data_df) / batch_size))
        shap_results, shap_raw = [], []
        try:
            with multiprocessing.Pool(cores, maxtasksperchild=10) as pool:
                constants = functools.partial(
                    self._create_shap_mp_helper, explainer=explainer, clf_name=clf_name
                )
                for cnt, result in enumerate(
                    pool.imap_unordered(constants, shap_data, chunksize=1)
                ):
                    print(
                        f"Concatenating multi-processed SHAP data (batch {cnt+1}/{len(shap_data)})"
                    )
                    proba = rf_clf.predict_proba(result[1])[:, 1].reshape(-1, 1)
                    shap_sum = np.sum(result[0], axis=1).reshape(-1, 1)
                    batch_shap_results = np.hstack(
                        (
                            result[0],
                            np.full((result[0].shape[0]), expected_value).reshape(
                                -1, 1
                            ),
                            shap_sum,
                            proba,
                            result[2],
                        )
                    )
                    shap_results.append(batch_shap_results)
                    shap_raw.append(result[1])
            pool.terminate()
            pool.join()
            shap_save_df = pd.DataFrame(
                data=np.row_stack(shap_results),
                columns=list(x_names)
                + ["Expected_value", "Sum", "Prediction_probability", clf_name],
            )
            raw_save_df = pd.DataFrame(
                data=np.row_stack(shap_raw), columns=list(x_names)
            )

            shap_timer.stop_timer()
            stdout_success(
                msg="SHAP calculations complete",
                elapsed_time=shap_timer.elapsed_time_str,
                source=self.__class__.__name__,
            )
            if save_path:
                shap_save_df.to_csv(self.out_df_shap_path)
                raw_save_df.to_csv(self.out_df_raw_path)
                _ = ShapAggregateStatisticsVisualizer(
                    config_path=ini_file_path,
                    classifier_name=clf_name,
                    shap_df=shap_save_df,
                    shap_baseline_value=int(expected_value * 100),
                    save_path=save_path,
                )
            else:
                return (shap_save_df, raw_save_df, int(expected_value * 100))

        except Exception as e:
            print(e.args)
            ShapWarning(
                msg="Multiprocessing SHAP values failed. Revert to single core. This will negatively affect run-time. ",
                source=self.__class__.__name__,
            )
            self.create_shap_log(
                ini_file_path=ini_file_path,
                rf_clf=rf_clf,
                x_df=x_df,
                y_df=y_df,
                x_names=x_names,
                clf_name=clf_name,
                cnt_present=cnt_present,
                cnt_absent=cnt_absent,
                save_path=save_path,
                save_it=len(x_df),
                save_file_no=save_file_no,
            )

    def check_df_dataset_integrity(
        self, df: pd.DataFrame, file_name: str, logs_path: Union[str, os.PathLike]
    ) -> None:
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

        self.model_dir_out = os.path.join(
            read_config_entry(
                config,
                ConfigKey.SML_SETTINGS.value,
                ConfigKey.MODEL_DIR.value,
                data_type=Dtypes.STR.value,
            ),
            "generated_models",
        )
        if not os.path.exists(self.model_dir_out):
            os.makedirs(self.model_dir_out)
        self.eval_out_path = os.path.join(self.model_dir_out, "model_evaluations")
        if not os.path.exists(self.eval_out_path):
            os.makedirs(self.eval_out_path)
        self.clf_name = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.CLASSIFIER.value,
            data_type=Dtypes.STR.value,
        )
        self.tt_size = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.TT_SIZE.value,
            data_type=Dtypes.FLOAT.value,
        )
        self.algo = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.MODEL_TO_RUN.value,
            data_type=Dtypes.STR.value,
            default_value="rf",
        )
        self.split_type = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value,
            data_type=Dtypes.STR.value,
            options=Options.TRAIN_TEST_SPLIT.value,
            default_value=Methods.SPLIT_TYPE_FRAMES.value,
        )
        self.under_sample_setting = (
            read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.UNDERSAMPLE_SETTING.value,
                data_type=Dtypes.STR.value,
            )
            .lower()
            .strip()
        )
        self.over_sample_setting = (
            read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.OVERSAMPLE_SETTING.value,
                data_type=Dtypes.STR.value,
            )
            .lower()
            .strip()
        )
        self.n_estimators = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.RF_ESTIMATORS.value,
            data_type=Dtypes.INT.value,
        )
        self.rf_max_depth = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.RF_MAX_DEPTH.value,
            data_type=Dtypes.INT.value,
            default_value=Dtypes.NONE.value,
        )
        if self.rf_max_depth == "None":
            self.rf_max_depth = None
        self.max_features = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.RF_MAX_FEATURES.value,
            data_type=Dtypes.STR.value,
        )
        self.criterion = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.RF_CRITERION.value,
            data_type=Dtypes.STR.value,
            options=Options.CLF_CRITERION.value,
        )
        self.min_sample_leaf = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.MIN_LEAF.value,
            data_type=Dtypes.INT.value,
        )
        self.compute_permutation_importance = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.PERMUTATION_IMPORTANCE.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_learning_curve = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.LEARNING_CURVE.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_precision_recall_curve = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.PRECISION_RECALL.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_example_decision_tree = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.EX_DECISION_TREE.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_classification_report = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.CLF_REPORT.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_features_importance_log = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.IMPORTANCE_LOG.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_features_importance_bar_graph = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.IMPORTANCE_LOG.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_example_decision_tree_fancy = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.EX_DECISION_TREE_FANCY.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.generate_shap_scores = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.SHAP_SCORES.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.save_meta_data = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.RF_METADATA.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.compute_partial_dependency = read_config_entry(
            config,
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
            MLParamKeys.PARTIAL_DEPENDENCY.value,
            data_type=Dtypes.STR.value,
            default_value=False,
        )
        self.save_train_test_frm_info = str_2_bool(
            read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value,
                data_type=Dtypes.STR.value,
                default_value="False",
            )
        )

        if self.under_sample_setting == Methods.RANDOM_UNDERSAMPLE.value:
            self.under_sample_ratio = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.UNDERSAMPLE_RATIO.value,
                data_type=Dtypes.FLOAT.value,
                default_value=Dtypes.NAN.value,
            )
            check_float(
                name=MLParamKeys.UNDERSAMPLE_RATIO.value, value=self.under_sample_ratio
            )
        else:
            self.under_sample_ratio = Dtypes.NAN.value
        if (self.over_sample_setting == Methods.SMOTEENN.value.lower()) or (
            self.over_sample_setting == Methods.SMOTE.value.lower()
        ):
            self.over_sample_ratio = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.OVERSAMPLE_RATIO.value,
                data_type=Dtypes.FLOAT.value,
                default_value=Dtypes.NAN.value,
            )
            check_float(
                name=MLParamKeys.OVERSAMPLE_RATIO.value, value=self.over_sample_ratio
            )
        else:
            self.over_sample_ratio = Dtypes.NAN.value

        if config.has_option(
            ConfigKey.CREATE_ENSEMBLE_SETTINGS.value, MLParamKeys.CLASS_WEIGHTS.value
        ):
            self.class_weights = read_config_entry(
                config,
                ConfigKey.CREATE_ENSEMBLE_SETTINGS.value,
                MLParamKeys.CLASS_WEIGHTS.value,
                data_type=Dtypes.STR.value,
                default_value=Dtypes.NONE.value,
            )
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

    def check_validity_of_meta_files(
        self, data_df: pd.DataFrame, meta_file_paths: List[Union[str, os.PathLike]]
    ):
        meta_dicts, errors = {}, []
        for config_cnt, path in enumerate(meta_file_paths):
            _, meta_file_name, _ = get_fn_ext(path)
            meta_dict = read_meta_file(path)
            meta_dict = {k.lower(): v for k, v in meta_dict.items()}
            errors.append(
                check_str(
                    name=meta_dict[MLParamKeys.CLASSIFIER_NAME.value],
                    value=meta_dict[MLParamKeys.CLASSIFIER_NAME.value],
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_str(
                    name=MLParamKeys.RF_CRITERION.value,
                    value=meta_dict[MLParamKeys.RF_CRITERION.value],
                    options=Options.CLF_CRITERION.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_str(
                    name=MLParamKeys.RF_MAX_FEATURES.value,
                    value=meta_dict[MLParamKeys.RF_MAX_FEATURES.value],
                    options=Options.CLF_MAX_FEATURES.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_str(
                    MLParamKeys.UNDERSAMPLE_SETTING.value,
                    meta_dict[MLParamKeys.UNDERSAMPLE_SETTING.value].lower(),
                    options=[x.lower() for x in Options.UNDERSAMPLE_OPTIONS.value],
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_str(
                    MLParamKeys.OVERSAMPLE_SETTING.value,
                    meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower(),
                    options=[x.lower() for x in Options.OVERSAMPLE_OPTIONS.value],
                    raise_error=False,
                )[1]
            )
            if MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value in meta_dict.keys():
                errors.append(
                    check_str(
                        name=meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value],
                        value=meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value],
                        options=Options.TRAIN_TEST_SPLIT.value,
                        raise_error=False,
                    )[1]
                )

            errors.append(
                check_int(
                    name=MLParamKeys.RF_ESTIMATORS.value,
                    value=meta_dict[MLParamKeys.RF_ESTIMATORS.value],
                    min_value=1,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_int(
                    name=MLParamKeys.MIN_LEAF.value,
                    value=meta_dict[MLParamKeys.MIN_LEAF.value],
                    raise_error=False,
                )[1]
            )
            if (
                meta_dict[MLParamKeys.LEARNING_CURVE.value]
                in Options.PERFORM_FLAGS.value
            ):
                errors.append(
                    check_int(
                        name=MLParamKeys.LEARNING_CURVE_K_SPLITS.value,
                        value=meta_dict[MLParamKeys.LEARNING_CURVE_K_SPLITS.value],
                        raise_error=False,
                    )[1]
                )
                errors.append(
                    check_int(
                        name=MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value,
                        value=meta_dict[MLParamKeys.LEARNING_CURVE_DATA_SPLITS.value],
                        raise_error=False,
                    )[1]
                )
            if (
                meta_dict[MLParamKeys.IMPORTANCE_BAR_CHART.value]
                in Options.PERFORM_FLAGS.value
            ):
                errors.append(
                    check_int(
                        name=MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value,
                        value=meta_dict[MLParamKeys.N_FEATURE_IMPORTANCE_BARS.value],
                        raise_error=False,
                    )[1]
                )
            if MLParamKeys.SHAP_SCORES.value in meta_dict.keys():
                if (
                    meta_dict[MLParamKeys.SHAP_SCORES.value]
                    in Options.PERFORM_FLAGS.value
                ):
                    errors.append(
                        check_int(
                            name=MLParamKeys.SHAP_PRESENT.value,
                            value=meta_dict[MLParamKeys.SHAP_PRESENT.value],
                            raise_error=False,
                        )[1]
                    )
                    errors.append(
                        check_int(
                            name=MLParamKeys.SHAP_ABSENT.value,
                            value=meta_dict[MLParamKeys.SHAP_ABSENT.value],
                            raise_error=False,
                        )[1]
                    )
            if MLParamKeys.RF_MAX_DEPTH.value in meta_dict.keys():
                if meta_dict[MLParamKeys.RF_MAX_DEPTH.value] != Dtypes.NONE.value:
                    errors.append(
                        check_int(
                            name=MLParamKeys.RF_MAX_DEPTH.value,
                            value=meta_dict[MLParamKeys.RF_MAX_DEPTH.value],
                            min_value=1,
                            raise_error=False,
                        )[1]
                    )
                else:
                    meta_dict[MLParamKeys.RF_MAX_DEPTH.value] = None
            else:
                meta_dict[MLParamKeys.RF_MAX_DEPTH.value] = None

            errors.append(
                check_float(
                    name=MLParamKeys.TT_SIZE.value,
                    value=meta_dict[MLParamKeys.TT_SIZE.value],
                    raise_error=False,
                )[1]
            )
            if (
                meta_dict[MLParamKeys.UNDERSAMPLE_SETTING.value].lower()
                == Methods.RANDOM_UNDERSAMPLE.value
            ):
                errors.append(
                    check_float(
                        name=MLParamKeys.UNDERSAMPLE_RATIO.value,
                        value=meta_dict[MLParamKeys.UNDERSAMPLE_RATIO.value],
                        raise_error=False,
                    )[1]
                )
                try:
                    present_len, absent_len = len(
                        data_df[
                            data_df[meta_dict[MLParamKeys.CLASSIFIER_NAME.value]] == 1
                        ]
                    ), len(
                        data_df[
                            data_df[meta_dict[MLParamKeys.CLASSIFIER_NAME.value]] == 0
                        ]
                    )
                    ratio_n = int(
                        present_len * meta_dict[MLParamKeys.UNDERSAMPLE_RATIO.value]
                    )
                    if absent_len < ratio_n:
                        errors.append(
                            f"The under-sample ratio of {meta_dict[MLParamKeys.UNDERSAMPLE_RATIO.value]} in \n classifier {meta_dict[MLParamKeys.CLASSIFIER_NAME.value]} demands {ratio_n} behavior-absent annotations."
                        )
                except:
                    pass

            if (
                meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower()
                == Methods.SMOTEENN.value.lower()
            ) or (
                meta_dict[MLParamKeys.OVERSAMPLE_SETTING.value].lower()
                == Methods.SMOTE.value.lower()
            ):
                errors.append(
                    check_float(
                        name=MLParamKeys.OVERSAMPLE_RATIO.value,
                        value=meta_dict[MLParamKeys.OVERSAMPLE_RATIO.value],
                        raise_error=False,
                    )[1]
                )

            errors.append(
                check_if_valid_input(
                    name=MLParamKeys.RF_METADATA.value,
                    input=meta_dict[MLParamKeys.RF_METADATA.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_if_valid_input(
                    MLParamKeys.EX_DECISION_TREE.value,
                    input=meta_dict[MLParamKeys.EX_DECISION_TREE.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_if_valid_input(
                    MLParamKeys.CLF_REPORT.value,
                    input=meta_dict[MLParamKeys.CLF_REPORT.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_if_valid_input(
                    MLParamKeys.IMPORTANCE_LOG.value,
                    input=meta_dict[MLParamKeys.IMPORTANCE_LOG.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_if_valid_input(
                    MLParamKeys.IMPORTANCE_BAR_CHART.value,
                    input=meta_dict[MLParamKeys.IMPORTANCE_BAR_CHART.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_if_valid_input(
                    MLParamKeys.PERMUTATION_IMPORTANCE.value,
                    input=meta_dict[MLParamKeys.PERMUTATION_IMPORTANCE.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_if_valid_input(
                    MLParamKeys.LEARNING_CURVE.value,
                    input=meta_dict[MLParamKeys.LEARNING_CURVE.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            errors.append(
                check_if_valid_input(
                    MLParamKeys.PRECISION_RECALL.value,
                    input=meta_dict[MLParamKeys.PRECISION_RECALL.value],
                    options=Options.RUN_OPTIONS_FLAGS.value,
                    raise_error=False,
                )[1]
            )
            if MLParamKeys.PARTIAL_DEPENDENCY.value in meta_dict.keys():
                errors.append(
                    check_if_valid_input(
                        MLParamKeys.PARTIAL_DEPENDENCY.value,
                        input=meta_dict[MLParamKeys.PARTIAL_DEPENDENCY.value],
                        options=Options.RUN_OPTIONS_FLAGS.value,
                        raise_error=False,
                    )[1]
                )
            if MLParamKeys.SHAP_MULTIPROCESS.value in meta_dict.keys():
                errors.append(
                    check_if_valid_input(
                        MLParamKeys.SHAP_MULTIPROCESS.value,
                        input=meta_dict[MLParamKeys.SHAP_MULTIPROCESS.value],
                        options=Options.RUN_OPTIONS_FLAGS.value,
                        raise_error=False,
                    )[1]
                )
            if meta_dict[MLParamKeys.RF_MAX_FEATURES.value] == Dtypes.NONE.value:
                meta_dict[MLParamKeys.RF_MAX_FEATURES.value] = None
            if MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value not in meta_dict.keys():
                meta_dict[MLParamKeys.TRAIN_TEST_SPLIT_TYPE.value] = (
                    Methods.SPLIT_TYPE_FRAMES.value
                )

            if MLParamKeys.CLASS_WEIGHTS.value in meta_dict.keys():
                if (
                    meta_dict[MLParamKeys.CLASS_WEIGHTS.value]
                    not in Options.CLASS_WEIGHT_OPTIONS.value
                ):
                    meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = None
                if meta_dict[MLParamKeys.CLASS_WEIGHTS.value] == "custom":
                    meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = ast.literal_eval(
                        meta_dict[MLParamKeys.CLASS_CUSTOM_WEIGHTS.value]
                    )
                    for k, v in meta_dict[MLParamKeys.CLASS_WEIGHTS.value].items():
                        meta_dict[MLParamKeys.CLASS_WEIGHTS.value][k] = int(v)
                if meta_dict[MLParamKeys.CLASS_WEIGHTS.value] == Dtypes.NONE.value:
                    meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = None
            else:
                meta_dict[MLParamKeys.CLASS_WEIGHTS.value] = None

            if MLParamKeys.CLASSIFIER_MAP.value in meta_dict.keys():
                meta_dict[MLParamKeys.CLASSIFIER_MAP.value] = ast.literal_eval(
                    meta_dict[MLParamKeys.CLASSIFIER_MAP.value]
                )
                for k, v in meta_dict[MLParamKeys.CLASSIFIER_MAP.value].items():
                    errors.append(check_int(name="MULTICLASS KEYS", value=k, raise_error=False)[1])
                    errors.append(check_str(name="MULTICLASS VALUES", value=v, raise_error=False)[1])

            else:
                meta_dict[MLParamKeys.CLASSIFIER_MAP.value] = None

            if MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value in meta_dict.keys():
                errors.append(
                    check_if_valid_input(
                        name="SAVE TRAIN AND TEST FRAME INDEXES",
                        input=meta_dict[MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value],
                        options=Options.RUN_OPTIONS_FLAGS.value,
                        raise_error=False,
                    )[1]
                )
            else:
                meta_dict[MLParamKeys.SAVE_TRAIN_TEST_FRM_IDX.value] = False

            errors = [x for x in errors if x != ""]
            if errors:
                option = TwoOptionQuestionPopUp(
                    question=f"{errors[0]} \n ({meta_file_name}) \n  Do you want to skip this meta file or terminate training ?",
                    title="META CONFIG FILE ERROR",
                    option_one="SKIP",
                    option_two="TERMINATE",
                )

                if option.selected_option == "SKIP":
                    continue
                else:
                    raise InvalidInputError(
                        msg=errors[0], source=self.__class__.__name__
                    )
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
                raise FeatureNumberMismatchError(
                    msg=f"The scaler {name} expects {scaler.n_features_in_} features. Got {len(data.columns)}.",
                    source=TrainModelMixin.scaler_transform.__name__,
                )

        return pd.DataFrame(
            scaler.inverse_transform(data), columns=data.columns
        ).set_index(data.index)

    @staticmethod
    def define_scaler(
        scaler_name: Literal["MIN-MAX", "STANDARD", "QUANTILE"]
    ) -> Union[MinMaxScaler, StandardScaler, QuantileTransformer]:
        """
        Defines a sklearn scaler object. See ``UMLOptions.SCALER_OPTIONS.value`` for accepted scalers.

        :example:
        >>> TrainModelMixin.define_scaler(scaler_name='MIN-MAX')
        """

        if scaler_name not in Options.SCALER_OPTIONS.value:
            raise InvalidInputError(
                msg=f"Scaler {scaler_name} not supported. Options: {Options.SCALER_OPTIONS.value}",
                source=self.__class__.__name__,
            )
        if scaler_name == Options.MIN_MAX_SCALER.value:
            return MinMaxScaler()
        elif scaler_name == Options.STANDARD_SCALER.value:
            return StandardScaler()
        elif scaler_name == Options.QUANTILE_SCALER.value:
            return QuantileTransformer()

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
    def find_low_variance_fields(
        data: pd.DataFrame, variance_threshold: float
    ) -> List[str]:
        """
        Finds fields with variance below provided threshold.

        :param pd.DataFrame data: Dataframe with continoues numerical features.
        :param float variance: Variance threshold (0.0-1.0).
        :return List[str]:
        """
        feature_selector = VarianceThreshold(
            threshold=round((variance_threshold / 100), 2)
        )
        feature_selector.fit(data)
        low_variance_fields = [
            c
            for c in data.columns
            if c not in data.columns[feature_selector.get_support()]
        ]
        if len(low_variance_fields) == len(data.columns):
            raise NoDataError(
                msg=f"All feature columns show a variance below the {variance_threshold} threshold. Thus, no data remain for analysis.",
                source=TrainModelMixin.find_low_variance_fields.__name__,
            )
        return low_variance_fields


# test = TrainModelMixin()
# test.read_all_files_in_folder(file_paths=['/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437C_c3_2022-11-01_13-16-23_color.csv', '/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437D_c4_2022-11-01_13-16-39_color.csv'],
#                               file_type='csv', classifier_names=['attack', 'non-agresive parallel swimming'])


# test = TrainModelMixin()
# test.read_all_files_in_folder_mp(file_paths=['/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437C_c3_2022-11-01_13-16-23_color.csv', '/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437D_c4_2022-11-01_13-16-39_color.csv'],
#                               file_type='csv', classifier_names=['attack', 'non-agresive parallel swimming'])
#
