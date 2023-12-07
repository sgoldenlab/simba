__author__ = "Simon Nilsson"

import shutil
import subprocess
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

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
from subprocess import call

import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import ShuffleSplit, learning_curve
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
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.plotting.shap_agg_stats_visualizer import \
    ShapAggregateStatisticsVisualizer
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_str)
from simba.utils.data import create_color_palette, detect_bouts
from simba.utils.enums import ConfigKey, Defaults, Dtypes, MetaKeys
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
                                    read_df)
from simba.utils.warnings import (MissingUserInputWarning,
                                  MultiProcessingFailedWarning,
                                  NoModuleWarning, NotEnoughDataWarning,
                                  ShapWarning)

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
    ) -> pd.DataFrame:
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

        :examples:
        >>> self.read_all_files_in_folder(file_paths=['targets_inserted/Video_1.csv', 'targets_inserted/Video_2.csv'], file_type='csv', classifier_names=['Attack'])
        """

        timer = SimbaTimer(start=True)
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
            df.index = [vid_name] * len(df)
            if classifier_names != None:
                for clf_name in classifier_names:
                    if not clf_name in df.columns:
                        raise MissingColumnsError(
                            msg=f"Data for video {vid_name} does not contain any annotations for behavior {clf_name}. Delete classifier {clf_name} from the SimBA project, or add annotations for behavior {clf_name} to the video {vid_name}",
                            source=self.__class__.__name__,
                        )
                    elif len(set(df[clf_name].unique()) - {0, 1}) > 0:
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
        return df_concat.astype(np.float32)

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
            self, df: pd.DataFrame, annotations_lst: List[str]
    ) -> pd.DataFrame:
        """
        Helper to drop fields that contain annotations which are not the target.

        :parameter pd.DataFrame df: Dataframe holding features and annotations.
        :parameter List[str] annotations_lst: column fields to be removed from df
        :return pd.DataFrame: Dataframe without non-target annotation columns

        :examples:
        >>> self.delete_other_annotation_columns(df=df, annotations_lst=['Sniffing'])
        """

        for a_col in annotations_lst:
            df = df.drop([a_col], axis=1)
        return df

    def split_df_to_x_y(
            self, df: pd.DataFrame, col_names: [str]
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Helper to split dataframe into features and target.

        :parameter pd.DataFrame df: Dataframe holding features and annotations.
        :parameter str clf_name: Name of target.

        :return pd.DataFrame: features
        :return pd.DataFrame: target

        :examples:
        >>> self.split_df_to_x_y(df=df, col_names='Attack')
        """

        df = deepcopy(df)
        ys = np.array([0]*len(df.index))
        for i,col_name in enumerate(col_names):
            y = df.pop(col_name)
            ys[y == 1] = i+1

        return df, pd.DataFrame(ys)

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
        return self.split_df_to_x_y(data_df, [y_train.name])

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

        print("Performing SMOTEEN oversampling...")
        smt = SMOTEENN(sampling_strategy=sample_ratio)
        return smt.fit_sample(x_train, y_train)

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
        return smt.fit_sample(x_train, y_train)

    def calc_permutation_importance(
            self,
            x_test: np.ndarray,
            y_test: np.ndarray,
            clf: RandomForestClassifier,
            feature_names: List[str],
            clf_name: str,
            save_dir: Union[str, os.PathLike],
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Helper to calculate feature permutation importance scores.

        :parameter np.ndarray x_test: 2d feature test data of shape len(frames) x len(features)
        :parameter np.ndarray y_test: 2d feature target test data of shape len(frames) x 1
        :parameter RandomForestClassifier clf: random forest classifier
        :parameter List[str] feature_names: Names of features in x_test
        :parameter str clf_name: Name of classifier in y_test
        :parameter str save_dir: Directory where to save results in CSV format
        :parameter Optional[int] save_file_no: If permutation importance calculation is part of a grid search, provide integer representing sequence.
        """

        print("Calculating feature permutation importances...")
        timer = SimbaTimer(start=True)
        p_importances = permutation_importance(
            clf, x_test, y_test, n_repeats=10, random_state=0
        )
        df = pd.DataFrame(
            np.column_stack(
                [
                    feature_names,
                    p_importances.importances_mean,
                    p_importances.importances_std,
                ]
            ),
            columns=[
                "FEATURE_NAME",
                "FEATURE_IMPORTANCE_MEAN",
                "FEATURE_IMPORTANCE_STDEV",
            ],
        )
        df = df.sort_values(by=["FEATURE_IMPORTANCE_MEAN"], ascending=False)
        if save_file_no != None:
            save_file_path = os.path.join(
                save_dir,
                clf_name
                + "_"
                + str(save_file_no + 1)
                + "_permutations_importances.csv",
            )
        else:
            save_file_path = os.path.join(
                save_dir, clf_name + "_permutations_importances.csv"
            )
        df.to_csv(save_file_path, index=False)
        timer.stop_timer()
        print(
            "Permutation importance calculation complete (elapsed time: {}s) ...".format(
                timer.elapsed_time_str
            )
        )

    def calc_learning_curve(
            self,
            x_y_df: pd.DataFrame,
            clf_name: str,
            shuffle_splits: int,
            dataset_splits: int,
            tt_size: float,
            rf_clf: RandomForestClassifier,
            save_dir: str,
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Helper to compute random forest learning curves with cross-validation.

        .. image:: _static/img/learning_curves.png
           :width: 400
           :align: center

        :parameter pd.DataFrame x_y_df: Dataframe holding features and target.
        :parameter str clf_name: Name of the classifier
        :parameter int shuffle_splits: Number of cross-validation datasets at each data split.
        :parameter int dataset_splits: Number of data splits.
        :parameter float tt_size: test size
        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """

        print("Calculating learning curves...")
        timer = SimbaTimer(start=True)
        x_df, y_df = self.split_df_to_x_y(x_y_df, [clf_name])
        cv = ShuffleSplit(n_splits=shuffle_splits, test_size=tt_size)
        if platform.system() == "Darwin":
            with parallel_backend("threading", n_jobs=-2):
                train_sizes, train_scores, test_scores = learning_curve(
                    estimator=rf_clf,
                    X=x_df,
                    y=y_df,
                    cv=cv,
                    scoring="f1",
                    shuffle=False,
                    verbose=0,
                    train_sizes=np.linspace(0.01, 1.0, dataset_splits),
                    error_score="raise",
                )
        else:
            train_sizes, train_scores, test_scores = learning_curve(
                estimator=rf_clf,
                X=x_df,
                y=y_df,
                cv=cv,
                scoring="f1",
                shuffle=False,
                n_jobs=-1,
                verbose=0,
                train_sizes=np.linspace(0.01, 1.0, dataset_splits),
                error_score="raise",
            )
        results_df = pd.DataFrame()
        results_df["FRACTION TRAIN SIZE"] = np.linspace(0.01, 1.0, dataset_splits)
        results_df["TRAIN_MEAN_F1"] = np.mean(train_scores, axis=1)
        results_df["TEST_MEAN_F1"] = np.mean(test_scores, axis=1)
        results_df["TRAIN_STDEV_F1"] = np.std(train_scores, axis=1)
        results_df["TEST_STDEV_F1"] = np.std(test_scores, axis=1)
        if save_file_no != None:
            self.learning_curve_save_path = os.path.join(
                save_dir, clf_name + "_" + str(save_file_no + 1) + "_learning_curve.csv"
            )
        else:
            self.learning_curve_save_path = os.path.join(
                save_dir, clf_name + "_learning_curve.csv"
            )
        results_df.to_csv(self.learning_curve_save_path, index=False)
        timer.stop_timer()
        print(
            "Learning curve calculation complete (elapsed time: {}s) ...".format(
                timer.elapsed_time_str
            )
        )

    def calc_pr_curve(
            self,
            rf_clf: RandomForestClassifier,
            x_df: pd.DataFrame,
            y_df: pd.DataFrame,
            clf_name: str,
            save_dir: str,
            save_file_no: Optional[int] = None,
    ) -> None:
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
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """

        print("Calculating PR curves...")
        timer = SimbaTimer(start=True)
        p = rf_clf.predict_proba(x_df)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_df, p, pos_label=1)
        pr_df = pd.DataFrame()
        pr_df["PRECISION"] = precision
        pr_df["RECALL"] = recall
        pr_df["F1"] = (
                2
                * pr_df["RECALL"]
                * pr_df["PRECISION"]
                / (pr_df["RECALL"] + pr_df["PRECISION"])
        )
        thresholds = list(thresholds)
        thresholds.insert(0, 0.00)
        pr_df["DISCRIMINATION THRESHOLDS"] = thresholds
        if save_file_no != None:
            self.pr_save_path = os.path.join(
                save_dir, clf_name + "_" + str(save_file_no + 1) + "_pr_curve.csv"
            )
        else:
            self.pr_save_path = os.path.join(save_dir, clf_name + "_pr_curve.csv")
        pr_df.to_csv(self.pr_save_path, index=False)
        timer.stop_timer()
        print(
            "Precision-recall curve calculation complete (elapsed time: {}s) ...".format(
                timer.elapsed_time_str
            )
        )

    def create_example_dt(
            self,
            rf_clf: RandomForestClassifier,
            clf_name: str,
            feature_names: List[str],
            class_names: List[str],
            save_dir: str,
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Helper to produce visualization of random forest decision tree using graphviz.

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter str clf_name: Classifier name.
        :parameter List[str] feature_names: List of feature names.
        :parameter List[str] class_names: List of classes. E.g., ['Attack absent', 'Attack present']
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """
        dot_path = self.find_graphviz_dot()
        if not dot_path:
            print("please install graphviz using the following link: https://graphviz.org/download/")
            return
        print(f"Visualizing example decision tree using graphviz using {dot_path}")
        estimator = rf_clf.estimators_[3]
        if save_file_no != None:
            dot_name = os.path.join(
                save_dir, str(clf_name) + "_" + str(save_file_no) + "_tree.dot"
            )
            file_name = os.path.join(
                save_dir, str(clf_name) + "_" + str(save_file_no) + "_tree.pdf"
            )
        else:
            dot_name = os.path.join(save_dir, str(clf_name) + "_tree.dot")
            file_name = os.path.join(save_dir, str(clf_name) + "_tree.png")
        export_graphviz(
            estimator,
            out_file=dot_name,
            filled=True,
            rounded=True,
            special_characters=False,
            impurity=False,
            class_names=class_names,
            feature_names=feature_names,
        )
        subprocess.run([dot_path, '-Tpng', dot_name, '-o', file_name])

    def find_graphviz_dot(self):
        # Check if dot is in PATH
        dot_path = shutil.which("dot")
        if dot_path:
            return dot_path

        common_paths = [
            r"C:\Program Files\Graphviz\bin\dot.exe",
            r"C:\Program Files (x86)\Graphviz\bin\dot.exe"
        ]

        # Check common paths
        for path in common_paths:
            if os.path.isfile(path):
                return path
        return None

    def create_clf_report(
            self,
            rf_clf: RandomForestClassifier,
            x_df: pd.DataFrame,
            y_df: pd.DataFrame,
            class_names: List[str],
            save_dir: str,
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Helper to create classifier truth table report.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_
            .. image:: _static/img/clf_report.png
               :width: 400
               :align: center

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter pd.DataFrame x_df: dataframe holding test features
        :parameter pd.DataFrame y_df: dataframe holding test target
        :parameter List[str] class_names: List of classes. E.g., ['Attack absent', 'Attack present']
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """

        print("Creating classification report visualization...")
        try:
            visualizer = ClassificationReport(rf_clf, classes=class_names, support=True)
            visualizer.score(x_df, y_df)
            if save_file_no != None:
                save_path = os.path.join(
                    save_dir,
                    class_names[1]
                    + "_"
                    + str(save_file_no)
                    + "_classification_report.png",
                )
            else:
                save_path = os.path.join(
                    save_dir, class_names[1] + "_classification_report.png"
                )
            visualizer.poof(outpath=save_path, clear_figure=True)
        except KeyError as e:
            NotEnoughDataWarning(
                msg=f"Not enough data to create classification report: {class_names[1]}",
                source=self.__class__.__name__,
            )

    def create_x_importance_log(
            self,
            rf_clf: RandomForestClassifier,
            x_names: List[str],
            clf_name: str,
            save_dir: str,
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Helper to save gini or entropy based feature importance scores.

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter List[str] x_names: Names of features.
        :parameter str clf_name: Name of classifier
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """

        print("Creating feature importance log...")
        importances = list(rf_clf.feature_importances_)
        feature_importances = [
            (feature, round(importance, 2))
            for feature, importance in zip(x_names, importances)
        ]
        df = pd.DataFrame(
            feature_importances, columns=["FEATURE", "FEATURE_IMPORTANCE"]
        ).sort_values(by=["FEATURE_IMPORTANCE"], ascending=False)
        if save_file_no != None:
            self.f_importance_save_path = os.path.join(
                save_dir,
                clf_name + "_" + str(save_file_no) + "_feature_importance_log.csv",
            )
        else:
            self.f_importance_save_path = os.path.join(
                save_dir, clf_name + "_feature_importance_log.csv"
            )
        df.to_csv(self.f_importance_save_path, index=False)

    def create_x_importance_bar_chart(
            self,
            rf_clf: RandomForestClassifier,
            x_names: list,
            clf_name: str,
            save_dir: str,
            n_bars: int,
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Helper to create a bar chart displaying the top N gini or entropy feature importance scores.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_
            .. image:: _static/img/gini_bar_chart.png
               :width: 400
               :align: center

        :parameter RandomForestClassifier rf_clf: sklearn RandomForestClassifier object.
        :parameter List[str] x_names: Names of features.
        :parameter str clf_name: Name of classifier.
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter int n_bars: Number of bars in the plot.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search
        """

        print("Creating feature importance bar chart...")
        self.create_x_importance_log(rf_clf, x_names, clf_name, save_dir)
        importances_df = pd.read_csv(
            os.path.join(save_dir, clf_name + "_feature_importance_log.csv")
        )
        importances_head = importances_df.head(n_bars)
        colors = create_color_palette(
            pallete_name="hot", increments=n_bars, as_rgb_ratio=True
        )
        colors = [x[::-1] for x in colors]
        ax = importances_head.plot.bar(
            x="FEATURE",
            y="FEATURE_IMPORTANCE",
            legend=False,
            rot=90,
            fontsize=6,
            color=colors,
        )
        plt.ylabel("Feature importances' (mean decrease impurity)", fontsize=6)
        plt.tight_layout()
        if save_file_no != None:
            save_file_path = os.path.join(
                save_dir,
                clf_name
                + "_"
                + str(save_file_no)
                + "_feature_importance_bar_graph.png",
            )
        else:
            save_file_path = os.path.join(
                save_dir, clf_name + "_feature_importance_bar_graph.png"
            )
        plt.savefig(save_file_path, dpi=600)
        plt.close("all")

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
    def split_and_group_df(
            df: pd.DataFrame, splits: int, include_split_order: bool = True
    ) -> (List[pd.DataFrame], int):
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

    def create_shap_log(
            self,
            ini_file_path: str,
            rf_clf: RandomForestClassifier,
            x_df: pd.DataFrame,
            y_df: pd.Series,
            x_names: List[str],
            clf_name: str,
            cnt_present: int,
            cnt_absent: int,
            save_path: str,
            save_it: int = 100,
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Compute SHAP values for a random forest classifier.

        This method computes SHAP (SHapley Additive exPlanations) values for a given random forest classifier.

        .. seealso::
           `Documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario1.md#train-predictive-classifiers-settings>`_

           .. image:: _static/img/shap.png
              :width: 400
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
        if save_file_no == None:
            self.out_df_shap_path = os.path.join(
                save_path, f"SHAP_values_{clf_name}.csv"
            )
            self.out_df_raw_path = os.path.join(
                save_path, f"RAW_SHAP_feature_values_{clf_name}.csv"
            )
        else:
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
                msg=f"Train data contains {str(len(target_df))} behavior-present annotations. This is less the number of frames you specified to calculate shap values for {str(cnt_present)}. SimBA will calculate shap scores for the {str(len(target_df))} behavior-present frames available",
                source=self.__class__.__name__,
            )
            cnt_present = len(target_df)
        if len(nontarget_df) < cnt_absent:
            NotEnoughDataWarning(
                msg=f"Train data contains {str(len(nontarget_df))} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for {str(cnt_absent)}. SimBA will calculate shap scores for the {str(len(target_df))} behavior-absent frames available",
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
            if (cnt % save_it == 0) or (cnt == len(shap_df) - 1) and (cnt != 0):
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
        _ = ShapAggregateStatisticsVisualizer(
            config_path=ini_file_path,
            classifier_name=clf_name,
            shap_df=out_df_shap,
            shap_baseline_value=int(expected_value * 100),
            save_path=save_path,
        )

    def print_machine_model_information(self, model_dict: dict) -> None:
        """
        Helper to print model information in tabular form.

        :parameter dict model_dict: dictionary holding model meta data in SimBA meta-config format.

        """

        table_view = [
            ["Model name", model_dict[MetaKeys.CLF_NAME.value]],
            ["Ensemble method", "RF"],
            ["Estimators (trees)", model_dict[MetaKeys.RF_ESTIMATORS.value]],
            ["Max features", model_dict[MetaKeys.RF_MAX_FEATURES.value]],
            ["Under sampling setting", model_dict[ConfigKey.UNDERSAMPLE_SETTING.value]],
            ["Under sampling ratio", model_dict[ConfigKey.UNDERSAMPLE_RATIO.value]],
            ["Over sampling setting", model_dict[ConfigKey.OVERSAMPLE_SETTING.value]],
            ["Over sampling ratio", model_dict[ConfigKey.OVERSAMPLE_RATIO.value]],
            ["criterion", model_dict[MetaKeys.CRITERION.value]],
            ["Min sample leaf", model_dict[MetaKeys.MIN_LEAF.value]],
        ]
        table = tabulate(table_view, ["Setting", "value"], tablefmt="grid")
        print(f"{table} {Defaults.STR_SPLIT_DELIMITER.value}TABLE")

    def create_meta_data_csv_training_one_model(
            self, meta_data_lst: list, clf_name: str, save_dir: Union[str, os.PathLike]
    ) -> None:
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

    def create_meta_data_csv_training_multiple_models(
            self, meta_data, clf_name, save_dir, save_file_no: Optional[int] = None
    ) -> None:
        print("Saving model meta data file...")
        save_path = os.path.join(save_dir, f"{clf_name}_{str(save_file_no)}_meta.csv")
        out_df = pd.DataFrame.from_dict(meta_data, orient="index").T
        out_df.to_csv(save_path)

    def save_rf_model(
            self,
            rf_clf: RandomForestClassifier,
            clf_name: str,
            save_dir: Union[str, os.PathLike],
            save_file_no: Optional[int] = None,
    ) -> None:
        """
        Helper to save pickled classifier object to disk.

        :parameter RandomForestClassifier rf_clf: sklearn random forest classifier
        :parameter str clf_name: Classifier name
        :parameter str save_dir: Directory where to save output in csv file format.
        :parameter Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.
        """
        if save_file_no != None:
            save_path = os.path.join(
                save_dir, clf_name + "_" + str(save_file_no) + ".sav"
            )
        else:
            save_path = os.path.join(save_dir, clf_name + ".sav")
        pickle.dump(rf_clf, open(save_path, "wb"))

    def get_model_info(
            self, config: configparser.ConfigParser, model_cnt: int
    ) -> Dict[int, Any]:
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
        labels = np.unique(y_df)
        if len(labels)== 1:
            if labels[0] == 0:
                raise FaultyTrainingSetError(
                    msg=f"All training annotations for classifier {str(y_df.name)} is labelled as ABSENT. A classifier has be be trained with both behavior PRESENT and ABSENT ANNOTATIONS.",
                    source=self.__class__.__name__,
                )
            if labels[0] == 1:
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
            print(f"Partial dependencies for {feature_name} complete...")

    def clf_predict_proba(
            self,
            clf: RandomForestClassifier,
            x_df: pd.DataFrame,
            model_name: Optional[str] = None,
            data_path: Optional[Union[str, os.PathLike]] = None,
    ) -> np.ndarray:
        """

        :param RandomForestClassifier clf: Random forest classifier object
        :param pd.DataFrame x_df: Features df
        :param Optional[str] model_name: Name of model
        :param Optional[str] data_path: Path to model on disk
        :return np.ndarray: 2D array with frame represented by rows and present/absent probabilities as columns
        :raises FeatureNumberMismatchError: If shape of x_df and clf.n_features_ show mismatch
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
        # if p_vals.shape[1] != 2:
        #     raise ClassifierInferenceError(
        #         msg=f"The classifier {model_name} (data path {data_path}) has not been created properly. See The SimBA GitHub FAQ page or Gitter for more information and suggested fixes. The classifier is not a binary classifier and does not predict two targets (absence and presence of behavior)",
        #         source=self.__class__.__name__,
        #     )
        return p_vals[:, 1]

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
        nan_target = y_df[y_df.isna().to_numpy()]
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

    def _read_data_file_helper(
            self, file_path: str, file_type: str, clf_names: Optional[List[str]] = None
    ):
        """
        Private function called by :meth:`simba.train_model_functions.read_all_files_in_folder_mp`
        """

        timer = SimbaTimer(start=True)
        _, vid_name, _ = get_fn_ext(file_path)
        df = read_df(file_path, file_type).dropna(axis=0, how="all").fillna(0)
        df.index = [vid_name] * len(df)
        if clf_names != None:
            for clf_name in clf_names:
                if not clf_name in df.columns:
                    raise ColumnNotFoundError(
                        column_name=clf_name,
                        file_name=file_path,
                        source=self.__class__.__name__,
                    )
                elif len(set(df[clf_name].unique()) - {0, 1}) > 0:
                    raise InvalidInputError(
                        msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {file_path} the {clf_name} field contains additional value(s): {list(set(df[clf_name].unique()) - {0, 1})}.",
                        source=self.__class__.__name__,
                    )
        timer.stop_timer()
        print(
            f"Reading complete {vid_name} (elapsed time: {timer.elapsed_time_str}s)..."
        )
        return df

    def read_all_files_in_folder_mp(
            self,
            file_paths: List[str],
            file_type: Literal["csv", "parquet", "pickle"],
            classifier_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
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

        """
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        cpu_cnt, _ = find_core_cnt()
        df_lst = []
        try:
            with ProcessPoolExecutor(int(np.ceil(cpu_cnt / 2))) as pool:
                for res in pool.map(
                        self._read_data_file_helper,
                        file_paths,
                        repeat(file_type),
                        repeat(classifier_names),
                ):
                    df_lst.append(res)
            df_concat = pd.concat(df_lst, axis=0).round(4)
            if "scorer" in df_concat.columns:
                df_concat = df_concat.drop(["scorer"], axis=1)
            if len(df_concat) == 0:
                raise NoDataError(
                    msg="SimBA found 0 observations (frames) in the project_folder/csv/targets_inserted directory",
                    source=self.read_all_files_in_folder_mp.__name__,
                )
            df_concat = df_concat.loc[
                        :, ~df_concat.columns.str.contains("^Unnamed")
                        ].astype(np.float32)
            memory_size = get_memory_usage_of_df(df=df_concat)
            print(
                f'Dataset size: {memory_size["megabytes"]}MB / {memory_size["gigabytes"]}GB'
            )
            return df_concat

        except BrokenProcessPool or AttributeError:
            MultiProcessingFailedWarning(
                msg="Multi-processing file read failed, reverting to single core (increased run-time).",
                source=self.__class__.__name__,
            )
            return self.read_all_files_in_folder(
                file_paths=file_paths,
                file_type=file_type,
                classifier_names=classifier_names,
            )

    @staticmethod
    def _read_data_file_helper_futures(
            annotation_file_path: str, features_dir: str, file_type: str, clf_names: Optional[List[str]] = None
    ):
        """
        Private function called by :meth:`simba.train_model_functions.read_all_files_in_folder_mp_futures`
        """

        timer = SimbaTimer(start=True)
        _, vid_name, _ = get_fn_ext(annotation_file_path)
        annotation_df = read_df(annotation_file_path, file_type).dropna(axis=0, how="all").fillna(0)
        if features_dir != "":
            features_file_path = os.path.join(features_dir, vid_name + ".csv")
            features_df = read_df(features_file_path, file_type).dropna(axis=0, how="all").fillna(0)
            features_df = pd.concat([features_df, annotation_df[clf_names]], axis=1)
        else:
            features_df = annotation_df
        features_df.index = [vid_name] * len(features_df)
        if clf_names != None:
            for clf_name in clf_names:
                if not clf_name in annotation_df.columns:
                    raise ColumnNotFoundError(column_name=clf_name, file_name=annotation_file_path)
                elif len(set(annotation_df[clf_name].unique()) - {0, 1}) > 0:
                    raise InvalidInputError(
                        msg=f"The annotation column for a classifier should contain only 0 or 1 values. However, in file {annotation_file_path} the {clf_name} field contains additional value(s): {list(set(annotation_df[clf_name].unique()) - {0, 1})}."
                    )
        timer.stop_timer()
        return features_df, vid_name, timer.elapsed_time_str

    def read_and_concatenate_all_files_in_folder_mp_futures(
            self,
            annotations_file_paths: List[str],
            features_dir: str,
            file_type: Literal["csv", "parquet", "pickle"],
            classifier_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Multiprocessing helper function to read in all data files in a folder to a single
        pd.DataFrame for downstream ML through concurrent.Futures. Asserts that all classifiers
        have annotation fields present in each dataframe.

        .. note::
           A ``concurrent.Futures`` alternative to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder_mp` which
           has uses ``multiprocessing.ProcessPoolExecutor`` and reported unstable on Linux machines.

           If multiprocess failure, reverts to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder`

        :parameter List[str] file_paths: List of file-paths
        :parameter List[str] file_paths: The filetype of ``file_paths`` OPTIONS: csv or parquet.
        :parameter Optional[List[str]] classifier_names: List of classifier names representing fields of human annotations. If not None, then assert that classifier names
            are present in each data file.
        :return pd.DataFrame: Concatenated dataframe of all data in ``file_paths``.

        """
        try:
            if platform.system() == "Darwin":
                multiprocessing.set_start_method("spawn")
            cpu_cnt, _ = find_core_cnt()
            df_lst = []
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=cpu_cnt
            ) as executor:
                results = [
                    executor.submit(
                        self._read_data_file_helper_futures,
                        annotation_file_path,
                        features_dir,
                        file_type,
                        classifier_names,
                    )
                    for annotation_file_path in annotations_file_paths
                ]
                for result in concurrent.futures.as_completed(results):
                    df_lst.append(result.result()[0])
                    print(
                        f"Reading complete {result.result()[1]} (elapsed time: {result.result()[2]}s)..."
                    )
            df_concat = pd.concat(df_lst, axis=0).round(4)
            if "scorer" in df_concat.columns:
                df_concat = df_concat.drop(["scorer"], axis=1)
            return df_concat

        except:
            MultiProcessingFailedWarning(
                msg="Multi-processing file read failed, reverting to single core (increased run-time on large datasets)."
            )
            return self.read_all_files_in_folder(
                file_paths=annotations_file_paths,
                file_type=file_type,
                classifier_names=classifier_names,
            )

    def read_all_files_in_folder_mp_futures(
            self,
            annotations_file_paths: List[str],
            file_type: Literal["csv", "parquet", "pickle"],
            classifier_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Multiprocessing helper function to read in all data files in a folder to a single
        pd.DataFrame for downstream ML through concurrent.Futures. Asserts that all classifiers
        have annotation fields present in each dataframe.

        .. note::
           A ``concurrent.Futures`` alternative to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder_mp` which
           has uses ``multiprocessing.ProcessPoolExecutor`` and reported unstable on Linux machines.

           If multiprocess failure, reverts to :meth:`simba.mixins.train_model_mixin.read_all_files_in_folder`

        :parameter List[str] file_paths: List of file-paths
        :parameter List[str] file_paths: The filetype of ``file_paths`` OPTIONS: csv or parquet.
        :parameter Optional[List[str]] classifier_names: List of classifier names representing fields of human annotations. If not None, then assert that classifier names
            are present in each data file.
        :return pd.DataFrame: Concatenated dataframe of all data in ``file_paths``.

        """
        try:
            if platform.system() == "Darwin":
                multiprocessing.set_start_method("spawn")
            cpu_cnt, _ = find_core_cnt()
            df_lst = []
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=cpu_cnt
            ) as executor:
                results = [
                    executor.submit(
                        self._read_data_file_helper_futures,
                        annotation_file_path,
                        "",
                        file_type,
                        classifier_names,
                    )
                    for annotation_file_path in annotations_file_paths
                ]
                for result in concurrent.futures.as_completed(results):
                    df_lst.append(result.result()[0])
                    print(
                        f"Reading complete {result.result()[1]} (elapsed time: {result.result()[2]}s)..."
                    )
            df_concat = pd.concat(df_lst, axis=0).round(4)
            if "scorer" in df_concat.columns:
                df_concat = df_concat.drop(["scorer"], axis=1)
            return df_concat

        except:
            MultiProcessingFailedWarning(
                msg="Multi-processing file read failed, reverting to single core (increased run-time on large datasets)."
            )
            return self.read_all_files_in_folder(
                file_paths=annotations_file_paths,
                file_type=file_type,
                classifier_names=classifier_names,
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
            save_path: str,
            batch_size: int = 10,
            save_file_no: Optional[int] = None,
    ) -> None:
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
        :param str save_dir: Directory where to save output in csv file format.
        :param Optional[int] save_file_no: If integer, represents the count of the classifier within a grid search. If none, the classifier is not
            part of a grid search.

        """
        shap_timer = SimbaTimer(start=True)
        data_df = pd.concat([x_df, y_df], axis=1)
        if save_file_no == None:
            self.out_df_shap_path = os.path.join(
                save_path, f"SHAP_values_{clf_name}.csv"
            )
            self.out_df_raw_path = os.path.join(
                save_path, f"RAW_SHAP_feature_values_{clf_name}.csv"
            )
        else:
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
                msg=f"Train data contains {str(len(target_df))} behavior-present annotations. This is less the number of frames you specified to calculate shap values for {str(cnt_present)}. SimBA will calculate shap scores for the {str(len(target_df))} behavior-present frames available",
                source=self.__class__.__name__,
            )
            cnt_present = len(target_df)
        if len(nontarget_df) < cnt_absent:
            NotEnoughDataWarning(
                msg=f"Train data contains {str(len(nontarget_df))} behavior-absent annotations. This is less the number of frames you specified to calculate shap values for {str(cnt_absent)}. SimBA will calculate shap scores for the {str(len(target_df))} behavior-absent frames available",
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
        print(
            f"Computing {len(shap_data_df)} SHAP values (MULTI-CORE BATCH SIZE: {batch_size}, FOLLOW PROGRESS IN OS TERMINAL)..."
        )
        shap_data, _ = self.split_and_group_df(
            df=shap_data_df, splits=int(len(shap_data_df) / batch_size)
        )
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
                        f"Concatenating multi-processed SHAP data (batch {cnt + 1}/{len(shap_data)})"
                    )
                    proba = rf_clf.predict_proba(result[1])[:, 1].reshape(-1, 1)
                    shap_sum = np.sum(result[0], axis=1).reshape(-1, 1)
                    shap_results.append(
                        np.hstack(
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
                    )
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
            shap_save_df.to_csv(self.out_df_shap_path)
            raw_save_df.to_csv(self.out_df_raw_path)
            shap_timer.stop_timer()
            stdout_success(
                msg="SHAP calculations complete",
                elapsed_time=shap_timer.elapsed_time_str,
                source=self.__class__.__name__,
            )
            _ = ShapAggregateStatisticsVisualizer(
                config_path=ini_file_path,
                classifier_name=clf_name,
                shap_df=shap_save_df,
                shap_baseline_value=int(expected_value * 100),
                save_path=save_path,
            )
        except:
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

# test = TrainModelMixin()
# test.read_all_files_in_folder(file_paths=['/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437C_c3_2022-11-01_13-16-23_color.csv', '/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437D_c4_2022-11-01_13-16-39_color.csv'],
#                               file_type='csv', classifier_names=['attack', 'non-agresive parallel swimming'])


# test = TrainModelMixin()
# test.read_all_files_in_folder_mp(file_paths=['/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437C_c3_2022-11-01_13-16-23_color.csv', '/Users/simon/Desktop/envs/troubleshooting/jake/project_folder/csv/targets_inserted/22-437D_c4_2022-11-01_13-16-39_color.csv'],
#                               file_type='csv', classifier_names=['attack', 'non-agresive parallel swimming'])
#
