import itertools
import os
from copy import deepcopy
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success

FEATURE_NAME = "FEATURE NAME"
FEATURE_IMPORTANCE = "IMPORTANCE"
F_STATISTIC = "F-STATISTIC"
MEASURE = "MEASURE"
P_VALUE = "P-VALUE"
CLUSTER = "CLUSTER"
PAIRED = "cluster_paired"
CORRELATION_METHODS = "correlation_methods"
GINI_IMPORTANCE = "gini_importance"
TUKEY = "tukey_posthoc"
METHOD = "method"
TARGET = "TARGET"
PEARSON = "pearson"
KENDALL = "kendall"
SHAP = "shap"
PLOTS = "plots"
CREATE = "create"
RUN = "run"
SAMPLE = "sample"
SPEARMAN = "spearman"
MEAN = "MEAN"
STDEV = "STANDARD DEVIATION"
PERMUTATION_IMPORTANCE = "permutation_importance"
DESCRIPTIVE_STATISTICS = "descriptive_statistics"
ANOVA_HEADERS = ["FEATURE NAME", "F-STATISTIC", "P-VALUE"]


class ClusterXAICalculator(UnsupervisedMixin, ConfigReader):
    def __init__(self, data_path: str, config_path: str, settings: dict):
        """
        Class for building RF models on top of cluster assignments, and calculating explainability metrics on RF models

        :param str config_path: path to SimBA configparser.ConfigParser project_config.ini
        :param str data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
        :param dict settings: dict holding which tests to use.

        :Example:
        >>> settings = {'gini_importance': True, 'permutation_importance': True, 'shap': {'method': 'cluster_paired', 'create': True, 'sample': 100}}
        >>> calculator = ClusterXAICalculator(config_path='unsupervised/project_folder/project_config.ini', data_path='unsupervised/cluster_models/quizzical_rhodes.pickle', settings=settings)
        >>> calculator.run()
        """

        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.settings, self.data_path = settings, data_path
        check_file_exist_and_readable(file_path=data_path)
        self.data = self.read_pickle(data_path=self.data_path)
        check_if_keys_exist_in_dict(
            data=self.data,
            key=[Unsupervised.METHODS.value, Clustering.CLUSTER_MODEL.value],
            name=self.data_path,
        )
        check_if_keys_exist_in_dict(
            data=self.settings,
            key=[GINI_IMPORTANCE, PERMUTATION_IMPORTANCE, SHAP],
            name=self.data_path,
        )
        self.save_path = os.path.join(
            self.logs_path,
            f"cluster_xai_statistics_{self.data[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}_{self.datetime}.{Formats.XLXS.value}",
        )

    def run(self):
        self.x_df = self.data[Unsupervised.METHODS.value][
            Unsupervised.SCALED_DATA.value
        ]
        self.cluster_data = self.data[Clustering.CLUSTER_MODEL.value][
            Unsupervised.MODEL.value
        ].labels_
        self.x_y_df = pd.concat(
            [
                self.x_df,
                pd.DataFrame(
                    self.cluster_data, columns=[CLUSTER], index=self.x_df.index
                ),
            ],
            axis=1,
        )
        self.cluster_cnt = self.get_cluster_cnt(
            data=self.cluster_data,
            clusterer_name=self.data[Clustering.CLUSTER_MODEL.value][
                Unsupervised.HASHED_NAME.value
            ],
            min_clusters=2,
        )
        with pd.ExcelWriter(self.save_path, mode="w") as writer:
            pd.DataFrame().to_excel(writer, sheet_name=" ", index=True)
        self.__train_rf_models()
        if self.settings[GINI_IMPORTANCE]:
            self.__gini_importance()
        if self.settings[PERMUTATION_IMPORTANCE]:
            self.__permutation_importance()
        if self.settings[SHAP][RUN]:
            self.__shap_values()
        self.timer.stop_timer()
        stdout_success(
            msg=f"Cluster XAI complete. Data saved at {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def __save_results(self, df: pd.DataFrame, name: str):
        with pd.ExcelWriter(self.save_path, mode="a") as writer:
            df.to_excel(writer, sheet_name=name, index=True)

    def __train_rf_models(self, n_estimators: Optional[int] = 100):
        print("Training ML model...")
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            n_jobs=-1,
            criterion="gini",
            min_samples_leaf=1,
            bootstrap=True,
            verbose=1,
        )
        self.rf_data = {}
        for clf_cnt, cluster_id in enumerate(self.x_y_df[CLUSTER].unique()):
            print(f"Training model {clf_cnt+1}/{self.cluster_cnt} ...")
            self.rf_data[cluster_id] = {}
            target_df = self.x_y_df[self.x_y_df[CLUSTER] == cluster_id].drop(
                [CLUSTER], axis=1
            )
            non_target_df = self.x_y_df[self.x_y_df[CLUSTER] != cluster_id].drop(
                [CLUSTER], axis=1
            )
            target_df[TARGET] = 1
            non_target_df[TARGET] = 0
            self.rf_data[cluster_id]["X"] = (
                pd.concat([target_df, non_target_df], axis=0)
                .reset_index(drop=True)
                .sample(frac=1)
            )
            self.rf_data[cluster_id]["Y"] = self.rf_data[cluster_id]["X"].pop(TARGET)
            rf_clf.fit(self.rf_data[cluster_id]["X"], self.rf_data[cluster_id]["Y"])
            self.rf_data[cluster_id][Unsupervised.MODEL.value] = deepcopy(rf_clf)

    def __gini_importance(self):
        print("Calculating cluster gini importances'...")
        timer = SimbaTimer(start=True)
        for cluster_id, cluster_data in self.rf_data.items():
            importances = list(
                cluster_data[Unsupervised.MODEL.value].feature_importances_
            )
            gini_data = [
                (feature, round(importance, 6))
                for feature, importance in zip(
                    self.data[Unsupervised.METHODS.value][
                        Unsupervised.FEATURE_NAMES.value
                    ],
                    importances,
                )
            ]
            df = (
                pd.DataFrame(gini_data, columns=[FEATURE_NAME, FEATURE_IMPORTANCE])
                .sort_values(by=[FEATURE_IMPORTANCE], ascending=False)
                .reset_index(drop=True)
            )
            self.__save_results(df=df, name=f"GINI CLUSTER {str(cluster_id)}")
        timer.stop_timer()
        stdout_success(
            msg=f"Cluster features gini importances' complete",
            elapsed_time=timer.elapsed_time_str,
        )

    def __permutation_importance(self):
        print("Calculating permutation importance...")
        timer = SimbaTimer(start=True)
        for cluster_id, cluster_data in self.rf_data.items():
            p_importances = permutation_importance(
                cluster_data[Unsupervised.MODEL.value],
                cluster_data["X"],
                cluster_data["Y"],
                n_repeats=5,
                random_state=0,
            )
            df = pd.DataFrame(
                np.column_stack(
                    [
                        self.data[Unsupervised.METHODS.value][
                            Unsupervised.FEATURE_NAMES.value
                        ],
                        p_importances.importances_mean,
                        p_importances.importances_std,
                    ]
                ),
                columns=[FEATURE_NAME, MEAN, STDEV],
            )
            df = df.sort_values(by=[MEAN], ascending=False).reset_index(drop=True)
            self.__save_results(df=df, name=f"PERMUTATION CLUSTER {str(cluster_id)}")
        timer.stop_timer()
        stdout_success(
            msg=f"Cluster features permutation importances' complete",
            elapsed_time=timer.elapsed_time_str,
        )

    def __shap_values(self):
        if self.settings[SHAP][METHOD] == PAIRED:
            print("Calculating paired-clusters shap values ...")
            timer = SimbaTimer(start=True)
            cluster_combinations = list(
                itertools.combinations(list(self.rf_data.keys()), 2)
            )
            for cluster_one_id, cluster_two_id in cluster_combinations:
                explainer = shap.TreeExplainer(
                    self.rf_data[cluster_one_id]["MODEL"],
                    data=None,
                    model_output="raw",
                    feature_perturbation="tree_path_dependent",
                )
                if self.settings[SHAP][SAMPLE] > (
                    len(self.rf_data[cluster_one_id]["X"])
                    or len(self.rf_data[cluster_two_id]["X"])
                ):
                    self.settings[SHAP][SAMPLE] = min(
                        len(self.rf_data[cluster_one_id]["X"]),
                        len(self.rf_data[cluster_two_id]["X"]),
                    )
                cluster_one_sample = self.rf_data[cluster_one_id]["X"].sample(
                    self.settings[SHAP][SAMPLE], replace=False
                )
                cluster_two_sample = self.rf_data[cluster_two_id]["X"].sample(
                    self.settings[SHAP][SAMPLE], replace=False
                )
                cluster_one_shap = pd.DataFrame(
                    explainer.shap_values(cluster_one_sample, check_additivity=False)[
                        1
                    ],
                    columns=self.rf_data[cluster_one_id]["X"].columns,
                )
                cluster_two_shap = pd.DataFrame(
                    explainer.shap_values(cluster_two_sample, check_additivity=False)[
                        1
                    ],
                    columns=self.rf_data[cluster_two_id]["X"].columns,
                )
                mean_df_cluster_one, stdev_df_cluster_one = pd.DataFrame(
                    cluster_one_shap.mean(), columns=["MEAN"]
                ), pd.DataFrame(cluster_one_shap.std(), columns=["STDEV"])
                mean_df_cluster_two, stdev_df_cluster_two = pd.DataFrame(
                    cluster_two_shap.mean(), columns=["MEAN"]
                ), pd.DataFrame(cluster_two_shap.std(), columns=["STDEV"])
                results_cluster_one = mean_df_cluster_one.join(
                    stdev_df_cluster_one
                ).sort_values(by="MEAN", ascending=False)
                results_cluster_two = mean_df_cluster_two.join(
                    stdev_df_cluster_two
                ).sort_values(by="MEAN", ascending=False)
                self.__save_results(
                    df=results_cluster_one,
                    name=f"SHAP CLUSTER {str(cluster_one_id)} vs. {str(cluster_two_id)}",
                )
                self.__save_results(
                    df=results_cluster_two,
                    name=f"SHAP CLUSTER {str(cluster_two_id)} vs. {str(cluster_one_id)}",
                )
            timer.stop_timer()
            stdout_success(
                msg=f"Paired clusters SHAP values complete",
                elapsed_time=timer.elapsed_time_str,
            )
