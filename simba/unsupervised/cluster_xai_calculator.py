import itertools
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from simba.mixins.config_reader import ConfigReader
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.plotting.shap_agg_stats_visualizer import \
    ShapAggregateStatisticsVisualizer
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict, check_instance)
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_unique_values_in_iterable, read_pickle

FEATURE_NAME = "FEATURE NAME"
FEATURE_IMPORTANCE = "IMPORTANCE"
F_STATISTIC = "F-STATISTIC"
MEASURE = "MEASURE"
P_VALUE = "P-VALUE"
CLUSTER = "CLUSTER"
PAIRED = "Paired clusters"
ONE_AGAINST_ALL = "One-against-all"
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
    """
    Class for building RF models on top of cluster assignments, and calculating latent space explainability metrics based on RF models.

    :param Union[str, os.PathLike] config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param Union[str, os.PathLike] data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param Dict[str, Any] settings: Dict holding which explainability tests to use.

    :Example:
    >>> settings = {'gini_importance': True, 'permutation_importance': True, 'shap': {'method': 'cluster_paired', 'create': True, 'sample': 100}}
    >>> calculator = ClusterXAICalculator(config_path='unsupervised/project_folder/project_config.ini', data_path='unsupervised/cluster_models/quizzical_rhodes.pickle', settings=settings)
    >>> calculator.run()
    """

    def __init__(
        self,
        data_path: Union[str, os.PathLike],
        config_path: Union[str, os.PathLike],
        settings: Dict[str, Any],
    ):
        check_file_exist_and_readable(file_path=data_path)
        check_file_exist_and_readable(file_path=config_path)
        check_instance(
            source=f"{self.__class__.__name__} settings",
            instance=settings,
            accepted_types=(dict,),
        )
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.settings, self.data_path = settings, data_path
        self.data = read_pickle(data_path=self.data_path)
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
        self.mdl_file_name = self.data[Clustering.CLUSTER_MODEL.value][
            Unsupervised.HASHED_NAME.value
        ]
        self.cluster_cnt = get_unique_values_in_iterable(
            data=self.cluster_data,
            name=self.mdl_file_name,
            min=2,
        )

        with pd.ExcelWriter(self.save_path, mode="w") as writer:
            pd.DataFrame().to_excel(writer, sheet_name=" ", index=True)

        if (
            self.settings[GINI_IMPORTANCE]
            or self.settings[PERMUTATION_IMPORTANCE]
            or (self.settings[SHAP][METHOD] == PAIRED)
        ):
            self.__train_paired_rf_models()
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

    def __train_paired_rf_models(self, n_estimators: Optional[int] = 100):
        print("Training RF ML model...")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            n_jobs=-1,
            criterion="gini",
            min_samples_leaf=1,
            bootstrap=True,
            verbose=1,
        )
        self.rf_data = {}
        cluster_permutations = list(
            itertools.permutations(list(self.x_y_df[CLUSTER].unique()), 2)
        )
        for clf_cnt, (x1, x2) in enumerate(cluster_permutations):
            print(f"Training model {clf_cnt + 1}/{len(cluster_permutations)} ...")
            self.rf_data[x1] = {}
            target_df = self.x_y_df[self.x_y_df[CLUSTER] == x1].drop([CLUSTER], axis=1)
            non_target_df = self.x_y_df[self.x_y_df[CLUSTER] == x2].drop(
                [CLUSTER], axis=1
            )
            target_df[TARGET] = 1
            non_target_df[TARGET] = 0
            self.rf_data[x1]["X"] = pd.concat(
                [target_df, non_target_df], axis=0
            ).reset_index(drop=True)
            self.rf_data[x1]["Y"] = self.rf_data[x1]["X"].pop(TARGET)
            clf = TrainModelMixin().clf_fit(
                clf=clf, x_df=self.rf_data[x1]["X"], y_df=self.rf_data[x1]["Y"]
            )
            self.rf_data[x1][Unsupervised.MODEL.value] = deepcopy(clf)

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

    def __train_all_against_one_rf_models(self, n_estimators: Optional[int] = 100):
        """Private helper to create random forest classifiers training observations in one cluster against all observations in other clusters"""
        all_against_one_rf_mdls = {}
        for cluster_id in sorted(self.x_y_df[CLUSTER].unique()):
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                n_jobs=-1,
                criterion="gini",
                min_samples_leaf=1,
                bootstrap=True,
                verbose=1,
            )
            cluster_df = self.x_y_df[self.x_y_df[CLUSTER] == cluster_id].drop(
                [CLUSTER], axis=1
            )
            noncluster_df = self.x_y_df[self.x_y_df[CLUSTER] != cluster_id].drop(
                [CLUSTER], axis=1
            )
            cluster_df[TARGET] = 1
            noncluster_df[TARGET] = 0
            x = pd.concat([cluster_df, noncluster_df], axis=0)
            y = x.pop(TARGET)
            all_against_one_rf_mdls[cluster_id] = TrainModelMixin().clf_fit(
                clf=clf, x_df=x, y_df=y
            )
        return all_against_one_rf_mdls

    def __shap_values(self):
        if self.settings[SHAP][METHOD] == PAIRED:
            print("Computing paired-clusters shap values ...")
            timer = SimbaTimer(start=True)
            cluster_combinations = list(
                itertools.combinations(list(self.rf_data.keys()), 2)
            )
            for cluster_one_id, cluster_two_id in cluster_combinations:
                mdl_name = f"SHAP CLUSTER {str(cluster_one_id)} vs. {str(cluster_two_id)} {self.mdl_file_name}"
                print(f"Computing {mdl_name} values ...")
                cluster_one_sample = self.x_y_df[
                    self.x_y_df[CLUSTER] == cluster_one_id
                ].drop(CLUSTER, axis=1)
                cluster_two_sample = self.x_y_df[
                    self.x_y_df[CLUSTER] == cluster_two_id
                ].drop(CLUSTER, axis=1)
                sample_n = min(
                    self.settings[SHAP][SAMPLE],
                    len(cluster_one_sample),
                    len(cluster_two_sample),
                )
                cluster_one_sample = cluster_one_sample.sample(sample_n, replace=False)
                cluster_two_sample = cluster_two_sample.sample(sample_n, replace=False)
                cluster_one_sample[mdl_name] = 1
                cluster_two_sample[mdl_name] = 0
                x_df = pd.concat(
                    [cluster_one_sample, cluster_two_sample], axis=0
                ).reset_index(drop=True)
                y_df = x_df.pop(mdl_name)
                shap_df, _, expected_value = TrainModelMixin().create_shap_log_mp(
                    ini_file_path=self.config_path,
                    rf_clf=self.rf_data[cluster_one_id]["MODEL"],
                    x_df=x_df,
                    y_df=y_df,
                    x_names=list(x_df.columns),
                    clf_name=mdl_name,
                    cnt_present=sample_n,
                    cnt_absent=sample_n,
                )
                _ = ShapAggregateStatisticsVisualizer(
                    config_path=self.config_path,
                    classifier_name=mdl_name,
                    shap_df=shap_df,
                    shap_baseline_value=expected_value,
                    save_path=None,
                )
                cluster_one_shap = shap_df[shap_df[mdl_name] == 1]
                cluster_two_shap = shap_df[shap_df[mdl_name] == 0]
                mean_df_cluster_one, stdev_df_cluster_one = pd.DataFrame(
                    cluster_one_shap.mean(), columns=["MEAN"]
                ), pd.DataFrame(cluster_one_shap.std(), columns=["STDEV"])
                mean_df_cluster_two, stdev_df_cluster_two = pd.DataFrame(
                    cluster_two_shap.mean(), columns=["MEAN"]
                ), pd.DataFrame(cluster_two_shap.std(), columns=["STDEV"])
                mean_df_cluster_two["MEAN"] = mean_df_cluster_two["MEAN"] * -1
                results_cluster_two = (
                    mean_df_cluster_two.join(stdev_df_cluster_two)
                    .sort_values(by="MEAN", ascending=False)
                    .drop(
                        [mdl_name, "Expected_value", "Sum", "Prediction_probability"],
                        axis=0,
                    )
                )
                results_cluster_one = (
                    mean_df_cluster_one.join(stdev_df_cluster_one)
                    .sort_values(by="MEAN", ascending=False)
                    .drop(
                        [mdl_name, "Expected_value", "Sum", "Prediction_probability"],
                        axis=0,
                    )
                )
                self.__save_results(df=results_cluster_one, name=mdl_name)
                self.__save_results(df=results_cluster_two, name=mdl_name)
            timer.stop_timer()
            stdout_success(
                msg=f"Paired clusters SHAP values complete",
                elapsed_time=timer.elapsed_time_str,
            )

        elif self.settings[SHAP][METHOD] == ONE_AGAINST_ALL:
            timer = SimbaTimer(start=True)
            print("Calculating one-against-all shap values ...")
            mdls = self.__train_all_against_one_rf_models()
            for cluster_id, cluster_mdl in mdls.items():
                shap_mdl_name = (
                    f"SHAP CLUSTER {cluster_id} vs. ALL {self.mdl_file_name}"
                )
                print(f"Computing SHAP for cluster {shap_mdl_name}...")
                cluster_one_sample = self.x_y_df[
                    self.x_y_df[CLUSTER] == cluster_id
                ].drop(CLUSTER, axis=1)
                cluster_two_sample = self.x_y_df[
                    self.x_y_df[CLUSTER] != cluster_id
                ].drop(CLUSTER, axis=1)
                sample_n = min(
                    self.settings[SHAP][SAMPLE],
                    len(cluster_one_sample),
                    len(cluster_one_sample),
                )
                cluster_one_sample = cluster_one_sample.sample(sample_n, replace=False)
                cluster_two_sample = cluster_two_sample.sample(sample_n, replace=False)
                cluster_one_sample[shap_mdl_name] = 1
                cluster_two_sample[shap_mdl_name] = 0
                x_df = pd.concat(
                    [cluster_one_sample, cluster_two_sample], axis=0
                ).reset_index(drop=True)
                y_df = x_df.pop(shap_mdl_name)
                shap_df, _, expected_value = TrainModelMixin().create_shap_log_mp(
                    ini_file_path=self.config_path,
                    rf_clf=mdls[cluster_id],
                    x_df=x_df,
                    y_df=y_df,
                    x_names=list(x_df.columns),
                    clf_name=shap_mdl_name,
                    cnt_present=sample_n,
                    cnt_absent=sample_n,
                )
                _ = ShapAggregateStatisticsVisualizer(
                    config_path=self.config_path,
                    classifier_name=shap_mdl_name,
                    shap_df=shap_df,
                    shap_baseline_value=expected_value,
                    save_path=None,
                )
                mean_df, stdev_df = pd.DataFrame(
                    shap_df.mean(), columns=["MEAN"]
                ), pd.DataFrame(shap_df.std(), columns=["STDEV"])
                shap_df = (
                    mean_df.join(stdev_df)
                    .sort_values(by="MEAN", ascending=False)
                    .drop(
                        [
                            shap_mdl_name,
                            "Expected_value",
                            "Sum",
                            "Prediction_probability",
                        ],
                        axis=0,
                    )
                )
                self.__save_results(df=shap_df, name=shap_mdl_name)
                timer.stop_timer()
                stdout_success(
                    msg=f"SHAP one-vs-all values complete",
                    elapsed_time=timer.elapsed_time_str,
                )

        else:
            raise InvalidInputError(
                msg=f"Shap parameter {self.settings[SHAP][METHOD]} not recognized",
                source=self.__class__.__name__,
            )


# settings = {"gini_importance": False, "permutation_importance": False, "shap": {"method": PAIRED, "run": True, "sample": 10}}
# calculator = ClusterXAICalculator(
#     config_path="/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini",
#     data_path="/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/error_xai/determined_wing.pickle",
#     settings=settings)
# # #
# # #
# calculator.run()
