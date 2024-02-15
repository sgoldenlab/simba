__author__ = "Simon Nilsson"

import os
from typing import Union, Dict

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import check_file_exist_and_readable, check_if_keys_exist_in_dict
from simba.utils.enums import Methods
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
ANOVA = "anova"
KENDALL = "kendall"
SHAP = "shap"
PLOTS = "plots"
CREATE = "create"
SPEARMAN = "spearman"
MEAN = "MEAN"
STDEV = "STANDARD DEVIATION"
PERMUTATION_IMPORTANCE = "permutation_importance"
DESCRIPTIVE_STATISTICS = "descriptive_statistics"
ANOVA_HEADERS = ["FEATURE NAME", "F-STATISTIC", "P-VALUE"]

class ClusterFrequentistCalculator(UnsupervisedMixin, ConfigReader):
    """
    Class for computing frequentist statitics based on cluster assignment labels (for explainability purposes).

    :param str config_path: path to SimBA configparser.ConfigParser project_config.ini
    :param str data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
    :param dict settings: dict holding which statistical tests to use

    :example:
    >>> settings = {'scaled': True, 'ANOVA': True, 'tukey_posthoc': True, 'descriptive_statistics': True}
    >>> calculator = ClusterFrequentistCalculator(config_path='unsupervised/project_folder/project_config.ini', data_path='unsupervised/cluster_models/quizzical_rhodes.pickle', settings=settings)
    >>> calculator.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike],
                 settings: Dict[str, bool]):

        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.settings = settings
        check_file_exist_and_readable(file_path=data_path)
        self.data = self.read_pickle(data_path=data_path)
        self.save_path = os.path.join(self.logs_path, f"cluster_descriptive_statistics_{self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value]}_{self.datetime}.xlsx")
        check_if_keys_exist_in_dict(data=self.data, key=[Clustering.CLUSTER_MODEL.value, Unsupervised.METHODS.value], name=data_path)
        check_if_keys_exist_in_dict(data=settings, key=[ANOVA, DESCRIPTIVE_STATISTICS, TUKEY], name='settings')
    def run(self):
        self.x_data = self.data[Unsupervised.METHODS.value][Unsupervised.SCALED_DATA.value]
        self.cluster_data = self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value].labels_
        if not self.settings[Unsupervised.SCALED.value]:
            self.x_data = TrainModelMixin.scaler_inverse_transform(data=self.x_data, scaler=self.data[Unsupervised.METHODS.value][Unsupervised.SCALER.value])
        self.x_y_df = pd.concat([self.x_data, pd.DataFrame(self.cluster_data, columns=[CLUSTER], index=self.x_data.index)], axis=1)
        self.cluster_cnt = UnsupervisedMixin.get_cluster_cnt(data=self.cluster_data, clusterer_name=self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value], min_clusters=2)
        with pd.ExcelWriter(self.save_path, mode="w") as writer:
            pd.DataFrame().to_excel(writer, sheet_name=" ", index=True)
        if self.settings[ANOVA]:
            self.__one_way_anovas()
        if self.settings[DESCRIPTIVE_STATISTICS]:
            self.__descriptive_stats()
        if self.settings[TUKEY]:
            self.__tukey_posthoc()
        self.timer.stop_timer()
        stdout_success(msg=f"Cluster statistics complete. Data saved at {self.save_path}", elapsed_time=self.timer.elapsed_time_str)

    def __save_results(self, df: pd.DataFrame, name: str):
        with pd.ExcelWriter(self.save_path, mode="a") as writer:
            df.to_excel(writer, sheet_name=name, index=True)

    def __one_way_anovas(self):
        print("Calculating ANOVAs...")
        timer = SimbaTimer(start=True)
        self.anova_results = pd.DataFrame(columns=ANOVA_HEADERS)
        for feature_name in self.data[Unsupervised.METHODS.value][
            Unsupervised.FEATURE_NAMES.value
        ]:
            stats_data = (
                self.x_y_df[[feature_name, "CLUSTER"]]
                .sort_values(by=["CLUSTER"])
                .values
            )
            stats_data = np.split(
                stats_data[:, 0], np.unique(stats_data[:, 1], return_index=True)[1][1:]
            )
            f_val, p_val = f_oneway(*stats_data)
            self.anova_results.loc[len(self.anova_results)] = [
                feature_name,
                f_val,
                p_val,
            ]
        self.anova_results = self.anova_results.sort_values(by=[P_VALUE]).set_index(
            FEATURE_NAME
        )
        self.anova_results[P_VALUE] = self.anova_results[P_VALUE].round(5)
        self.__save_results(df=self.anova_results, name=Methods.ANOVA.value)
        timer.stop_timer()
        stdout_success(
            msg=f"ANOVAs saved in {self.save_path}", elapsed_time=timer.elapsed_time_str
        )

    def __descriptive_stats(self):
        print("Calculating descriptive statistics..")
        timer = SimbaTimer(start=True)
        self.descriptive_results = []
        for feature_name in self.data[Unsupervised.METHODS.value][
            Unsupervised.FEATURE_NAMES.value
        ]:
            agg = (
                self.x_y_df.groupby([CLUSTER])[feature_name]
                .agg(["mean", "std", "sem"])
                .T
            )
            agg[FEATURE_NAME] = feature_name
            agg = (
                agg.reset_index(drop=False)
                .set_index(FEATURE_NAME)
                .rename(columns={"index": MEASURE})
            )
            self.descriptive_results.append(pd.DataFrame(agg))
        self.descriptive_results = pd.concat(self.descriptive_results, axis=0)
        self.__save_results(df=self.descriptive_results, name=DESCRIPTIVE_STATISTICS)
        timer.stop_timer()
        stdout_success(
            msg=f"Descriptive statistics saved in {self.save_path}",
            elapsed_time=timer.elapsed_time_str,
        )

    def __tukey_posthoc(self):
        print("Calculating tukey posthocs...")
        timer = SimbaTimer(start=True)
        self.post_hoc_results = []
        for feature_name in self.data[Unsupervised.METHODS.value][
            Unsupervised.FEATURE_NAMES.value
        ]:
            data = pairwise_tukeyhsd(self.x_y_df[feature_name], self.x_y_df[CLUSTER])
            df = pd.DataFrame(
                data=data._results_table.data[1:], columns=data._results_table.data[0]
            )
            df[P_VALUE] = psturng(
                np.abs(data.meandiffs / data.std_pairs),
                len(data.groupsunique),
                data.df_total,
            )
            df[FEATURE_NAME] = feature_name
            df = df.reset_index(drop=True).set_index(FEATURE_NAME)
            self.post_hoc_results.append(df)
        self.post_hoc_results = pd.concat(self.post_hoc_results, axis=0)
        self.__save_results(df=self.post_hoc_results, name=TUKEY)
        timer.stop_timer()
        stdout_success(
            msg=f"Tukey post-hocs' statistics saved in {self.save_path}",
            elapsed_time=timer.elapsed_time_str,
        )