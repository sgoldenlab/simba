__author__ = "Simon Nilsson"


import os
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import f_oneway
import shap
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.enums import Methods
from simba.utils.printing import stdout_success, SimbaTimer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.utils.checks import check_file_exist_and_readable

FEATURE_NAME = 'FEATURE NAME'
FEATURE_IMPORTANCE = 'IMPORTANCE'
F_STATISTIC = 'F-STATISTIC'
MEASURE = 'MEASURE'
P_VALUE = 'P-VALUE'
CLUSTER = 'CLUSTER'
PAIRED = 'cluster_paired'
CORRELATION_METHODS = 'correlation_methods'
GINI_IMPORTANCE = 'gini_importance'
TUKEY = 'tukey_posthoc'
METHOD = 'method'
TARGET = 'TARGET'
PEARSON = 'pearson'
KENDALL = 'kendall'
SHAP = 'shap'
PLOTS = 'plots'
CREATE = 'create'
SPEARMAN = 'spearman'
MEAN = 'MEAN'
STDEV = 'STANDARD DEVIATION'
PERMUTATION_IMPORTANCE = 'permutation_importance'
DESCRIPTIVE_STATISTICS = 'descriptive_statistics'
ANOVA_HEADERS = ['FEATURE NAME', 'F-STATISTIC', 'P-VALUE']

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
                 config_path: str,
                 data_path: str,
                 settings: dict):


        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.settings = settings
        check_file_exist_and_readable(file_path=data_path)
        self.data = self.read_pickle(data_path=data_path)
        self.save_path = os.path.join(self.logs_path, f'cluster_descriptive_statistics_{self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value]}_{self.datetime}.xlsx')
        self.check_key_exist_in_object(object=self.data, key=Clustering.CLUSTER_MODEL.value)


    def run(self):
        self.x_data = self.data[Unsupervised.METHODS.value][Unsupervised.SCALED_DATA.value]
        self.cluster_data = self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value].labels_
        if not self.settings[Unsupervised.SCALED.value]:
            self.feature_data = self.scaler_inverse_transform(scaler=self.data[Unsupervised.METHODS.value][Unsupervised.SCALER.value], data=self.x_data)
        self.x_y_df = pd.concat([self.x_data, pd.DataFrame(self.cluster_data, columns=[CLUSTER], index=self.x_data.index)], axis=1)
        self.cluster_cnt = self.get_cluster_cnt(data=self.cluster_data, clusterer_name=self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value], minimum_clusters=2)
        with pd.ExcelWriter(self.save_path, mode='w') as writer:
            pd.DataFrame().to_excel(writer, sheet_name=' ', index=True)
        if self.settings[Methods.ANOVA.value]:
            self.__one_way_anovas()
        if self.settings[DESCRIPTIVE_STATISTICS]:
            self.__descriptive_stats()
        if self.settings[TUKEY]:
            self.__tukey_posthoc()
        self.timer.stop_timer()
        stdout_success(msg=f'Cluster statistics complete. Data saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

    def __save_results(self, df: pd.DataFrame, name: str):
        with pd.ExcelWriter(self.save_path, mode='a') as writer:
            df.to_excel(writer, sheet_name=name, index=True)

    def __one_way_anovas(self):
        print('Calculating ANOVAs...')
        timer = SimbaTimer(start=True)
        self.anova_results = pd.DataFrame(columns=ANOVA_HEADERS)
        for feature_name in self.data[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value]:
            stats_data = self.x_y_df[[feature_name, 'CLUSTER']].sort_values(by=['CLUSTER']).values
            stats_data = np.split(stats_data[:, 0], np.unique(stats_data[:, 1], return_index=True)[1][1:])
            f_val, p_val = f_oneway(*stats_data)
            self.anova_results.loc[len(self.anova_results)] = [feature_name, f_val, p_val]
        self.anova_results = self.anova_results.sort_values(by=[P_VALUE]).set_index(FEATURE_NAME)
        self.anova_results[P_VALUE] = self.anova_results[P_VALUE].round(5)
        self.__save_results(df=self.anova_results, name=Methods.ANOVA.value)
        timer.stop_timer()
        stdout_success(msg=f'ANOVAs saved in {self.save_path}', elapsed_time=timer.elapsed_time_str)


    def __descriptive_stats(self):
        print('Calculating descriptive statistics..')
        timer = SimbaTimer(start=True)
        self.descriptive_results = []
        for feature_name in self.data[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value]:
            agg = self.x_y_df.groupby([CLUSTER])[feature_name].agg(['mean','std', 'sem']).T
            agg[FEATURE_NAME] = feature_name
            agg = agg.reset_index(drop=False).set_index(FEATURE_NAME).rename(columns={'index': MEASURE})
            self.descriptive_results.append(pd.DataFrame(agg))
        self.descriptive_results = pd.concat(self.descriptive_results, axis=0)
        self.__save_results(df=self.descriptive_results, name=DESCRIPTIVE_STATISTICS)
        timer.stop_timer()
        stdout_success(msg=f'Descriptive statistics saved in {self.save_path}', elapsed_time=timer.elapsed_time_str)

    def __tukey_posthoc(self):
        print('Calculating tukey posthocs...')
        timer = SimbaTimer(start=True)
        self.post_hoc_results = []
        for feature_name in self.data[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value]:
            data = pairwise_tukeyhsd(self.x_y_df[feature_name], self.x_y_df[CLUSTER])
            df = pd.DataFrame(data=data._results_table.data[1:], columns=data._results_table.data[0])
            df[P_VALUE] = psturng(np.abs(data.meandiffs / data.std_pairs), len(data.groupsunique), data.df_total)
            df[FEATURE_NAME] = feature_name
            df = df.reset_index(drop=True).set_index(FEATURE_NAME)
            self.post_hoc_results.append(df)
        self.post_hoc_results = pd.concat(self.post_hoc_results, axis=0)
        self.__save_results(df=self.post_hoc_results, name=TUKEY)
        timer.stop_timer()
        stdout_success(msg=f"Tukey post-hocs' statistics saved in {self.save_path}", elapsed_time=timer.elapsed_time_str)

class EmbeddingCorrelationCalculator(UnsupervisedMixin, ConfigReader):
    def __init__(self,
                 data_path: str,
                 config_path: str,
                 settings: dict):

        """
        Class for correlating dimensionality reduction features with original features (for explainability purposes)

        :param str config_path: path to SimBA configparser.ConfigParser project_config.ini
        :param str data_path: path to pickle holding unsupervised results in ``data_map.yaml`` format.
        :param dict settings: dict holding which statistical tests to use and how to create plots.

        :Example:
        >>> settings = {'correlation_methods': ['pearson', 'kendall', 'spearman'], 'plots': {'create': True, 'correlations': 'pearson', 'palette': 'jet'}}
        >>> calculator = EmbeddingCorrelationCalculator(config_path='unsupervised/project_folder/project_config.ini', data_path='unsupervised/cluster_models/quizzical_rhodes.pickle', settings=settings)
        >>> calculator.run()
        """

        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        check_file_exist_and_readable(file_path=data_path)
        self.settings, self.data_path = settings, data_path
        self.data = self.read_pickle(data_path=self.data_path)
        self.save_path = os.path.join(self.logs_path, f'embedding_correlations_{self.data[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}_{self.datetime}.csv')

    def run(self):
        print('Calculating embedding correlations...')
        self.x_df = self.data[Unsupervised.METHODS.value][Unsupervised.SCALED_DATA.value]
        self.y_df = pd.DataFrame(self.data[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].embedding_, columns=['X', 'Y'], index=self.x_df.index)
        results = pd.DataFrame()
        for correlation_method in self.settings[CORRELATION_METHODS]:
            results[f'{correlation_method}_Y'] = self.x_df.corrwith(self.y_df['Y'], method=correlation_method)
            results[f'{correlation_method}_X'] = self.x_df.corrwith(self.y_df['X'], method=correlation_method)
        results.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(msg=f'Embedding correlations saved in {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

        if self.settings[PLOTS][CREATE]:
            print('Creating embedding correlation plots...')
            df = pd.concat([self.x_df, self.y_df], axis=1)
            save_dir = os.path.join(self.logs_path, 'embedding_correlation_plots')
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            for feature_cnt, feature_name in enumerate(self.data[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value]):
                color_bar = plt.cm.ScalarMappable(cmap=self.settings[PLOTS]['palette'])
                color_bar.set_array([])
                plot = sns.scatterplot(data=df, x="X", y="Y", hue=feature_name, cmap=self.settings[PLOTS]['palette'])
                plot.get_legend().remove()
                plot.figure.colorbar(color_bar, label=feature)
                plt.suptitle(feature_name, x=0.5, y=0.92)
                save_path = os.path.join(save_dir, f'{feature_name}.png')
                plot.figure.savefig(save_path, bbox_inches="tight")
                plot.clear()
                plt.close()
                print(f'Saving image {str(feature_cnt+1)}/{str(len(df.columns))} ({feature_name})')
        stdout_success(msg=f'Embedding correlation calculations complete')



class ClusterXAICalculator(UnsupervisedMixin, ConfigReader):
    def __init__(self,
                 data_path: str,
                 config_path: str,
                 settings: dict):

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
        self.save_path = os.path.join(self.logs_path, f'cluster_xai_statistics_{self.data[Unsupervised.DR_MODEL.value][Unsupervised.HASHED_NAME.value]}_{self.datetime}.xlsx')

    def run(self):
        self.x_df = self.data[Unsupervised.METHODS.value][Unsupervised.SCALED_DATA.value]
        self.cluster_data = self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value].labels_
        self.x_y_df = pd.concat([self.x_df, pd.DataFrame(self.cluster_data, columns=[CLUSTER], index=self.x_df.index)], axis=1)
        self.cluster_cnt = self.get_cluster_cnt(data=self.cluster_data, clusterer_name=self.data[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value], minimum_clusters=2)
        with pd.ExcelWriter(self.save_path, mode='w') as writer:
            pd.DataFrame().to_excel(writer, sheet_name=' ', index=True)
        self.__train_rf_models()
        print(self.settings[GINI_IMPORTANCE])
        if self.settings[GINI_IMPORTANCE]:
            self.__gini_importance()
        if self.settings[PERMUTATION_IMPORTANCE]:
            self.__permutation_importance()
        if self.settings[SHAP][CREATE]:
            self.__shap_values()
        self.timer.stop_timer()
        stdout_success(msg=f'Cluster XAI complete. Data saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)


    def __save_results(self, df: pd.DataFrame, name: str):
        with pd.ExcelWriter(self.save_path, mode='a') as writer:
            df.to_excel(writer, sheet_name=name, index=True)

    def __train_rf_models(self):
        print('Training ML model...')
        rf_clf = RandomForestClassifier(n_estimators=100,
                                        max_features='sqrt',
                                        n_jobs=-1, criterion='gini',
                                        min_samples_leaf=1,
                                        bootstrap=True,
                                        verbose=1)
        self.rf_data = {}
        for clf_cnt, cluster_id in enumerate(self.x_y_df[CLUSTER].unique()):
            print(f'Training model {clf_cnt+1}/{self.cluster_cnt}')
            self.rf_data[cluster_id] = {}
            target_df = self.x_y_df[self.x_y_df[CLUSTER] == cluster_id].drop([CLUSTER], axis=1)
            non_target_df = self.x_y_df[self.x_y_df[CLUSTER] != cluster_id].drop([CLUSTER], axis=1)
            target_df[TARGET] = 1
            non_target_df[TARGET] = 0
            self.rf_data[cluster_id]['X'] = pd.concat([target_df, non_target_df], axis=0).reset_index(drop=True).sample(frac=1)
            self.rf_data[cluster_id]['Y'] = self.rf_data[cluster_id]['X'].pop(TARGET)
            rf_clf.fit(self.rf_data[cluster_id]['X'], self.rf_data[cluster_id]['Y'])
            self.rf_data[cluster_id][Unsupervised.MODEL.value] = deepcopy(rf_clf)

    def __gini_importance(self):
        print("Calculating cluster gini importances'...")
        timer = SimbaTimer(start=True)
        for cluster_id, cluster_data in self.rf_data.items():
            importances = list(cluster_data[Unsupervised.MODEL.value].feature_importances_)
            gini_data = [(feature, round(importance, 6)) for feature, importance in zip(self.data[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value], importances)]
            df = pd.DataFrame(gini_data, columns=[FEATURE_NAME, FEATURE_IMPORTANCE]).sort_values(by=[FEATURE_IMPORTANCE], ascending=False).reset_index(drop=True)
            self.__save_results(df=df, name=f'GINI CLUSTER {str(cluster_id)}')
        timer.stop_timer()
        stdout_success(msg=f"Cluster features gini importances' complete", elapsed_time=timer.elapsed_time_str)

    def __permutation_importance(self):
        print('Calculating permutation importance...')
        timer = SimbaTimer(start=True)
        for cluster_id, cluster_data in self.rf_data.items():
            p_importances = permutation_importance(cluster_data[Unsupervised.MODEL.value], cluster_data['X'], cluster_data['Y'], n_repeats=5, random_state=0)
            df = pd.DataFrame(np.column_stack([self.data[Unsupervised.METHODS.value][Unsupervised.FEATURE_NAMES.value], p_importances.importances_mean, p_importances.importances_std]), columns=[FEATURE_NAME, MEAN, STDEV])
            df = df.sort_values(by=[MEAN], ascending=False).reset_index(drop=True)
            self.__save_results(df=df, name=f'PERMUTATION CLUSTER {str(cluster_id)}')
        timer.stop_timer()
        stdout_success(msg=f"Cluster features permutation importances' complete", elapsed_time=timer.elapsed_time_str)

    def __shap_values(self):
        if self.settings[SHAP][METHOD] == PAIRED:
            print('Calculating paired-clusters shap values ...')
            timer = SimbaTimer(start=True)
            cluster_combinations = list(itertools.combinations(list(self.rf_data.keys()), 2))
            for (cluster_one_id, cluster_two_id) in cluster_combinations:
                explainer = shap.TreeExplainer(self.rf_data[cluster_one_id]['MODEL'], data=None, model_output='raw', feature_perturbation='tree_path_dependent')
                if self.settings['shap']['sample'] > (len(self.rf_data[cluster_one_id]['X']) or len(self.rf_data[cluster_two_id]['X'])):
                    self.settings['shap']['sample'] = min(len(self.rf_data[cluster_one_id]['X']), len(self.rf_data[cluster_two_id]['X']))
                cluster_one_sample = self.rf_data[cluster_one_id]['X'].sample(self.settings['shap']['sample'], replace=False)
                cluster_two_sample = self.rf_data[cluster_two_id]['X'].sample(self.settings['shap']['sample'], replace=False)
                cluster_one_shap = pd.DataFrame(explainer.shap_values(cluster_one_sample, check_additivity=False)[1], columns=self.rf_data[cluster_one_id]['X'].columns)
                cluster_two_shap = pd.DataFrame(explainer.shap_values(cluster_two_sample, check_additivity=False)[1], columns=self.rf_data[cluster_two_id]['X'].columns)
                mean_df_cluster_one, stdev_df_cluster_one = pd.DataFrame(cluster_one_shap.mean(), columns=['MEAN']), pd.DataFrame(cluster_one_shap.std(), columns=['STDEV'])
                mean_df_cluster_two, stdev_df_cluster_two = pd.DataFrame(cluster_two_shap.mean(), columns=['MEAN']), pd.DataFrame(cluster_two_shap.std(), columns=['STDEV'])
                results_cluster_one = mean_df_cluster_one.join(stdev_df_cluster_one).sort_values(by='MEAN', ascending=False)
                results_cluster_two = mean_df_cluster_two.join(stdev_df_cluster_two).sort_values(by='MEAN', ascending=False)
                self.__save_results(df=results_cluster_one, name=f'SHAP CLUSTER {str(cluster_one_id)} vs. {str(cluster_two_id)}')
                self.__save_results(df=results_cluster_two, name=f'SHAP CLUSTER {str(cluster_two_id)} vs. {str(cluster_one_id)}')
            timer.stop_timer()
            stdout_success(msg=f"Paired clusters SHAP values complete", elapsed_time=timer.elapsed_time_str)


# settings = {'gini_importance': True, 'permutation_importance': True, 'shap': {'method': 'cluster_paired', 'create': True, 'sample': 100}}
# test = ClusterXAICalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                             data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/quizzical_rhodes.pickle',
#                             settings=settings)
# test.run()




# settings = {'correlation_methods': ['pearson', 'kendall', 'spearman'], 'plots': {'create': True, 'correlations': 'pearson', 'palette': 'jet'}}
# test = EmbeddingCorrelationCalculator(data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/quizzical_rhodes.pickle',
#                                    config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                                    settings=settings)
# test.run()


# settings = {'scaled': True, 'ANOVA': True, 'tukey_posthoc': True, 'descriptive_statistics': True}
# test = ClusterFrequentistCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                                    data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/quizzical_rhodes.pickle',
#                                    settings=settings)
# test.run()







# settings = {'scaled': True, 'anova': True, 'tukey_posthoc': True, 'descriptive_statistics': True}
# test = ClusterStatisticsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                                    data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/nostalgic_albattani.pickle',
#                                    settings=settings)
# test.run()



