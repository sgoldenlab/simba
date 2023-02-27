import pandas as pd
from simba.read_config_unit_tests import (read_config_file,
                                          read_project_path_and_file_type,
                                          check_file_exist_and_readable)
from simba.unsupervised.misc import (read_pickle,
                                     get_cluster_cnt)
from sklearn.inspection import permutation_importance
from simba.misc_tools import SimbaTimer
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
from copy import deepcopy
from statsmodels.stats.libqsturng import psturng
from sklearn.ensemble import RandomForestClassifier
import itertools
import warnings
import shap
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class ClusterFrequentistCalculator(object):
    def __init__(self,
                 config_path: str,
                 data_path: str,
                 settings: dict):

        self.config = read_config_file(ini_path=config_path)
        self.settings = settings
        self.project_path, _ = read_project_path_and_file_type(config=self.config)
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.save_path = os.path.join(self.project_path, 'logs', f'cluster_descriptive_statistics_{self.datetime}.xlsx')
        check_file_exist_and_readable(file_path=data_path)
        self.data = read_pickle(data_path=data_path)
        self.feature_data = self.data['EMBEDDER']['MODEL']._raw_data
        if not settings['scaled']:
            self.feature_data = self.data['EMBEDDER']['SCALER'].inverse_transform(self.feature_data)
        self.feature_data = pd.DataFrame(data=self.feature_data, columns=self.data['EMBEDDER']['OUT_FEATURE_NAMES'])
        self.cluster_data = self.data['MODEL'].labels_
        _ = get_cluster_cnt(data=self.cluster_data, clusterer_name=self.data['NAME'], minimum_clusters=2)
        self.feature_data['CLUSTER'] = self.cluster_data
        self.timer = SimbaTimer()
        self.timer.start_timer()

        with pd.ExcelWriter(self.save_path, mode='w') as writer:
            pd.DataFrame().to_excel(writer, sheet_name=' ', index=True)

    def __one_way_anovas(self):
        print('Computing ANOVAs...')
        self.anova_results = pd.DataFrame(columns=['FEATURE NAME', 'F-STATISTIC', 'P-VALUE'])
        for feature_name in self.data['EMBEDDER']['OUT_FEATURE_NAMES']:
            stats_data = self.feature_data[[feature_name, 'CLUSTER']].sort_values(by=['CLUSTER']).values
            stats_data = np.split(stats_data[:, 0], np.unique(stats_data[:, 1], return_index=True)[1][1:])
            f_val, p_val = f_oneway(*stats_data)
            self.anova_results.loc[len(self.anova_results)] = [feature_name, f_val, p_val]
        self.anova_results = self.anova_results.sort_values(by=['P-VALUE']).set_index('FEATURE NAME')
        self.anova_results['P-VALUE'] = self.anova_results['P-VALUE'].round(5)
        self.__save_results(df=self.anova_results, name='ANOVA')

    def __descriptive_stats(self):
        print('Computing descriptive statistics..')
        self.descriptive_results = []
        for feature_name in self.data['EMBEDDER']['OUT_FEATURE_NAMES']:
            agg = self.feature_data.groupby(['CLUSTER'])[feature_name].agg(['mean','std', 'sem']).T
            agg['FEATURE_NAME'] = feature_name
            agg = agg.reset_index(drop=False).set_index('FEATURE_NAME').rename(columns={'index': 'MEASSURE'})
            self.descriptive_results.append(pd.DataFrame(agg))
        self.descriptive_results = pd.concat(self.descriptive_results, axis=0)
        self.__save_results(df=self.descriptive_results, name='DESCRIPTIVE STATISTICS')

    def __tukey_posthoc(self):
        print('Computing tukey posthocs...')
        self.post_hoc_results = []
        for feature_name in self.data['EMBEDDER']['OUT_FEATURE_NAMES']:
            data = pairwise_tukeyhsd(self.feature_data[feature_name], self.feature_data['CLUSTER'])
            df = pd.DataFrame(data=data._results_table.data[1:], columns=data._results_table.data[0])
            df['P-VALUE'] = psturng(np.abs(data.meandiffs / data.std_pairs), len(data.groupsunique), data.df_total)
            df['FEATURE_NAME'] = feature_name
            df = df.reset_index(drop=True).set_index('FEATURE_NAME')
            self.post_hoc_results.append(df)
        self.post_hoc_results = pd.concat(self.post_hoc_results, axis=0)
        self.__save_results(df=self.post_hoc_results, name='TUKEY POST-HOC')

    def __save_results(self, df: pd.DataFrame, name: str):
        with pd.ExcelWriter(self.save_path, mode='a') as writer:
            df.to_excel(writer, sheet_name=name, index=True)

    def run(self):
        if self.settings['descriptive_statistics']:
            self.__descriptive_stats()
        if self.settings['anova']:
            self.__one_way_anovas()
        if self.settings['tukey_posthoc']:
            self.__tukey_posthoc()
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: Cluster statistics complete. Data saved at {self.save_path} (elapsed time: {self.timer.elapsed_time_str}s)')

class ClusterXAICalculator(object):
    def __init__(self,
                 data_path: str,
                 config_path: str,
                 settings: dict):

        self.config = read_config_file(ini_path=config_path)
        self.settings = settings
        self.project_path, _ = read_project_path_and_file_type(config=self.config)
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.save_path = os.path.join(self.project_path, 'logs', f'cluster_xai_statistics_{self.datetime}.xlsx')
        check_file_exist_and_readable(file_path=data_path)
        self.data = read_pickle(data_path=data_path)
        self.feature_data = self.data['EMBEDDER']['MODEL']._raw_data
        self.feature_data = pd.DataFrame(data=self.feature_data, columns=self.data['EMBEDDER']['OUT_FEATURE_NAMES'])
        self.cluster_data = self.data['MODEL'].labels_
        _ = get_cluster_cnt(data=self.cluster_data, clusterer_name=self.data['NAME'], minimum_clusters=2)
        self.feature_data['CLUSTER'] = self.cluster_data
        self.timer = SimbaTimer()
        self.timer.start_timer()

        with pd.ExcelWriter(self.save_path, mode='w') as writer:
            pd.DataFrame().to_excel(writer, sheet_name=' ', index=True)

    def __train_rf_models(self):
        rf_clf = RandomForestClassifier(n_estimators=20,
                                             max_features='sqrt',
                                             n_jobs=-1, criterion='gini',
                                             min_samples_leaf=1,
                                             bootstrap=True,
                                             verbose=1)
        self.rf_data = {}
        for cluster_id in self.feature_data['CLUSTER'].unique():
            self.rf_data[cluster_id] = {}
            target_df = self.feature_data[self.feature_data['CLUSTER'] == cluster_id].drop(['CLUSTER'], axis=1)
            nontarget_df = self.feature_data[self.feature_data['CLUSTER'] != cluster_id].drop(['CLUSTER'], axis=1)
            target_df['TARGET'] = 1
            nontarget_df['TARGET'] = 0
            self.rf_data[cluster_id]['X'] = pd.concat([target_df, nontarget_df], axis=0).reset_index(drop=True).sample(frac=1)
            self.rf_data[cluster_id]['Y'] = self.rf_data[cluster_id]['X'].pop('TARGET')
            rf_clf.fit(self.rf_data[cluster_id]['X'], self.rf_data[cluster_id]['Y'])
            self.rf_data[cluster_id]['MODEL'] = deepcopy(rf_clf)


    def __gini_importance(self):
        print('Computing cluster gini...')
        for cluster_id, cluster_data in self.rf_data.items():
            importances = list(cluster_data['MODEL'].feature_importances_)
            gini_data = [(feature, round(importance, 6)) for feature, importance in zip(self.data['EMBEDDER']['OUT_FEATURE_NAMES'], importances)]
            df = pd.DataFrame(gini_data, columns=['FEATURE', 'FEATURE_IMPORTANCE']).sort_values(by=['FEATURE_IMPORTANCE'], ascending=False).reset_index(drop=True)
            self.__save_results(df=df, name=f'GINI CLUSTER {str(cluster_id)}')

    def __permutation_importance(self):
        print('Computing permutation importance...')
        for cluster_id, cluster_data in self.rf_data.items():
            p_importances = permutation_importance(cluster_data['MODEL'], cluster_data['X'], cluster_data['Y'], n_repeats=5, random_state=0)
            df = pd.DataFrame(np.column_stack([self.data['EMBEDDER']['OUT_FEATURE_NAMES'], p_importances.importances_mean, p_importances.importances_std]), columns=['FEATURE_NAME', 'FEATURE_IMPORTANCE_MEAN', 'FEATURE_IMPORTANCE_STDEV'])
            df = df.sort_values(by=['FEATURE_IMPORTANCE_MEAN'], ascending=False).reset_index(drop=True)
            self.__save_results(df=df, name=f'PERMUTATION CLUSTER {str(cluster_id)}')

    def __save_results(self, df: pd.DataFrame, name: str):
        with pd.ExcelWriter(self.save_path, mode='a') as writer:
            df.to_excel(writer, sheet_name=name, index=True)

    def __shap_values(self):
        if self.settings['shap']['method'] == 'cluster-wise':
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

    def run(self):
        self.__train_rf_models()
        if self.settings['gini_importance']:
            self.__gini_importance()
        if self.settings['permutation_importance']:
            self.__permutation_importance()
        if self.settings['shap']:
            self.__shap_values()

class EmbeddingCorrelationCalculator(object):
    def __init__(self,
                 data_path: str,
                 config_path: str,
                 settings: dict):

        check_file_exist_and_readable(file_path=data_path)
        config = read_config_file(ini_path=config_path)
        self.project_path, _ = read_project_path_and_file_type(config=config)
        self.logs_path = os.path.join(self.project_path, 'logs')
        data = read_pickle(data_path=data_path)
        results = pd.DataFrame()
        if data['MODEL'].__class__.__name__ is 'UMAP':
            df = pd.DataFrame(data['MODEL']._raw_data, columns=data['OUT_FEATURE_NAMES'])
            embedder_name = data['HASH']
            embedding = df.join(pd.DataFrame(data['MODEL'].embedding_, columns=['X', 'Y']))
        elif data['MODEL'].__class__.__name__ is 'HDBSCAN':
            df = pd.DataFrame(data['EMBEDDER']['MODEL']._raw_data, columns=data['OUT_FEATURE_NAMES'])
            embedder_name = data['EMBEDDER']['HASH']
            embedding = df.join(pd.DataFrame(data['EMBEDDER']['MODEL'].embedding_, columns=['X', 'Y']))
        for method in settings['correlations']:
            results[f'{method}_Y'] = df.corrwith(embedding['Y'], method=method)
            results[f'{method}_X'] = df.corrwith(embedding['X'], method=method)
        save_path = os.path.join(self.logs_path, embedder_name + '_embedding_stats.csv')
        results.to_csv(save_path)

        if settings['plots']['create']:
            plots_dir = os.path.join(self.logs_path, 'embedding_correlations')
            if not os.path.exists(plots_dir): os.makedirs(plots_dir)
            for feature_cnt, feature in enumerate(df.columns):
                color_bar = plt.cm.ScalarMappable(cmap=settings['plots']['palette'])
                color_bar.set_array([])
                plot = sns.scatterplot(data=embedding, x="X", y="Y", hue=feature, cmap=settings['plots']['palette'])
                plot.get_legend().remove()
                plot.figure.colorbar(color_bar, label=feature)
                plt.suptitle(feature, x=0.5, y=0.92)
                save_path = os.path.join(plots_dir, feature + '.png')
                plot.figure.savefig(save_path, bbox_inches="tight")
                plot.clear()
                plt.close()
                print(f'Saving image {str(feature_cnt+1)}/{str(len(df.columns))} ({feature})')

        print(f'SIMBA COMPLETE: Data saved at {save_path}')

# settings = {'correlations': ['pearson', 'kendall', 'spearman'], 'plots': {'create': True, 'correlations': 'pearson', 'palette': 'jet'}}
# _ = EmbeddingCorrelationCalculator(data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/dr_models/funny_heisenberg.pickle',
#                                    config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                                    settings=settings)




# settings = {'gini_importance': False, 'permutation_importance': False, 'shap': {'method': 'cluster-wise', 'run': True, 'sample': 100}}
# test = ClusterXAICalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                             data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/amazing_burnell.pickle',
#                             settings=settings)
# test.run()




# settings = {'scaled': True, 'anova': True, 'tukey_posthoc': True, 'descriptive_statistics': True}
# test = ClusterFrequentistCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                                    data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/nostalgic_albattani.pickle',
#                                    settings=settings)
# test.run()







# settings = {'scaled': True, 'anova': True, 'tukey_posthoc': True, 'descriptive_statistics': True}
# test = ClusterStatisticsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                                    data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/nostalgic_albattani.pickle',
#                                    settings=settings)
# test.run()



