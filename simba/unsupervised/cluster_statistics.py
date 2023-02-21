import pandas as pd
from simba.read_config_unit_tests import (read_config_file,
                                          read_project_path_and_file_type,
                                          check_file_exist_and_readable)
from simba.unsupervised.misc import (read_pickle,
                                     get_cluster_cnt)
import os
from datetime import datetime
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
from statsmodels.stats.libqsturng import psturng
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class ClusterStatisticsCalculator(object):
    def __init__(self,
                 config_path: str,
                 data_path: str,
                 settings: dict):

        self.config = read_config_file(ini_path=config_path)
        self.settings = settings
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
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
        with pd.ExcelWriter(self.save_path, mode='w') as writer:
            pd.DataFrame().to_excel(writer, sheet_name=' ', index=True)

    def __one_way_anovas(self):
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
        self.descriptive_results = []
        for feature_name in self.data['EMBEDDER']['OUT_FEATURE_NAMES']:
            agg = self.feature_data.groupby(['CLUSTER'])[feature_name].agg(['mean','std', 'sem']).T
            agg['FEATURE_NAME'] = feature_name
            agg = agg.reset_index(drop=False).set_index('FEATURE_NAME').rename(columns={'index': 'MEASSURE'})
            self.descriptive_results.append(pd.DataFrame(agg))
        self.descriptive_results = pd.concat(self.descriptive_results, axis=0)
        self.__save_results(df=self.descriptive_results, name='DESCRIPTIVE STATISTICS')

    def __tukey_posthoc(self):
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


# settings = {'scaled': True, 'anova': True, 'tukey_posthoc': True, 'descriptive_statistics': True}
# test = ClusterStatisticsCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/project_folder/project_config.ini',
#                                    data_path='/Users/simon/Desktop/envs/troubleshooting/unsupervised/cluster_models/nostalgic_albattani.pickle',
#                                    settings=settings)
# test.run()



