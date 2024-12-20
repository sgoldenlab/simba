from typing import Union
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import itertools
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import check_valid_boolean, check_file_exist_and_readable, check_if_keys_exist_in_dict
from simba.utils.read_write import find_files_of_filetypes_in_directory, read_pickle, get_unique_values_in_iterable
from simba.utils.errors import InvalidInputError
from simba.utils.enums import UML
from simba.utils.printing import SimbaTimer
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng



class ClusterFrequentistCalculator():
    """
    Class for computing frequentist statitics based on cluster assignment labels for explainability purposes.

    :param Union[str, os.PathLike] data_path: path to pickle or directory of pickles holding unsupervised results in ``simba.unsupervised.data_map.yaml`` format.
    :param dict settings: Dict holding which statistical tests to use, with test name as keys and booleans as values.

    :example:
    """
    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 save_path: Union[str, os.PathLike],
                 scaled: bool = True,
                 anova: bool = True,
                 kruskal_wallis: bool = True,
                 tukey: bool = True,
                 descriptive: bool = True,
                 pairwise: bool = True):


        check_valid_boolean(value=[scaled, anova, tukey, descriptive, pairwise], source=self.__class__.__name__)
        if len(list({scaled, anova, tukey, descriptive, kruskal_wallis})) == 1 and not list({scaled, anova, tukey, descriptive, kruskal_wallis})[0]:
            raise InvalidInputError(msg='No statistical tests chosen', source=self.__class__.__name__)
        if os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.pickle', '.pkl'], raise_error=True)
        else:
            check_file_exist_and_readable(file_path=data_path)
            self.data_paths = [data_path]
        self.scaled, self.anova, self.tukey, self.descriptive, self.pairwise, self.kruskal = scaled, anova, tukey, descriptive, pairwise, kruskal_wallis
        with pd.ExcelWriter(save_path, mode="w") as writer:
            pd.DataFrame().to_excel(writer, sheet_name=" ", index=True)


    def run(self):
        for file_cnt, file_path in enumerate(self.data_paths):
            data = read_pickle(data_path=file_path)
            check_if_keys_exist_in_dict(data=data, key=[UML.CLUSTER_MODEL.value, UML.DATA.value])
            name = data[UML.CLUSTER_MODEL.value][UML.HASHED_NAME.value]
            if not self.scaled:
                features = data[UML.DATA.value][UML.UNSCALED_TRAIN_DATA.value].reset_index(drop=True)
            else:
                features = data[UML.DATA.value][UML.SCALED_TRAIN_DATA.value].reset_index(drop=True)
            lbls = data[UML.CLUSTER_MODEL.value][UML.MODEL.value].labels_
            cluster_cnt = get_unique_values_in_iterable(data=lbls, name=name, min=2)
            unique_cluster_lbls = np.unique(lbls)
            if self.pairwise:
                cluster_comb = list(itertools.combinations(unique_cluster_lbls, 2))
                cluster_comb = [(x[0], (x[1],)) for x in cluster_comb]
            else:
                cluster_comb= [(x, tuple(y for y in unique_cluster_lbls if y != x)) for x in unique_cluster_lbls]
            anova_results, kruskal_results, tukey_results = [], [], []
            for target, base in cluster_comb:
                target_X = features.loc[np.argwhere(lbls == target).flatten()].values
                non_target_X = features.loc[np.argwhere(np.isin(lbls, base)).flatten()].values
                if self.anova:
                    anova_df = pd.DataFrame(features.columns, columns=['FEATURE'])
                    anova_df[['GROUP_1', 'GROUP_2']] = target, str(base)
                    anova_df['F-STATISTIC'], anova_df['P-VALUE'] = f_oneway(target_X, non_target_X)
                    anova_results.append(anova_df)
                if self.kruskal:
                    kruskal_df = pd.DataFrame(features.columns, columns=['FEATURE'])
                    kruskal_df[['GROUP_1', 'GROUP_2']] = target, str(base)
                    kruskal_df['STATISTIC'], kruskal_df['P-VALUE'] = kruskal(target_X, non_target_X)
                    kruskal_results.append(kruskal_df)
            if self.tukey:
                tukey_df = deepcopy(features)
                tukey_df['lbls'] = lbls
                for x in features.columns:
                    data = pairwise_tukeyhsd(tukey_df[x], tukey_df['lbls'])
                    df = pd.DataFrame(data=data._results_table.data[1:], columns=data._results_table.data[0])
                    df['P-VALUE'] = psturng(np.abs(data.meandiffs / data.std_pairs), len(data.groupsunique), data.df_total)
                    df['FEATURE'] = x
                    tukey_results.append(tukey_df)
            if self.descriptive:
                for cluster_id in unique_cluster_lbls:
                    target_X = features.loc[np.argwhere(lbls == target).flatten()].values




                    #print()
                    #anova_df['F-STATISTIC', 'P-VALUE'] = f_vals, p_vals




                 #= pair


        #
        #
        # if self.settings[ANOVA]:
        #     self.__one_way_anovas()






calculator = ClusterFrequentistCalculator(data_path=r"C:\troubleshooting\nastacia_unsupervised\cluster_data\dreamy_moser.pickle", scaled=True, save_path=r"C:\troubleshooting\nastacia_unsupervised\cluster_statistics\stats.xlsx", pairwise=False)
calculator.run()


