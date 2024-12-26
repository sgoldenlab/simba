import itertools
import os
from typing import Union

import numpy as np
import pandas as pd
from simba.mixins.statistics_mixin import Statistics
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict,
                                check_valid_boolean, check_if_dir_exists)
from simba.utils.enums import UML
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory, get_unique_values_in_iterable, read_pickle, df_to_xlsx_sheet)


class ClusterFrequentistCalculator():
    """
    Class for computing frequentist statitics based on cluster assignment labels for explainability purposes.

    :param Union[str, os.PathLike] data_path: path to pickle or directory of pickles holding unsupervised results in ``simba.unsupervised.data_map.yaml`` format. Can be greated by ...
    :param Union[str, os.PathLike] save_path: Location wehere to store the results in MS Excel format.
    :param bool scaled: If True, uses the scaled data (used to fit the model). Else, uses the raw feature bout values.
    :param bool anova: If True, computes one-way ANOVAs.
    :param bool kruskal_wallis: If True, computes Kruskal-Wallis comparisons.
    :param bool tukey: If True, computes Tukey-HSD post-hoc cluster comparisons.
    :param bool descriptive: If True, computes descriptive statistics for each feature value in each cluster (mean, sem, stdev, min, max).
    :param bool pairwise: If True, computes one-way ANOVAs and Kruskal-Wallis comparisons where each cluster is compared to each of the other clusters. Else computes one against all others.

    :example:
    >>> calculator = ClusterFrequentistCalculator(data_path=r"/Users/simon/Downloads/academic_elgamal.pickle", scaled=False, save_path=r"/Users/simon/Desktop/test.xlsx", pairwise=True, tukey=False)
    >>> calculator.run()
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

        self.timer = SimbaTimer(start=True)
        check_valid_boolean(value=[scaled, anova, tukey, descriptive, pairwise], source=self.__class__.__name__)
        check_if_dir_exists(in_dir=os.path.dirname(save_path), source=self.__class__.__name__)
        if len(list({scaled, anova, tukey, descriptive, kruskal_wallis})) == 1 and not list({scaled, anova, tukey, descriptive, kruskal_wallis})[0]:
            raise InvalidInputError(msg='No statistical tests chosen', source=self.__class__.__name__)
        if os.path.isdir(data_path):
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.pickle', '.pkl'], raise_error=True)
        else:
            check_file_exist_and_readable(file_path=data_path)
            self.data_paths = [data_path]
        self.scaled, self.anova, self.tukey, self.descriptive, self.pairwise, self.kruskal = scaled, anova, tukey, descriptive, pairwise, kruskal_wallis
        self.save_path = save_path

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
            _ = get_unique_values_in_iterable(data=lbls, name=name, min=2)
            unique_cluster_lbls = np.unique(lbls)
            if self.anova or self.kruskal:
                if self.pairwise:
                    cluster_comb = list(itertools.combinations(unique_cluster_lbls, 2))
                    cluster_comb = [(x[0], (x[1],)) for x in cluster_comb]
                else:
                    cluster_comb= [(x, tuple(y for y in unique_cluster_lbls if y != x)) for x in unique_cluster_lbls]
                self.anova_results, self.kruskal_results = [], []
                for target, nontargets in cluster_comb:
                    target_X = features.loc[np.argwhere(lbls == target).flatten()].values
                    non_target_X = features.loc[np.argwhere(np.isin(lbls, nontargets)).flatten()].values
                    if self.anova:
                        anova_df = Statistics.one_way_anova_scipy(x=target_X, y=non_target_X, variable_names=list(features.columns), x_name=str(target), y_name=str(nontargets))
                        self.anova_results.append(anova_df)
                    if self.kruskal:
                        kruskal_df = Statistics.kruskal_scipy(x=target_X, y=non_target_X, variable_names=list(features.columns), x_name=str(target), y_name=str(nontargets))
                        self.kruskal_results.append(kruskal_df)
                if self.anova:
                    self.__save_results(df=pd.concat(self.anova_results, axis=0).reset_index(drop=True).sort_values(['GROUP_1', 'GROUP_2', 'P-VALUE']), name='ANOVA')
                if self.kruskal:
                    self.__save_results(df=pd.concat(self.kruskal_results, axis=0).reset_index(drop=True).sort_values(['GROUP_1', 'GROUP_2', 'P-VALUE']), name='KRUSKAL WALLIS')
            if self.tukey:
                tukey_results = Statistics.pairwise_tukeyhsd_scipy(data=features.values, group=lbls, variable_names=list(features.columns), verbose=True)
                self.__save_results(df=tukey_results, name='TUKEY_HSD')
            if self.descriptive:
                def sem(x):
                    return np.std(x, ddof=1) / np.sqrt(len(x))
                features['GROUP'] =  lbls
                numeric_columns = [col for col in features.select_dtypes(include='number').columns if col != 'GROUP']
                descriptive_result = features.groupby('GROUP').agg({col: ['mean', sem, 'std', 'min', 'max'] for col in numeric_columns})
                descriptive_result.columns = [f'{col[0]}_{col[1]}' if col[1] != '<lambda>' or '<lambda_0>' else f'{col[0]}_sem' for col in descriptive_result.columns]
                self.__save_results(df=descriptive_result, name='DESCRIPTIVE_STATISTICS')
            self.timer.stop_timer()
            stdout_success(msg=f'Cluster descriptive statistics saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

    def __save_results(self, df: pd.DataFrame, name: str):
        df_to_xlsx_sheet(xlsx_path=self.save_path, df=df, sheet_name=name, create_file=True)

# calculator = ClusterFrequentistCalculator(data_path=r"/Users/simon/Downloads/academic_elgamal.pickle", scaled=False, save_path=r"/Users/simon/Desktop/test.xlsx", pairwise=True, tukey=False)
# calculator.run()


