import itertools
import os
from typing import Union

import pandas as pd

from simba.mixins.statistics_mixin import Statistics
from simba.utils.checks import (check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_valid_boolean)
from simba.utils.enums import UML, Formats
from simba.utils.errors import CountError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (create_empty_xlsx_file, df_to_xlsx_sheet,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_pickle)

# HOMOGENEITY_SCORE, V_MEASURE_SCORE, COMPLETENESS_SCORE, SILHOUETTE_SCORE

class ClustererComparsionMetricCalculator():
    """
    :example:
    >>> x = ClustererComparsionMetricCalculator(data_dir=r'/Users/simon/Desktop/tests_', save_path=r'/Users/simon/Desktop/asdasdasd.xlsx')
    >>> x.run()
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 save_path: Union[str, os.PathLike],
                 ami: bool = True,
                 ari: bool = True,
                 fowlkes_mellow: bool = True):

        check_if_dir_exists(in_dir=data_dir)
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        check_valid_boolean(value=[ami, ari, fowlkes_mellow], source=self.__class__.__name__)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f".{Formats.PICKLE.value}", '.pkl'])
        if len(self.data_paths) < 2:
            raise CountError(msg=f"Cluster comparisons require at least two models. Found {len(self.data_paths)} in {data_dir}", source=self.__class__.__name__)
        self.ami, self.ari, self.fowlkes_mellow, self.save_path = ami, ari, fowlkes_mellow, save_path

    def run(self):
        timer = SimbaTimer(start=True)
        create_empty_xlsx_file(xlsx_path=self.save_path)

        data = {}
        for file_cnt, file_path in enumerate(self.data_paths):
            data_ = read_pickle(data_path=file_path, verbose=True)
            check_if_keys_exist_in_dict(data=data_, key=[UML.CLUSTER_MODEL.value, UML.DR_MODEL.value])
            name = data_[UML.CLUSTER_MODEL.value][UML.HASHED_NAME.value]
            name = get_fn_ext(filepath=file_path)[1]
            data[name] = data_[UML.CLUSTER_MODEL.value][UML.MODEL.value].labels_
        cluster_comb = list(itertools.combinations(list(data.keys()), 2))

        if self.ami:
            ami_results = pd.DataFrame(columns=['CLUSTERER_1', 'CLUSTERER_2', 'ADJUSTED_MUTUAL_INFORMATION'])
            for x, y in cluster_comb:
                ami_results.loc[len(ami_results)] = [x, y, Statistics.adjusted_mutual_info(x=data[x], y=data[y])]
            df_to_xlsx_sheet(xlsx_path=self.save_path, df=ami_results, sheet_name='AMI', create_file=True)
        if self.ari:
            ari_results = pd.DataFrame(columns=['CLUSTERER_1', 'CLUSTERER_2', 'ADJUSTED_RAND_INDEX'])
            for x, y in cluster_comb:
                ari_results.loc[len(ari_results)] = [x, y, Statistics.adjusted_rand(x=data[x], y=data[y])]
            df_to_xlsx_sheet(xlsx_path=self.save_path, df=ari_results, sheet_name='ARI', create_file=True)
        if self.fowlkes_mellow:
            fm_results = pd.DataFrame(columns=['CLUSTERER_1', 'CLUSTERER_2', 'FOWLKES_MELLOW'])
            for x, y in cluster_comb:
                fm_results.loc[len(fm_results)] = [x, y, Statistics.fowlkes_mallows(x=data[x], y=data[y])]
            df_to_xlsx_sheet(xlsx_path=self.save_path, df=fm_results, sheet_name='FOWLKES_MELLOW', create_file=True)

        timer.stop_timer()
        stdout_success(f"Cluster comparisons saved at {self.save_path}", elapsed_time=timer.elapsed_time_str)



# x = ClustererComparsionMetricCalculator(data_dir=r'/Users/simon/Desktop/tests_', save_path=r'/Users/simon/Desktop/asdasdasd.xlsx')
# x.run()
