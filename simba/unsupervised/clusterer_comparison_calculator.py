import os
from itertools import combinations, product
from typing import List, Union

import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from simba.mixins.config_reader import ConfigReader
from simba.mixins.statistics_mixin import Statistics
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict, check_valid_lst)
from simba.utils.enums import Formats
from simba.utils.errors import CountError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_pickle)

ADJ_MUTUAL_INFO = "adjusted mutual information"
FOWLKES_MALLOWS = "fowlkes mallows"
ADJ_RAND_INDEX = "adjusted rand index"
STATS_OPTIONS = [ADJ_MUTUAL_INFO, FOWLKES_MALLOWS, ADJ_RAND_INDEX]


class ClustererComparisonCalculator(ConfigReader, Statistics):
    """

    Methods to compute matrix of comparisons between different cluster models.

    .. note::
       `Example of exepcted output <https://github.com/sgoldenlab/simba/blob/master/misc/clusterer_comparisons_20240401112101.xlsx>`__.

    :example:
    >>> cluster_comparer = ClustererComparisonCalculator(config_path="/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini", data_dir='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters', statistics=STATS_OPTIONS)
    >>> cluster_comparer.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_dir: Union[str, os.PathLike],
        statistics: List[Literal[ADJ_MUTUAL_INFO, FOWLKES_MALLOWS, ADJ_RAND_INDEX]],
    ):

        ConfigReader.__init__(
            self, config_path=config_path, read_video_info=False, create_logger=False
        )
        check_if_dir_exists(in_dir=data_dir)
        check_file_exist_and_readable(file_path=config_path)
        self.data_paths = find_files_of_filetypes_in_directory(
            directory=data_dir, extensions=[f".{Formats.PICKLE.value}"]
        )
        if len(self.data_paths) < 2:
            raise CountError(
                msg=f"Cluster comparisons require at least two models. Found {len(self.data_paths)} in {data_dir}",
                source=self.__class__.__name__,
            )
        check_valid_lst(
            data=statistics,
            source=self.__class__.__name__,
            valid_dtypes=(str,),
            valid_values=STATS_OPTIONS,
        )
        self.statistics, self.config_path, self.data_dir = (
            statistics,
            config_path,
            data_dir,
        )
        self.save_path = os.path.join(
            self.logs_path,
            f"clusterer_comparisons_{self.datetime}.{Formats.XLXS.value}",
        )
        with pd.ExcelWriter(self.save_path, mode="w") as writer:
            pd.DataFrame().to_excel(writer, sheet_name=" ", index=True)

    def __save_results(self, df: pd.DataFrame, name: str):
        with pd.ExcelWriter(self.save_path, mode="a") as writer:
            df.to_excel(writer, sheet_name=name, index=True)

    def run(self):
        self.data, obs_cnts = {}, {}
        for file_cnt, file_path in enumerate(self.data_paths):
            self.data_ = read_pickle(data_path=file_path, verbose=True)
            check_if_keys_exist_in_dict(
                data=self.data_,
                key=[Unsupervised.METHODS.value, Clustering.CLUSTER_MODEL.value],
                name=file_path,
            )
            mdl_name = self.data_[Clustering.CLUSTER_MODEL.value][
                Unsupervised.HASHED_NAME.value
            ]
            cluster_data = self.data_[Clustering.CLUSTER_MODEL.value][
                Unsupervised.MODEL.value
            ].labels_
            self.data[mdl_name] = cluster_data
            obs_cnts[mdl_name] = cluster_data.shape[0]

        counts = list(set([v for k, v in obs_cnts.items()]))
        if len(counts) > 1:
            raise CountError(
                msg=f"Cluster comparisons require models built using the data. Found different number of observations in the different mdoels in {self.data_dir}. {obs_cnts}",
                source=self.__class__.__name__,
            )

        for statistic in self.statistics:
            results = {}
            for i in self.data.keys():
                print(f"Computing {statistic} statistics for {i}...")
                results[i] = {}
                for j in self.data.keys():
                    x, y = self.data[i], self.data[j]
                    if statistic == ADJ_RAND_INDEX:
                        s = Statistics.adjusted_rand(x=x, y=y)
                    elif statistic == ADJ_MUTUAL_INFO:
                        s = Statistics.adjusted_mutual_info(x=x, y=y)
                    else:
                        s = Statistics.fowlkes_mallows(x=x, y=y)
                    results[i][j] = s
            df = pd.DataFrame.from_dict(results, orient="index")
            self.__save_results(df=df, name=statistic)

        self.timer.stop_timer()
        stdout_success(
            f"Cluster comparisons saved at {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# x = ClustererComparison(config_path="/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini", data_dir='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/clusters', statistics=STATS_OPTIONS)
# x.run()
