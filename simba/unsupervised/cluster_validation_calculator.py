import os
from typing import Callable, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.statistics_mixin import Statistics
from simba.unsupervised.enums import Clustering, Unsupervised
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_keys_exist_in_dict,
                                check_valid_extension)
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_unique_values_in_iterable, read_pickle)

CLUSTERER_NAME = "CLUSTERER_NAME"
CLUSTER_COUNT = "CLUSTER_COUNT"
EMBEDDER_NAME = "EMBEDDER_NAME"
DUNN_INDEX = "DUNN_INDEX"


class ClusterValidators(ConfigReader):

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        data_path: Union[str, os.PathLike],
        validator_func: Callable[[np.ndarray, np.ndarray], None],
    ):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        self.validator = validator_func
        if os.path.isdir(data_path):
            check_if_dir_exists(in_dir=data_path)
            self.data_paths = find_files_of_filetypes_in_directory(
                directory=data_path,
                extensions=[f".{Formats.PICKLE.value}"],
                raise_error=True,
            )
        else:
            check_valid_extension(
                path=data_path, accepted_extensions=Formats.PICKLE.value
            )
            self.data_paths = [data_path]
        self.validator_name = validator_func.__name__
        self.save_path = os.path.join(
            self.logs_path, f"{self.validator_name}_{self.datetime}.xlsx"
        )
        with pd.ExcelWriter(self.save_path, mode="w") as writer:
            pd.DataFrame().to_excel(writer, sheet_name=" ", index=True)

    def run(self):
        print(
            f"Analyzing {self.validator_name} for {len(self.data_paths)} clusterer(s)..."
        )
        self.results = {}
        for file_cnt, file_path in enumerate(self.data_paths):
            model_timer = SimbaTimer(start=True)
            v = read_pickle(data_path=file_path)
            self.results[file_cnt] = {}
            self.results[file_cnt][CLUSTERER_NAME] = v[Clustering.CLUSTER_MODEL.value][
                Unsupervised.HASHED_NAME.value
            ]
            self.results[file_cnt][EMBEDDER_NAME] = v[Unsupervised.DR_MODEL.value][
                Unsupervised.HASHED_NAME.value
            ]
            check_if_keys_exist_in_dict(
                data=v, key=[Clustering.CLUSTER_MODEL.value], name=file_path
            )
            print(
                f"Performing {self.validator_name} for cluster model {v[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value]}..."
            )
            x = v[Unsupervised.DR_MODEL.value][Unsupervised.MODEL.value].embedding_
            y = v[Clustering.CLUSTER_MODEL.value][Unsupervised.MODEL.value].labels_
            cluster_cnt = get_unique_values_in_iterable(
                data=y,
                name=v[Clustering.CLUSTER_MODEL.value][Unsupervised.HASHED_NAME.value],
                min=1,
            )
            validator_statistic = self.validator(
                x.astype(np.float32), y.astype(np.int64)
            )
            self.results[file_cnt] = {
                **self.results[file_cnt],
                **v[Clustering.CLUSTER_MODEL.value][Unsupervised.PARAMETERS.value],
                **v[Unsupervised.DR_MODEL.value][Unsupervised.PARAMETERS.value],
            }
            self.results[file_cnt] = {
                **self.results[file_cnt],
                **{self.validator_name: validator_statistic},
                **{CLUSTER_COUNT: cluster_cnt},
            }
            model_timer.stop_timer()
            stdout_success(
                msg=f"DUNN INDEX complete for model {self.results[file_cnt][CLUSTERER_NAME]} ...",
                elapsed_time=model_timer.elapsed_time_str,
            )

    def save(self):
        for k, v in self.results.items():
            df = pd.DataFrame.from_dict(v, orient="index", columns=["VALUE"])
            with pd.ExcelWriter(self.save_path, mode="a") as writer:
                df.to_excel(writer, sheet_name=v[CLUSTERER_NAME], index=True)
        self.timer.stop_timer()
        stdout_success(
            msg=f"ALL {self.validator_name} calculations complete and saved in {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# calculator = ClusterValidators(config_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/project_config.ini',
#                                data_path='/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/small_clusters',
#                                validator_func=Statistics.calinski_harabasz)
# calculator.run()
# calculator.save()
