import os
from typing import Union
import numpy as np
import pandas as pd
from copy import deepcopy
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.read_write import read_df, write_df
from simba.utils.printing import SimbaTimer


class MitraCorrelationChecks(ConfigReader):

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path)

    def run(self):
        results = []
        for file_path in self.target_file_paths:
            df = read_df(file_path=file_path, file_type=self.file_type).astype(np.float32)
            results.append(df)
        results = pd.concat(results, axis=0).reset_index(drop=True)
        for clf in self.clf_names:
            df = results.corrwith(results[clf]).sort_values(ascending=False)
            df.to_csv(f'/Users/simon/Desktop/envs/simba/troubleshooting/mitra/correlations/{clf}.csv')
            print(f'Saved {clf}')

x = MitraCorrelationChecks('/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/project_config.ini')
x.run()