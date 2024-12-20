import os
from typing import Union
import pandas as pd
import numpy as np
from copy import deepcopy
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.read_write import read_df, write_df, find_files_of_filetypes_in_directory, get_fn_ext
from simba.utils.printing import SimbaTimer

class MitraDownSampler(ConfigReader):

    def __init__(self,
                 config_path: os.PathLike,
                 data_path: os.PathLike):

        ConfigReader.__init__(self, config_path=config_path)
        self.data_path = data_path
        self.data_paths = find_files_of_filetypes_in_directory(directory=self.data_path, extensions=['.' + self.file_type])

    def run(self):
        for file_path in self.data_paths:
            df = read_df(file_path=file_path, file_type=self.file_type)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            for clf in self.clf_names:
                save_path = os.path.join(self.targets_folder, clf, video_name + '.' + self.file_type)
                if not os.path.isdir(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                annot = df[df[clf] == 1]
                n_samples = int(len(annot) * 10)
                not_annot = df[df[clf] == 0]
                if n_samples > len(not_annot):
                   n_samples = len(not_annot)
                if n_samples < 5000:
                    n_samples = 5000
                not_annot = not_annot.sample(n=n_samples)
                idx = list(not_annot.index) + list(annot.index)
                out = df.loc[idx, :].sort_index()
                print(len(out), len(annot), n_samples, len(not_annot), clf, video_name)
                write_df(df=out.astype(np.float32), file_type=self.file_type, save_path=save_path)





x = MitraDownSampler(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/project_config.ini',
                     data_path='/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/targets_inserted/originals')
x.run()