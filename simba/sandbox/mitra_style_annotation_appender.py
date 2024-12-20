import os
from typing import Union
import pandas as pd
from copy import deepcopy
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_file_exist_and_readable, check_if_dir_exists
from simba.utils.read_write import read_df, write_df
from simba.utils.printing import SimbaTimer

class MitraStyleAnnotationAppender(ConfigReader):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Union[str, os.PathLike],
                 features_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path)
        check_file_exist_and_readable(file_path=data_path)
        check_if_dir_exists(in_dir=features_dir)
        check_if_dir_exists(in_dir=save_dir)
        self.data_path = data_path
        self.save_dir, self.features_dir = save_dir, features_dir

    def run(self):
        total_cnt = {}
        df_dict = pd.read_excel(self.data_path, sheet_name=None)
        for file_name, file_df in df_dict.items():
            video_timer = SimbaTimer(start=True)
            data_path = os.path.join(self.features_dir, file_name + '.csv')
            if os.path.isfile(data_path):
                data_df = read_df(file_path=data_path, file_type='csv')
                out_df = deepcopy(data_df)
                save_path = os.path.join(self.save_dir, file_name + '.csv')
                df = pd.DataFrame(file_df.values[1:, :3], columns=['BEHAVIOR', 'START', 'STOP'])
                df['BEHAVIOR'] = df['BEHAVIOR'].str.lower()
                for clf in self.clf_names:
                    if clf not in total_cnt.keys():
                        total_cnt[clf] = 0
                    clf_df = df[df['BEHAVIOR'] == clf].sort_values(['START'])
                    out_df[clf] = 0
                    if len(clf_df) > 0:
                        annot_idx = list(clf_df.apply(lambda x: list(range(int(x["START"]), int(x["STOP"]) + 1)), 1))
                        annot_idx = [x for xs in annot_idx for x in xs]
                        if len(annot_idx) > 0:
                            out_df.loc[annot_idx, clf] = 1
                    total_cnt[clf] += out_df[clf].sum()
                #write_df(df=out_df, file_type=self.file_type, save_path=save_path)
                video_timer.stop_timer()
                print(total_cnt)
                print(f'{file_name} saved..')

        print(total_cnt)

            #
            #
            #
            #
            #
            #

            #

data_path = r"C:\troubleshooting\mitra\Start-Stop Annotations.xlsx"
features_dir = r"C:\troubleshooting\mitra\project_folder\videos\bg_removed\rotated\laying_down_features\APPENDED"
save_dir = r"C:\troubleshooting\mitra\project_folder\videos\bg_removed\rotated\laying_down_features\APPENDED\targets_inserted"
config_path = r"C:\troubleshooting\mitra\project_folder\project_config.ini"

x = MitraStyleAnnotationAppender(data_path=data_path, features_dir=features_dir, save_dir=save_dir, config_path=config_path)
x.run()





