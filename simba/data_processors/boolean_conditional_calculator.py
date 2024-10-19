import os
from copy import deepcopy
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_all_file_names_are_represented_in_video_log, check_if_df_field_is_boolean, check_instance)
from simba.utils.errors import MissingColumnsError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (get_fn_ext, read_data_paths, read_df, read_video_info, str_2_bool)
from simba.utils.data import detect_bouts


class BooleanConditionalCalculator(ConfigReader):
    """
    Compute descriptive statistics (e.g., the time in seconds and number of frames) of multiple Boolean fields fullfilling user-defined conditions.

    For example, computedescriptive statistics for when Animal 1 is inside the shape Rectangle_1 while at the same time directing towards shape Polygon_1,
    while at the same time Animal 2 is outside shape Rectangle_1 and directing towards Polygon_1.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format.
    :param Dict[str, Union[bool, str]] rules: Rules with field names as keys and bools (or string representations of bools) as values.
    :param Optional[Union[str, os.PathLike, None]] data_paths: Optional data paths to be processsed. If None, all CSVs inside the `projecet_folder/csv/outlier_corrected_movement_location` are analysed.

    .. note:
       `Example expected aggregate output table <https://github.com/sgoldenlab/simba/blob/master/misc/Conditional_aggregate_statistics_20231004130314.csv>`__.
       `Example expected detailed output table <https://github.com/sgoldenlab/simba/blob/master/misc/Detailed_conditional_aggregate_statistics_20241011123409.csv>`__.

    :examples:
    >>> rules = {'Rectangle_1 Simon in zone': 'TRUE', 'Polygon_1 JJ in zone': 'TRUE'} #  OR {'Rectangle_1 Simon in zone': True, 'Polygon_1 JJ in zone': True}
    >>> conditional_bool_rule_calculator = BooleanConditionalCalculator(rules=rules, config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
    >>> conditional_bool_rule_calculator.run()
    >>> conditional_bool_rule_calculator.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 rules: Dict[str, Union[bool, str]],
                 data_paths: Optional[Union[str, os.PathLike, None]] = None):

        ConfigReader.__init__(self, config_path=config_path)
        check_instance(source=self.__class__.__name__, instance=rules, accepted_types=(dict,))
        self.save_path = os.path.join(self.logs_path, f"Conditional_aggregate_statistics_{self.datetime}.csv")
        self.detailed_save_path = os.path.join(self.logs_path, f"Detailed_conditional_aggregate_statistics_{self.datetime}.csv")
        self.data_paths = read_data_paths(path=data_paths, default=self.outlier_corrected_paths, default_name=self.feature_file_paths, file_type=self.file_type)
        self.rules = rules
        self.output_df = pd.DataFrame(columns=["VIDEO"] + list(self.rules.keys()) + ["TIME (s)", "FRAMES (count)"])
        self.bout_df_cols = ["VIDEO"] + list(self.rules.keys()) + ["START FRAME", "END FRAME", "START TIME", "END TIME" ,"BOUT TIME"]
        self.bout_dfs = []

    def _check_integrity_of_rule_columns(self):
        for behavior in self.rules.keys():
            if behavior not in self.df.columns:
                raise MissingColumnsError(msg=f"File is missing the column {behavior} which is required for your conditional aggregate statistics {self.file_path}", source=self.__class__.__name__)
            check_if_df_field_is_boolean(df=self.df, field=behavior, df_name=self.file_path)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.feature_file_paths):
            self.file_path = file_path
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            _, _, self.fps = read_video_info(vid_info_df=self.video_info_df, video_name=self.video_name)
            self.df = read_df(file_path=file_path, file_type=self.file_type)
            self._check_integrity_of_rule_columns()
            self.df = self.df[list(self.rules.keys())]
            self.sliced_df = deepcopy(self.df)
            values_str = []
            for k, v in self.rules.items():
                if str_2_bool(v):
                    self.sliced_df = self.sliced_df[self.sliced_df[k] == 1]
                else:
                    self.sliced_df = self.sliced_df[self.sliced_df[k] == 0]
                values_str.append(v)
            time_s = round(len(self.sliced_df) / self.fps, 4)
            if len(self.sliced_df) > 0:
                bout_df = pd.DataFrame(data=np.zeros((len(self.df))), columns=['behavior'])
                bout_df.iloc[self.sliced_df.index] = 1
                bout_df = detect_bouts(data_df=bout_df, target_lst=['behavior'], fps=self.fps)
                bout_df[list(self.rules.keys())] = list(self.rules.values())
                bout_df['VIDEO'] = self.video_name
                bout_df = bout_df.rename(columns={'Start_time': 'START TIME', 'End Time': 'END TIME', 'Start_frame': 'START FRAME', 'End_frame': 'END FRAME', 'Bout_time': 'BOUT TIME'})
                self.bout_dfs.append(bout_df[self.bout_df_cols])
            self.output_df.loc[len(self.output_df)] = ([self.video_name] + list(self.rules.values()) + [time_s] + [len(self.sliced_df)])


    def save(self):
        self.output_df.to_csv(self.save_path, index=False)
        self.timer.stop_timer()
        stdout_success(msg=f"Boolean conditional data saved at at {self.save_path}!", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)
        if len(self.bout_dfs) > 0:
            self.bout_dfs = pd.concat(self.bout_dfs, axis=0).reset_index(drop=True)
            self.bout_dfs.to_csv(self.detailed_save_path, index=False)
            stdout_success(msg=f"Detailed boolean conditional data saved at at {self.save_path}!", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


# rules = {'Polygon_2 Animal_1 nose in zone': True, 'Polygon_2 Animal_1 facing': False}
# runner = BooleanConditionalCalculator(rules=rules, config_path=r"C:\troubleshooting\spontenous_alternation\project_folder\project_config.ini")
# runner.run()
# runner.save()
# #
#


