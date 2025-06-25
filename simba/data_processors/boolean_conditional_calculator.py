import os
from copy import deepcopy
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_if_df_field_is_boolean, check_if_dir_exists, check_instance,
    check_valid_boolean, check_valid_dataframe, check_valid_dict)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats
from simba.utils.errors import NoDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, read_video_info,
                                    str_2_bool)


class BooleanConditionalCalculator(ConfigReader):
    """
    Compute descriptive statistics (e.g., the time in seconds and number of frames) of multiple Boolean fields fullfilling user-defined conditions.

    For example, computedescriptive statistics for when Animal 1 is inside the shape Rectangle_1 while at the same time directing towards shape Polygon_1,
    while at the same time Animal 2 is outside shape Rectangle_1 and directing towards Polygon_1.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format.
    :param Dict[str, Union[bool, str]] rules: Rules with field names as keys and bools (or string representations of bools) as values.
    :param Optional[Union[str, os.PathLike, None]] data_path: Optional data paths to be processsed. Can be a directory or file path. If None, all CSVs inside the `projecet_folder/csv/outlier_corrected_movement_location` are analysed.
    :param Optional[Union[str, os.PathLike]] agg_save_path: Optional location where to save the aggregate results as CSV file. If None, then results are saved in project logs folder under the ``Detailed_conditional_aggregate_statistics_{self.datetime}.csv`` filename.
    :param Optional[Union[str, os.PathLike]] detailed_save_path: Optional location where to save the detailed results as CSV file (bout level data). If None, then results are saved in project logs folder under the ``Detailed_conditional_aggregate_statistics_{self.datetime}.csv`` filename.

    .. note:
       `Example expected aggregate output table <https://github.com/sgoldenlab/simba/blob/master/misc/Conditional_aggregate_statistics_20231004130314.csv>`__.
       `Example expected detailed output table <https://github.com/sgoldenlab/simba/blob/master/misc/Detailed_conditional_aggregate_statistics_20241011123409.csv>`__.

    :example I:
    >>> rules = {'Rectangle_1 Simon in zone': 'TRUE', 'Polygon_1 JJ in zone': 'TRUE'} #  OR {'Rectangle_1 Simon in zone': True, 'Polygon_1 JJ in zone': True}
    >>> conditional_bool_rule_calculator = BooleanConditionalCalculator(rules=rules, config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
    >>> conditional_bool_rule_calculator.run()
    >>> conditional_bool_rule_calculator.save()


    :example II:
    >>> rules = {'Stimulus 2 Animal_1 in zone': True, 'Stimulus 6 Animal_1 in zone': 'falsE'}
    >>> runner = BooleanConditionalCalculator(rules=rules, config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini", data_path=r'C:\troubleshooting\RAT_NOR\project_folder\csv\features_extracted')
    >>> runner.run()
    >>> runner.save()


    :references:
       .. [1] Shonka, Sophia, and Michael J Hylin. “Younger Is Better But Only for Males: Social Behavioral Development Following Juvenile Traumatic Brain Injury to the Prefrontal Cortex,” n.d.

    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 rules: Dict[str, Union[bool, str]],
                 data_path: Optional[Union[str, os.PathLike, None]] = None,
                 agg_save_path: Optional[Union[str, os.PathLike]] = None,
                 detailed_save_path: Optional[Union[str, os.PathLike]] = None,
                 verbose: bool = True):

        ConfigReader.__init__(self, config_path=config_path)
        check_instance(source=self.__class__.__name__, instance=rules, accepted_types=(dict,))
        check_valid_dict(x=rules, valid_key_dtypes=(str,), valid_values_dtypes=(str, bool,), min_len_keys=2, source=f'{self.__class__.__name__} rules')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        if data_path is not None:
            if not os.path.isfile(data_path) and not os.path.isdir(data_path):
                raise NoDataError(msg=f'The data_path {data_path} is not a valid file-path or directory', source=self.__class__.__name__)
            elif os.path.isdir(data_path):
                self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=[f'.{self.file_type}'], as_dict=False, raise_error=False, raise_warning=True)
            else:
                self.data_paths = [data_path]
        else:
            data_path = self.features_dir
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=[f'.{self.file_type}'], as_dict=False, raise_error=False, raise_warning=True)
        if len(self.data_paths) == 0:
            raise NoDataError(msg=f'The data_path {data_path} has no valid data files', source=self.__class__.__name__)
        if agg_save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(agg_save_path))
        else:
            agg_save_path = os.path.join(self.logs_path, f"Conditional_aggregate_statistics_{self.datetime}.csv")
        if detailed_save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(detailed_save_path))
        else:
            detailed_save_path = os.path.join(self.logs_path, f"Detailed_conditional_aggregate_statistics_{self.datetime}.csv")
        self.agg_save_path, self.detailed_save_path, self.rules = agg_save_path, detailed_save_path, rules
        self.output_df = pd.DataFrame(columns=["VIDEO"] + list(self.rules.keys()) + ["TIME (s)", "FRAMES (count)"])
        self.bout_df_cols = ["VIDEO"] + list(self.rules.keys()) + ["START FRAME", "END FRAME", "START TIME", "END TIME" ,"BOUT TIME"]
        self.bout_dfs, self.rule_cols, self.verbose = [], list(self.rules.keys()), verbose
        self.rules = {k: str_2_bool(v) for k, v in self.rules.items()}


    def _slice_df(self, df: pd.DataFrame, rules: dict) -> pd.DataFrame:
        sliced_df = deepcopy(df)
        for k, v in rules.items():
            sliced_df = sliced_df[sliced_df[k] == 1] if v else sliced_df[sliced_df[k] == 0]
        return sliced_df



    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        for file_cnt, file_path in enumerate(self.data_paths):
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            if self.verbose: print(f'Analyzing conditional boolean statistics in {self.video_name}...({file_cnt+1}/{len(self.data_paths)})')
            _, _, self.fps = read_video_info(vid_info_df=self.video_info_df, video_name=self.video_name)
            self.df = read_df(file_path=file_path, file_type=self.file_type)
            check_valid_dataframe(df=self.df, source=file_path, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.rule_cols)
            for rule_col in self.rule_cols: check_if_df_field_is_boolean(df=self.df, field=rule_col, df_name=file_path)
            self.sliced_df = self._slice_df(df=self.df, rules=self.rules)
            time_s = round(len(self.sliced_df) / self.fps, 4)
            if len(self.sliced_df) > 0:
                bout_df = pd.DataFrame(data=np.zeros((len(self.df))), columns=['behavior'])
                bout_df.iloc[self.sliced_df.index] = 1
                bout_df = detect_bouts(data_df=bout_df, target_lst=['behavior'], fps=self.fps)
                bout_df = bout_df.assign(**{k: v for k, v in self.rules.items()})
                bout_df['VIDEO'] = self.video_name
                bout_df = bout_df.rename(columns={'Start_time': 'START TIME', 'End Time': 'END TIME', 'Start_frame': 'START FRAME', 'End_frame': 'END FRAME', 'Bout_time': 'BOUT TIME'})
                self.bout_dfs.append(bout_df[self.bout_df_cols])
            self.output_df.loc[len(self.output_df)] = ([self.video_name] + list(self.rules.values()) + [time_s] + [len(self.sliced_df)])


    def save(self):
        self.output_df.to_csv(self.agg_save_path, index=False)
        self.timer.stop_timer()
        stdout_success(msg=f"Boolean conditional data saved at at {self.agg_save_path}!", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)
        if len(self.bout_dfs) > 0:
            self.bout_dfs = pd.concat(self.bout_dfs, axis=0).reset_index(drop=True)
            self.bout_dfs.to_csv(self.detailed_save_path, index=False)
            stdout_success(msg=f"Detailed boolean conditional data saved at at {self.detailed_save_path}!", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


#'Stimulus 2 Animal_1 in zone', 'Stimulus 2 Animal_1 facing'
# rules = {'Stimulus 2 Animal_1 in zone': True, 'Stimulus 6 Animal_1 in zone': 'falsE'}
# runner = BooleanConditionalCalculator(rules=rules, config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini", data_path=r'C:\troubleshooting\RAT_NOR\project_folder\csv\features_extracted')
# runner.run()
# runner.save()
#



