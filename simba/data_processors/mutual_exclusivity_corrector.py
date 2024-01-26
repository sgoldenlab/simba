import glob
import os
import shutil
from copy import deepcopy
from typing import Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_that_column_exist)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df
from simba.utils.warnings import IdenticalInputWarning

SUBORDINATES = "subordinates"
WINNER = "winner"
TIE_BREAKER = "tie_breaker"
THRESHOLD = "threshold"
THRESHOLD_DETERMINATOR = "clf_threshold"
HIGHEST_PROBABILITY = "highest_probability"
SKIP_FILES_WITH_IDENTICAL = "skip_files_with_identical"


class MutualExclusivityCorrector(ConfigReader):
    """
    Refactor classification results according to user-defined mutual exclusivity rules.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/mutual_exclusivity_heuristic_rules.md>`__.


    Examples
    ----------
    >>> rules = {1: {'rule_type': 'threshold_determinator','determinator': 'Attack', 'threshold': 0.5, 'subordinates': ['Sniffing']}, 2: {'rule_type': 'threshold_determinator', 'determinator': 'Attack', 'threshold': 0.0, 'subordinates': ['Sniffing', 'Rear']}}
    >>> exclusivity_corrector = MutualExclusivityCorrector(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', rules=rules)
    >>> exclusivity_corrector.run()
    >>> rules = {1: {'rule_type': 'highest_probability', 'subordinates': ['body', 'face'], 'winner': 'body', 'skip_files_with_identical': True}}
    >>> exclusivity_corrector = MutualExclusivityCorrector(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', rules=rules)
    >>> exclusivity_corrector.run()
    """

    def __init__(self, rules: dict, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        self.rules, self.save_dir = rules, None
        check_if_filepath_list_is_empty(
            filepaths=self.machine_results_paths,
            error_msg=f"The {self.machine_results_dir} directory is empty",
        )
        self.save_dir = os.path.join(
            self.machine_results_dir, f"Prior_to_mutual_exclusivity_{self.datetime}"
        )

    def run(self):
        for file_cnt, file_path in enumerate(self.machine_results_paths):
            self.video_name = get_fn_ext(filepath=file_path)[1]
            video_timer = SimbaTimer(start=True)
            print(f"Analysing mutual exclusivity in {self.video_name}..")
            self.input_df = read_df(file_path=file_path, file_type=self.file_type)
            self.data_df = deepcopy(self.input_df)
            for rule_cnt, rule_data in self.rules.items():
                self.rule_cnt, self.rule_data = rule_cnt, rule_data
                if rule_data["rule_type"] == THRESHOLD_DETERMINATOR:
                    check_that_column_exist(
                        df=self.data_df,
                        column_name=rule_data[WINNER],
                        file_name=file_path,
                    )
                    check_that_column_exist(
                        df=self.data_df,
                        column_name=f"Probability_{rule_data[WINNER]}",
                        file_name=file_path,
                    )
                    for subordinate in rule_data[SUBORDINATES]:
                        check_that_column_exist(
                            df=self.data_df,
                            column_name=f"Probability_{subordinate}",
                            file_name=file_path,
                        )
                    self.threshold_determinator()

                if rule_data["rule_type"] == HIGHEST_PROBABILITY:
                    self.probability_cols = [
                        f"Probability_{x}" for x in rule_data[SUBORDINATES]
                    ]
                    for subordinate in rule_data[SUBORDINATES]:
                        check_that_column_exist(
                            df=self.data_df,
                            column_name=subordinate,
                            file_name=file_path,
                        )
                    self.highest_probability_determinator()

            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            shutil.move(
                file_path, os.path.join(self.save_dir, os.path.basename(file_path))
            )
            write_df(df=self.data_df, file_type=self.file_type, save_path=file_path)
            video_timer.stop_timer()
            stdout_success(
                msg=f"Mutual exclusivity complete video {self.video_name}. ({file_cnt+1}/{len(self.machine_results_paths)})",
                elapsed_time=video_timer.elapsed_time_str,
            )

        self.timer.stop_timer()
        stdout_success(
            msg=f"Mutual exclusivity performed on {len(self.machine_results_paths)} file(s). Results are saved in the project_folder/csv/machine_results directory. Copies of files PRIOR to applying mutual exclusivity rules are saved in {self.save_dir}",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def threshold_determinator(self):
        sums_df = self.data_df[
            self.rule_data[SUBORDINATES] + [self.rule_data[WINNER]]
        ].sum(axis=1)
        rows = list(
            sums_df.loc[(sums_df == len(self.rule_data[SUBORDINATES]) + 1)].index
        )
        rows = [
            x
            for x in rows
            if self.data_df.loc[x, f"Probability_{self.rule_data[WINNER]}"]
            > self.rule_data[THRESHOLD]
        ]
        for subordinate in self.rule_data[SUBORDINATES]:
            self.data_df.loc[rows, subordinate] = 0

    def highest_probability_determinator(self):
        sums = self.data_df[self.rule_data[SUBORDINATES]].sum(axis=1)
        overlap_df = self.data_df[self.probability_cols][
            self.data_df.index.isin(
                list(sums.index[sums == len(self.rule_data[SUBORDINATES])])
            )
        ]
        if len(overlap_df) > 0:
            identical_rows = list(
                overlap_df.index[overlap_df.apply(lambda x: min(x) == max(x), 1)]
            )
            if len(identical_rows) > 0 and self.rule_data[SKIP_FILES_WITH_IDENTICAL]:
                IdenticalInputWarning(
                    msg=f'Skipping rule {self.rule_cnt} for video {self.video_name} in frames {identical_rows}. Identical rows found and "SKIP ON EQUAL" selected...'
                )
                overlap_df = overlap_df.drop(index=identical_rows)
            elif (len(identical_rows) > 0) and not self.rule_data[
                SKIP_FILES_WITH_IDENTICAL
            ]:
                overlap_df.loc[
                    identical_rows, f"Probability_{self.rule_data[TIE_BREAKER]}"
                ] = (overlap_df[f"Probability_{self.rule_data[TIE_BREAKER]}"] + 1)
            overlap_array = (
                overlap_df.values == overlap_df.values.max(axis=1)[:, None]
            ).astype(int)
            results = pd.DataFrame(
                overlap_array,
                columns=self.rule_data[SUBORDINATES],
                index=overlap_df.index,
            )
            self.data_df.update(results)


# rules = {1: {'rule_type': 'clf_threshold',
#              'winner': 'Attack',
#              'threshold': 0.0,
#              'subordinates': ['Sniffing', 'Attack']},
#          2: {'rule_type': 'clf_threshold',
#              'winner': 'Sniffing',
#              'threshold': 0.0,
#              'subordinates': ['Sniffing', 'Rear', 'Attack']}}

# rules = {1: {'rule_type': 'clf_threshold',
#              'winner': 'Attack',
#              'threshold': 0,
#              'subordinates': ['Sniffing']},
#          2: {'rule_type': 'clf_threshold',
#              'winner': 'Sniffing',
#              'threshold': 0.0,
#              'subordinates': ['Rear', 'Attack']}}
#
# rules = {1: {'rule_type': 'highest_probability',
#              'tie_breaker': 'Attack',
#              'subordinates': ['Sniffing', 'Attack'],
#              'skip_files_with_identical': None},
#          2: {'rule_type': 'highest_probability',
#              'tie_breaker': 'Rear',
#              'subordinates': ['Rear', 'Attack'],
#              'skip_files_with_identical': None}}

# rules = {1: {'rule_type': 'highest_probability', 'subordinates': ['Sniffing', 'Attack'], 'tie_breaker': 'Attack', 'skip_files_with_identical': False}}
# test = MutualExclusivityCorrector(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', rules=rules)
# test.run()
