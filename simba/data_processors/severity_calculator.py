__author__ = "Simon Nilsson"

import os
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numba import jit

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df
from simba.utils.warnings import NoDataFoundWarning


class SeverityCalculator(ConfigReader):
    """
    Computes the "severity" of classification frame events based on how much
    the animals are moving. Frames are scored as less or more severe at lower and higher movements, respectively.

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter dict settings: how to calculate the severity. E.g., {'brackets': 10, 'clf': 'Attack', 'animals': ['Simon', 'JJ'], 'time': True, 'frames': False}.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md>`__.

    Examples
    ----------
    >>> settings = {'brackets': 10, 'clf': 'Attack', 'animals': ['Simon', 'JJ'], 'time': True, 'frames': False}
    >>> processor = SeverityCalculator(config_path='project_folder/project_config.ini', settings=settings)
    >>> processor.run()
    >>> processor.save()
    """

    def __init__(self, config_path: Union[str, os.PathLike], settings: Dict):
        ConfigReader.__init__(self, config_path=config_path)
        self.settings = settings
        check_if_filepath_list_is_empty(
            filepaths=self.machine_results_paths,
            error_msg=f"SIMBA ERROR: Cannot process severity. {self.machine_results_dir} directory is empty",
        )
        save_name = os.path.join(
            f'severity_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
        )
        self.save_path = os.path.join(self.logs_path, save_name)
        self.results = {}

    @staticmethod
    @jit(nopython=True)
    def __euclidean_distance(bp_1_x_vals, bp_2_x_vals, bp_1_y_vals, bp_2_y_vals):
        return np.sqrt(
            (bp_1_x_vals - bp_2_x_vals) ** 2 + (bp_1_y_vals - bp_2_y_vals) ** 2
        )

    def run(self):
        for file_path in self.machine_results_paths:
            _, video_name, _ = get_fn_ext(file_path)
            self.results[video_name] = {}
            df = read_df(file_path=file_path, file_type=self.file_type)
            if self.settings["clf"] not in df.columns:
                NoDataFoundWarning(
                    msg=f'Skipping file {video_name} - {self.settings["clf"]} data not present in file'
                )
                continue
            _, _, fps = self.read_video_info(video_name=video_name)
            for animal_name, animal_bodyparts in self.animal_bp_dict.items():
                animal_df = df[animal_bodyparts["X_bps"] + animal_bodyparts["Y_bps"]]
                shifted = animal_df.shift(periods=1).fillna(0)
                movement = pd.DataFrame()
                for bp_x, bp_y in zip(
                    animal_bodyparts["X_bps"], animal_bodyparts["Y_bps"]
                ):
                    movement[bp_x.rstrip("_x")] = self.__euclidean_distance(
                        animal_df[bp_x].values,
                        shifted[bp_x].values,
                        animal_df[bp_y].values,
                        shifted[bp_y].values,
                    )
                movement["sum"] = movement.sum(axis=1)
                movement["sum"].iloc[0] = 0
                df[animal_name] = movement["sum"]
            df["movement"] = df[self.settings["animals"]].sum(axis=1)
            df["bin"] = pd.qcut(
                x=df["movement"],
                q=self.settings["brackets"],
                labels=list(range(1, self.settings["brackets"] + 1)),
            )
            clf_df = (
                df["bin"][df[self.settings["clf"]] == 1]
                .astype(int)
                .reset_index(drop=True)
            )
            for i in range(0, self.settings["brackets"]):
                if self.settings["frames"]:
                    self.results[video_name][f"Grade {str(i + 1)} (frames)"] = len(
                        clf_df[clf_df == i]
                    )
                if self.settings["time"]:
                    self.results[video_name][f"Grade {str(i + 1)} (s)"] = round(
                        (len(clf_df[clf_df == i]) / fps), 4
                    )

    def save(self):
        out_df = pd.DataFrame(columns=["VIDEO", "MEASUREMENT", "VALUE"])
        for video_name, video_data in self.results.items():
            for grade, grade_data in video_data.items():
                out_df.loc[len(out_df)] = [video_name, grade, grade_data]
        out_df.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(
            msg=f"Severity data saved at {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# settings = {'brackets': 10, 'clf': 'Attack', 'animals': ['Simon', 'JJ'], 'time': True, 'frames': True}
# processor = SeverityCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', settings=settings)
# processor.run()
# processor.save()
