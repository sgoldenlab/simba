__author__ = "Simon Nilsson"

import os
from statistics import mean
from typing import List, Optional

import numpy as np
import pandas as pd

from simba.feature_extractors.perimeter_jit import jitted_centroid
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_that_column_exist)
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df


class MovementCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Compute aggregate movement statistics from pose-estimation data in SimBA project.

    :parameters str config_path: path to SimBA project config file in Configparser format
    :param List[str] body_parts: Body-parts to use for movement calculations OR ``Animal_name CENTER OF GRAVITY``. If ``Animal_name CENTER OF GRAVITY``, then SimBA will approximate animal centroids through convex hull.
    :param float threshold: Filter body-part detection below set threshold (Value 0-1). Default: 0.00
    :param List[str] or None file_paths: Files to calucalte movements for. If None, then all files in ``project_folder/csv/outlier_corrected_movement_location`` directory.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

    :examples:
    >>> body_parts=['Animal_1 CENTER OF GRAVITY']
    >>> movement_processor = MovementCalculator(config_path='project_folder/project_config.ini', body_parts=body_parts)
    >>> movement_processor.run()
    >>> movement_processor.save()

    """

    def __init__(
        self,
        config_path: str,
        body_parts: List[str],
        threshold: float = 0.00,
        file_paths: Optional[List[str]] = None,
    ):
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self)
        self.save_path = os.path.join(
            self.logs_path, "Movement_log_{}.csv".format(self.datetime)
        )
        self.file_paths, self.body_parts, self.threshold = (
            file_paths,
            body_parts,
            threshold,
        )
        if not self.file_paths:
            check_if_filepath_list_is_empty(
                filepaths=self.outlier_corrected_paths,
                error_msg=f"SIMBA ERROR: Cannot process movement. ZERO data files found in the {self.outlier_corrected_dir} directory.",
            )
            self.file_paths = self.outlier_corrected_paths
        print(f"Processing {len(self.file_paths)} video(s)...")

    def __find_body_part_columns(self):
        self.body_parts_dict, self.bp_list = {}, []
        for bp_cnt, bp_name in enumerate(self.body_parts):
            if not bp_name.endswith("CENTER OF GRAVITY"):
                animal_name = self.find_animal_name_from_body_part_name(
                    bp_name=bp_name, bp_dict=self.animal_bp_dict
                )
                self.body_parts_dict[bp_cnt] = {
                    "ANIMAL NAME": animal_name,
                    "BODY-PART": bp_name,
                    "BODY-PART HEADERS": [
                        f"{bp_name}_x",
                        f"{bp_name}_y",
                        f"{bp_name}_p",
                    ],
                }
                self.bp_list.extend((self.body_parts_dict[bp_cnt]["BODY-PART HEADERS"]))
            else:
                pass

    def __find_polygons(self, data):
        print(data.shape)

    def run(self):
        self.results = pd.DataFrame(
            columns=["VIDEO", "ANIMAL", "BODY-PART", "MEASURE", "VALUE"]
        )
        self.movement_dfs = {}
        for file_path in self.file_paths:
            self.__find_body_part_columns()
            _, video_name, _ = get_fn_ext(file_path)
            print(f"Analysing {video_name}...")
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=video_name)
            self.movement_dfs[video_name] = pd.DataFrame()
            if self.bp_list:
                check_that_column_exist(df=self.data_df, column_name=self.bp_list, file_name=file_path)
                self.data_df = self.data_df[self.bp_list]
                for animal_cnt, animal_data in self.body_parts_dict.items():
                    animal_df = self.data_df[animal_data["BODY-PART HEADERS"]]
                    if self.threshold > 0.00:
                        animal_df = animal_df[
                            animal_df[animal_data["BODY-PART HEADERS"][-1]]
                            >= self.threshold
                        ]
                    animal_df = animal_df.iloc[:, 0:2].reset_index(drop=True)
                    animal_df = self.create_shifted_df(df=animal_df)
                    bp_time_1 = animal_df[
                        [
                            animal_data["BODY-PART HEADERS"][0],
                            animal_data["BODY-PART HEADERS"][1],
                        ]
                    ].values.astype(float)
                    bp_time_2 = animal_df[
                        [
                            animal_data["BODY-PART HEADERS"][0] + "_shifted",
                            animal_data["BODY-PART HEADERS"][1] + "_shifted",
                        ]
                    ].values.astype(float)
                    self.movement = pd.Series(
                        self.framewise_euclidean_distance(
                            location_1=bp_time_1,
                            location_2=bp_time_2,
                            px_per_mm=self.px_per_mm,
                        )
                    )
                    self.movement.loc[0] = 0
                    self.movement_dfs[video_name][
                        f'{animal_data["ANIMAL NAME"]} {animal_data["BODY-PART"]}'
                    ] = self.movement
                    distance = round((self.movement.sum() / 10), 4)
                    velocity_lst = []
                    for df in np.array_split(
                        self.movement, int(len(self.movement) / self.fps)
                    ):
                        velocity_lst.append(df.sum())
                    self.results.loc[len(self.results)] = [
                        video_name,
                        animal_data["ANIMAL NAME"],
                        animal_data["BODY-PART"],
                        "Distance (cm)",
                        distance,
                    ]
                    self.results.loc[len(self.results)] = [
                        video_name,
                        animal_data["ANIMAL NAME"],
                        animal_data["BODY-PART"],
                        "Velocity (cm/s)",
                        round((mean(velocity_lst) / 10), 4),
                    ]

            else:
                for animal in self.body_parts:
                    animal_name = animal.split("CENTER OF GRAVITY")[0].strip()
                    x, y = (
                        self.data_df[self.animal_bp_dict[animal_name]["X_bps"]],
                        self.data_df[self.animal_bp_dict[animal_name]["Y_bps"]],
                    )
                    z = pd.concat([x, y], axis=1)[
                        [item for items in zip(x.columns, y.columns) for item in items]
                    ]
                    df = pd.DataFrame(
                        jitted_centroid(
                            points=np.reshape(z.values, (len(z / 2), -1, 2)).astype(
                                np.float32
                            )
                        ),
                        columns=["X", "Y"],
                    )
                    df = self.dataframe_savgol_smoother(df=df, fps=self.fps).astype(int)
                    df_shifted = df.shift(1)
                    df_shifted = df_shifted.combine_first(df).add_suffix("_shifted")
                    self.movement = pd.Series(
                        self.framewise_euclidean_distance(
                            location_1=df.values.astype(np.float32),
                            location_2=df_shifted.values.astype(np.float32),
                            px_per_mm=self.px_per_mm,
                        )
                    )
                    self.movement.loc[0] = 0
                    self.movement_dfs[video_name][
                        f'{animal_name} {"GRAVITY CENTER"}'
                    ] = self.movement
                    distance = round((self.movement.sum() / 10), 4)
                    velocity_lst = []
                    for df in np.array_split(
                        self.movement, int(len(self.movement) / self.fps)
                    ):
                        velocity_lst.append(df.sum())
                    self.results.loc[len(self.results)] = [
                        video_name,
                        animal_name,
                        "GRAVITY CENTER",
                        "Distance (cm)",
                        distance,
                    ]
                    self.results.loc[len(self.results)] = [
                        video_name,
                        animal_name,
                        "GRAVITY CENTER",
                        "Velocity (cm/s)",
                        round((mean(velocity_lst) / 10), 4),
                    ]

    def save(self):
        self.results.set_index("VIDEO").to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(
            msg=f"Movement log saved in {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = MovementCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                           body_parts=['Simon CENTER OF GRAVITY'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()


# test.save()

# test = MovementCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini',
#                           body_parts=['Animal_1 CENTER OF GRAVITY'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()
# test.save()
