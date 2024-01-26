__author__ = "Simon Nilsson"

import glob
import itertools
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.errors import BodypartColumnNotFoundError, NoFilesFoundError
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_config_entry, read_df
from simba.utils.warnings import NoDataFoundWarning


class ROIAnalyzer(ConfigReader, FeatureExtractionMixin):
    """

    Analyze movements, entries, exits, and time-spent-in user-defined ROIs. Results are stored in the
    'project_folder/logs' directory of the SimBA project.

    :param str ini_path: Path to SimBA project config file in Configparser format.
    :param Optional[str] data_path: Path to folder holding the data used to caluclate ROI aggregate statistics. If None, then `project_folder/
        csv/outlier_corrected_movement_location`. Deafult: None.
    :param Optional[dict] settings: If dict, the animal body-parts and the probability threshold. If None, then the data is read from the
        project_config.ini. Defalt: None.
    :param Optional[bool] calculate_distances: If True, then calculate movements aggregate statistics (distances and velocities) inside ROIs. Results
                                               are saved in ``project_folder/logs/`` directory. Default: False.

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> settings = {'body_parts': {'Simon': 'Ear_left_1', 'JJ': 'Ear_left_2'}, 'threshold': 0.4}
    >>> roi_analyzer = ROIAnalyzer(ini_path='MyProjectConfig', data_path='outlier_corrected_movement_location', settings=settings, calculate_distances=True)
    >>> roi_analyzer.run()
    >>> roi_analyzer.save()
    """

    def __init__(
        self,
        ini_path: str,
        data_path: Optional[str] = None,
        settings: Optional[dict] = None,
        calculate_distances: Optional[bool] = False,
    ):
        ConfigReader.__init__(self, config_path=ini_path)
        FeatureExtractionMixin.__init__(self)
        self.calculate_distances, self.settings = calculate_distances, settings
        if not os.path.exists(self.detailed_roi_data_dir):
            os.makedirs(self.detailed_roi_data_dir)
        if data_path != None:
            self.input_folder = os.path.join(self.project_path, "csv", data_path)
            self.files_found = glob.glob(self.input_folder + "/*." + self.file_type)
            if len(self.files_found) == 0:
                raise NoFilesFoundError(
                    msg=f"No data files found in {self.input_folder}"
                )

        if not self.settings:
            self.roi_config = dict(self.config.items(ConfigKey.ROI_SETTINGS.value))
            if "animal_1_bp" not in self.roi_config.keys():
                raise BodypartColumnNotFoundError(msg="Please analyze ROI data FIRST.")
            self.settings = {}
            self.settings["threshold"] = read_config_entry(
                self.config,
                ConfigKey.ROI_SETTINGS.value,
                ConfigKey.PROBABILITY_THRESHOLD.value,
                Dtypes.FLOAT.value,
                0.00,
            )
            self.settings["body_parts"] = {}
            self.__check_that_roi_config_data_is_valid()
            for animal_name, bp in self.roi_bp_config.items():
                self.settings["body_parts"][animal_name] = bp

        self.body_part_to_animal_lookup = {}
        for animal_cnt, body_part_name in self.settings["body_parts"].items():
            animal_name = self.find_animal_name_from_body_part_name(
                bp_name=body_part_name, bp_dict=self.animal_bp_dict
            )
            self.body_part_to_animal_lookup[animal_cnt] = animal_name

        self.bp_dict, self.bp_names = {}, []
        for animal_name, bp in self.settings["body_parts"].items():
            self.bp_dict[animal_name] = []
            self.bp_dict[animal_name].extend(
                [f'{bp}_{"x"}', f'{bp}_{"y"}', f'{bp}_{"p"}']
            )
            self.bp_names.extend([f'{bp}_{"x"}', f'{bp}_{"y"}', f'{bp}_{"p"}'])
        self.read_roi_data()

    def __check_that_roi_config_data_is_valid(self):
        all_bps = list(set([x[:-2] for x in self.bp_headers]))
        self.roi_bp_config = {}
        for k, v in self.roi_config.items():
            if "".join([i for i in k if not i.isdigit()]) == "animal__bp":
                id = int("".join(c for c in k if c.isdigit())) - 1
                try:
                    self.roi_bp_config[self.multi_animal_id_list[id]] = v
                except:
                    pass
        for animal, bp in self.roi_bp_config.items():
            if bp not in all_bps:
                raise BodypartColumnNotFoundError(
                    msg=f"Project config setting [{ConfigKey.ROI_SETTINGS.value}][{animal}] is not a valid body-part. Please make sure you have analyzed ROI data."
                )

    def __get_bouts(self, lst=None):
        lst = list(lst)
        return lst[0], lst[-1]

    def run(self):
        """
        Method to analyze ROI statistics.

        Returns
        -------
        Attribute: list
            dist_lst, list of pd.DataFrame holding ROI-dependent movement statistics.
        """
        (
            self.time_dict,
            self.entries_dict,
            self.entries_exit_dict,
            self.movement_dict,
        ) = ({}, {}, {}, {})
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(file_path)
            (
                self.time_dict[video_name],
                self.entries_dict[video_name],
                self.entries_exit_dict[video_name],
            ) = ({}, {}, {})
            print("Analysing ROI data for video {}...".format(video_name))
            self.video_recs = self.rectangles_df.loc[
                self.rectangles_df["Video"] == video_name
            ]
            self.video_circs = self.circles_df.loc[
                self.circles_df["Video"] == video_name
            ]
            self.video_polys = self.polygon_df.loc[
                self.polygon_df["Video"] == video_name
            ]
            video_shapes = list(
                itertools.chain(
                    self.video_recs["Name"].unique(),
                    self.video_circs["Name"].unique(),
                    self.video_polys["Name"].unique(),
                )
            )

            if video_shapes == 0:
                NoDataFoundWarning(
                    msg=f"Skipping video {video_name}: No user-defined ROI data found for this video..."
                )
                continue

            else:
                video_settings, pix_per_mm, self.fps = self.read_video_info(
                    video_name=video_name
                )
                self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
                self.data_df.columns = self.bp_headers
                data_df_sliced = self.data_df[self.bp_names]
                self.video_length_s = data_df_sliced.shape[0] / self.fps
                for animal_name in self.bp_dict:
                    animal_df = self.data_df[self.bp_dict[animal_name]]
                    (
                        self.time_dict[video_name][animal_name],
                        self.entries_dict[video_name][animal_name],
                    ) = ({}, {})
                    self.entries_exit_dict[video_name][animal_name] = {}
                    for _, row in self.video_recs.iterrows():
                        top_left_x, top_left_y, shape_name = (
                            row["topLeftX"],
                            row["topLeftY"],
                            row["Name"],
                        )
                        self.entries_exit_dict[video_name][animal_name][shape_name] = {}
                        bottom_right_x, bottom_right_y = (
                            row["Bottom_right_X"],
                            row["Bottom_right_Y"],
                        )
                        slice_x = animal_df[
                            animal_df[self.bp_dict[animal_name][0]].between(
                                top_left_x, bottom_right_x, inclusive=True
                            )
                        ]
                        slice_y = slice_x[
                            slice_x[self.bp_dict[animal_name][1]].between(
                                top_left_y, bottom_right_y, inclusive=True
                            )
                        ]
                        slice = (
                            slice_y[
                                slice_y[self.bp_dict[animal_name][2]]
                                >= self.settings["threshold"]
                            ]
                            .reset_index()
                            .rename(columns={"index": "frame_no"})
                        )
                        bouts = [
                            self.__get_bouts(g)
                            for _, g in itertools.groupby(
                                list(slice["frame_no"]),
                                key=lambda n, c=itertools.count(): n - next(c),
                            )
                        ]
                        self.time_dict[video_name][animal_name][shape_name] = round(
                            len(slice) / self.fps, 3
                        )
                        self.entries_dict[video_name][animal_name][shape_name] = len(
                            bouts
                        )
                        self.entries_exit_dict[video_name][animal_name][shape_name][
                            "Entry_times"
                        ] = list(map(lambda x: x[0], bouts))
                        self.entries_exit_dict[video_name][animal_name][shape_name][
                            "Exit_times"
                        ] = list(map(lambda x: x[1], bouts))

                    for _, row in self.video_circs.iterrows():
                        center_x, center_y, radius, shape_name = (
                            row["centerX"],
                            row["centerY"],
                            row["radius"],
                            row["Name"],
                        )
                        self.entries_exit_dict[video_name][animal_name][shape_name] = {}
                        animal_df["distance"] = np.sqrt(
                            (animal_df[self.bp_dict[animal_name][0]] - center_x) ** 2
                            + (animal_df[self.bp_dict[animal_name][1]] - center_y) ** 2
                        )
                        slice = (
                            animal_df.loc[
                                (animal_df["distance"] <= radius)
                                & (
                                    animal_df[self.bp_dict[animal_name][2]]
                                    >= self.settings["threshold"]
                                )
                            ]
                            .reset_index()
                            .rename(columns={"index": "frame_no"})
                        )
                        bouts = [
                            self.__get_bouts(g)
                            for _, g in itertools.groupby(
                                list(slice["frame_no"]),
                                key=lambda n, c=itertools.count(): n - next(c),
                            )
                        ]
                        self.time_dict[video_name][animal_name][shape_name] = round(
                            len(slice) / self.fps, 3
                        )
                        self.entries_dict[video_name][animal_name][shape_name] = len(
                            bouts
                        )
                        self.entries_exit_dict[video_name][animal_name][shape_name][
                            "Entry_times"
                        ] = list(map(lambda x: x[0], bouts))
                        self.entries_exit_dict[video_name][animal_name][shape_name][
                            "Exit_times"
                        ] = list(map(lambda x: x[1], bouts))

                    for _, row in self.video_polys.iterrows():
                        polygon_shape, shape_name = (
                            Polygon(
                                list(zip(row["vertices"][:, 0], row["vertices"][:, 1]))
                            ),
                            row["Name"],
                        )
                        self.entries_exit_dict[video_name][animal_name][shape_name] = {}
                        points_arr = animal_df[
                            [self.bp_dict[animal_name][0], self.bp_dict[animal_name][1]]
                        ].to_numpy()
                        contains_func = np.vectorize(
                            lambda p: polygon_shape.contains(Point(p)),
                            signature="(n)->()",
                        )
                        inside_frame_no = [
                            j
                            for sub in np.argwhere(contains_func(points_arr))
                            for j in sub
                        ]
                        slice = (
                            animal_df.loc[
                                (animal_df.index.isin(inside_frame_no))
                                & (
                                    animal_df[self.bp_dict[animal_name][2]]
                                    >= self.settings["threshold"]
                                )
                            ]
                            .reset_index()
                            .rename(columns={"index": "frame_no"})
                        )
                        bouts = [
                            self.__get_bouts(g)
                            for _, g in itertools.groupby(
                                list(slice["frame_no"]),
                                key=lambda n, c=itertools.count(): n - next(c),
                            )
                        ]
                        self.time_dict[video_name][animal_name][shape_name] = round(
                            len(slice) / self.fps, 3
                        )
                        self.entries_dict[video_name][animal_name][shape_name] = len(
                            bouts
                        )
                        self.entries_exit_dict[video_name][animal_name][shape_name][
                            "Entry_times"
                        ] = list(map(lambda x: x[0], bouts))
                        self.entries_exit_dict[video_name][animal_name][shape_name][
                            "Exit_times"
                        ] = list(map(lambda x: x[1], bouts))

                if self.calculate_distances:
                    self.movement_dict[video_name] = {}
                    for animal, shape_dicts in self.entries_exit_dict[
                        video_name
                    ].items():
                        self.movement_dict[video_name][animal] = {}
                        for shape_name, shape_data in shape_dicts.items():
                            d = pd.DataFrame.from_dict(
                                shape_data, orient="index"
                            ).T.values.tolist()
                            for entry in d:
                                df = self.data_df[self.bp_dict[animal][0:2]][
                                    self.data_df.index.isin(
                                        list(range(entry[0], entry[1] + 1))
                                    )
                                ]
                                df = self.create_shifted_df(df=df)
                                df["Movement"] = (
                                    np.sqrt(
                                        (df.iloc[:, 0] - df.iloc[:, 2]) ** 2
                                        + (df.iloc[:, 1] - df.iloc[:, 3]) ** 2
                                    )
                                ) / pix_per_mm
                                self.movement_dict[video_name][animal][shape_name] = (
                                    df["Movement"].sum() / 10
                                )
            self.__transpose_dicts_to_dfs()

    def compute_framewise_distance_to_roi_centroids(self):
        """
        Method to compute frame-wise distances between ROI centroids and animal body-parts.

        Returns
        -------
        Attribute: dict
            roi_centroid_distance
        """

        self.roi_centroid_distance = {}
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(file_path)
            self.roi_centroid_distance[video_name] = {}
            video_recs = self.rectangles_df.loc[
                self.rectangles_df["Video"] == video_name
            ]
            video_circs = self.circles_df.loc[self.circles_df["Video"] == video_name]
            video_polys = self.polygon_df.loc[self.polygon_df["Video"] == video_name]
            data_df = read_df(file_path, self.file_type).reset_index(drop=True)
            data_df.columns = self.bp_headers
            for animal_name in self.bp_dict:
                self.roi_centroid_distance[video_name][animal_name] = {}
                animal_df = data_df[self.bp_dict[animal_name]]
                for _, row in video_recs.iterrows():
                    center_cord = (
                        (
                            int(
                                row["Bottom_right_Y"]
                                - ((row["Bottom_right_Y"] - row["topLeftY"]) / 2)
                            )
                        ),
                        (
                            int(
                                row["Bottom_right_X"]
                                - ((row["Bottom_right_X"] - row["topLeftX"]) / 2)
                            )
                        ),
                    )
                    self.roi_centroid_distance[video_name][animal_name][row["Name"]] = (
                        np.sqrt(
                            (animal_df[self.bp_dict[animal_name][0]] - center_cord[0])
                            ** 2
                            + (animal_df[self.bp_dict[animal_name][1]] - center_cord[1])
                            ** 2
                        )
                    )

                for _, row in video_circs.iterrows():
                    center_cord = (row["centerX"], row["centerY"])
                    self.roi_centroid_distance[video_name][animal_name][row["Name"]] = (
                        np.sqrt(
                            (animal_df[self.bp_dict[animal_name][0]] - center_cord[0])
                            ** 2
                            + (animal_df[self.bp_dict[animal_name][1]] - center_cord[1])
                            ** 2
                        )
                    )

                for _, row in video_polys.iterrows():
                    polygon_shape = Polygon(
                        list(zip(row["vertices"][:, 0], row["vertices"][:, 1]))
                    )
                    center_cord = polygon_shape.centroid.coords[0]
                    self.roi_centroid_distance[video_name][animal_name][row["Name"]] = (
                        np.sqrt(
                            (animal_df[self.bp_dict[animal_name][0]] - center_cord[0])
                            ** 2
                            + (animal_df[self.bp_dict[animal_name][1]] - center_cord[1])
                            ** 2
                        )
                    )

    def __transpose_dicts_to_dfs(self):
        self.entries_df = pd.DataFrame(columns=["VIDEO", "ANIMAL", "SHAPE", "ENTRIES"])
        for video_name, video_data in self.entries_dict.items():
            for animal_name, animal_data in video_data.items():
                for shape_name, shape_data in animal_data.items():
                    self.entries_df.loc[len(self.entries_df)] = [
                        video_name,
                        animal_name,
                        shape_name,
                        shape_data,
                    ]
        self.entries_df["ANIMAL"] = self.entries_df["ANIMAL"].map(
            self.body_part_to_animal_lookup
        )

        self.time_df = pd.DataFrame(columns=["VIDEO", "ANIMAL", "SHAPE", "TIME"])
        for video_name, video_data in self.time_dict.items():
            for animal_name, animal_data in video_data.items():
                for shape_name, shape_data in animal_data.items():
                    self.time_df.loc[len(self.time_df)] = [
                        video_name,
                        animal_name,
                        shape_name,
                        shape_data,
                    ]
        self.time_df["ANIMAL"] = self.time_df["ANIMAL"].map(
            self.body_part_to_animal_lookup
        )

        self.detailed_df = pd.DataFrame(
            columns=[
                "VIDEO",
                "ANIMAL",
                "BODY-PART",
                "SHAPE",
                "ENTRY FRAMES",
                "EXIT FRAMES",
            ]
        )
        for video_name, video_data in self.entries_exit_dict.items():
            for animal, animal_data in video_data.items():
                body_part = self.settings["body_parts"][animal]
                for shape_name, shape_data in animal_data.items():
                    df = pd.DataFrame.from_dict(shape_data).rename(
                        columns={
                            "Entry_times": "ENTRY FRAMES",
                            "Exit_times": "EXIT FRAMES",
                        }
                    )
                    df["VIDEO"] = video_name
                    df["ANIMAL"] = animal
                    df["BODY-PART"] = body_part
                    df["SHAPE"] = shape_name
                    self.detailed_df = pd.concat([self.detailed_df, df], axis=0)
        self.detailed_df["ANIMAL"] = self.detailed_df["ANIMAL"].map(
            self.body_part_to_animal_lookup
        )
        self.detailed_df = self.detailed_df[
            ["VIDEO", "ANIMAL", "BODY-PART", "SHAPE", "ENTRY FRAMES", "EXIT FRAMES"]
        ]

        if self.calculate_distances:
            self.movements_df = pd.DataFrame(
                columns=["VIDEO", "ANIMAL", "SHAPE", "MOVEMENT INSIDE SHAPE (CM)"]
            )
            for video_name, video_data in self.movement_dict.items():
                for animal_name, animal_data in video_data.items():
                    for shape_name, shape_data in animal_data.items():
                        self.movements_df.loc[len(self.movements_df)] = [
                            video_name,
                            animal_name,
                            shape_name,
                            shape_data,
                        ]
            self.movements_df["ANIMAL"] = self.movements_df["ANIMAL"].map(
                self.body_part_to_animal_lookup
            )

    def save(self):
        """
        Method to save ROI data to disk. ROI latency and ROI entry data is saved in the "project_folder/logs/" directory.
        If ``calculate_distances`` is True, ROI movement data is saved in the "project_folder/logs/" directory.

        Returns
        -------
        None
        """

        self.entries_df.to_csv(
            os.path.join(self.logs_path, f'{"ROI_entry_data"}_{self.datetime}.csv')
        )
        self.time_df.to_csv(
            os.path.join(self.logs_path, f'{"ROI_time_data"}_{self.datetime}.csv')
        )
        self.detailed_df.to_csv(
            os.path.join(self.logs_path, f'{"Detailed_ROI_data"}_{self.datetime}.csv')
        )
        stdout_success(
            msg='ROI time, ROI entry, and Detailed ROI data, have been saved in the "project_folder/logs/" directory in CSV format.'
        )
        if self.calculate_distances:
            self.movements_df.to_csv(
                os.path.join(
                    self.logs_path, f'{"ROI_movement_data"}_{self.datetime}.csv'
                )
            )
            stdout_success(
                msg='ROI movement data saved in the "project_folder/logs/" directory'
            )

        self.timer.stop_timer()
        stdout_success(
            msg="ROI analysis complete", elapsed_time=self.timer.elapsed_time_str
        )


# test = ROIAnalyzer(ini_path = r"/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/zebrafish/project_folder/project_config.ini",
#                    data_path = "outlier_corrected_movement_location",
#                    calculate_distances=True)


# settings = {'body_parts': {'animal_1_bp': 'Ear_left_1', 'animal_2_bp': 'Ear_left_2', 'animal_3_bp': 'Ear_right_1',}, 'threshold': 0.4}
# test = ROIAnalyzer(ini_path = r"/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini",
#                    data_path = "outlier_corrected_movement_location",
#                    settings=settings,
#                    calculate_distances=True)
# test.run()
# test.save()


# settings = {'body_parts': {'Simon': 'Ear_left_1', 'JJ': 'Ear_left_2'}, 'threshold': 0.4}
# test = ROIAnalyzer(ini_path = r"/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                    data_path = "outlier_corrected_movement_location",
#                    settings=settings,
#                    calculate_distances=True)
# test.read_roi_dfs()
# test.analyze_ROIs()
# test.save_data()


# settings = {'body_parts': {'animal_1_bp': 'Ear_left_1', 'animal_2_bp': 'Ear_left_2'}, 'threshold': 0.4}
# test = ROIAnalyzer(ini_path = r"/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                    data_path = "outlier_corrected_movement_location",
#                    calculate_distances=True)
# test.run()
# test.analyze_ROIs()
# test.save_data()
