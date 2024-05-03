__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_float, check_that_column_exist,
    check_valid_lst)
from simba.utils.data import detect_bouts, slice_roi_dict_for_video
from simba.utils.enums import Keys
from simba.utils.errors import (CountError, MissingColumnsError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_data_paths, read_df
from simba.utils.warnings import NoDataFoundWarning


class ROIAnalyzer(ConfigReader, FeatureExtractionMixin):
    """
    Analyze movements, entries, exits, and time-spent-in user-defined ROIs. Results are stored in the
    'project_folder/logs' directory of the SimBA project.

    :param str config_path: Path to SimBA project config file in Configparser format.
    :param Optional[str] data_path: Path to folder or file holding the data used to calculate ROI aggregate statistics. If None, then defaults to the `project_folder/csv/outlier_corrected_movement_location` directory of the SimBA project. Default: None.
    :param Optional[bool] calculate_distances: If True, then calculate movements aggregate statistics (distances and velocities) inside ROIs. Results are saved in ``project_folder/logs/`` directory. Default: False.
    :param Optional[bool] detailed_bout_data: If True, saves a file with a row for every entry into each ROI for each animal in each video. Results are saved in ``project_folder/logs/`` directory. Default: False.
    :param Optional[float] threshold: Float between 0 and 1. Body-part locations detected below this confidence threshold are filtered. Default: 0.0.
    :param Optional[float] threshold: List of body-parts to perform ROI analysis on.

    .. note::
       `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    :example:
    >>> test = ROIAnalyzer(config_path = r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini", calculate_distances=True, detailed_bout_data=True, body_parts=['Nose_1', 'Nose_2'], threshold=0.0)
    >>> test.run()
    >>> test.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Optional[Union[str, os.PathLike, List[str]]] = None,
                 detailed_bout_data: Optional[bool] = False,
                 calculate_distances: Optional[bool] = False,
                 threshold: Optional[float] = 0.0,
                 body_parts: Optional[List[str]] = None):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        self.read_roi_data()
        FeatureExtractionMixin.__init__(self)
        if detailed_bout_data and (not os.path.exists(self.detailed_roi_data_dir)):
            os.makedirs(self.detailed_roi_data_dir)
        self.data_paths = read_data_paths(path=data_path,
                                          default=self.outlier_corrected_paths,
                                          default_name=self.outlier_corrected_dir,
                                          file_type=self.file_type)

        check_float(name="Body-part probability threshold", value=threshold, min_value=0.0, max_value=1.0)
        check_valid_lst(
            data=body_parts,
            source=f"{self.__class__.__name__} body-parts",
            valid_dtypes=(str,),
        )
        if len(set(body_parts)) != len(body_parts):
            raise CountError(
                msg=f"All body-part entries have to be unique. Got {body_parts}",
                source=self.__class__.__name__,
            )
        self.bp_dict, self.bp_lk = {}, {}
        for bp in body_parts:
            animal = self.find_animal_name_from_body_part_name(
                bp_name=bp, bp_dict=self.animal_bp_dict
            )
            self.bp_dict[animal] = [f'{bp}_{"x"}', f'{bp}_{"y"}', f'{bp}_{"p"}']
            self.bp_lk[animal] = bp
        self.roi_headers = [v for k, v in self.bp_dict.items()]
        self.roi_headers = [item for sublist in self.roi_headers for item in sublist]
        self.calculate_distances, self.threshold = calculate_distances, threshold
        self.detailed_bout_data = detailed_bout_data

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        self.movements_df = pd.DataFrame(columns=["VIDEO", "ANIMAL", "SHAPE", "MEASUREMENT", "VALUE"])
        self.entry_results = pd.DataFrame(columns=["VIDEO", "ANIMAL", "SHAPE", "ENTRY COUNT"])
        self.time_results = pd.DataFrame(columns=["VIDEO", "ANIMAL", "SHAPE", "TIME (S)"])
        self.roi_bout_results = []
        self.detailed_df = None
        for file_cnt, file_path in enumerate(self.data_paths):
            _, video_name, _ = get_fn_ext(file_path)
            print(f"Analysing ROI data for video {video_name}...")
            video_settings, pix_per_mm, self.fps = self.read_video_info(video_name=video_name)
            self.sliced_roi_dict, video_shape_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=video_name)
            if len(video_shape_names) == 0:
                NoDataFoundWarning(msg=f"Skipping video {video_name}: No user-defined ROI data found for this video...")
                continue
            else:
                self.data_df = read_df(file_path, self.file_type).reset_index(drop=True)
                if len(self.bp_headers) != len(self.data_df.columns):
                    raise MissingColumnsError(msg=f"The data file {file_path} contains {len(self.data_df.columns)} body-part columns, but the project is made for {len(self.bp_headers)} body-part columns as suggested by the {self.body_parts_path} file", source=self.__class__.__name__)
                self.data_df.columns = self.bp_headers
                check_that_column_exist(df=self.data_df, column_name=self.roi_headers, file_name=file_path)
                for animal_name, bp_names in self.bp_dict.items():
                    animal_df = self.data_df[self.bp_dict[animal_name]].reset_index(drop=True)
                    animal_bout_results = {}
                    for _, row in self.sliced_roi_dict[Keys.ROI_RECTANGLES.value].iterrows():
                        roi_coords = np.array([[row["topLeftX"], row["topLeftY"]], [row["Bottom_right_X"], row["Bottom_right_Y"]]])
                        animal_df[row["Name"]] = (FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=animal_df.values[:, 0:2], roi_coords=roi_coords))
                        animal_df.loc[animal_df[bp_names[2]] < self.threshold, row["Name"]] = 0
                        roi_bouts = detect_bouts(data_df=animal_df, target_lst=[row["Name"]], fps=self.fps)
                        roi_bouts["ANIMAL"] = animal_name
                        roi_bouts["VIDEO"] = video_name
                        self.roi_bout_results.append(roi_bouts)
                        animal_bout_results[row["Name"]] = roi_bouts
                        self.entry_results.loc[len(self.entry_results)] = [video_name,animal_name,row["Name"],len(roi_bouts)]
                        self.time_results.loc[len(self.time_results)] = [video_name,animal_name,row["Name"],roi_bouts["Bout_time"].sum()]
                    for _, row in self.sliced_roi_dict[Keys.ROI_CIRCLES.value].iterrows():

                        center_x, center_y = row["centerX"], row["centerY"]
                        animal_df[f'{row["Name"]}_distance'] = (FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=animal_df.values[:, 0:2], location_2=np.array([center_x, center_y]), px_per_mm=1))
                        animal_df[row["Name"]] = 0
                        animal_df.loc[animal_df[f'{row["Name"]}_distance'] <= row["radius"], row["Name"]] = 1
                        animal_df.loc[animal_df[bp_names[2]] < self.threshold, row["Name"]] = 0
                        roi_bouts = detect_bouts(data_df=animal_df, target_lst=[row["Name"]], fps=self.fps)
                        roi_bouts["ANIMAL"] = animal_name
                        roi_bouts["VIDEO"] = video_name
                        self.roi_bout_results.append(roi_bouts)
                        animal_bout_results[row["Name"]] = roi_bouts
                        self.entry_results.loc[len(self.entry_results)] = [video_name, animal_name, row["Name"], len(roi_bouts)]
                        self.time_results.loc[len(self.time_results)] = [video_name,animal_name,row["Name"],roi_bouts["Bout_time"].sum()]
                    for _, row in self.sliced_roi_dict[Keys.ROI_POLYGONS.value].iterrows():
                        roi_coords = np.array(list(zip(row["vertices"][:, 0], row["vertices"][:, 1])))
                        animal_df[row["Name"]] = (
                            FeatureExtractionMixin.framewise_inside_polygon_roi(
                                bp_location=animal_df.values[:, 0:2],
                                roi_coords=roi_coords,
                            )
                        )
                        animal_df.loc[
                            animal_df[bp_names[2]] < self.threshold, row["Name"]
                        ] = 0
                        roi_bouts = detect_bouts(
                            data_df=animal_df, target_lst=[row["Name"]], fps=self.fps
                        )
                        roi_bouts["ANIMAL"] = animal_name
                        roi_bouts["VIDEO"] = video_name
                        self.roi_bout_results.append(roi_bouts)
                        animal_bout_results[row["Name"]] = roi_bouts
                        self.entry_results.loc[len(self.entry_results)] = [
                            video_name,
                            animal_name,
                            row["Name"],
                            len(roi_bouts),
                        ]
                        self.time_results.loc[len(self.time_results)] = [
                            video_name,
                            animal_name,
                            row["Name"],
                            roi_bouts["Bout_time"].sum(),
                        ]
                    if self.calculate_distances:
                        for roi_name, roi_data in animal_bout_results.items():
                            if len(roi_data) == 0:
                                self.movements_df.loc[len(self.movements_df)] = [
                                    video_name,
                                    animal_name,
                                    roi_name,
                                    "Movement (cm)",
                                    0,
                                ]
                                self.movements_df.loc[len(self.movements_df)] = [
                                    video_name,
                                    animal_name,
                                    roi_name,
                                    "Average velocity (cm/s)",
                                    "None",
                                ]
                            else:
                                distances, velocities = [], []
                                roi_frames = roi_data[
                                    ["Start_frame", "End_frame"]
                                ].values
                                for event in roi_frames:
                                    event_pose = animal_df.loc[
                                        np.arange(event[0], event[1] + 1), bp_names
                                    ]
                                    event_pose = event_pose[
                                        event_pose[bp_names[2]] > self.threshold
                                    ][bp_names[:2]].values
                                    if event_pose.shape[0] > 1:
                                        distance, velocity = (
                                            FeatureExtractionSupplemental.distance_and_velocity(
                                                x=event_pose,
                                                fps=self.fps,
                                                pixels_per_mm=pix_per_mm,
                                                centimeters=True,
                                            )
                                        )
                                        distances.append(distance)
                                        velocities.append(velocity)
                                self.movements_df.loc[len(self.movements_df)] = [
                                    video_name,
                                    animal_name,
                                    roi_name,
                                    "Movement (cm)",
                                    sum(distances),
                                ]
                                self.movements_df.loc[len(self.movements_df)] = [
                                    video_name,
                                    animal_name,
                                    roi_name,
                                    "Average velocity (cm/s)",
                                    np.average(velocities),
                                ]
        if len(self.roi_bout_results) > 1:
            self.detailed_df = pd.concat(self.roi_bout_results, axis=0)
            self.detailed_df = self.detailed_df.rename(columns={"Event": "SHAPE NAME", "Start_time": "START TIME", "End Time": "END TIME", "Start_frame": "START FRAME", "End_frame": "END FRAME", "Bout_time": "DURATION (S)"})
            self.detailed_df["BODY-PART"] = self.detailed_df["ANIMAL"].map(self.bp_lk)
            self.detailed_df = self.detailed_df[["VIDEO", "ANIMAL", "BODY-PART", "SHAPE NAME", "START TIME", "END TIME", "START FRAME", "END FRAME", "DURATION (S)"]]

    def save(self):
        self.entry_results["BODY-PART"] = self.entry_results["ANIMAL"].map(self.bp_lk)
        self.time_results["BODY-PART"] = self.time_results["ANIMAL"].map(self.bp_lk)
        self.entry_results = self.entry_results[["VIDEO", "ANIMAL", "BODY-PART", "SHAPE", "ENTRY COUNT"]]
        self.time_results = self.time_results[["VIDEO", "ANIMAL", "BODY-PART", "SHAPE", "TIME (S)"]]
        self.entry_results.to_csv(os.path.join(self.logs_path, f'{"ROI_entry_data"}_{self.datetime}.csv')
        )
        self.time_results.to_csv(
            os.path.join(self.logs_path, f'{"ROI_time_data"}_{self.datetime}.csv')
        )
        if self.detailed_bout_data and self.detailed_df is not None:
            detailed_path = os.path.join(self.logs_path, f'{"Detailed_ROI_data"}_{self.datetime}.csv')
            self.detailed_df.to_csv(detailed_path)
            print(f"Detailed ROI data saved at {detailed_path}...")
        if self.calculate_distances:
            movement_path = os.path.join(self.logs_path, f'{"ROI_movement_data"}_{self.datetime}.csv')
            self.movements_df["BODY-PART"] = self.movements_df["ANIMAL"].map(self.bp_lk)
            self.movements_df = self.movements_df[["VIDEO", "ANIMAL", "BODY-PART", "SHAPE", "MEASUREMENT", "VALUE"]]
            self.movements_df.to_csv(movement_path)
            print(f"ROI aggregate movement data saved at {movement_path}...")
        stdout_success(msg=f"ROI time and ROI entry saved in the {self.logs_path} directory in CSV format.")


# test = ROIAnalyzer(config_path = r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                    data_path=None,
#                    calculate_distances=True,
#                    detailed_bout_data=True,
#                    body_parts=['Nose_1', 'Nose_2'],
#                    threshold=0.0)
# test.run()
# test.save()


#
# test = ROIAnalyzer(ini_path = r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                    data_path = "outlier_corrected_movement_location",
#                    calculate_distances=True,
#                    settings={'threshold': 0.00, 'body_parts': {'Animal_1': 'Nose_1'}})
# test.run()
# test.save()

# test = ROIAnalyzer(ini_path = r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                    data_path = "outlier_corrected_movement_location",
#                    calculate_distances=True)
# test.run()

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
