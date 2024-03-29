import argparse
import os
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import multiprocessing
import platform

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_lst, check_video_has_rois)
from simba.utils.errors import (AnimalNumberError, InvalidInputError,
                                NoFilesFoundError)
from simba.utils.printing import stdout_success
from simba.utils.read_write import (get_file_name_info_in_directory,
                                    get_fn_ext, read_df)
from simba.utils.warnings import NoFileFoundWarning

TAIL_END = "tail_end"


class SpontaneousAlternationCalculator(ConfigReader):
    """
    Compute spontaneous alternations based on specified ROIs and animal detection parameters.

    .. note::
       This method computes spontaneous alternation by fitting the smallest viable geometry (a.k.a. shape, polygon) that
       encompasses the animal key-points (but see buffer parameter below). It then checks the percent overlap between the
       animal geometry and the defined arm and center geometries in each frame. If the percent overlap is more or equal to the
       specified threshold, then the animal considered to visiting the relevant arm. The animal is considered exiting an arm
       when the percent overlap with a different ROI is above the threshold.

    .. attention::
       Requires SimBA project with (i) only one tracked animal, (ii) at least three pose-estmated body-parts, and (iii) defined
       ROIs representing the arms and the center of the maze.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param List[str] arm_names: List of ROI names representing the arms.
    :param str center_name: Name of the ROI representing the center of the maze
    :param Optional[int] animal_area: Value between 51 and 100, representing the percent of the animal body that has to be situated in a ROI for it to be considered an entry.
    :param Optional[float] threshold: Value between 0.0 and 1.0. Body-parts with detection probabilities below this value will be (if possible) filtered when constructing the animal geometry.
    :param Optional[int] buffer: Millimeters area for which the animal geometry should be increased in size. Useful if the animal geometry does not fully cover the animal.
    :param Optional[bool] detailed_data: If True, saves an additional CSV for each analyzed video with detailed data pertaining each error type and alternation sequence.
    :param Optional[Union[str, os.PathLike]] data_path: Directory of path to the file to be analyzed. If None, then ``project_folder/outlier_corrected_movement_location`` directory.

    :example:
    >>> x = SpontaneousAlternationCalculator(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini', arm_names=['A', 'B', 'C'], center_name='Center', threshold=0.0, animal_area=100, buffer=2, detailed_data=True)
    >>> x.run()
    >>> x.save()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        arm_names: List[str],
        center_name: str,
        animal_area: Optional[int] = 80,
        threshold: Optional[float] = 0.0,
        buffer: Optional[int] = 2,
        verbose: Optional[bool] = False,
        detailed_data: Optional[bool] = False,
        data_path: Optional[Union[str, os.PathLike]] = None,
    ):

        ConfigReader.__init__(self, config_path=config_path)
        if self.animal_cnt != 1:
            raise AnimalNumberError(
                msg=f"Spontaneous alternation can only be calculated in 1 animal projects. Your project has {self.animal_cnt} animals.",
                source=self.__class__.__name__,
            )
        if len(self.body_parts_lst) < 3:
            raise InvalidInputError(
                msg=f"Spontaneous alternation can only be calculated in projects with 3 or more tracked body-parts. Found {len(self.body_parts_lst)}.",
                source=self.__class__.__name__,
            )
        check_valid_lst(
            data=arm_names,
            source=SpontaneousAlternationCalculator.__name__,
            valid_dtypes=(str,),
            min_len=2,
        )
        check_str(name="CENTER NAME", value=center_name)
        check_int(name="ANIMAL AREA", value=animal_area, min_value=51, max_value=100)
        check_float(name="THRESHOLD", value=threshold, min_value=0.0, max_value=1.0)
        check_int(name="BUFFER", value=buffer, min_value=1)
        if data_path is None:
            data_path = self.outlier_corrected_dir
            file_paths = get_file_name_info_in_directory(
                directory=data_path, file_type=self.file_type
            )
        elif os.path.isdir(data_path):
            check_if_dir_exists(in_dir=data_path)
            file_paths = get_file_name_info_in_directory(
                directory=data_path, file_type=self.file_type
            )
        else:
            check_file_exist_and_readable(file_path=data_path)
            file_paths = {get_fn_ext(filepath=data_path)[1]: data_path}

        self.read_roi_data()
        files_w_missing_rois = list(
            set(file_paths.keys()) - set(self.video_names_w_rois)
        )
        self.files_w_rois = [
            x for x in list(file_paths.keys()) if x in self.video_names_w_rois
        ]
        if len(self.files_w_rois) == 0:
            raise NoFilesFoundError(
                msg=f"No ROI definitions found for any of the data fat {arm_names}",
                source=__class__.__name__,
            )
        if len(files_w_missing_rois) > 0:
            NoFileFoundWarning(
                msg=f"{len(files_w_missing_rois)} file(s) at {data_path} are missing ROI definitions and will be skipped when performing spontaneous alternation calculations: {files_w_missing_rois}",
                source=__class__.__name__,
            )
        check_video_has_rois(
            roi_dict=self.roi_dict,
            video_names=self.files_w_rois,
            roi_names=arm_names + [center_name],
        )
        self.file_paths = list(
            {file_paths[k] for k in self.files_w_rois if k in file_paths}
        )
        self.threshold, self.buffer, self.animal_area, self.detailed_data = (
            threshold,
            buffer,
            animal_area,
            detailed_data,
        )
        self.verbose, self.center_name, self.arm_names = verbose, center_name, arm_names

    def run(self):
        roi_geos, self.roi_clrs = GeometryMixin.simba_roi_to_geometries(
            rectangles_df=self.rectangles_df,
            circles_df=self.circles_df,
            polygons_df=self.polygon_df,
            color=True,
        )
        self.roi_geos = {k: v for k, v in roi_geos.items() if k in self.files_w_rois}
        self.results = {}
        for file_path in self.file_paths:
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            _, px_per_mm, fps = self.read_video_info(video_name=self.video_name)

            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            bp_df = self.data_df[
                [
                    x
                    for x in self.bp_headers
                    if not x.endswith("_p") and not TAIL_END in x.lower()
                ]
            ]
            p_df = self.data_df[
                [
                    x
                    for x in self.bp_headers
                    if x.endswith("_p") and not TAIL_END in x.lower()
                ]
            ]
            bp_arr = bp_df.values.reshape(
                len(bp_df), int(len(bp_df.columns) / 2), 2
            ).astype(np.int64)
            p_arr = p_df.values.reshape(len(p_df), len(p_df.columns), 1)
            if self.threshold > 0.0:
                bp_arr = GeometryMixin.filter_low_p_bps_for_shapes(
                    x=bp_arr, p=p_arr, threshold=self.threshold
                ).reshape(bp_arr.shape[0], -1, 2)
            self.animal_polygons = GeometryMixin().multiframe_bodyparts_to_polygon(
                data=bp_arr,
                parallel_offset=self.buffer,
                pixels_per_mm=px_per_mm,
                video_name=self.video_name,
                animal_name="Animal",
                verbose=self.verbose,
            )
            self.roi_df = pd.DataFrame()
            for geo_name, geo in self.roi_geos[self.video_name].items():
                roi_geo = [geo for x in range(len(self.animal_polygons))]
                pct_overlap = np.array(
                    GeometryMixin().multiframe_compute_pct_shape_overlap(
                        shape_1=self.animal_polygons,
                        shape_2=roi_geo,
                        denominator="shape_1",
                        verbose=self.verbose,
                        animal_names=geo_name,
                        video_name=self.video_name,
                    )
                )
                frames_in_roi = np.zeros(pct_overlap.shape)
                frames_in_roi[np.argwhere(pct_overlap >= self.animal_area)] = 1
                self.roi_df[geo_name] = frames_in_roi

            self.video_results = FeatureExtractionSupplemental.spontaneous_alternations(
                data=self.roi_df, arm_names=self.arm_names, center_name=self.center_name
            )
            self.results[self.video_name] = self.video_results

    def save(self):
        print("Running alternation analysis...")
        results_df = pd.DataFrame(
            columns=[
                "VIDEO NAME",
                "ALTERNATION RATE",
                "ALTERNATION COUNT",
                "ERROR COUNT",
                "SAME ARM RETURN ERRORS",
                "ALTERNATE ARM RETURN ERRORS",
            ]
        )
        save_path = os.path.join(
            self.logs_path, f"spontaneous_alternation_{self.datetime}.csv"
        )
        for video_name, d in self.results.items():
            results_df.loc[len(results_df)] = [
                video_name,
                d["pct_alternation"],
                d["alternation_cnt"],
                d["error_cnt"],
                d["same_arm_returns_cnt"],
                d["alternate_arm_returns_cnt"],
            ]
        results_df.set_index("VIDEO NAME").to_csv(save_path)
        stdout_success(
            msg=f"Spontaneous alternation data for {len(list(self.results.keys()))} video(s) saved at {save_path}"
        )
        if self.detailed_data:
            save_dir = os.path.join(
                self.logs_path, f"detailed_spontaneous_alternation_data_{self.datetime}"
            )
            sliced_keys = [
                "same_arm_returns_dict",
                "alternate_arm_returns_dict",
                "alternations_dict",
            ]
            replace_keys = {
                "same_arm_returns_dict": "same arm return",
                "alternate_arm_returns_dict": "alternate arm return",
                "alternations_dict": "alternations",
            }
            os.makedirs(save_dir)
            for video_name, d in self.results.items():
                save_path = os.path.join(save_dir, f"{video_name}.csv")
                sliced_data = {k: v for k, v in d.items() if k in sliced_keys}
                sliced_data = {
                    replace_keys.get(key, key): value
                    for key, value in sliced_data.items()
                }
                row_idx = [(o, i) for o, i in sliced_data.items() for i in i.keys()]
                values = [v for i in sliced_data.values() for v in i.values()]
                values = [
                    ["" if len(sublist) == 0 else ", ".join(map(str, sublist))]
                    for sublist in values
                ]
                multi_index = pd.MultiIndex.from_tuples(
                    row_idx, names=["Behavior", "Arm"]
                )
                df = pd.DataFrame(values, index=multi_index, columns=["Frames"])
                df.to_csv(save_path)
            stdout_success(
                msg=f"Detailed spontaneous alternation data for {len(list(self.results.keys()))} video(s) saved at {save_dir}"
            )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='SimBA Custom Feature Extractor')
#     parser.add_argument('--config_path', type=str, help='SimBA project config path')
#     parser.add_argument('--arm_names', nargs='+', help='List of ROI arm names')
#     parser.add_argument('--center_name', type=str, help='Name of ROI center')
#     parser.add_argument('--animal_area', type=int, default=80, help='Size of animal geometry required for scored entry/exit')
#     parser.add_argument('--threshold', type=float, default=0.0, help='Body-part threshold')
#     parser.add_argument('--buffer', type=int, default=2, help='Added buffer to animal geometry')
#     parser.add_argument('--verbose', type=bool, default=False, help='If process print outs')
#     parser.add_argument('--detailed_data', default=False, type=bool, help='If to create additional detailed data')
#     parser.add_argument('--data_path', type=bool, default=None, help='Path to directory or file.')
#     args = parser.parse_args()
#     x = SpontaneousAlternationCalculator(config_path=args.config_path,
#                                          arm_names=args.arm_names,
#                                          center_name=args.center_name,
#                                          threshold=args.threshold,
#                                          animal_area=args.animal_area,
#                                          buffer=args.buffer,
#                                          verbose=args.verbose,
#                                          detailed_data=args.detailed_data,
#                                          data_path=args.data_path)
#     x.run()
#     x.save()


# config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini'
# arm_names = ['A', 'B', 'C']
# center_name = 'Center'
# threshold = 0.0
# animal_area = 60
# buffer = 2
# verbose = True
#
# x = SpontaneousAlternationCalculator(config_path=config_path,
#                                      arm_names=arm_names,
#                                      center_name=center_name,
#                                      threshold=threshold,
#                                      animal_area=animal_area,
#                                      buffer=buffer,
#                                      verbose=verbose)
# x.run()
# x.save()
