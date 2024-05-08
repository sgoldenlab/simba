__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.roi_tools.ROI_directing_analyzer import DirectingROIAnalyzer
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_that_column_exist, check_valid_lst)
from simba.utils.data import slice_roi_dict_for_video
from simba.utils.enums import Keys, TagNames
from simba.utils.errors import (BodypartColumnNotFoundError, CountError,
                                NoFilesFoundError, ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_data_paths, read_df,
                                    write_df)


class ROIFeatureCreator(ConfigReader, FeatureExtractionMixin):
    """
    Compute features based on the relationships between the location of the animals and the location of
    user-defined ROIs. This includes the distance to the ROIs, if the animals are inside the ROIs, and if the
    animals are directing towards the ROIs (if viable)

    .. note::
        `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param List[str] body_parts: List of the body-parts to use as proxy for animal location(s).
    :param Optional[Union[str, os.PathLike]] data_path: Path to folder or file holding the data used to calculate ROI aggregate statistics. If None, then defaults to the `project_folder/csv/outlier_corrected_movement_location` directory of the SimBA project. Default: None.
    :param Optional[bool] append_data: If True, adds the features to the data in the `project_folder/csv/features_extracted` directory. Else, held in memory.


    :example:
    >>> roi_featurizer = ROIFeatureCreator(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
    >>>                                    body_parts=['Nose_1', 'Nose_2'])
    >>> roi_featurizer.run()
    >>> roi_featurizer.save()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        body_parts: List[str],
        data_path: Optional[Union[str, os.PathLike]] = None,
        append_data: Optional[bool] = False,
    ):

        check_valid_lst(
            data=body_parts,
            source=f"{self.__class__.__name__} body-parts",
            valid_dtypes=(str,),
            min_len=1,
        )
        if len(set(body_parts)) != len(body_parts):
            raise CountError(
                msg=f"All body-part entries have to be unique. Got {body_parts}",
                source=self.__class__.__name__,
            )
        log_event(
            logger_name=str(__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        self.read_roi_data()
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(
                expected_file_path=self.roi_coordinates_path
            )
        for bp in body_parts:
            if bp not in self.body_parts_lst:
                raise BodypartColumnNotFoundError(
                    msg=f"The body-part {bp} is not a valid body-part in the SimBA project. Options: {self.body_parts_lst}",
                    source=self.__class__.__name__,
                )
        self.bp_lk = {}
        for cnt, bp in enumerate(body_parts):
            animal = self.find_animal_name_from_body_part_name(
                bp_name=bp, bp_dict=self.animal_bp_dict
            )
            self.bp_lk[cnt] = [
                animal,
                bp,
                [f'{bp}_{"x"}', f'{bp}_{"y"}', f'{bp}_{"p"}'],
            ]
        self.roi_directing_viable = self.check_directionality_viable()[0]
        log_event(
            logger_name=str(__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        self.data_paths = read_data_paths(
            path=data_path,
            default=self.outlier_corrected_paths,
            default_name=self.outlier_corrected_dir,
            file_type=self.file_type,
        )
        if self.roi_directing_viable:
            print("Directionality calculations are VIABLE.")
            self.directing_analyzer = DirectingROIAnalyzer(
                config_path=config_path, data_path=self.data_paths
            )
            self.directing_analyzer.run()
            self.dr = self.directing_analyzer.results_df
        else:
            self.directing_analyzer = None
            self.dr = None
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(
                msg=f"No data found in the {self.outlier_corrected_dir} directory",
                source=self.__class__.__name__,
            )
        if len(self.feature_file_paths) == 0:
            raise NoFilesFoundError(
                msg=f"No data found in the {self.features_dir} directory",
                source=self.__class__.__name__,
            )
        self.append_data = append_data
        print(
            f"Processing {len(self.outlier_corrected_paths)} video(s) for ROI features..."
        )

    def run(self):
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.outlier_corrected_paths
        )
        self.summary = pd.DataFrame(
            columns=["VIDEO", "ANIMAL", "SHAPE NAME", "MEASUREMENT", "VALUE"]
        )
        if self.append_data:
            _o_paths = set([get_fn_ext(x)[1] for x in self.outlier_corrected_paths])
            _f_paths = set([get_fn_ext(x)[1] for x in self.feature_file_paths])
            o_paths = _o_paths.difference(_f_paths)
            f_paths = _f_paths.difference(_o_paths)
            if len(o_paths) != 0 or len(f_paths) != 0:
                raise NoFilesFoundError(
                    msg=f"Before appending ROI features, make sure each video is represented in both the {self.outlier_corrected_dir} and {self.features_dir} directory, you have data files that are only represented in one folder {o_paths} {f_paths}",
                    source=self.__class__.__name__,
                )
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            features_file_path = os.path.join(
                self.features_dir, f"{self.video_name}.{self.file_type}"
            )
            self.out_df = read_df(
                file_path=features_file_path, file_type=self.file_type
            )
            _, self.pixels_per_mm, self.fps = self.read_video_info(
                video_name=self.video_name
            )
            data_df = read_df(file_path=file_path, file_type=self.file_type)
            self.video_roi_dict, self.shape_names = slice_roi_dict_for_video(
                data=self.roi_dict, video_name=self.video_name
            )
            for animal_cnt, animal_data in self.bp_lk.items():
                animal_name, body_part_name, bp_cols = animal_data
                check_that_column_exist(
                    df=data_df, column_name=bp_cols, file_name=file_path
                )
                animal_df = data_df[bp_cols]
                for _, row in self.video_roi_dict[Keys.ROI_RECTANGLES.value].iterrows():
                    roi_border = np.array(
                        [
                            [row["topLeftX"], row["topLeftY"]],
                            [row["Bottom_right_X"], row["Bottom_right_Y"]],
                        ]
                    )
                    roi_name = row["Name"]
                    roi_center = np.array([row["Center_X"], row["Center_Y"]])
                    c = f"{roi_name} {animal_name} {body_part_name} distance"
                    self.out_df[c] = (
                        FeatureExtractionMixin.framewise_euclidean_distance_roi(
                            location_1=animal_df.values[:, 0:2],
                            location_2=roi_center,
                            px_per_mm=self.pixels_per_mm,
                        )
                    )
                    self.summary.loc[len(self.summary)] = [
                        self.video_name,
                        animal_name,
                        roi_name,
                        "Average distance (mm)",
                        self.out_df[c].mean().round(4),
                    ]
                    c = f"{roi_name} {animal_name} {body_part_name} in zone"
                    self.out_df[c] = (
                        FeatureExtractionMixin.framewise_inside_rectangle_roi(
                            bp_location=animal_df.values[:, 0:2], roi_coords=roi_border
                        )
                    )
                for _, row in self.video_roi_dict[Keys.ROI_CIRCLES.value].iterrows():
                    roi_center = np.array([row["centerX"], row["centerY"]])
                    roi_name, radius = row["Name"], row["radius"]
                    c = f"{roi_name} {animal_name} {body_part_name} distance"
                    self.out_df[c] = (
                        FeatureExtractionMixin.framewise_euclidean_distance_roi(
                            location_1=animal_df.values[:, 0:2],
                            location_2=roi_center,
                            px_per_mm=self.pixels_per_mm,
                        )
                    )
                    self.summary.loc[len(self.summary)] = [
                        self.video_name,
                        animal_name,
                        roi_name,
                        "Average distance (mm)",
                        self.out_df[c].mean().round(4),
                    ]
                    in_zone = f"{roi_name} {animal_name} {body_part_name} in zone"
                    self.out_df[in_zone] = 0
                    self.out_df.loc[self.out_df[c] <= row["radius"], in_zone] = 1
                for _, row in self.roi_dict[Keys.ROI_POLYGONS.value].iterrows():
                    roi_coords = np.array(
                        list(zip(row["vertices"][:, 0], row["vertices"][:, 1]))
                    )
                    roi_center = np.array([row["Center_X"], row["Center_Y"]])
                    roi_name = row["Name"]
                    c = f"{roi_name} {animal_name} {body_part_name} distance"
                    self.out_df[c] = (
                        FeatureExtractionMixin.framewise_euclidean_distance_roi(
                            location_1=animal_df.values[:, 0:2],
                            location_2=roi_center,
                            px_per_mm=self.pixels_per_mm,
                        )
                    )
                    self.summary.loc[len(self.summary)] = [
                        self.video_name,
                        animal_name,
                        roi_name,
                        "Average distance (mm)",
                        self.out_df[c].mean().round(4),
                    ]
                    c = f"{roi_name} {animal_name} {body_part_name} in zone"
                    self.out_df[c] = (
                        FeatureExtractionMixin.framewise_inside_polygon_roi(
                            bp_location=animal_df.values[:, 0:2], roi_coords=roi_coords
                        )
                    )
                if self.roi_directing_viable:
                    animal_dr = self.dr.loc[
                        (self.dr["Video"] == self.video_name)
                        & (self.dr["Animal"] == animal_name)
                    ]
                    for shape_name in self.shape_names:
                        c = f"{shape_name} {animal_name} facing"
                        animal_shape_idx = list(
                            animal_dr.loc[
                                (animal_dr["ROI"] == shape_name)
                                & (animal_dr["Directing_BOOL"] == 1)
                            ]["Frame"]
                        )
                        self.out_df[c] = 0
                        self.out_df.loc[animal_shape_idx, c] = 1
                        self.summary.loc[len(self.summary)] = [
                            self.video_name,
                            animal_name,
                            shape_name,
                            "Total direction time (s)",
                            round((self.out_df[c].sum() / self.fps), 4),
                        ]
            video_timer.stop_timer()
            if self.append_data:
                write_df(
                    df=self.out_df,
                    file_type=self.file_type,
                    save_path=features_file_path,
                )
                print(
                    f"New file with ROI features created at  {features_file_path} saved (File {file_cnt+1}/{len(self.outlier_corrected_paths)}), elapsed time: {video_timer.elapsed_time_str}s"
                )

        self.timer.stop_timer()
        stdout_success(
            msg=f"ROI features analysed for {len(self.data_paths)} videos",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def save(self):
        save_path = os.path.join(
            self.logs_path, f"ROI_features_summary_{self.datetime}.csv"
        )
        self.summary.to_csv(save_path)
        print(f"ROI feature summary data saved at {save_path}")
        self.timer.stop_timer()
        stdout_success(
            msg=f"{len(self.outlier_corrected_paths)} new files with ROI features saved in {self.features_dir}",
            elapsed_time=self.timer.elapsed_time_str,
        )


#
# roi_featurizer = ROIFeatureCreator(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                    body_parts=['Nose_1', 'Nose_2'],
#                                    data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Together_1.csv')
# roi_featurizer.run()


# roi_featurizer = ROIFeatureCreator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# roi_featurizer.roi_directing_viable
# roi_featurizer.run()
# roi_featurizer.save()


# roi_featurizer = ROIFeatureCreator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
# roi_featurizer.run()
# roi_featurizer.save()
