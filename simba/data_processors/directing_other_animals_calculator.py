__author__ = "Simon Nilsson"

import itertools
import os
from typing import Optional, Union
from copy import deepcopy
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (check_valid_lst, check_valid_boolean, check_all_file_names_are_represented_in_video_log, check_file_exist_and_readable, check_that_dir_has_list_of_filenames, check_valid_dataframe)
from simba.utils.enums import TagNames, Paths, Keys, Formats
from simba.utils.errors import AnimalNumberError, CountError, InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (get_fn_ext, read_df, write_df, create_directory)
from simba.utils.lookups import create_directionality_cords

NOSE, EAR_LEFT, EAR_RIGHT = Keys.NOSE.value, Keys.EAR_LEFT.value, Keys.EAR_RIGHT.value
X_BPS, Y_BPS = Keys.X_BPS.value, Keys.Y_BPS.value

class DirectingOtherAnimalsAnalyzer(ConfigReader, FeatureExtractionMixin):
    """
    Calculate when animals are directing towards body-parts of other animals. Results are stored in
    the ``project_folder/logs/directionality_dataframes`` directory of the SimBA project.

    .. note::
       `Example expected bool table <https://github.com/sgoldenlab/simba/blob/master/misc/boolean_directionaly_example.csv>`__.
       `Example expected summary table <https://github.com/sgoldenlab/simba/blob/master/misc/detailed_summary_directionality_example.csv>`__.
       `Example expected aggregate statistics table <https://github.com/sgoldenlab/simba/blob/master/misc/direction_data_aggregates_example.csv>`__.

    .. important::
       Requires the pose-estimation data for the ``left ear``, ``right ear`` and ``nose`` of each individual animals.
       `Github Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data>`__.
       `Expected output <https://github.com/sgoldenlab/simba/blob/master/misc/Direction_data_example.csv>`__.

    ..  youtube:: d6pAatreb1E
       :width: 640
       :height: 480
       :align: center

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param Optional[Union[str, os.PathLike, None]] data_paths: Optional paths to input data files. If None, uses outlier corrected paths from project. Default: None.
    :param Optional[bool] bool_tables: If True, creates boolean output tables. Default: True.
    :param Optional[bool] summary_tables: If True, creates summary tables including approximate location of eye of observer and the location of observed body-parts and frames when observation was detected. Default: False.
    :param Optional[bool] append_bool_tables_to_features: If True, appends boolean tables to feature data files. Default: False.
    :param Optional[bool] aggregate_statistics_tables: If True, creates summary statistics tables of how much time each animal spent observing the other animals. Default: False.
    :param Optional[str] left_ear_name: Name of left ear body-part. If None, SimBA will attempt to auto-detect. Default: None.
    :param Optional[str] right_ear_name: Name of right ear body-part. If None, SimBA will attempt to auto-detect. Default: None.
    :param Optional[str] nose_name: Name of nose body-part. If None, SimBA will attempt to auto-detect. Default: None.

    :example:
    >>> directing_analyzer = DirectingOtherAnimalsAnalyzer(config_path='MyProjectConfig')
    >>> directing_analyzer.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: Optional[Union[str, os.PathLike, None]] = None,
                 bool_tables: Optional[bool] = True,
                 summary_tables: Optional[bool] = False,
                 append_bool_tables_to_features: Optional[bool] = False,
                 aggregate_statistics_tables: Optional[bool] = False,
                 verbose: Optional[bool] = True,
                 left_ear_name: Optional[str] = None,
                 right_ear_name: Optional[str] = None,
                 nose_name: Optional[str] = None):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        if data_paths is None:
            data_paths = deepcopy(self.outlier_corrected_paths)
            if len(data_paths) == 0: raise NoFilesFoundError(msg=f'No data files cound be found in the {self.outlier_corrected_dir} directory', source=self.__class__.__name__)
        elif isinstance(data_paths, str):
            check_file_exist_and_readable(file_path=data_paths)
            data_paths = [data_paths]
        elif isinstance(data_paths, list):
            check_valid_lst(data=data_paths, source=f'{self.__class__.__class__} data_paths', min_len=1, valid_dtypes=(str,))
        else:
            raise InvalidInputError(msg=f'data_path is nota valid file path, or list of file paths, or None. Got: {type(data_paths)}', source=self.__class__.__name__)
        for file_path in data_paths:
            check_file_exist_and_readable(file_path=file_path, raise_error=True)
        self.data_paths = data_paths
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        if self.animal_cnt < 2:
            raise AnimalNumberError("Cannot analyze directionality between animals in a project with less than two animals.", source=self.__class__.__name__)
        check_valid_boolean(value=bool_tables, source=f'{self.__class__.__name__} bool_tables', raise_error=True)
        check_valid_boolean(value=summary_tables, source=f'{self.__class__.__name__} summary_tables', raise_error=True)
        check_valid_boolean(value=append_bool_tables_to_features, source=f'{self.__class__.__name__} append_bool_tables_to_features', raise_error=True)
        check_valid_boolean(value=aggregate_statistics_tables, source=f'{self.__class__.__name__} aggregate_statistics_tables', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        self.animal_permutations = list(itertools.permutations(self.animal_bp_dict, 2))
        self.bool_tables, self.summary_tables, self.aggregate_statistics_tables, self.append_bool_tables_to_features = bool_tables, summary_tables, aggregate_statistics_tables, append_bool_tables_to_features
        self.verbose = verbose
        if self.append_bool_tables_to_features:
            check_that_dir_has_list_of_filenames(dir=self.features_dir, file_name_lst=self.outlier_corrected_paths,file_type=self.file_type)
        passed_bps = [left_ear_name, right_ear_name, nose_name]
        if sum(p is None for p in passed_bps) not in (0, len(passed_bps)):
            raise InvalidInputError(msg="left_ear_name, right_ear_name, and nose_name must either all be None or all be provided as strings", source=self.__class__.__name__)
        if isinstance(left_ear_name, str) and isinstance(right_ear_name, str) and isinstance(nose_name, str):
            self.direct_bp_dict = create_directionality_cords(bp_dict=self.animal_bp_dict, left_ear_name=left_ear_name, nose_name=nose_name, right_ear_name=right_ear_name)
        else:
            if not self.check_directionality_viable()[0]:
                raise InvalidInputError(msg="You are not tracking the necessary body-parts to calculate direction. Either (i) pass the body-parts names, or (ii) name the body-parts properly so SimBA can automatically detect the left ear, right ear, and nose body-part names.", source=self.__class__.__name__)
            self.direct_bp_dict = self.check_directionality_cords()
        print(f"Processing {len(self.data_paths)} video(s)...")

    def transpose_results(self):
        self.directionality_df_dict = {}
        for video_name, video_data in self.results.items():
            out_df_lst = []
            for animal_permutation, permutation_data in video_data.items():
                for bp_name, bp_data in permutation_data.items():
                    directing_df = (bp_data[bp_data["Directing_BOOL"] == 1].reset_index().rename(columns={"index": "Frame_#", bp_name + "_x": "Animal_2_bodypart_x", bp_name + "_y": "Animal_2_bodypart_y"}))
                    directing_df.insert(loc=0, column="Video", value=video_name)
                    out_df_lst.append(directing_df)
            self.directionality_df_dict[video_name] = pd.concat(out_df_lst, axis=0).drop("Directing_BOOL", axis=1)



    def run(self):
        if self.aggregate_statistics_tables:
            check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        self.results = {}
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            self.results[video_name] = {}
            print(f"Analyzing directionality between animals in video {video_name}... (file {file_cnt+1}/{len(self.data_paths)})")
            data_df = read_df(file_path=file_path, file_type=self.file_type)
            for animal_permutation in self.animal_permutations:
                self.results[video_name][f"{animal_permutation[0]} directing towards {animal_permutation[1]}"] = {}
                first_animal_bps, second_animal_bps = (self.direct_bp_dict[animal_permutation[0]], self.animal_bp_dict[animal_permutation[1]])
                for k, v in first_animal_bps.items():
                    check_valid_dataframe(df=data_df, source=f'{self.__class__.__name__} {file_path}', required_fields=list(v.values()), valid_dtypes=Formats.NUMERIC_DTYPES.value)
                first_ear_left_arr = data_df[[first_animal_bps[EAR_LEFT][X_BPS], first_animal_bps[EAR_RIGHT][Y_BPS]]].values.astype(np.int32)
                first_ear_right_arr = data_df[[first_animal_bps[EAR_RIGHT][X_BPS], first_animal_bps[EAR_RIGHT][Y_BPS]]].values.astype(np.int32)
                first_nose_arr = data_df[[first_animal_bps[NOSE][X_BPS], first_animal_bps[NOSE][Y_BPS]]].values.astype(np.int32)
                second_animal_x_bps, second_animal_y_bps = (second_animal_bps[X_BPS], second_animal_bps[Y_BPS])
                for x_bp, y_bp in zip(second_animal_x_bps, second_animal_y_bps):
                    target_cord_arr = data_df[[x_bp, y_bp]].values
                    direction_data = self.jitted_line_crosses_to_nonstatic_targets(left_ear_array=first_ear_left_arr, right_ear_array=first_ear_right_arr, nose_array=first_nose_arr, target_array=target_cord_arr)
                    x_min = np.minimum(direction_data[:, 1], first_nose_arr[:, 0])
                    y_min = np.minimum(direction_data[:, 2], first_nose_arr[:, 1])
                    delta_x = abs((direction_data[:, 1] - first_nose_arr[:, 0]) / 2)
                    delta_y = abs((direction_data[:, 2] - first_nose_arr[:, 1]) / 2)
                    x_middle, y_middle = np.add(x_min, delta_x), np.add(y_min, delta_y)
                    direction_data = np.concatenate((y_middle.reshape(-1, 1), direction_data), axis=1)
                    direction_data = np.concatenate((x_middle.reshape(-1, 1), direction_data), axis=1)
                    direction_data = np.delete(direction_data, [2, 3, 4], 1)
                    direction_data = np.hstack((direction_data, target_cord_arr))
                    bp_data = pd.DataFrame(direction_data, columns=["Eye_x", "Eye_y", "Directing_BOOL", x_bp, y_bp])
                    bp_data = bp_data[["Eye_x", "Eye_y", x_bp, y_bp, "Directing_BOOL"]]
                    bp_data.insert(loc=0, column="Animal_2_body_part", value=x_bp[:-2])
                    bp_data.insert(loc=0, column="Animal_2", value=animal_permutation[1])
                    bp_data.insert(loc=0, column="Animal_1", value=animal_permutation[0])
                    self.results[video_name][f"{animal_permutation[0]} directing towards {animal_permutation[1]}"][x_bp[:-2]] = bp_data
            #video_timer.stop_timer()
            #print(f"Direction analysis complete for video {video_name} ({file_cnt + 1}/{len(self.outlier_corrected_paths)}, elapsed time: {video_timer.elapsed_time_str}s)...")

        if self.bool_tables:
            save_dir = os.path.join(self.logs_path, f"Animal_directing_animal_booleans_{self.datetime}")
            create_directory(paths=save_dir)
            bool_timer = SimbaTimer(start=True)
            for video_cnt, (video_name, video_data) in enumerate(self.results.items()):
                if self.verbose: print(f"Saving boolean directing tables for video {video_name} (Video {video_cnt + 1}/{len(self.results.keys())})...")
                video_df = pd.DataFrame()
                for animal_permutation, animal_permutation_data in video_data.items():
                    for (body_part_name, body_part_data) in animal_permutation_data.items():
                        video_df[f"{animal_permutation}_{body_part_name}"] = (body_part_data["Directing_BOOL"])
                if self.append_bool_tables_to_features:
                    if self.verbose: print(f"Adding directionality tables to features data for video {video_name}...")
                    df = read_df(file_path=os.path.join(self.features_dir, f"{video_name}.{self.file_type}"), file_type=self.file_type)
                    if len(df) != len(video_df):
                        raise CountError(msg=f"Failed to join data files as they contains different number of frames: the file representing video {video_name} in directory {self.outlier_corrected_dir} contains {len(video_df)} frames, and the file representing video {video_name} in directory {self.features_dir} contains {len(df)} frames.")
                    df = pd.concat([df.reset_index(drop=True), video_df.reset_index(drop=True)], axis=1)
                    write_df( df=df, file_type=self.file_type, save_path=os.path.join(self.features_dir, f"{video_name}.{self.file_type}" ))
                video_df.to_csv(os.path.join(save_dir, f"{video_name}.csv"))
            bool_timer.stop_timer()
            if self.verbose: stdout_success(msg=f"All boolean tables saved in {save_dir}!", source=self.__class__.__name__, elapsed_time=bool_timer.elapsed_time_str)

        if self.aggregate_statistics_tables:
            agg_stat_timer = SimbaTimer(start=True)
            if self.verbose: print("Computing summary statistics...")
            save_path = os.path.join(self.logs_path, f"Direction_aggregate_summary_data_{self.datetime}.csv")
            out_df = pd.DataFrame(columns=["VIDEO", "ANIMAL PERMUTATION", "VALUE (S)"])
            for video_name, video_data in self.results.items():
                _, _, fps = self.read_video_info(video_name=video_name)
                for animal_permutation, permutation_data in video_data.items():
                    idx_directing = set()
                    for bp_name, bp_data in permutation_data.items():
                        idx_directing.update(list(bp_data.index[bp_data["Directing_BOOL"] == 1]))
                    value = round(len(idx_directing) / fps, 3)
                    out_df.loc[len(out_df)] = [video_name, animal_permutation, value]

            self.out_df = out_df.sort_values(by=["VIDEO", "ANIMAL PERMUTATION"]).set_index("VIDEO")
            self.out_df.to_csv(save_path)
            agg_stat_timer.stop_timer()
            if self.verbose: stdout_success(msg=f"Summary directional statistics saved at {save_path}", source=self.__class__.__name__, elapsed_time=agg_stat_timer.elapsed_time_str)

        if self.summary_tables:
            self.transpose_results()
            save_dir = os.path.join(self.logs_path, f"detailed_directionality_summary_dataframes_{self.datetime}")
            create_directory(paths=save_dir)
            for video_cnt, (video_name, video_data) in enumerate(self.directionality_df_dict.items()):
                save_name = os.path.join(save_dir, f"{video_name}.csv")
                video_data.to_csv(save_name)
            if self.verbose: stdout_success(f"All detailed directional data saved in the {save_dir} directory!", source=self.__class__.__name__)
        self.timer.stop_timer()
        if self.verbose: stdout_success(msg="All directional data saved in SimBA project", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)




# test = DirectingOtherAnimalsAnalyzer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                      bool_tables=True,
#                                      summary_tables=True,
#                                      aggregate_statistics_tables=True,
#                                      append_bool_tables_to_features=False,
#                                      data_paths=None,
#                                      nose_name='Nose',
#                                      left_ear_name='Ear_left',
#                                      right_ear_name='Ear_right')
# test.run()


# test = DirectingOtherAnimalsAnalyzer(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                      bool_tables=True,
#                                      summary_tables=True,
#                                      aggregate_statistics_tables=True,
#                                      append_bool_tables_to_features=False,
#                                      data_paths=None,
#                                      nose_name=None,
#                                      left_ear_name=None,
#                                      right_ear_name=None)
# test.run()
