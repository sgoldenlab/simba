__author__ = "Simon Nilsson"

import itertools
import json
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from simba.data_processors.interpolation_smoothing import Interpolate, Smooth
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pose_importer_mixin import PoseImporterMixin
from simba.utils.enums import Methods
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_all_videos_in_project,
                                    get_video_meta_data, write_df)


class SLEAPImporterSLP(ConfigReader, PoseImporterMixin):
    """
    Class for importing SLEAP pose-estimation data into a SimBA project.

    .. note::
      Importing SLEAP .SLP files into SimBA come at long runtimes. For fater runtimes, use
      :meth:`simba.pose_importers.sleap_h5_importer.SLEAPImporterH5` or :meth:`simba.pose_importers.sleap_csv_importer.SLEAPImporterCSV`

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str data_folder: Path to folder containing SLEAP data in `csv` format.
    :parameter List[str] id_lst: Animal names. This will be ignored in one animal projects and default to ``Animal_1``.
    :parameter str interpolation_settings: String defining the pose-estimation interpolation method. OPTIONS: 'None', 'Animal(s): Nearest',
        'Animal(s): Linear', 'Animal(s): Quadratic','Body-parts: Nearest', 'Body-parts: Linear',
        'Body-parts: Quadratic'.
    :parameter str smoothing_settings: Dictionary defining the pose estimation smoothing method. EXAMPLE: {'Method': 'Savitzky Golay',
        'Parameters': {'Time_window': '200'}}.

    Example
    ----------
    >>> slp_importer = SLEAPImporterSLP(project_path="MyConfigPath", data_folder=r'MySLPDataFolder', actor_IDs=['Mouse_1', 'Mouse_2'], interpolation_settings="Body-parts: Nearest", smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}})
    >>> slp_importer.run()

    References
    ----------
    .. [1] Pereira et al., SLEAP: A deep learning system for multi-animal pose tracking, `Nature Methods`,
           2022.

    """

    def __init__(
        self,
        project_path: str,
        data_folder: str,
        id_lst: list,
        interpolation_settings: str,
        smoothing_settings: dict,
    ):
        ConfigReader.__init__(self, config_path=project_path, read_video_info=False)
        PoseImporterMixin.__init__(self)
        self.interpolation_settings, self.smoothing_settings = (
            interpolation_settings,
            smoothing_settings,
        )
        self.data_folder, self.id_lst = data_folder, id_lst
        self.import_log_path = os.path.join(
            self.logs_path, f"data_import_log_{self.datetime}.csv"
        )
        self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
        self.input_data_paths = self.find_data_files(
            dir=self.data_folder, extensions=[".slp"]
        )
        self.data_and_videos_lk = self.link_video_paths_to_data_paths(
            data_paths=self.input_data_paths, video_paths=self.video_paths
        )
        if self.pose_setting is Methods.USER_DEFINED.value:
            self.__update_config_animal_cnt()
        if self.animal_cnt > 1:
            self.check_multi_animal_status()
            self.animal_bp_dict = self.create_body_part_dictionary(
                self.multi_animal_status,
                self.id_lst,
                self.animal_cnt,
                self.x_cols,
                self.y_cols,
                self.p_cols,
                self.clr_lst,
            )
            self.update_bp_headers_file()
        print(f"Importing {len(list(self.data_and_videos_lk.keys()))} file(s)...")

    def __h5_to_dict(self, name, obj):
        attr = list(obj.attrs.items())
        if name == "metadata":
            jsonList = attr[1][1]
            jsonList = jsonList.decode("utf-8")
            final_dictionary = json.loads(jsonList)
            final_dictionary = dict(final_dictionary)
            return final_dictionary

    def __check_that_all_animals_exist_in_frame(self):
        existing_animals = list(self.frame_dict.keys())
        missing_animals = [
            x for x in range(self.animal_cnt) if x not in existing_animals
        ]
        for missing_animal in missing_animals:
            self.frame_dict[missing_animal] = (
                [0] * ((len(self.analysis_dict["ordered_bps"]))) * 3
            )

    def __fill_missing_indexes(self):
        missing_indexes = list(
            set(list(range(0, self.video_info["frame_count"])))
            - set(list(self.data_df.index))
        )
        missing_df = pd.DataFrame(
            0, index=missing_indexes, columns=self.analysis_dict["xyp_headers"]
        )
        self.data_df = pd.concat([self.data_df, missing_df], axis=0)

    def __run_interpolation(self):
        print(
            f"Interpolating missing values in video {self.video_name} (Method: {self.interpolation_settings})..."
        )
        _ = Interpolate(
            input_path=self.save_path,
            config_path=self.config_path,
            method=self.interpolation_settings,
            initial_import_multi_index=True,
        )

    def __run_smoothing(self):
        print(
            f'Performing {self.smoothing_settings["Method"]} smoothing on video {self.video_name}...'
        )
        Smooth(
            config_path=self.config_path,
            input_path=self.save_path,
            time_window=int(self.smoothing_settings["Parameters"]["Time_window"]),
            smoothing_method=self.smoothing_settings,
        )

    def run(self):
        self.analysis_dict = defaultdict(list)
        self.save_paths_lst = []
        for file_cnt, (video_name, video_data) in enumerate(
            self.data_and_videos_lk.items()
        ):
            print(f"Analysing {video_name}...")
            video_timer = SimbaTimer(start=True)
            self.video_name = video_name
            in_h5 = h5py.File(video_data["DATA"], "r")
            self.sleap_dict = in_h5.visititems(self.__h5_to_dict)
            self.video_info = get_video_meta_data(video_path=video_data["VIDEO"])
            self.analysis_dict["bp_names"] = []
            self.analysis_dict["ordered_ids"] = []
            self.analysis_dict["ordered_bps"] = []
            self.analysis_dict["xy_headers"] = []
            self.analysis_dict["xyp_headers"] = []
            self.analysis_dict["animals_in_each_frame"] = []
            for bp in self.sleap_dict["nodes"]:
                self.analysis_dict["bp_names"].append(bp["name"])
            for orderVar in self.sleap_dict["skeletons"][0]["nodes"]:
                self.analysis_dict["ordered_ids"].append((orderVar["id"]))
            for index in self.analysis_dict["ordered_ids"]:
                self.analysis_dict["ordered_bps"].append(
                    self.analysis_dict["bp_names"][index]
                )

            with h5py.File(video_data["DATA"], "r") as file:
                self.analysis_dict["frames"] = file["frames"][:]
                self.analysis_dict["instances"] = file["instances"][:]
                self.analysis_dict["predicted_points"] = np.reshape(
                    file["pred_points"][:], (file["pred_points"][:].size, 1)
                )
            self.analysis_dict["no_frames"] = len(self.analysis_dict["frames"])

            for c in itertools.product(self.id_lst, self.analysis_dict["ordered_bps"]):
                x, y, p = (
                    str("{}_{}_x".format(c[0], c[1])),
                    str("{}_{}_y".format(c[0], c[1])),
                    (str("{}_{}_p".format(c[0], c[1]))),
                )
                self.analysis_dict["xy_headers"].extend((x, y))
                self.analysis_dict["xyp_headers"].extend((x, y, p))
            #
            self.data_df = pd.DataFrame(columns=self.analysis_dict["xyp_headers"])
            frames_lst = [l.tolist() for l in self.analysis_dict["frames"]]
            self.analysis_dict["animals_in_each_frame"] = [
                x[4] - x[3] for x in frames_lst
            ]
            self.__create_tracks()

            self.initialize_multi_animal_ui(
                animal_bp_dict=self.animal_bp_dict,
                video_info=self.video_info,
                data_df=self.data_df,
                video_path=video_data["VIDEO"],
            )
            if self.animal_cnt > 1:
                self.multianimal_identification()
            self.save_path = os.path.join(
                os.path.join(self.input_csv_dir, f"{self.video_name}.{self.file_type}")
            )
            write_df(
                df=self.out_df,
                file_type=self.file_type,
                save_path=self.save_path,
                multi_idx_header=True,
            )
            if self.interpolation_settings != "None":
                self.__run_interpolation()
            if self.smoothing_settings["Method"] != "None":
                self.__run_smoothing()
            video_timer.stop_timer()
            stdout_success(
                msg=f"Video {video_name} data imported...",
                elapsed_time=video_timer.elapsed_time_str,
            )
        self.timer.stop_timer()
        stdout_success(
            msg="All SLEAP SLP data files imported",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def __create_tracks(self):
        start_frame = 0
        for frame_cnt, frame in enumerate(range(self.analysis_dict["no_frames"])):
            frame_idx = self.analysis_dict["frames"][frame_cnt][2]
            self.frame_dict = {}
            print(
                f"Restructuring SLEAP frame: {frame_cnt}/{self.analysis_dict['no_frames']}, Video: {self.video_name}"
            )
            self.cnt_animals_frm = self.analysis_dict["animals_in_each_frame"][frame]
            if self.cnt_animals_frm == 0:
                self.frame_dict[0] = [0] * len(self.analysis_dict["xyp_headers"])
                end_frame = start_frame + (
                    len(self.analysis_dict["ordered_bps"]) * self.cnt_animals_frm
                )

            else:
                end_frame = start_frame + (
                    len(self.analysis_dict["ordered_bps"]) * self.cnt_animals_frm
                )
                start_animal, end_animal = 0, len(self.analysis_dict["ordered_bps"])
                frame_arr = self.analysis_dict["predicted_points"][
                    start_frame:end_frame
                ]
                for instance_counter, animal in enumerate(range(self.cnt_animals_frm)):
                    currRow = []
                    animal_arr = frame_arr[start_animal:end_animal]
                    track_id = self.analysis_dict["instances"][instance_counter][4]
                    for bp in animal_arr:
                        currRow.extend((bp[0][0], bp[0][1], bp[0][4]))
                    self.frame_dict[track_id] = currRow
                    start_animal += len(self.analysis_dict["ordered_bps"])
                    end_animal += len(self.analysis_dict["ordered_bps"])

            if self.animal_cnt > 1:
                self.__check_that_all_animals_exist_in_frame()
            frame_lst = [
                item for sublist in list(self.frame_dict.values()) for item in sublist
            ]
            start_frame = end_frame
            try:
                self.data_df.loc[frame_idx] = frame_lst
            except ValueError:
                break

        self.data_df.fillna(0, inplace=True)
        self.__fill_missing_indexes()
        self.data_df.sort_index(inplace=True)
        self.data_df.columns = self.bp_headers


# test = SLEAPImporterSLP(project_path="/Users/simon/Desktop/envs/simba/troubleshooting/sleap_two_animals/project_folder/project_config.ini",
#                         data_folder=r'/Users/simon/Desktop/envs/simba/troubleshooting/sleap_two_animals/slp_import',
#                         id_lst=['Simon', 'JJ'],
#                         interpolation_settings="Body-parts: Nearest",
#                         smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.run()
#
# print('All SLEAP imports complete.')


# test = SLEAPImporterSLP(project_path="/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/data',
#                    id_lst=['Simon', 'Nastacia', 'JJ', 'Sam', 'Liana'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.run()
# if test.animals_no > 1:
#     test.visualize_sleap()
# test.save_df()
# test.perform_interpolation()
# test.perform_smothing()
# print('All SLEAP imports complete.')


# test = ImportSLEAP(project_path="/Users/simon/Desktop/envs/troubleshooting/sleap_cody/project_folder/project_config.ini",
#                    data_folder=r'/Users/simon/Desktop/envs/troubleshooting/sleap_cody/import_data',
#                    actor_IDs=['Animal_1', 'Animal_2'],
#                    interpolation_settings="Body-parts: Nearest",
#                    smoothing_settings = {'Method': 'Savitzky Golay', 'Parameters': {'Time_window': '200'}}) #Savitzky Golay
# test.initate_import_slp()
# if test.animals_no > 1:
#     test.visualize_sleap()
# test.save_df()
# test.perform_interpolation()
# test.perform_smothing()
# print('All SLEAP imports complete.')
