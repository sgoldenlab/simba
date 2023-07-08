__author__ = "Simon Nilsson"

import os
from datetime import datetime
from typing import Dict, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_that_column_exist)
from simba.utils.data import (create_color_palettes, detect_bouts, find_bins,
                              get_video_meta_data)
from simba.utils.errors import InvalidVideoFileError, NoDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df
from simba.utils.warnings import NoDataFoundWarning


class SeverityFrameCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Computes the "severity" of classification frame events based on how much
    the animals are moving. Frames are scored as less or more severe at lower and higher movements, respectively.

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter dict settings: how to calculate the severity. E.g., {'brackets': 10, 'clf': 'Attack', 'animals': ['Simon', 'JJ'], 'time': True, 'frames': False}.

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md>`__.

    Examples
    ----------
    >>> settings = {'brackets': 10, 'clf': 'Attack', 'animals': ['Simon', 'JJ'], 'time': True, 'frames': False, 'normalization': 'ALL VIDEOS', 'save_bin_definitions': True}
    >>> processor = SeverityFrameCalculator(config_path='project_folder/project_config.ini', settings=settings)
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
        self.movement_animal_bp_dict = {
            k: self.animal_bp_dict[k] for k in self.settings["animals"]
        }
        self.color_lst_lst = create_color_palettes(
            no_animals=len(list(self.animal_bp_dict.keys())),
            map_size=len(self.bp_headers),
        )
        definitions_save_name = os.path.join(
            f'severity_bin_definitions_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
        )
        self.save_path = os.path.join(self.logs_path, save_name)
        self.definitions_path = os.path.join(self.logs_path, definitions_save_name)
        self.results = {}

    def __calculate_movements(self):
        self.movements = {}
        for file_cnt, file_path in enumerate(self.machine_results_paths):
            _, video_name, _ = get_fn_ext(file_path)
            print(
                f"Analyzing movements in {video_name} ({file_cnt+1}/{len(self.machine_results_paths)})..."
            )
            _, px_per_mm, fps = self.read_video_info(video_name=video_name)
            df = read_df(file_path=file_path, file_type=self.file_type)
            if self.settings["clf"] not in df.columns:
                NoDataFoundWarning(
                    msg=f'Skipping file {video_name} - {self.settings["clf"]} data not present in file'
                )
                continue
            video_movement = np.full((len(df)), 0)
            for animal_name, animal_bodyparts in self.movement_animal_bp_dict.items():
                animal_df = df[animal_bodyparts["X_bps"] + animal_bodyparts["Y_bps"]]
                animal_df = self.create_shifted_df(df=animal_df)
                for bp_x, bp_y in zip(
                    animal_bodyparts["X_bps"], animal_bodyparts["Y_bps"]
                ):
                    video_movement = np.add(
                        video_movement,
                        self.euclidean_distance(
                            animal_df[bp_x].values,
                            animal_df[f"{bp_x}_shifted"].values,
                            animal_df[bp_y].values,
                            animal_df[f"{bp_y}_shifted"].values,
                            px_per_mm,
                        ),
                    )
            self.movements[video_name] = video_movement.astype(np.int)

    def run(self):
        self.__calculate_movements()
        self.visualization_data = {}
        self.video_bins_info = find_bins(
            data=self.movements,
            bracket_type=self.settings["bracket_type"],
            bracket_cnt=self.settings["brackets"],
            normalization_method=self.settings["normalization"],
        )
        for file_cnt, file_path in enumerate(self.machine_results_paths):
            _, video_name, _ = get_fn_ext(file_path)
            print(
                f"Matching brackets to movements in video {video_name} ({file_cnt+1}/{len(self.machine_results_paths)})..."
            )
            self.results[video_name] = {}
            _, px_per_mm, fps = self.read_video_info(video_name=video_name)
            df = read_df(file_path=file_path, file_type=self.file_type).astype(int)
            check_that_column_exist(
                df=df, column_name=self.settings["clf"], file_name=file_path
            )
            move_df = pd.DataFrame(self.movements[video_name], columns=["MOVEMENT"])
            video_bins = np.array((0))
            video_bins = np.hstack(
                (video_bins, self.video_bins_info[video_name][:, -1])
            )
            hist = np.histogram([self.movements[video_name]], bins=video_bins)
            move_df["BIN"] = np.fmin(
                np.digitize(self.movements[video_name], hist[1]),
                self.settings["brackets"],
            )
            df = pd.concat([df, move_df], axis=1)
            if self.settings["visualize"]:
                self.visualization_data[video_name] = df
            clf_df = (
                df["BIN"][df[self.settings["clf"]] == 1]
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
        self.__save()
        if self.settings["visualize"]:
            self.__visualize()

    def __save(self):
        print("Saving data..")
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

        if self.settings["save_bin_definitions"]:
            results = pd.DataFrame(columns=["VIDEO", "SEVERITY_BIN", ">=", "<"])
            for video_name, video_bins in self.video_bins_info.items():
                for bin_cnt, video_bin in enumerate(video_bins):
                    results.loc[len(results)] = [
                        video_name,
                        bin_cnt + 1,
                        video_bin[0],
                        video_bin[1],
                    ]
            save_path = os.path.join(
                self.logs_path, f"severity_bin_definitions_{self.datetime}.csv"
            )
            results.to_csv(save_path)
            stdout_success(msg=f"Severity bracket definitions saved at {save_path}")

    def __visualize(self):
        bouts_df_lst = []
        video_timer, save_dir = SimbaTimer(start=True), os.path.join(
            self.project_path, "frames", "output", "severity_bouts"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for video_name, video_data in self.visualization_data.items():
            bouts_df = detect_bouts(
                data_df=video_data, target_lst=[self.settings["clf"]], fps=-1
            )
            bouts_df.insert(0, "VIDEO", video_name)
            bouts_df_lst.append(bouts_df)
        bouts_df = pd.concat(bouts_df_lst, axis=0)
        if self.settings["visualize_event_cnt"] == "ALL CLIPS":
            self.settings["visualize_event_cnt"] = len(bouts_df)
        elif self.settings["visualize_event_cnt"] > len(bouts_df):
            self.settings["visualize_event_cnt"] = len(bouts_df)
            print(
                f'User specified {self.settings["visualize_event_cnt"]} visualization but only {len(self.results)} bouts where found. Creating {len(self.results)} videos...'
            )
        bouts_df = bouts_df.sample(n=self.settings["visualize_event_cnt"]).reset_index(
            drop=True
        )
        for idx, r in bouts_df.iterrows():
            video_name, start_frm, end_frame = (
                r["VIDEO"],
                r["Start_frame"],
                r["End_frame"],
            )
            print(start_frm)
            bout_df = self.visualization_data[video_name].iloc[start_frm:end_frame]
            clip_path = os.path.join(
                save_dir,
                f'{video_name}_{self.settings["clf"]}_{start_frm}_{end_frame}.mp4',
            )
            video_path = self.find_video_of_file(
                video_dir=self.video_dir, filename=video_name
            )
            video_meta_data = get_video_meta_data(video_path)
            self.space_scale, self.radius_scale, self.res_scale, self.font_scale = (
                60,
                12,
                1500,
                1.1,
            )
            self.max_dim = max(video_meta_data["width"], video_meta_data["height"])
            self.circle_scale = int(self.radius_scale / (self.res_scale / self.max_dim))
            self.font_size = float(self.font_scale / (self.res_scale / self.max_dim))
            video_fps = int(video_meta_data["fps"] * self.settings["video_speed"])
            if video_fps < 1:
                video_fps = 1
            cap = cv2.VideoCapture(video_path)
            writer = cv2.VideoWriter(
                clip_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                video_fps,
                (int(video_meta_data["width"]), int(video_meta_data["height"])),
            )
            event_frm_count, frm_cnt, current_frm = (
                (end_frame - start_frm),
                0,
                start_frm,
            )
            cap.set(1, current_frm)

            while current_frm < end_frame:
                ret, img = cap.read()
                if self.settings["show_pose"]:
                    frm_pose = bout_df.loc[current_frm]
                    cv2.putText(
                        img,
                        f'Severity bin: {frm_pose["BIN"]}',
                        (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        self.font_size,
                        (255, 0, 0),
                        2,
                    )
                    for animal_cnt, (animal_name, animal_body_parts) in enumerate(
                        self.animal_bp_dict.items()
                    ):
                        for bp_cnt, (x_name, y_name) in enumerate(
                            zip(animal_body_parts["X_bps"], animal_body_parts["Y_bps"])
                        ):
                            x = frm_pose[[x_name, y_name]].values
                            cv2.circle(
                                img,
                                (x[0], x[1]),
                                0,
                                self.color_lst_lst[animal_cnt][bp_cnt],
                                self.circle_scale,
                            )

                if not ret:
                    raise InvalidVideoFileError(
                        msg=f'Could not find frame {current_frm} in video {video_path}. Video {video_path} contains {video_meta_data["frame_count"]} frames.'
                    )
                writer.write(img)
                frm_cnt += 1
                print(
                    f"Frame {str(frm_cnt)} / {str(event_frm_count)}, Event {idx + 1}/{str(len(bouts_df))}, Video {video_name}"
                )
                current_frm += 1
            writer.release()
        video_timer.stop_timer()
        stdout_success(
            msg=f"Videos complete: saved in {save_dir}",
            elapsed_time=video_timer.elapsed_time_str,
        )


# settings = {'brackets': 10,
#             'clf': 'Attack',
#             'animals': ['Simon', 'JJ'],
#             'normalization': 'ALL VIDEOS', #BY VIDEO
#             'bracket_type': "QUANTILE",
#             'time': True,
#             'frames': True,
#             'save_bin_definitions': True,
#             'visualize': True,
#             'visualize_event_cnt': 'ALL CLIPS',
#             'video_speed': 0.1,
#             'show_pose': True}
# processor = SeverityFrameCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', settings=settings)
# processor.run()
