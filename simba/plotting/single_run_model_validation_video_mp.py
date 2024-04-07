__author__ = "Simon Nilsson"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import functools
import multiprocessing
import os
import platform
from typing import Any, Dict, List, Optional, Union

import cv2
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_keys_exist_in_dict, check_int,
                                check_valid_extension)
from simba.utils.data import plug_holes_shortest_bout
from simba.utils.enums import TextOptions
from simba.utils.printing import stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df, read_pickle,
                                    write_df)


def validation_video_mp(
    data: pd.DataFrame,
    bp_dict: dict,
    video_save_dir: str,
    settings: dict,
    video_path: str,
    video_meta_data: dict,
    gantt_setting: Union[int, None],
    final_gantt: Optional[np.ndarray],
    clf_data: np.ndarray,
    clrs: List[List],
    clf_name: str,
    bouts_df: pd.DataFrame,
):

    def create_gantt(bouts_df: pd.DataFrame, clf_name: str, image_index: int, fps: int):

        fig, ax = plt.subplots(
            figsize=(final_gantt.shape[1] / dpi, final_gantt.shape[0] / dpi)
        )
        matplotlib.font_manager._get_font.cache_clear()
        relRows = bouts_df.loc[bouts_df["End_frame"] <= image_index]
        for i, event in enumerate(relRows.groupby("Event")):
            data_event = event[1][["Start_time", "Bout_time"]]
            ax.broken_barh(data_event.values, (4, 4), facecolors="red")
        xLength = (round(image_index / fps)) + 1
        if xLength < 10:
            xLength = 10

        ax.set_xlim(0, xLength)
        ax.set_ylim([0, 12])
        ax.set_xlabel("Session (s)", fontsize=12)
        ax.set_ylabel(clf_name, fontsize=12)
        ax.set_title(f"{clf_name} GANTT CHART", fontsize=12)
        ax.set_yticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.grid(True)
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.array(np.uint8(np.array(canvas.renderer._renderer)))[:, :, :3]
        plt.close(fig)
        return img

    dpi = plt.rcParams["figure.dpi"]
    fourcc, font = cv2.VideoWriter_fourcc(*"mp4v"), cv2.FONT_HERSHEY_COMPLEX
    cap = cv2.VideoCapture(video_path)
    group = data["group"].iloc[0]
    start_frm, current_frm, end_frm = data.index[0], data.index[0], data.index[-1]
    video_save_path = os.path.join(video_save_dir, f"{group}.mp4")
    if gantt_setting is not None:
        video_size = (
            int(video_meta_data["width"] + final_gantt.shape[1]),
            int(video_meta_data["height"]),
        )
        writer = cv2.VideoWriter(
            video_save_path, fourcc, video_meta_data["fps"], video_size
        )
    else:
        video_size = (int(video_meta_data["width"]), int(video_meta_data["height"]))
        writer = cv2.VideoWriter(
            video_save_path, fourcc, video_meta_data["fps"], video_size
        )
    cap.set(1, start_frm)
    while (current_frm <= end_frm) & (current_frm <= video_meta_data["frame_count"]):
        clf_frm_cnt = np.sum(clf_data[0:current_frm])
        ret, img = cap.read()
        if ret:
            if settings["pose"]:
                for animal_cnt, (animal_name, animal_data) in enumerate(
                    bp_dict.items()
                ):
                    for bp_cnt, bp in enumerate(range(len(animal_data["X_bps"]))):
                        x_header, y_header = (
                            animal_data["X_bps"][bp],
                            animal_data["Y_bps"][bp],
                        )
                        animal_cords = tuple(
                            data.loc[current_frm, [x_header, y_header]]
                        )
                        cv2.circle(
                            img,
                            (int(animal_cords[0]), int(animal_cords[1])),
                            0,
                            clrs[animal_cnt][bp_cnt],
                            settings["styles"]["circle size"],
                        )
            if settings["animal_names"]:
                for animal_cnt, (animal_name, animal_data) in enumerate(
                    bp_dict.items()
                ):
                    x_header, y_header = (
                        animal_data["X_bps"][0],
                        animal_data["Y_bps"][0],
                    )
                    animal_cords = tuple(data.loc[current_frm, [x_header, y_header]])
                    cv2.putText(
                        img,
                        animal_name,
                        (int(animal_cords[0]), int(animal_cords[1])),
                        font,
                        settings["styles"]["font size"],
                        clrs[animal_cnt][0],
                        1,
                    )

            target_timer = round((1 / video_meta_data["fps"]) * clf_frm_cnt, 2)
            cv2.putText(
                img,
                "Timer",
                (TextOptions.BORDER_BUFFER_Y.value, settings["styles"]["space_scale"]),
                font,
                settings["styles"]["font size"],
                TextOptions.COLOR.value,
                2,
            )
            addSpacer = 2
            cv2.putText(
                img,
                (f"{clf_name} {target_timer}s"),
                (10, settings["styles"]["space_scale"] * addSpacer),
                font,
                settings["styles"]["font size"],
                TextOptions.COLOR.value,
                2,
            )
            addSpacer += 1
            cv2.putText(
                img,
                "Ensemble prediction",
                (
                    TextOptions.BORDER_BUFFER_Y.value,
                    settings["styles"]["space_scale"] * addSpacer,
                ),
                font,
                settings["styles"]["font size"],
                (0, 255, 0),
                2,
            )
            addSpacer += 2
            if clf_data[current_frm] == 1:
                cv2.putText(
                    img,
                    clf_name,
                    (10, +settings["styles"]["space_scale"] * addSpacer),
                    font,
                    settings["styles"]["font size"],
                    TextOptions.COLOR.value,
                    2,
                )
            addSpacer += 1
            if gantt_setting == 1:
                img = np.concatenate((img, final_gantt), axis=1)
            elif gantt_setting == 2:
                gantt_img = create_gantt(
                    bouts_df, clf_name, current_frm, video_meta_data["fps"]
                )
                img = np.concatenate((img, gantt_img), axis=1)
            img = cv2.resize(img, video_size, interpolation=cv2.INTER_LINEAR)
            writer.write(np.uint8(img))
            current_frm += 1
            print(f"Multi-processing video frame {current_frm} on core {group}...")
    cap.release()
    writer.release()

    return group


class ValidateModelOneVideoMultiprocess(ConfigReader, PlottingMixin, TrainModelMixin):
    """
    Create classifier validation video for a single input video. Results are stored in the
    ``project_folder/frames/output/validation`` directory.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str feature_file_path: path to SimBA file (parquet or CSV) containing pose-estimation and feature fields.
    :param str model_path: path to pickled classifier object
    :param float discrimination_threshold: classification threshold.
    :param int shortest_bout: Allowed classified bout length expressed in milliseconds. E.g., `1000` will shift frames classified
        as containing the behavior, but occuring in a bout shorter than `1000`, from `target present to `target absent`.
    :param str create_gantt:
        If SimBA should create gantt charts alongside the validation video. OPTIONS: 'None', 'Gantt chart: final frame only (slightly faster)',
        'Gantt chart: video'.
    :param dict settings: User style settings for video. E.g., {'pose': True, 'animal_names': True, 'styles': None}
    :param int cores: Number of cores to use.

    :example:
    >>> test = ValidateModelOneVideoMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
    >>>                              feature_file_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/features_extracted/Together_1.csv',
    >>>                              model_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/models/generated_models/Attack.sav',
    >>>                              discrimination_threshold=0.6,
    >>>                              shortest_bout=50,
    >>>                              cores=6,
    >>>                              settings={'pose': True, 'animal_names': True, 'styles': None},
    >>>                              create_gantt=None)
    >>> test.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        feature_file_path: Union[str, os.PathLike],
        model_path: Union[str, os.PathLike],
        settings: Dict[str, Any],
        cores: Optional[int] = -1,
        discrimination_threshold: Optional[float] = 0.0,
        shortest_bout: Optional[int] = 0,
        create_gantt: Optional[Union[int, None]] = None,
    ):

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        TrainModelMixin.__init__(self)
        check_int(
            name=f"{self.__class__.__name__} shortest_bout",
            value=shortest_bout,
            min_value=0,
        )
        check_float(
            name=f"{self.__class__.__name__} discrimination_threshold",
            value=discrimination_threshold,
            min_value=0,
            max_value=1.0,
        )
        check_int(
            name=f"{self.__class__.__name__} cores", value=shortest_bout, min_value=-1
        )
        check_file_exist_and_readable(file_path=feature_file_path)
        check_file_exist_and_readable(file_path=model_path)
        check_if_keys_exist_in_dict(
            data=settings,
            key=["pose", "animal_names", "styles"],
            name=f"{self.__class__.__name__} settings",
        )
        if (cores > find_core_cnt()[0]) or (cores == -1):
            cores = find_core_cnt()[0]
        if create_gantt is not None:
            check_int(
                name=f"{self.__class__.__name__} create gantt",
                value=create_gantt,
                max_value=2,
                min_value=1,
            )
        _, self.feature_filename, feature_ext = get_fn_ext(feature_file_path)
        _, self.model_filename, model_ext = get_fn_ext(model_path)
        check_valid_extension(
            path=feature_file_path, accepted_extensions=[self.file_type]
        )
        check_valid_extension(path=model_path, accepted_extensions=["sav"])
        if not os.path.exists(self.single_validation_video_save_dir):
            os.makedirs(self.single_validation_video_save_dir)
        if not os.path.exists(self.clf_data_validation_dir):
            os.makedirs(self.clf_data_validation_dir)
        _, _, self.fps = self.read_video_info(video_name=self.feature_filename)
        self.clf_name = os.path.basename(model_path).replace(".sav", "")
        self.video_path = self.find_video_of_file(self.video_dir, self.feature_filename)
        self.clf_data_save_path = os.path.join(
            self.clf_data_validation_dir, self.feature_filename + ".csv"
        )
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.clf = read_pickle(data_path=model_path)
        self.data_df = read_df(file_path=feature_file_path, file_type=self.file_type)
        self.x_df = self.drop_bp_cords(df=self.data_df)
        self.temp_dir = os.path.join(self.single_validation_video_save_dir, "temp")
        self.video_save_path = os.path.join(
            self.single_validation_video_save_dir, self.feature_filename + ".mp4"
        )
        if os.path.exists(self.temp_dir):
            shutil.rmtree(path=self.temp_dir)
        os.makedirs(self.temp_dir)
        (
            self.discrimination_threshold,
            self.cores,
            self.shortest_bout,
            self.create_gantt,
            self.settings,
            self.feature_file_path,
        ) = (
            discrimination_threshold,
            cores,
            shortest_bout,
            create_gantt,
            settings,
            feature_file_path,
        )
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)

    def __index_df_for_multiprocessing(
        self, data: List[np.ndarray]
    ) -> List[np.ndarray]:
        for cnt, df in enumerate(data):
            df["group"] = cnt
        return data

    def run(self):
        self.prob_col_name = f"Probability_{self.clf_name}"
        self.data_df[self.prob_col_name] = self.clf_predict_proba(
            clf=self.clf,
            x_df=self.x_df,
            model_name=self.clf_name,
            data_path=self.feature_file_path,
        )
        self.data_df[self.clf_name] = np.where(
            self.data_df[self.prob_col_name] > self.discrimination_threshold, 1, 0
        )
        if self.shortest_bout > 1:
            self.data_df = plug_holes_shortest_bout(
                data_df=self.data_df,
                clf_name=self.clf_name,
                fps=self.fps,
                shortest_bout=self.shortest_bout,
            )
        _ = write_df(
            df=self.data_df, file_type=self.file_type, save_path=self.clf_data_save_path
        )
        print(f"Predictions created for video {self.feature_filename}...")
        if self.create_gantt is not None:
            self.bouts_df = self.get_bouts_for_gantt(
                data_df=self.data_df, clf_name=self.clf_name, fps=self.fps
            )
            self.final_gantt_img = self.create_gantt_img(
                self.bouts_df,
                self.clf_name,
                len(self.data_df),
                self.fps,
                "Behavior gantt chart (entire session)",
            )
            self.final_gantt_img = self.resize_gantt(
                self.final_gantt_img, self.video_meta_data["height"]
            )
        else:
            self.bouts_df, self.final_gantt_img = None, None

        if self.settings["styles"] is None:
            self.settings["styles"] = {}
            max_dim = max(self.video_meta_data["width"], self.video_meta_data["height"])
            self.settings["styles"]["circle size"] = int(
                TextOptions.RADIUS_SCALER.value
                / (TextOptions.RESOLUTION_SCALER.value / max_dim)
            )
            self.settings["styles"]["font size"] = float(
                TextOptions.FONT_SCALER.value
                / (TextOptions.RESOLUTION_SCALER.value / max_dim)
            )
            self.settings["styles"]["space_scale"] = int(
                TextOptions.SPACE_SCALER.value
                / (TextOptions.RESOLUTION_SCALER.value / max_dim)
            )

        self.data_df = self.data_df.head(
            min(len(self.data_df), self.video_meta_data["frame_count"])
        )
        data = np.array_split(self.data_df, self.cores)
        frm_per_core = data[0].shape[0]
        data = self.__index_df_for_multiprocessing(data=data)
        pool = multiprocessing.Pool(self.cores, maxtasksperchild=self.maxtasksperchild)
        constants = functools.partial(
            validation_video_mp,
            bp_dict=self.animal_bp_dict,
            video_save_dir=self.temp_dir,
            settings=self.settings,
            video_meta_data=self.video_meta_data,
            video_path=self.video_path,
            gantt_setting=self.create_gantt,
            final_gantt=self.final_gantt_img,
            clf_data=self.data_df[self.clf_name].values,
            clrs=self.clr_lst,
            clf_name=self.clf_name,
            bouts_df=self.bouts_df,
        )
        print("Creating video...")
        for cnt, result in enumerate(
            pool.imap(constants, data, chunksize=self.multiprocess_chunksize)
        ):
            print(
                f"Image {int(frm_per_core * (cnt + 1))}/{len(self.data_df)}, Video {self.feature_filename}..."
            )
        pool.terminate()
        pool.join()
        concatenate_videos_in_folder(
            in_folder=self.temp_dir, save_path=self.video_save_path
        )
        self.timer.stop_timer()
        stdout_success(
            msg=f"Video {self.feature_filename} complete",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = ValidateModelOneVideoMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
#                              feature_file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/features_extracted/SI_DAY3_308_CD1_PRESENT.csv',
#                              model_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/models/generated_models/Running.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              cores=6,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt=None)
# test.run()

# test = ValidateModelOneVideoMultiprocess(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                              feature_file_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/features_extracted/Together_1.csv',
#                              model_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/models/generated_models/Attack.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              cores=6,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt=None)
# test.run()
