__author__ = "Simon Nilsson"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import warnings
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.data import plug_holes_shortest_bout
from simba.utils.enums import TagNames
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (get_fn_ext, get_video_meta_data, read_df,
                                    write_df)

plt.interactive(True)
plt.ioff()
warnings.simplefilter(action="ignore", category=FutureWarning)


class ValidateModelOneVideo(ConfigReader, PlottingMixin, TrainModelMixin):
    """
    Create classifier validation video for a single input video. Results are stored in the
    `project_folder/frames/output/validation directory`.

    .. note::
       For improved run-time, see :meth:`simba.sing_run_model_validation_video_mp.ValidateModelOneVideoMultiprocess` for multiprocess class.

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

    :example:
    >>> test = ValidateModelOneVideo(config_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
    >>>                                 feature_file_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/features_extracted/SF2.csv',
    >>>                                 model_path='/Users/simon/Desktop/envs/troubleshooting/naresh/models/generated_models/Top.sav',
    >>>                                 discrimination_threshold=0.6,
    >>>                                 shortest_bout=50,
    >>>                                 settings={'pose': True, 'animal_names': True, 'styles': None},
    >>>                                 create_gantt='Gantt chart: final frame only (slightly faster)')
    >>> test.run()
    """

    def __init__(
        self,
        config_path: str,
        feature_file_path: str,
        model_path: str,
        discrimination_threshold: float,
        shortest_bout: int,
        create_gantt: str,
        settings: Dict[str, Any],
    ):
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        TrainModelMixin.__init__(self)
        log_event(
            logger_name=str(__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        _, self.feature_filename, ext = get_fn_ext(feature_file_path)
        (
            self.discrimination_threshold,
            self.shortest_bout,
            self.create_gantt,
            self.settings,
        ) = (float(discrimination_threshold), shortest_bout, create_gantt, settings)
        if not os.path.exists(self.single_validation_video_save_dir):
            os.makedirs(self.single_validation_video_save_dir)
        _, _, self.fps = self.read_video_info(video_name=self.feature_filename)
        self.clf_name, self.feature_file_path = (
            os.path.basename(model_path).replace(".sav", ""),
            feature_file_path,
        )
        self.video_path = self.find_video_of_file(self.video_dir, self.feature_filename)
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.vid_output_path = os.path.join(
            self.single_validation_video_save_dir,
            f"{self.feature_filename} {self.clf_name}.avi",
        )
        self.clf_data_save_path = os.path.join(
            self.clf_data_validation_dir, self.feature_filename + ".csv"
        )
        self.clf = read_df(file_path=model_path, file_type="pickle")
        self.in_df = read_df(feature_file_path, self.file_type)

    def __run_clf(self):
        self.data_df = self.drop_bp_cords(df=self.in_df)
        self.prob_col_name = f"Probability_{self.clf_name}"
        self.data_df[self.prob_col_name] = self.clf_predict_proba(
            clf=self.clf,
            x_df=self.data_df,
            model_name=self.clf_name,
            data_path=self.feature_file_path,
        )
        self.data_df[self.clf_name] = np.where(
            self.data_df[self.prob_col_name] > self.discrimination_threshold, 1, 0
        )

    def __plug_bouts(self):
        self.data_df = plug_holes_shortest_bout(
            data_df=self.data_df,
            clf_name=self.clf_name,
            fps=self.fps,
            shortest_bout=self.shortest_bout,
        )

    def __save(self):
        write_df(
            df=self.data_df, file_type=self.file_type, save_path=self.clf_data_save_path
        )
        print(f"Predictions created for video {self.feature_filename}...")

    def __create_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if self.create_gantt == "None":
            writer = cv2.VideoWriter(
                self.vid_output_path,
                fourcc,
                self.fps,
                (self.video_meta_data["width"], self.video_meta_data["height"]),
            )
        else:
            self.bouts_df = self.get_bouts_for_gantt(
                data_df=self.data_df, clf_name=self.clf_name, fps=self.fps
            )
            self.gantt_img = self.create_gantt_img(
                self.bouts_df,
                self.clf_name,
                len(self.data_df),
                self.fps,
                "Behavior gantt chart (entire session)",
            )
            self.gantt_img = self.resize_gantt(
                self.gantt_img, self.video_meta_data["height"]
            )
            writer = cv2.VideoWriter(
                self.vid_output_path,
                fourcc,
                self.fps,
                (
                    int(self.video_meta_data["width"] + self.gantt_img.shape[1]),
                    int(self.video_meta_data["height"]),
                ),
            )

        if self.settings["styles"] is None:
            self.settings["styles"] = {}
            space_scaler, radius_scaler, resolution_scaler, font_scaler = (
                60,
                20,
                1500,
                1.5,
            )
            max_dim = max(self.video_meta_data["width"], self.video_meta_data["height"])
            self.settings["styles"]["circle size"] = int(
                radius_scaler / (resolution_scaler / max_dim)
            )
            self.settings["styles"]["font size"] = float(
                font_scaler / (resolution_scaler / max_dim)
            )
            self.settings["styles"]["space_scale"] = int(
                space_scaler / (resolution_scaler / max_dim)
            )

        frm_cnt, clf_frm_cnt = 0, 0
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                print(
                    "SIMBA WARNING: Some frames appear to be missing in the video vs the data file."
                )
                break
            clf_val = int(self.data_df.loc[frm_cnt, self.clf_name])
            clf_frm_cnt += clf_val
            if self.settings["pose"]:
                for animal_cnt, (animal_name, animal_data) in enumerate(
                    self.animal_bp_dict.items()
                ):
                    for bp_cnt, bp in enumerate(range(len(animal_data["X_bps"]))):
                        x_header, y_header = (
                            animal_data["X_bps"][bp],
                            animal_data["Y_bps"][bp],
                        )
                        animal_cords = tuple(
                            self.in_df.loc[
                                self.in_df.index[frm_cnt], [x_header, y_header]
                            ]
                        )
                        cv2.circle(
                            frame,
                            (int(animal_cords[0]), int(animal_cords[1])),
                            0,
                            self.clr_lst[animal_cnt][bp_cnt],
                            self.settings["styles"]["circle size"],
                        )
            if self.settings["animal_names"]:
                for animal_cnt, (animal_name, animal_data) in enumerate(
                    self.animal_bp_dict.items()
                ):
                    x_header, y_header = (
                        animal_data["X_bps"][0],
                        animal_data["Y_bps"][0],
                    )
                    animal_cords = tuple(
                        self.in_df.loc[self.in_df.index[frm_cnt], [x_header, y_header]]
                    )
                    cv2.putText(
                        frame,
                        animal_name,
                        (int(animal_cords[0]), int(animal_cords[1])),
                        self.font,
                        self.settings["styles"]["font size"],
                        self.clr_lst[animal_cnt][0],
                        1,
                    )
            target_timer = round((1 / self.fps) * clf_frm_cnt, 2)
            cv2.putText(
                frame,
                "Timer",
                (10, int(self.settings["styles"]["space_scale"])),
                self.font,
                self.settings["styles"]["font size"],
                (0, 255, 0),
                2,
            )
            addSpacer = 2
            cv2.putText(
                frame,
                (f"{self.clf_name} {target_timer}s"),
                (10, self.settings["styles"]["space_scale"] * addSpacer),
                self.font,
                self.settings["styles"]["font size"],
                (0, 0, 255),
                2,
            )
            addSpacer += 1
            cv2.putText(
                frame,
                "Ensemble prediction",
                (10, self.settings["styles"]["space_scale"] * addSpacer),
                self.font,
                self.settings["styles"]["font size"],
                (0, 255, 0),
                2,
            )
            addSpacer += 2
            if clf_val == 1:
                cv2.putText(
                    frame,
                    self.clf_name,
                    (10, +self.settings["styles"]["space_scale"] * addSpacer),
                    self.font,
                    self.settings["styles"]["font size"],
                    (2, 166, 249),
                    2,
                )
                addSpacer += 1
            if self.create_gantt == "Gantt chart: final frame only (slightly faster)":
                frame = np.concatenate((frame, self.gantt_img), axis=1)
            elif self.create_gantt == "Gantt chart: video":
                gantt_img = self.create_gantt_img(
                    self.bouts_df,
                    self.clf_name,
                    frm_cnt,
                    self.fps,
                    "Behavior gantt chart",
                )
                gantt_img = self.resize_gantt(
                    gantt_img, self.video_meta_data["video_height"]
                )
                frame = np.concatenate((frame, gantt_img), axis=1)
            elif self.create_gantt != "None":
                frame = cv2.resize(
                    frame,
                    (
                        int(self.video_meta_data["width"] + self.gantt_img.shape[1]),
                        self.video_meta_data["height"],
                    ),
                )
            writer.write(np.uint8(frame))
            print(
                "Frame created: {} / {}".format(
                    str(frm_cnt + 1), str(len(self.data_df))
                )
            )
            frm_cnt += 1

        cap.release()
        writer.release()
        self.timer.stop_timer()
        stdout_success(
            msg=f"Validation video saved at {self.vid_output_path}",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )

    def run(self):
        self.__run_clf()
        if self.shortest_bout > 1:
            self.__plug_bouts()
        self.__save()
        self.__create_video()


# test = ValidateModelOneVideo(config_path=r'/Users/simon/Desktop/envs/troubleshooting/dam_nest-c-only_ryan/project_folder/project_config.ini',
#                              feature_file_path='/Users/simon/Desktop/envs/troubleshooting/dam_nest-c-only_ryan/project_folder/csv/features_extracted/LBNF2_Ctrl_P04_4_2021-03-18_19-49-46c.csv',
#                              model_path='/Users/simon/Desktop/envs/troubleshooting/dam_nest-c-only_ryan/models/dam_in_nest.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt=None)
# test.run()


# test = ValidateModelOneVideo(config_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
#                              feature_file_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/features_extracted/SF2.csv',
#                              model_path='/Users/simon/Desktop/envs/troubleshooting/naresh/models/generated_models/Top.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt='Gantt chart: final frame only (slightly faster)')
# test.run()


# test = ValidateModelOneVideo(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                              feature_file_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/features_extracted/Together_1.csv',
#                              model_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/models/generated_models/Attack.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt='Gantt chart: final frame only (slightly faster)')
# test.run()


# test.perform_clf()
# test.plug_small_bouts()
# test.save_classification_data()
# test.create_video()

# test = ValidateModelOneVideo(ini_path='/Users/simon/Desktop/troubleshooting/Zebrafish/project_folder/project_config.ini',
#                  feature_file_path=r'/Users/simon/Desktop/troubleshooting/Zebrafish/project_folder/csv/features_extracted/20200730_AB_7dpf_850nm_0002.csv',
#                  model_path='/Users/simon/Desktop/troubleshooting/Zebrafish/models/generated_models/Rheotaxis.sav',
#                  d_threshold=0,
#                  shortest_bout=50,
#                  create_gantt='Gantt chart: final frame only (slightly faster)')
# test.perform_clf()
# test.plug_small_bouts()
# test.save_classification_data()
# test.create_video()


# inifile = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\project_config.ini"
# featuresPath = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\project_folder\csv\features_extracted\Together_1.csv"
# modelPath = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\DLC_two_mice\models\generated_models\Attack.sav"
#
# inifile = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Zebrafish\project_folder\project_config.ini"
# featuresPath = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Zebrafish\project_folder\csv\features_extracted\20200730_AB_7dpf_850nm_0002.csv"
# modelPath = r"Z:\DeepLabCut\DLC_extract\Troubleshooting\Zebrafish\models\validations\model_files\Rheotaxis_1.sav"
#
# dt = 0.4
# sb = 67
# generategantt = 'Gantt chart: video'
#
# test = ValidateModelOneVideo(ini_path=inifile,feature_file_path=featuresPath,model_path=modelPath,d_threshold=dt,shortest_bout=sb, create_gantt=generategantt)
# test.perform_clf()
# test.plug_small_bouts()
# test.save_classification_data()
# test.create_video()

# cv2.imshow('Window', frame)
# key = cv2.waitKey(3000)
# if key == 27:
#     cv2.destroyAllWindows()
