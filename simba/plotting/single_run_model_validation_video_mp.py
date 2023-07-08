__author__ = "Simon Nilsson"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import functools
import multiprocessing
import os
import platform
from typing import Any, Dict

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.data import plug_holes_shortest_bout
from simba.utils.printing import stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder, get_fn_ext,
                                    get_video_meta_data, read_df, write_df)


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
    >>> test = ValidateModelOneVideoMultiprocess(config_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini',
    >>>                                         feature_file_path=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/features_extracted/SF2.csv',
    >>>                                         model_path='/Users/simon/Desktop/envs/troubleshooting/naresh/models/generated_models/Top.sav',
    >>>                                         discrimination_threshold=0.6,
    >>>                                         shortest_bout=50,
    >>>                                         settings={'pose': True, 'animal_names': True, 'styles': None},
    >>>                                         create_gantt='Gantt chart: final frame only (slightly faster)',
    >>>                                         cores=5)
    >>> test.run()
    """

    def __init__(
        self,
        config_path: str,
        feature_file_path: str,
        model_path: str,
        discrimination_threshold: float,
        shortest_bout: int,
        cores: int,
        create_gantt: str,
        settings: Dict[str, Any],
    ):
        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        TrainModelMixin.__init__(self)
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        _, self.feature_filename, ext = get_fn_ext(feature_file_path)
        (
            self.discrimination_threshold,
            self.cores,
            self.shortest_bout,
            self.create_gantt,
            self.settings,
        ) = (
            float(discrimination_threshold),
            cores,
            shortest_bout,
            create_gantt,
            settings,
        )
        if self.create_gantt == "None":
            self.create_gantt = None
        if not os.path.exists(self.single_validation_video_save_dir):
            os.makedirs(self.single_validation_video_save_dir)
        _, _, self.fps = self.read_video_info(video_name=self.feature_filename)
        self.clf_name = os.path.basename(model_path).replace(".sav", "")
        self.video_path = self.find_video_of_file(self.video_dir, self.feature_filename)
        self.clf_data_save_path = os.path.join(
            self.clf_data_validation_dir, self.feature_filename + ".csv"
        )
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.clf = read_df(file_path=model_path, file_type="pickle")
        self.in_df = read_df(feature_file_path, self.file_type)
        self.feature_file_path = feature_file_path
        self.temp_dir = os.path.join(self.single_validation_video_save_dir, "temp")
        self.video_save_path = os.path.join(
            self.single_validation_video_save_dir, self.feature_filename + ".mp4"
        )
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def __run_clf(self):
        self.prob_col_name = f"Probability_{self.clf_name}"
        self.in_df[self.prob_col_name] = self.clf_predict_proba(
            clf=self.clf,
            x_df=self.in_df,
            model_name=self.clf_name,
            data_path=self.feature_file_path,
        )
        self.in_df[self.clf_name] = np.where(
            self.in_df[self.prob_col_name] > self.discrimination_threshold, 1, 0
        )

    def __plug_bouts(self):
        self.data_df = plug_holes_shortest_bout(
            data_df=self.in_df,
            clf_name=self.clf_name,
            fps=self.fps,
            shortest_bout=self.shortest_bout,
        )

    def __save(self):
        write_df(
            df=self.data_df, file_type=self.file_type, save_path=self.clf_data_save_path
        )
        print(f"Predictions created for video {self.feature_filename}...")

    def __index_df_for_multiprocessing(self, data: list) -> list:
        for cnt, df in enumerate(data):
            df["group"] = cnt
        return data

    def __create_video(self):
        self.final_gantt_img = None
        self.bouts_df = None
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

        data = np.array_split(self.in_df, self.cores)
        frm_per_core = data[0].shape[0]
        data = self.__index_df_for_multiprocessing(data=data)
        with multiprocessing.Pool(
            self.cores, maxtasksperchild=self.maxtasksperchild
        ) as pool:
            constants = functools.partial(
                self.validation_video_mp,
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
            try:
                for cnt, result in enumerate(
                    pool.imap(constants, data, chunksize=self.multiprocess_chunksize)
                ):
                    print(
                        "Image {}/{}, Video {}...".format(
                            str(int(frm_per_core * (result + 1))),
                            str(len(self.data_df)),
                            self.feature_filename,
                        )
                    )
                print(
                    "Joining {} multiprocessed video...".format(self.feature_filename)
                )
            except:
                pass
        concatenate_videos_in_folder(
            in_folder=self.temp_dir, save_path=self.video_save_path
        )
        self.timer.stop_timer()
        pool.terminate()
        pool.join()
        stdout_success(
            msg=f"Video {self.feature_filename} complete",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def run(self):
        self.__run_clf()
        if self.shortest_bout > 1:
            self.__plug_bouts()
        else:
            self.data_df = self.in_df
        self.__save()
        self.__create_video()


# test = ValidateModelOneVideoMultiprocess(config_path=r'/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                              feature_file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/features_extracted/Together_1.csv',
#                              model_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/models/generated_models/Attack.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              cores=6,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt=None)
# test.run()
