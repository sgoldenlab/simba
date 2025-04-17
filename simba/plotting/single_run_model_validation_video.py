__author__ = "Simon Nilsson"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import warnings
from typing import Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float, check_if_keys_exist_in_dict, check_int, check_video_and_data_frm_count_align, check_valid_boolean)
from simba.utils.data import plug_holes_shortest_bout
from simba.utils.enums import Formats, TagNames, TextOptions
from simba.utils.printing import log_event, stdout_success
from simba.utils.read_write import (get_fn_ext, get_video_meta_data, read_df, read_pickle, write_df)

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

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 feature_path: Union[str, os.PathLike],
                 model_path: Union[str, os.PathLike],
                 show_pose: bool = True,
                 show_animal_names: bool = False,
                 font_size: Optional[bool] = None,
                 circle_size: Optional[int] = None,
                 text_spacing: Optional[int] = None,
                 text_thickness: Optional[int] = None,
                 text_opacity: Optional[float] = None,
                 discrimination_threshold: Optional[float] = 0.0,
                 shortest_bout: Optional[int] = 0.0,
                 create_gantt: Optional[Union[None, int]] = None):

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        TrainModelMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        check_file_exist_and_readable(file_path=config_path)
        check_file_exist_and_readable(file_path=feature_path)
        check_file_exist_and_readable(file_path=model_path)
        check_valid_boolean(value=[show_pose], source=f'{self.__class__.__name__} show_pose', raise_error=True)
        check_valid_boolean(value=[show_animal_names], source=f'{self.__class__.__name__} show_animal_names', raise_error=True)
        if font_size is not None: check_int(name=f'{self.__class__.__name__} font_size', value=font_size)
        if circle_size is not None: check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size)
        if text_spacing is not None: check_int(name=f'{self.__class__.__name__} text_spacing', value=text_spacing)
        if text_opacity is not None: check_float(name=f'{self.__class__.__name__} text_opacity', value=text_opacity, min_value=0.1)
        if text_thickness is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=text_thickness, min_value=0.1)
        check_float(name=f"{self.__class__.__name__} discrimination_threshold", value=discrimination_threshold, min_value=0, max_value=1.0)
        check_int(name=f"{self.__class__.__name__} shortest_bout", value=shortest_bout, min_value=0)
        if create_gantt is not None:
            check_int(name=f"{self.__class__.__name__} create gantt", value=create_gantt, max_value=2, min_value=1)
        if not os.path.exists(self.single_validation_video_save_dir):
            os.makedirs(self.single_validation_video_save_dir)
        _, self.feature_filename, ext = get_fn_ext(feature_path)
        self.video_path = self.find_video_of_file(self.video_dir, self.feature_filename)
        self.video_meta_data = get_video_meta_data(video_path=self.video_path)
        self.clf_name, self.feature_file_path = (os.path.basename(model_path).replace(".sav", ""), feature_path)
        self.vid_output_path = os.path.join(self.single_validation_video_save_dir, f"{self.feature_filename} {self.clf_name}.mp4")
        self.clf_data_save_path = os.path.join(self.clf_data_validation_dir, f"{self.feature_filename }.csv")
        self.show_pose, self.show_animal_names = show_pose, show_animal_names
        self.font_size, self.circle_size, self.text_spacing = font_size, circle_size, text_spacing
        self.text_opacity, self.text_thickness = text_opacity, text_thickness
        self.clf = read_pickle(data_path=model_path, verbose=True)
        self.data_df = read_df(feature_path, self.file_type)
        self.feature_path =  feature_path
        self.x_df = self.drop_bp_cords(df=self.data_df)
        self.discrimination_threshold, self.shortest_bout, self.create_gantt = float(discrimination_threshold), shortest_bout, create_gantt
        check_video_and_data_frm_count_align(video=self.video_path, data=self.data_df, name=self.feature_filename, raise_error=False)

    def _get_styles(self):
        self.video_text_thickness = TextOptions.TEXT_THICKNESS.value if self.text_thickness is None else int(max(self.text_thickness, 1))
        longest_str = str(max(['TIMERS:', 'ENSEMBLE PREDICTION:'] + self.clf_names, key=len))
        optimal_font_size, _, optimal_spacing_scale = self.get_optimal_font_scales(text=longest_str, accepted_px_width=int(self.video_meta_data["width"] / 3), accepted_px_height=int(self.video_meta_data["height"] / 10), text_thickness=self.video_text_thickness)
        optimal_circle_size = self.get_optimal_circle_size(frame_size=(self.video_meta_data["width"], self.video_meta_data["height"]), circle_frame_ratio=100)
        self.video_circle_size = optimal_circle_size if self.circle_size is None else int(self.circle_size)
        self.video_font_size = optimal_font_size if self.font_size is None else self.font_size
        self.video_space_size = optimal_spacing_scale if self.text_spacing is None else int(max(self.text_spacing, 1))
        self.video_text_opacity = 0.8 if self.text_opacity is None else float(self.text_opacity)

    def run(self):
        self.prob_col_name = f"Probability_{self.clf_name}"
        self.data_df[self.prob_col_name] = self.clf_predict_proba(clf=self.clf, x_df=self.x_df, model_name=self.clf_name, data_path=self.feature_path)
        self.data_df[self.clf_name] = np.where(self.data_df[self.prob_col_name] > self.discrimination_threshold, 1, 0)
        if self.shortest_bout > 1:
            self.data_df = plug_holes_shortest_bout(data_df=self.data_df, clf_name=self.clf_name, fps=self.video_meta_data['fps'], shortest_bout=self.shortest_bout)
        _ = write_df(df=self.data_df, file_type=self.file_type, save_path=self.clf_data_save_path)
        print(f"Behavior predictions created for model {self.clf_name} and video {self.feature_filename}...")
        cap = cv2.VideoCapture(self.video_path)
        fourcc, font = cv2.VideoWriter_fourcc(*"mp4v"), cv2.FONT_HERSHEY_COMPLEX
        if self.create_gantt is not None:
            self.bouts_df = self.get_bouts_for_gantt(data_df=self.data_df, clf_name=self.clf_name, fps=self.video_meta_data['fps'])
            self.final_gantt_img = self.create_gantt_img(bouts_df=self.bouts_df, clf_name=self.clf_name, image_index=len(self.data_df), fps=self.video_meta_data['fps'], gantt_img_title=f"Behavior gantt chart (entire session, length (s): {self.video_meta_data['video_length_s']}, frames: {self.video_meta_data['frame_count']})")
            self.final_gantt_img = self.resize_gantt(self.final_gantt_img, self.video_meta_data["height"])
            video_size = (int(self.video_meta_data["width"] + self.final_gantt_img.shape[1]), int(self.video_meta_data["height"]))
            writer = cv2.VideoWriter(self.vid_output_path, fourcc, self.video_meta_data["fps"], video_size)
        else:
            video_size = (int(self.video_meta_data["width"]), int(self.video_meta_data["height"]))
            writer = cv2.VideoWriter(self.vid_output_path, fourcc, self.video_meta_data["fps"], video_size)

        self._get_styles()
        self.data_df = self.data_df.head(min(len(self.data_df), self.video_meta_data["frame_count"]))
        frm_cnt, clf_frm_cnt = 0, 0
        print(f"Creating single core validation visualization for video {self.video_meta_data['video_name']} and classifier {self.clf_name}...")
        while (cap.isOpened()) and (frm_cnt < len(self.data_df)):
            ret, frame = cap.read()
            clf_val = int(self.data_df.loc[frm_cnt, self.clf_name])
            clf_frm_cnt += clf_val
            if self.show_pose:
                for animal_cnt, (animal_name, animal_data) in enumerate(self.animal_bp_dict.items()):
                    for bp_cnt, bp in enumerate(range(len(animal_data["X_bps"]))):
                        x_header, y_header = (animal_data["X_bps"][bp], animal_data["Y_bps"][bp])
                        animal_cords = tuple(self.data_df.loc[self.data_df.index[frm_cnt], [x_header, y_header]])
                        cv2.circle(frame, (int(animal_cords[0]), int(animal_cords[1])), self.video_circle_size, self.clr_lst[animal_cnt][bp_cnt], -1)
            if self.show_animal_names:
                for animal_cnt, (animal_name, animal_data) in enumerate(self.animal_bp_dict.items()):
                    x_header, y_header = ( animal_data["X_bps"][0], animal_data["Y_bps"][0])
                    animal_cords = tuple(self.data_df.loc[self.data_df.index[frm_cnt], [x_header, y_header]])
                    cv2.putText(frame, animal_name, (int(animal_cords[0]), int(animal_cords[1])), self.font, self.video_font_size, self.clr_lst[animal_cnt][0], self.video_text_thickness)
            target_timer = round((1 / self.video_meta_data['fps']) * clf_frm_cnt, 2)
            frame = PlottingMixin().put_text(img=frame, text="TIMER:", pos=(TextOptions.BORDER_BUFFER_Y.value, self.video_space_size), font_size=self.video_font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_color=(255, 255, 255))
            addSpacer = 2
            frame = PlottingMixin().put_text(img=frame, text=(f"{self.clf_name} {target_timer}s"), pos=(TextOptions.BORDER_BUFFER_Y.value, self.video_space_size * addSpacer), font_size=self.video_font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_color=(255, 255, 255))
            addSpacer += 1
            frame = PlottingMixin().put_text(img=frame, text="ENSEMBLE PREDICTION:", pos=(TextOptions.BORDER_BUFFER_Y.value, self.video_space_size * addSpacer), font_size=self.video_font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_color=(255, 255, 255))
            addSpacer += 2
            if clf_val == 1:
                frame = PlottingMixin().put_text(img=frame, text=self.clf_name, pos=(TextOptions.BORDER_BUFFER_Y.value, self.video_space_size * addSpacer), font_size=self.video_font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_color=TextOptions.COLOR.value)
                addSpacer += 1
            if self.create_gantt == 1:
                frame = np.concatenate((frame, self.final_gantt_img), axis=1)
            elif self.create_gantt == 2:
                gantt_img = self.create_gantt_img(bouts_df=self.bouts_df, clf_name=self.clf_name, image_index=len(self.data_df), fps=self.video_meta_data['fps'], gantt_img_title=f"Behavior gantt chart: {self.clf_name}")
                gantt_img = self.resize_gantt(gantt_img, self.video_meta_data["height"])
                frame = np.concatenate((frame, gantt_img), axis=1)
            frame = cv2.resize(frame, video_size, interpolation=cv2.INTER_LINEAR)
            writer.write(np.uint8(frame))
            print(f"Frame created: for video {self.feature_filename} ({frm_cnt + 1} / {len(self.data_df)})...")
            frm_cnt += 1

        cap.release()
        writer.release()
        self.timer.stop_timer()
        stdout_success(msg=f"Validation video saved at {self.vid_output_path}", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


# test = ValidateModelOneVideo(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini",
#                              feature_path=r"D:\troubleshooting\mitra\project_folder\csv\features_extracted\592_MA147_CNO1_0515.csv",
#                              model_path=r"C:\troubleshooting\mitra\models\generated_models\grroming_undersample_2_1000\grooming.sav",
#                              create_gantt=1,
#                              show_pose=True,
#                              show_animal_names=True)
# test.run()
# ValidationVideoPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                      feature_path=r"C:\troubleshooting\mitra\project_folder\csv\features_extracted\501_MA142_Gi_CNO_0521.csv",
#                      model_path=r"C:\troubleshooting\mitra\models\generated_models\grroming_undersample_2_1000\grooming.sav",
#                      discrimination_threshold=0.4,
#                      shortest_bout=500)







# test = ValidateModelOneVideo(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/project_config.ini',
#                              feature_file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/features_extracted/SI_DAY3_308_CD1_PRESENT.csv',
#                              model_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/models/generated_models/Running.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt=None)
# test.run()

# test = ValidateModelOneVideo(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                              feature_file_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/features_extracted/Together_1.csv',
#                              model_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/models/generated_models/Attack.sav',
#                              discrimination_threshold=0.6,
#                              shortest_bout=50,
#                              settings={'pose': True, 'animal_names': True, 'styles': None},
#                              create_gantt=2)
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
