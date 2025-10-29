__author__ = "Simon Nilsson"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import functools
import multiprocessing
import os
import platform
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import cv2
import imutils
import pandas as pd

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int, check_str, check_valid_boolean,
                                check_video_and_data_frm_count_align)
from simba.utils.data import create_color_palette, plug_holes_shortest_bout
from simba.utils.enums import Options, TextOptions
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory, find_core_cnt,
                                    get_fn_ext, get_video_meta_data, read_df,
                                    read_pickle, write_df)
from simba.utils.warnings import FrameRangeWarning, NoDataFoundWarning


def _validation_video_mp(data: pd.DataFrame,
                        bp_dict: dict,
                        video_save_dir: str,
                        video_path: str,
                        text_thickness: int,
                        text_opacity: float,
                        font_size: int,
                        text_spacing: int,
                        circle_size: int,
                        show_pose: bool,
                        show_animal_bounding_boxes: bool,
                        show_animal_names: bool,
                        gantt_setting: Union[int, None],
                        final_gantt: Optional[np.ndarray],
                        clf_data: np.ndarray,
                        clrs: List[List],
                        clf_name: str,
                        bouts_df: pd.DataFrame,
                        conf_data: np.ndarray):

    def _put_text(img: np.ndarray,
                  text: str,
                  pos: Tuple[int, int],
                  font_size: int,
                  font_thickness: Optional[int] = 2,
                  font: Optional[int] = cv2.FONT_HERSHEY_DUPLEX,
                  text_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
                  text_color_bg: Optional[Tuple[int, int, int]] = (0, 0, 0),
                  text_bg_alpha: float = 0.8):

        x, y = pos
        text_size, px_buffer = cv2.getTextSize(text, font, font_size, font_thickness)
        w, h = text_size
        overlay, output = img.copy(), img.copy()
        cv2.rectangle(overlay, (x, y-h), (x + w, y + px_buffer), text_color_bg, -1)
        cv2.addWeighted(overlay, text_bg_alpha, output, 1 - text_bg_alpha, 0, output)
        cv2.putText(output, text, (x, y), font, font_size, text_color, font_thickness)
        return output


    def _create_gantt(bouts_df: pd.DataFrame,
                      clf_name: str,
                      image_index: int,
                      fps: int,
                      header_font_size: int = 24,
                      label_font_size: int = 12):

        fig, ax = plt.subplots(figsize=(final_gantt.shape[1] / dpi, final_gantt.shape[0] / dpi))
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
        ax.set_xlabel("Session (s)", fontsize=label_font_size)
        ax.set_ylabel(clf_name, fontsize=label_font_size)
        ax.set_title(f"{clf_name} GANTT CHART", fontsize=header_font_size)
        ax.set_yticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.grid(True)
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.array(np.uint8(np.array(canvas.renderer._renderer)))[:, :, :3]
        plt.close(fig)
        return img

    dpi = plt.rcParams["figure.dpi"]
    fourcc, font = cv2.VideoWriter_fourcc(*"mp4v"), cv2.FONT_HERSHEY_DUPLEX
    cap = cv2.VideoCapture(video_path)
    video_meta_data = get_video_meta_data(video_path=video_path, fps_as_int=False)
    batch_id, batch_data = data[0], data[1]
    start_frm, current_frm, end_frm = batch_data.index[0], batch_data.index[0], batch_data.index[-1]
    video_save_path = os.path.join(video_save_dir, f"{batch_id}.mp4")
    if gantt_setting is not None:
        video_size = (int(video_meta_data["width"] + final_gantt.shape[1]), int(video_meta_data["height"]))
        writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], video_size)
    else:
        video_size = (int(video_meta_data["width"]), int(video_meta_data["height"]))
        writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], video_size)
    cap.set(1, start_frm)
    while (current_frm <= end_frm) & (current_frm <= video_meta_data["frame_count"]):
        clf_frm_cnt = np.sum(clf_data[0:current_frm])
        ret, img = cap.read()
        if ret:
            frm_timer = SimbaTimer(start=True)
            if show_pose:
                for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                    for bp_cnt, bp in enumerate(range(len(animal_data["X_bps"]))):
                        x_header, y_header = (animal_data["X_bps"][bp], animal_data["Y_bps"][bp])
                        animal_cords = tuple(batch_data.loc[current_frm, [x_header, y_header]])
                        cv2.circle(img, (int(animal_cords[0]), int(animal_cords[1])), circle_size, clrs[animal_cnt][bp_cnt], -1)
            if show_animal_names:
                for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                    x_header, y_header = (animal_data["X_bps"][0], animal_data["Y_bps"][0],)
                    animal_cords = tuple(batch_data.loc[current_frm, [x_header, y_header]])
                    cv2.putText(img, animal_name, (int(animal_cords[0]), int(animal_cords[1])), font, font_size, clrs[animal_cnt][0], text_thickness)
            if show_animal_bounding_boxes:
                for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                    animal_headers = [val for pair in zip(animal_data["X_bps"], animal_data["Y_bps"]) for val in pair]
                    animal_cords = batch_data.loc[current_frm, animal_headers].values.reshape(-1, 2).astype(np.int32)
                    try:
                        bbox = GeometryMixin().keypoints_to_axis_aligned_bounding_box(keypoints=animal_cords.reshape(-1, len(animal_cords), 2).astype(np.int32))
                        cv2.polylines(img, [bbox], True, clrs[animal_cnt][0], thickness=text_thickness, lineType=-1)
                    except:
                        pass
            target_timer = round((1 / video_meta_data["fps"]) * clf_frm_cnt, 2)
            img = _put_text(img=img, text="BEHAVIOR TIMER:", pos=(TextOptions.BORDER_BUFFER_Y.value, text_spacing), font_size=font_size, font_thickness=TextOptions.TEXT_THICKNESS.value)
            addSpacer = 2
            img = _put_text(img=img, text=f"{clf_name} {target_timer}s", pos=(TextOptions.BORDER_BUFFER_Y.value, text_spacing * addSpacer), font_size=font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_bg_alpha=text_opacity)
            addSpacer += 1
            if conf_data is not None:
                img = _put_text(img=img, text=f"{clf_name} PROBABILITY: {round(conf_data[current_frm], 4)}", pos=(TextOptions.BORDER_BUFFER_Y.value, text_spacing * addSpacer), font_size=font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_bg_alpha=text_opacity)
                addSpacer += 1
            img = _put_text(img=img, text="ENSEMBLE PREDICTION:", pos=(TextOptions.BORDER_BUFFER_Y.value, text_spacing * addSpacer), font_size=font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_bg_alpha=text_opacity)
            addSpacer += 1
            if clf_data[current_frm] == 1:
                img = _put_text(img=img, text=clf_name, pos=(TextOptions.BORDER_BUFFER_Y.value, text_spacing * addSpacer), font_size=font_size, font_thickness=TextOptions.TEXT_THICKNESS.value, text_color=TextOptions.COLOR.value, text_bg_alpha=text_opacity)
            addSpacer += 1
            if gantt_setting == 1:
                img = np.concatenate((img, final_gantt), axis=1)
            elif gantt_setting == 2:
                gantt_img = _create_gantt(bouts_df, clf_name, current_frm, video_meta_data["fps"], header_font_size=9, label_font_size=12)
                gantt_img = imutils.resize(gantt_img, height=video_meta_data["height"])
                img = np.concatenate((img, gantt_img), axis=1)
            img = cv2.resize(img, video_size, interpolation=cv2.INTER_LINEAR)
            writer.write(np.uint8(img))
            current_frm += 1
            frm_timer.stop_timer()
            print(f"Multi-processing video frame {current_frm} on core {batch_id}...(elapsed time: {frm_timer.elapsed_time_str}s)")
        else:
            FrameRangeWarning(msg=f'Frame {current_frm} could not be read in video {video_path}. The video contains {video_meta_data["frame_count"]} frames while the data file contains data for {len(batch_data)} frames. Consider re-encoding the video, or make sure the pose-estimation data and associated video contains the same number of frames. ', source=_validation_video_mp.__name__)
            break

    cap.release()
    writer.release()
    return batch_id


class ValidateModelOneVideoMultiprocess(ConfigReader, PlottingMixin, TrainModelMixin):
    """
    Create classifier validation video for a single input video using multiprocessing for improved performance.
    
    This class generates validation videos that overlay classifier predictions, pose estimations, and 
    optional Gantt charts onto the original video using multiple CPU cores for faster processing. 
    Results are stored in the `project_folder/frames/output/validation` directory.

    .. note::
       This multiprocess version provides significant speed improvements over the single-core 
       :class:`simba.plotting.single_run_model_validation_video.ValidateModelOneVideo` class.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param Union[str, os.PathLike] feature_path: Path to SimBA file (parquet or CSV) containing pose-estimation and feature data.
    :param Union[str, os.PathLike] model_path: Path to pickled classifier object (.sav file).
    :param bool show_pose: If True, overlay pose estimation keypoints on the video. Default: True.
    :param bool show_animal_names: If True, display animal names near the first body part. Default: False.
    :param Optional[int] font_size: Font size for text overlays. If None, automatically calculated based on video dimensions.
    :param Optional[str] bp_palette: Optional name of the palette to use to color the animal body-parts (e.g., Pastel1). If None, ``spring`` is used.


    :param Optional[int] circle_size: Size of pose estimation circles. If None, automatically calculated based on video dimensions.
    :param Optional[int] text_spacing: Spacing between text lines. If None, automatically calculated.
    :param Optional[int] text_thickness: Thickness of text overlay. If None, uses default value.
    :param Optional[float] text_opacity: Opacity of text overlays (0.1-1.0). If None, defaults to 0.8.
    :param float discrimination_threshold: Classification probability threshold (0.0-1.0). Default: 0.0.
    :param int shortest_bout: Minimum classified bout length in milliseconds. Bouts shorter than this will be reclassified as absent. Default: 0.
    :param int core_cnt: Number of CPU cores to use for processing. If -1, uses all available cores. Default: -1.
    :param Optional[Union[None, int]] create_gantt: Gantt chart creation option:
        
        - None: No Gantt chart
        - 1: Static Gantt chart (final frame only, faster)
        - 2: Dynamic Gantt chart (updated per frame)


    ..  youtube:: UOLSj7DGKRo
       :width: 640
       :height: 480
       :align: center

    .. video:: _static/img/T1.webm
       :width: 1000
       :autoplay:
       :loop:

    :example:
    >>> # Create multiprocess validation video with dynamic Gantt chart
    >>> validator = ValidateModelOneVideoMultiprocess(
    ...     config_path=r'/path/to/project_config.ini',
    ...     feature_path=r'/path/to/features.csv',
    ...     model_path=r'/path/to/classifier.sav',
    ...     show_pose=True,
    ...     show_animal_names=True,
    ...     discrimination_threshold=0.6,
    ...     shortest_bout=500,
    ...     core_cnt=4,
    ...     create_gantt=2
    ... )
    >>> validator.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 feature_path: Union[str, os.PathLike],
                 model_path: Union[str, os.PathLike],
                 show_pose: bool = True,
                 show_animal_names: bool = False,
                 show_animal_bounding_boxes: bool = False,
                 show_clf_confidence: bool = False,
                 font_size: Optional[bool] = None,
                 circle_size: Optional[int] = None,
                 text_spacing: Optional[int] = None,
                 text_thickness: Optional[int] = None,
                 text_opacity: Optional[float] = None,
                 bp_palette: Optional[str] = None,
                 discrimination_threshold: float = 0.0,
                 shortest_bout: int = 0.0,
                 core_cnt: int = -1,
                 create_gantt: Optional[Union[None, int]] = None):


        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        TrainModelMixin.__init__(self)
        check_file_exist_and_readable(file_path=config_path)
        check_file_exist_and_readable(file_path=feature_path)
        check_file_exist_and_readable(file_path=model_path)
        check_valid_boolean(value=[show_pose], source=f'{self.__class__.__name__} show_pose', raise_error=True)
        check_valid_boolean(value=[show_animal_names], source=f'{self.__class__.__name__} show_animal_names', raise_error=True)
        check_valid_boolean(value=[show_animal_bounding_boxes], source=f'{self.__class__.__name__} show_animal_bounding_boxes', raise_error=True)
        check_valid_boolean(value=[show_clf_confidence], source=f'{self.__class__.__name__} show_clf_confidence', raise_error=True)
        check_int(name=f"{self.__class__.__name__} core_cnt", value=core_cnt, min_value=-1, unaccepted_vals=[0])
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
        if bp_palette is not None:
            self.bp_palette = []
            check_str(name=f'{self.__class__.__name__} bp_palette', value=bp_palette, options=(Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value))
            for animal in range(self.animal_cnt):
                self.bp_palette.append(create_color_palette(pallete_name=bp_palette, increments=(int(len(self.body_parts_lst)/self.animal_cnt) +1), as_int=True))
        else:
            self.bp_palette = deepcopy(self.clr_lst)
        _, self.feature_filename, ext = get_fn_ext(feature_path)
        self.video_path = self.find_video_of_file(self.video_dir, self.feature_filename)
        self.video_meta_data = get_video_meta_data(video_path=self.video_path, fps_as_int=False)
        self.clf_name, self.feature_file_path = (os.path.basename(model_path).replace(".sav", ""), feature_path)
        self.vid_output_path = os.path.join(self.single_validation_video_save_dir, f"{self.feature_filename} {self.clf_name}.mp4")
        self.clf_data_save_path = os.path.join(self.clf_data_validation_dir, f"{self.feature_filename }.csv")
        self.show_pose, self.show_animal_names = show_pose, show_animal_names
        self.font_size, self.circle_size, self.text_spacing, self.show_clf_confidence = font_size, circle_size, text_spacing, show_clf_confidence
        self.text_opacity, self.text_thickness, self.show_animal_bounding_boxes = text_opacity, text_thickness, show_animal_bounding_boxes
        self.clf = read_pickle(data_path=model_path, verbose=True)
        self.data_df = read_df(feature_path, self.file_type)
        self.x_df = self.drop_bp_cords(df=self.data_df)
        self.discrimination_threshold, self.shortest_bout, self.create_gantt = float(discrimination_threshold), shortest_bout, create_gantt
        check_video_and_data_frm_count_align(video=self.video_path, data=self.data_df, name=self.feature_filename, raise_error=False)
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.temp_dir = os.path.join(self.single_validation_video_save_dir, "temp")
        self.video_save_path = os.path.join(self.single_validation_video_save_dir, f"{self.feature_filename}.mp4")
        create_directory(paths=self.temp_dir, overwrite=True)
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)

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
        self.data_df[self.prob_col_name] = self.clf_predict_proba(clf=self.clf, x_df=self.x_df, model_name=self.clf_name, data_path=self.feature_file_path)
        self.data_df[self.clf_name] = np.where(self.data_df[self.prob_col_name] > self.discrimination_threshold, 1, 0)
        if self.shortest_bout > 1:
            self.data_df = plug_holes_shortest_bout(data_df=self.data_df, clf_name=self.clf_name, fps=self.video_meta_data['fps'], shortest_bout=self.shortest_bout)
        _ = write_df(df=self.data_df, file_type=self.file_type, save_path=self.clf_data_save_path)
        print(f"Predictions created for video {self.feature_filename} (creating video, follow progressin OS terminal)...")
        self._get_styles()
        if self.create_gantt is not None:
            self.bouts_df = self.get_bouts_for_gantt(data_df=self.data_df, clf_name=self.clf_name, fps=self.video_meta_data['fps'])
            self.final_gantt_img = self.create_gantt_img(self.bouts_df ,self.clf_name,len(self.data_df), self.video_meta_data['fps'],f"Behavior gantt chart (entire session, length (s): {self.video_meta_data['video_length_s']}, frames: {self.video_meta_data['frame_count']})", header_font_size=9, label_font_size=12)
            self.final_gantt_img = self.resize_gantt(self.final_gantt_img, self.video_meta_data["height"])
        else:
            self.bouts_df, self.final_gantt_img = None, None
        conf_data = self.data_df[self.prob_col_name].values if self.show_clf_confidence else None

        self.data_df = self.data_df.head(min(len(self.data_df), self.video_meta_data["frame_count"]))
        data = np.array_split(self.data_df, self.core_cnt)
        data = [(i, j) for i, j in enumerate(data)]

        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_validation_video_mp,
                                          bp_dict=self.animal_bp_dict,
                                          video_save_dir=self.temp_dir,
                                          text_thickness=self.video_text_thickness,
                                          text_opacity=self.video_text_opacity,
                                          font_size=self.video_font_size,
                                          text_spacing=self.video_space_size,
                                          circle_size=self.video_circle_size,
                                          video_path=self.video_path,
                                          show_pose=self.show_pose,
                                          show_animal_names=self.show_animal_names,
                                          show_animal_bounding_boxes=self.show_animal_bounding_boxes,
                                          gantt_setting=self.create_gantt,
                                          final_gantt=self.final_gantt_img,
                                          clf_data=self.data_df[self.clf_name].values,
                                          clrs=self.bp_palette,
                                          clf_name=self.clf_name,
                                          bouts_df=self.bouts_df,
                                          conf_data=conf_data)

            for cnt, result in enumerate(pool.imap(constants, data, chunksize=self.multiprocess_chunksize)):
                print(f"Image batch {result} complete, Video {self.feature_filename}...")
        pool.terminate()
        pool.join()
        concatenate_videos_in_folder(in_folder=self.temp_dir, save_path=self.video_save_path)
        self.timer.stop_timer()
        stdout_success(msg=f"Video complete, saved at {self.video_save_path}", elapsed_time=self.timer.elapsed_time_str)

#
# if __name__ == "__main__":
#     test = ValidateModelOneVideoMultiprocess(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini",
#                                              feature_path=r"D:\troubleshooting\mitra\project_folder\csv\features_extracted\592_MA147_CNO1_0515.csv",
#                                              model_path=r"C:\troubleshooting\mitra\models\validations\rearing_5\rearing.sav",
#                                              create_gantt=2,
#                                              show_pose=True,
#                                              show_animal_names=True,
#                                              core_cnt=13,
#                                              show_clf_confidence=True,
#                                              discrimination_threshold=0.20)
#     test.run()


#
# if __name__ == "__main__":
#     test = ValidateModelOneVideoMultiprocess(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                                     feature_file_path=r"C:\troubleshooting\mitra\project_folder\csv\features_extracted\844_MA131_gq_CNO_0624.csv",
#                                     model_path=r"C:\troubleshooting\mitra\models\validations\lay-on-belly_1\lay-on-belly.sav",
#                                     discrimination_threshold=0.35,
#                                     shortest_bout=200,
#                                     cores=-1,
#                                     settings={'pose': True, 'animal_names': False, 'styles': None},
#                                     create_gantt=2)
#     test.run()





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
