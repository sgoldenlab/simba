__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
import platform
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_float, check_if_valid_rgb_tuple,
                                check_int, check_nvidea_gpu_available,
                                check_str, check_that_column_exist,
                                check_valid_boolean,
                                check_video_and_data_frm_count_align)
from simba.utils.data import create_color_palette, detect_bouts
from simba.utils.enums import ConfigKey, Dtypes, Options, TagNames, TextOptions
from simba.utils.errors import (InvalidInputError, NoDataError,
                                NoSpecifiedOutputError)
from simba.utils.lookups import get_current_time
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory,
                                    find_all_videos_in_project, find_core_cnt,
                                    get_fn_ext, get_video_meta_data,
                                    read_config_entry, read_df)
from simba.utils.warnings import FrameRangeWarning


def _multiprocess_sklearn_video(data: pd.DataFrame,
                                bp_dict: dict,
                                video_save_dir: str,
                                frame_save_dir: str,
                                clf_cumsum: dict,
                                rotate: bool,
                                video_path: str,
                                print_timers: bool,
                                video_setting: bool,
                                frame_setting: bool,
                                pose_threshold: float,
                                show_pose: bool,
                                show_animal_names: bool,
                                show_bbox: bool,
                                circle_size: int,
                                font_size: int,
                                space_size: int,
                                text_thickness: int,
                                text_opacity: float,
                                text_bg_clr: Tuple[int, int, int],
                                text_color: Tuple[int, int, int],
                                pose_clr_lst: List[Tuple[int, int, int]],
                                show_gantt: Optional[int],
                                bouts_df: Optional[pd.DataFrame],
                                final_gantt: Optional[np.ndarray],
                                gantt_clrs: List[Tuple[float, float, float]],
                                clf_names: List[str]):

    fourcc, font = cv2.VideoWriter_fourcc(*"mp4v"), cv2.FONT_HERSHEY_DUPLEX
    video_meta_data = get_video_meta_data(video_path=video_path)
    if rotate:
        video_meta_data["height"], video_meta_data["width"] = (video_meta_data['width'], video_meta_data['height'])
    cap = cv2.VideoCapture(video_path)
    batch, data = data
    start_frm, current_frm, end_frm = (data["index"].iloc[0], data["index"].iloc[0], data["index"].iloc[-1])
    if video_setting:
        video_save_path = os.path.join(video_save_dir, f"{batch}.mp4")
        if show_gantt is None:
            video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
        else:
            video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (int(video_meta_data["width"] + final_gantt.shape[1]), video_meta_data["height"]))
    cap.set(1, start_frm)
    while current_frm < end_frm:
        ret, img = cap.read()
        if ret:
            clr_cnt = 0
            for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                if show_pose:
                    for bp_no in range(len(animal_data["X_bps"])):
                        x_bp, y_bp, p_bp = (animal_data["X_bps"][bp_no], animal_data["Y_bps"][bp_no], animal_data["P_bps"][bp_no])
                        bp_cords = data.loc[current_frm, [x_bp, y_bp, p_bp]]
                        if bp_cords[p_bp] >= pose_threshold:
                            img = cv2.circle(img, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), circle_size, pose_clr_lst[clr_cnt], -1)
                        clr_cnt += 1
                if show_animal_names:
                    x_bp, y_bp, p_bp = (animal_data["X_bps"][0], animal_data["Y_bps"][0], animal_data["P_bps"][0])
                    bp_cords = data.loc[current_frm, [x_bp, y_bp, p_bp]]
                    img = cv2.putText(img, animal_name, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), font, font_size, pose_clr_lst[0],  text_thickness)
                if show_bbox:
                    animal_headers = [val for pair in zip(animal_data["X_bps"], animal_data["Y_bps"]) for val in pair]
                    animal_cords = data.loc[current_frm, animal_headers].values.reshape(-1, 2).astype(np.int32)
                    try:
                        bbox = GeometryMixin().keypoints_to_axis_aligned_bounding_box(keypoints=animal_cords.reshape(-1, len(animal_cords), 2).astype(np.int32))
                        img = cv2.polylines(img, [bbox], True, pose_clr_lst[animal_cnt], thickness=circle_size, lineType=cv2.LINE_AA)
                    except Exception as e:
                        pass
            if rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if show_gantt == 1:
                img = np.concatenate((img, final_gantt), axis=1)
            elif show_gantt == 2:
                bout_rows = bouts_df.loc[bouts_df["End_frame"] <= current_frm]
                gantt_plot = PlottingMixin().make_gantt_plot(x_length=current_frm + 1,
                                                       bouts_df=bout_rows,
                                                       clf_names=clf_names,
                                                       fps=video_meta_data['fps'],
                                                       width=video_meta_data['width'],
                                                       height=video_meta_data['height'],
                                                       font_size=12,
                                                       font_rotation=90,
                                                       video_name=video_meta_data['video_name'],
                                                       save_path=None,
                                                       palette=gantt_clrs)
                img = np.concatenate((img, gantt_plot), axis=1)
            if print_timers:
                img = PlottingMixin().put_text(img=img, text="TIMERS:", pos=(TextOptions.BORDER_BUFFER_Y.value, ((video_meta_data["height"] - video_meta_data["height"]) + space_size)), font_size=font_size, font_thickness=text_thickness, font=font, text_bg_alpha=text_opacity, text_color_bg=text_bg_clr, text_color=text_color)
            add_spacer = 2
            for clf_name, clf_time_df in clf_cumsum.items():
                frame_results = clf_time_df.loc[current_frm]
                clf_time = round(frame_results / video_meta_data['fps'], 2)
                if print_timers:
                    img = PlottingMixin().put_text(img=img, text=f"{clf_name} {clf_time}",pos=(TextOptions.BORDER_BUFFER_Y.value, ((video_meta_data["height"] - video_meta_data["height"]) + space_size * add_spacer)), font_size=font_size,  font_thickness=text_thickness, font=font, text_bg_alpha=text_opacity, text_color_bg=text_bg_clr, text_color=text_color)
                    add_spacer += 1
            img = PlottingMixin().put_text(img=img, text="ENSEMBLE PREDICTION:", pos=(TextOptions.BORDER_BUFFER_Y.value, ((video_meta_data["height"] - video_meta_data["height"]) + space_size * add_spacer)), font_size=font_size, font_thickness=text_thickness, font=font, text_bg_alpha=text_opacity, text_color_bg=text_bg_clr, text_color=text_color)
            add_spacer += 1
            for clf_name in clf_cumsum.keys():
                if data.loc[current_frm, clf_name] == 1:
                    img = PlottingMixin().put_text(img=img, text=clf_name, pos=(TextOptions.BORDER_BUFFER_Y.value, (video_meta_data["height"] - video_meta_data["height"]) + space_size * add_spacer), font_size=font_size, font_thickness=text_thickness, font=font, text_color=TextOptions.COLOR.value, text_bg_alpha=text_opacity)
                    add_spacer += 1
            if video_setting:
                video_writer.write(img.astype(np.uint8))
            if frame_setting:
                frame_save_name = os.path.join(frame_save_dir, f"{current_frm}.png")
                cv2.imwrite(frame_save_name, img)
            current_frm += 1
            print(f"Multi-processing video frame {current_frm} on core {batch}...")
        else:
            FrameRangeWarning(msg=f'Could not read frame {current_frm} in video {video_path}. Stopping video creation.')
            break

    cap.release()
    if video_setting:
        video_writer.release()
    return batch


class PlotSklearnResultsMultiProcess(ConfigReader, TrainModelMixin, PlottingMixin):
    """
    Plot classification results on videos. Results are stored in the
    `project_folder/frames/output/sklearn_results` directory of the SimBA project.

    .. seealso::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-10-sklearn-visualization__.
        For non-multiptocess class, see :meth:`simba.plotting.plot_clf_results.PlotSklearnResultsSingleCore`.

    .. image:: _static/img/sklearn_visualization.gif
       :width: 600
       :align: center

    .. video:: _static/img/T1.webm
       :width: 1000
       :autoplay:
       :loop:

    ..  youtube:: Frq6mMcaHBc
       :width: 640
       :height: 480
       :align: center

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Optional[bool] video_setting: If True, SimBA will create compressed videos. Default True.
    :param Optional[bool] frame_setting: If True, SimBA will create individual frames. Default True.
    :param Optional[int] cores: Number of cores to use. Pass ``-1`` for all available cores.
    :param Optional[str] video_file_path: Path to video file to create classification visualizations for. If None, then all the videos in the csv/machine_results will be used. Default None.
    :param Optional[Union[Dict[str, float], bool]] text_settings: Dictionary holding the circle size, font size, spacing size, and text thickness of the printed text. If None, then these are autocomputed.
    :param Optional[bool] rotate: If True, the output video will be rotated 90 degrees from the input. Default False.
    :param Optional[bool] show_bbox: If True, axis-aligned bounding boxes created encompassing each anmals pose and displayed. Default True.
    :param Optional[str] palette: The name of the palette used for the pose-estimation key-points. Default ``Set1``.
    :param Optional[bool] print_timers: If True, the output video will have the cumulative time of the classified behaviours overlaid. Default True.

    :example:
    >>> text_settings = {'circle_scale': 5, 'font_size': 0.528, 'spacing_scale': 28, 'text_thickness': 2}
    >>> clf_plotter = PlotSklearnResultsMultiProcess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
    >>>                                              video_setting=True,
    >>>                                              frame_setting=False,
    >>>                                              rotate=False,
    >>>                                              video_file_path='Trial    10.mp4',
    >>>                                              cores=5,
    >>>                                              text_settings=False)
    >>> clf_plotter.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_setting: bool = True,
                 frame_setting: bool = False,
                 video_paths: Optional[Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]]] = None,
                 rotate: bool = False,
                 animal_names: bool = False,
                 show_pose: bool = True,
                 font_size: Optional[Union[int, float]] = None,
                 space_size: Optional[Union[int, float]] = None,
                 text_thickness: Optional[Union[int, float]] = None,
                 text_opacity: Optional[Union[int, float]] = None,
                 circle_size: Optional[Union[int, float]] = None,
                 pose_palette: Optional[str] = 'Set1',
                 print_timers: bool = True,
                 show_bbox: bool = False,
                 show_gantt: Optional[int] = None,
                 text_clr: Tuple[int, int, int] = (255, 255, 255),
                 text_bg_clr: Tuple[int, int, int] = (0, 0, 0),
                 gpu: bool = False,
                 core_cnt: int = -1):


        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        PlottingMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        for i in [video_setting, frame_setting, rotate, print_timers, animal_names, show_pose, gpu, show_bbox]:
            check_valid_boolean(value=i, source=self.__class__.__name__, raise_error=True)
        if (not video_setting) and (not frame_setting):
            raise NoSpecifiedOutputError(msg="Please choose to create a video and/or frames. SimBA found that you ticked neither video and/or frames", source=self.__class__.__name__)
        if font_size is not None: check_float(name=f'{self.__class__.__name__} font_size', value=font_size, min_value=0.1)
        if space_size is not None: check_float(name=f'{self.__class__.__name__} space_size', value=space_size, min_value=0.1)
        if text_thickness is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=text_thickness, min_value=0.1)
        if circle_size is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=circle_size, min_value=0.1)
        if circle_size is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=circle_size, min_value=0.1)
        if text_opacity is not None: check_float(name=f'{self.__class__.__name__} text_opacity', value=text_opacity, min_value=0.1)
        pose_palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        check_str(name=f'{self.__class__.__name__} pose_palette', value=pose_palette, options=pose_palettes)
        self.clr_lst = create_color_palette(pallete_name=pose_palette, increments=len(self.body_parts_lst)+1)
        check_if_valid_rgb_tuple(data=text_clr, source=f'{self.__class__.__name__} text_clr')
        check_if_valid_rgb_tuple(data=text_bg_clr, source=f'{self.__class__.__name__} text_bg_clr')
        if show_gantt is not None:
            check_int(name=f"{self.__class__.__name__} show_gantt", value=show_gantt, max_value=2, min_value=1)
        self.video_paths, self.print_timers = video_paths, print_timers
        if self.video_paths is None:
            self.video_paths = find_all_videos_in_project(videos_dir=self.video_dir)
            if len(self.video_paths) == 0:
                raise NoDataError(msg=f'Cannot create classification videos. No videos exist in {self.video_dir} directory', source=self.__class__.__name__)
        self.video_setting, self.frame_setting, self.rotate = video_setting, frame_setting, rotate
        self.circle_size, self.font_size, self.animal_names, self.text_opacity = circle_size, font_size, animal_names, text_opacity
        self.text_thickness, self.space_size, self.show_pose, self.pose_palette = text_thickness, space_size, show_pose, pose_palette
        self.text_color, self.text_bg_color, self.show_bbox, self.show_gantt = text_clr, text_bg_clr, show_bbox, show_gantt
        self.gpu = True if check_nvidea_gpu_available() and gpu else False
        self.pose_threshold = read_config_entry(self.config, ConfigKey.THRESHOLD_SETTINGS.value, ConfigKey.SKLEARN_BP_PROB_THRESH.value, Dtypes.FLOAT.value, 0.00)
        if not os.path.exists(self.sklearn_plot_dir):
            os.makedirs(self.sklearn_plot_dir)
        if isinstance(self.video_paths, str): self.video_paths = [video_paths]
        elif isinstance(self.video_paths, list): self.video_paths = video_paths
        else:
            raise InvalidInputError(msg=f'video_paths has to be a path of a list of paths. Got {type(video_paths)}', source=self.__class__.__name__)
        for video_path in self.video_paths:
            video_name = get_fn_ext(filepath=video_path)[1]
            data_path = os.path.join(self.machine_results_dir, f'{video_name}.{self.file_type}')
            if not os.path.isfile(data_path): raise NoDataError(msg=f'Cannot create classification videos for {video_name}. Expected classification data at location {data_path} but file does not exist', source=self.__class__.__name__)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.core_cnt = find_core_cnt()[0] if int(core_cnt) == -1 or int(core_cnt) > find_core_cnt()[0] else int(core_cnt)
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)

    def __get_print_settings(self):
        optimal_circle_size = self.get_optimal_circle_size(frame_size=(self.video_meta_data["width"], self.video_meta_data["height"]), circle_frame_ratio=100)
        longest_str = str(max(['TIMERS:', 'ENSEMBLE PREDICTION:'] + self.clf_names, key=len))
        self.video_text_thickness = TextOptions.TEXT_THICKNESS.value if self.text_thickness is None else int(max(self.text_thickness, 1))
        optimal_font_size, _, optimal_spacing_scale = self.get_optimal_font_scales(text=longest_str, accepted_px_width=int(self.video_meta_data["width"] / 3), accepted_px_height=int(self.video_meta_data["height"] / 10), text_thickness=self.video_text_thickness)
        self.video_circle_size = optimal_circle_size if self.circle_size is None else int(max(1, self.circle_size))
        self.video_font_size = optimal_font_size if self.font_size is None else self.font_size
        self.video_space_size = optimal_spacing_scale if self.space_size is None else int(max(self.space_size, 1))
        self.video_text_opacity = 0.8 if self.text_opacity is None else float(self.text_opacity)

    def run(self):
        print(f'Creating {len(self.video_paths)} classification visualizations using {self.core_cnt} cores... ({get_current_time()})')
        for video_cnt, video_path in enumerate(self.video_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(video_path)
            self.data_path = os.path.join(self.machine_results_dir, f'{self.video_name}.{self.file_type}')
            self.data_df = read_df(self.data_path, self.file_type).reset_index(drop=True).fillna(0)
            if self.show_pose: check_that_column_exist(df=self.data_df, column_name=self.bp_col_names, file_name=self.data_path)
            self.video_meta_data = get_video_meta_data(video_path=video_path)
            height, width = deepcopy(self.video_meta_data["height"]), deepcopy(self.video_meta_data["width"])
            self.save_path = os.path.join(self.sklearn_plot_dir, f"{self.video_name}.mp4")
            self.video_frame_dir, self.video_temp_dir = None, None
            if self.video_setting:
                self.video_save_path = os.path.join(self.sklearn_plot_dir, f"{self.video_name}.mp4")
                self.video_temp_dir = os.path.join(self.sklearn_plot_dir, self.video_name, "temp")
                create_directory(paths=self.video_temp_dir, overwrite=True)
            if self.frame_setting:
                self.video_frame_dir = os.path.join(self.sklearn_plot_dir, self.video_name)
                create_directory(paths=self.video_temp_dir, overwrite=True)
            if self.rotate:
                self.video_meta_data["height"], self.video_meta_data["width"] = (width, height)
            check_video_and_data_frm_count_align(video=video_path, data=self.data_df, name=self.video_name, raise_error=False)
            check_that_column_exist(df=self.data_df, column_name=self.clf_names, file_name=self.data_path)
            self.__get_print_settings()
            if self.show_gantt is not None:
                self.gantt_clrs = create_color_palette(pallete_name=self.pose_palette, increments=len(self.clf_names) + 1, as_int=True, as_rgb_ratio=True)
                self.bouts_df = detect_bouts(data_df=self.data_df, target_lst=list(self.clf_names), fps=int(self.video_meta_data["fps"]))
                self.final_gantt_img = PlottingMixin().make_gantt_plot(x_length=len(self.data_df) + 1, bouts_df=self.bouts_df, clf_names=self.clf_names, fps=self.video_meta_data["fps"], width=self.video_meta_data["width"], height=self.video_meta_data["height"], font_size=12, font_rotation=90, video_name=self.video_meta_data["video_name"], save_path=None, palette=self.gantt_clrs)
                self.final_gantt_img = self.resize_gantt(self.final_gantt_img, self.video_meta_data["height"])
            else:
                self.bouts_df, self.final_gantt_img, self.gantt_clrs = None, None, None


            self.clf_cumsums = {}
            for clf_name in self.clf_names:
                self.clf_cumsums[clf_name] = self.data_df[clf_name].cumsum()

            self.data_df["index"] = self.data_df.index
            data = np.array_split(self.data_df, self.core_cnt)
            data = [(cnt, x) for (cnt, x) in enumerate(data)]

            with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
                constants = functools.partial(_multiprocess_sklearn_video,
                                              bp_dict=self.animal_bp_dict,
                                              video_save_dir=self.video_temp_dir,
                                              frame_save_dir=self.video_frame_dir,
                                              clf_cumsum=self.clf_cumsums,
                                              rotate=self.rotate,
                                              video_path=video_path,
                                              print_timers=self.print_timers,
                                              video_setting=self.video_setting,
                                              frame_setting=self.frame_setting,
                                              pose_threshold=self.pose_threshold,
                                              show_pose=self.show_pose,
                                              show_animal_names=self.animal_names,
                                              circle_size=self.video_circle_size,
                                              font_size=self.video_font_size,
                                              space_size=self.video_space_size,
                                              text_thickness=self.video_text_thickness,
                                              text_opacity=self.video_text_opacity,
                                              text_bg_clr=self.text_bg_color,
                                              text_color=self.text_color,
                                              pose_clr_lst=self.clr_lst,
                                              show_bbox=self.show_bbox,
                                              show_gantt=self.show_gantt,
                                              bouts_df=self.bouts_df,
                                              final_gantt=self.final_gantt_img,
                                              gantt_clrs=self.gantt_clrs,
                                              clf_names=self.clf_names)

                for cnt, result in enumerate(pool.imap(constants, data, chunksize=self.multiprocess_chunksize)):
                    print(f"Image batch {result} complete, Video {(video_cnt + 1)}/{len(self.video_paths)}...")

            if self.video_setting:
                print(f"Joining {self.video_name} multiprocessed video...")
                concatenate_videos_in_folder(in_folder=self.video_temp_dir, save_path=self.video_save_path, gpu=self.gpu)
                video_timer.stop_timer()
                pool.terminate()
                pool.join()
                print(f"Video {self.video_name} complete (elapsed time: {video_timer.elapsed_time_str}s)...")

        self.timer.stop_timer()
        if self.video_setting:
            stdout_success(msg=f"{len(self.video_paths)} videos saved in {self.sklearn_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)
        if self.frame_setting:
            stdout_success(f"Frames for {len(self.video_paths)} videos saved in sub-folders within {self.sklearn_plot_dir} directory", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)


# if __name__ == "__main__":
#     clf_plotter = PlotSklearnResultsMultiProcess(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                                                 video_setting=True,
#                                                 frame_setting=False,
#                                                 video_paths=r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/Together_1.mp4",
#                                                 print_timers=True,
#                                                 rotate=False,
#                                                 animal_names=False,
#                                                 show_bbox=True,
#                                                 show_gantt=None)
#     clf_plotter.run()




#text_settings = {'circle_scale': 5, 'font_size': 0.528, 'spacing_scale': 28, 'text_thickness': 2}
# clf_plotter = PlotSklearnResultsMultiProcess(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini',
#                                              video_setting=True,
#                                              frame_setting=False,
#                                              rotate=False,
#                                              video_file_path='592_MA147_Gq_CNO_0515.mp4',
#                                              cores=-1,
#                                              text_settings=False)
# clf_plotter.run()
#

# clf_plotter = PlotSklearnResultsMultiProcess(config_path='/Users/simon/Desktop/envs/troubleshooting/DLC_2_Black_animals/project_folder/project_config.ini', video_setting=True, frame_setting=False, rotate=False, video_file_path='Together_1.avi', cores=5)
# clf_plotter.run()
