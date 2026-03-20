__author__ = "Simon Nilsson; sronilsson@gmail.com"

import functools
import multiprocessing
import os
import platform
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.plotting_mixin import PlottingMixin
from simba.mixins.train_model_mixin import TrainModelMixin
from simba.utils.checks import (check_float, check_if_df_field_is_boolean,
                                check_if_valid_rgb_tuple, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst)
from simba.utils.data import (create_color_palette, detect_bouts, get_cpu_pool,
                              terminate_cpu_pool)
from simba.utils.enums import ConfigKey, Dtypes, Options, TagNames, TextOptions
from simba.utils.errors import FrameRangeError, InvalidInputError, NoDataError
from simba.utils.printing import log_event, stdout_information, stdout_success
from simba.utils.read_write import (create_directory, find_core_cnt,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_config_entry,
                                    read_df, read_frm_of_video,
                                    seconds_to_timestamp)
from simba.utils.warnings import FrameRangeWarning

START_TIME, END_TIME = 'start_time', 'end_time'
SECONDS, HHMMSSSSSS = ['seconds', 'hh:mm:ss.ssss']


def _multiprocess_annotation_video(bout_df: pd.DataFrame,
                                   bp_dict: dict,
                                   save_dir: str,
                                   video_timer: Optional[str],
                                   show_pose: bool,
                                   show_animal_names: bool,
                                   print_settings: dict,
                                   text_bg_clr: Tuple[int, int, int],
                                   text_color: Tuple[int, int, int],
                                   pose_clr_lst: Tuple[int, int, int],
                                   bbox: Optional[str],
                                   verbose: bool,
                                   pose_data: dict):


    batch_id, bout_df = bout_df
    fourcc, font = cv2.VideoWriter_fourcc(*"mp4v"), cv2.FONT_HERSHEY_DUPLEX
    for bout_id, bout in bout_df.iterrows():
        video_path, video_name = bout['VIDEO_PATH'], bout['VIDEO_NAME']
        clf_start, clf_end = bout['clf_start'], bout['clf_end']
        bout_start, bout_end = bout['bout_start'], bout['bout_end']
        behavior_range, event = list(range(clf_start, clf_end+1)), bout['Event']
        clip_pose_data = pose_data[video_name][event]
        video_print_settings = print_settings[video_name]
        video_meta_data = get_video_meta_data(video_path=video_path)
        video_cap = cv2.VideoCapture(video_path)
        video_save_path = os.path.join(save_dir, f'{video_name}_{event}_{bout_id}.mp4')
        print(video_save_path)
        video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
        for frm_id in range(bout_start, bout_end+1):
            img = read_frm_of_video(video_path=video_cap, frame_index=frm_id, raise_error=True)
            clr_cnt = 0
            for animal_cnt, (animal_name, animal_data) in enumerate(bp_dict.items()):
                if show_pose:
                    for bp_no in range(len(animal_data["X_bps"])):
                        x_bp, y_bp, p_bp = (animal_data["X_bps"][bp_no], animal_data["Y_bps"][bp_no], animal_data["P_bps"][bp_no])
                        bp_cords = clip_pose_data.loc[frm_id, [x_bp, y_bp, p_bp]]
                        img = cv2.circle(img, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), video_print_settings['circle_size'], pose_clr_lst[clr_cnt], -1)
                        clr_cnt += 1
                if show_animal_names:
                    x_bp, y_bp, p_bp = (animal_data["X_bps"][0], animal_data["Y_bps"][0], animal_data["P_bps"][0])
                    bp_cords = clip_pose_data.loc[frm_id, [x_bp, y_bp, p_bp]]
                    img = cv2.putText(img, animal_name, (int(bp_cords[x_bp]), int(bp_cords[y_bp])), font, video_print_settings['font_size'], pose_clr_lst[0],  video_print_settings['text_thickness'])
                if bbox is not None:
                    animal_headers = [val for pair in zip(animal_data["X_bps"], animal_data["Y_bps"]) for val in pair]
                    animal_cords = clip_pose_data.loc[frm_id, animal_headers].values.reshape(-1, 2).astype(np.int32)
                    try:
                        if bbox == Options.AXIS_ALIGNED.value:
                            animal_bbox = GeometryMixin().keypoints_to_axis_aligned_bounding_box(keypoints=animal_cords.reshape(-1, len(animal_cords), 2).astype(np.int32))
                        else:
                            animal_bbox = GeometryMixin().minimum_rotated_rectangle(shape=animal_cords, buffer=None)
                            animal_bbox = np.round(np.array(animal_bbox.exterior.coords)).astype(np.int32)
                        img = cv2.polylines(img, [animal_bbox], True, pose_clr_lst[animal_cnt], thickness=video_print_settings['circle_size'], lineType=cv2.LINE_AA)
                    except Exception as e:
                        pass
            time = round(frm_id / video_meta_data['fps'], 3)
            print_time = f'{time}s' if video_timer == 'seconds' else seconds_to_timestamp(seconds=time, hh_mm_ss_sss=True)
            img = PlottingMixin().put_text(img=img, text=f"VIDEO TIME: {print_time}", pos=(TextOptions.BORDER_BUFFER_Y.value, ((video_meta_data["height"] - video_meta_data["height"]) + video_print_settings['space_size'])), font_size=video_print_settings['font_size'], font_thickness=video_print_settings['text_thickness'], font=font, text_bg_alpha=video_print_settings['text_opacity'], text_color_bg=text_bg_clr, text_color=text_color)
            add_spacer = 2
            if frm_id in behavior_range:
                img = PlottingMixin().put_text(img=img, text=f"{event} ANNOTATED as present", pos=(TextOptions.BORDER_BUFFER_Y.value, ((video_meta_data["height"] - video_meta_data["height"]) + video_print_settings['space_size'] * add_spacer)), font_size=video_print_settings['font_size'], font_thickness=video_print_settings['text_thickness'], font=font, text_bg_alpha=video_print_settings['text_opacity'], text_color_bg=text_bg_clr, text_color=TextOptions.FLAMINGO.value)
            video_writer.write(img.astype(np.uint8))
            if verbose: stdout_information(msg=f"Multi-processing ANNOTATION frame {frm_id} behavior {event} bout {bout_id} (time-stamp: {time}, core batch: {batch_id}, video name: {video_meta_data['video_name']})...")
        video_cap.release()
        video_writer.release()
    return batch_id

class PlotAnnotatedBouts(ConfigReader, TrainModelMixin, PlottingMixin):
    """
    Create per-bout annotation videos from classifier target files.

    For each selected classifier and video, detected annotation bouts are exported as individual MP4 clips. Optional pre/post windows can extend each bout. The rendered clips can include pose points, animal labels, bounding boxes, and a timer overlay.

    :param Union[str, os.PathLike] config_path: Path to the SimBA ``project_config.ini`` file.
    :param Optional[Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]]] data_paths: Target annotation file path(s). If ``None``, all target files in the project are used.
    :param bool animal_names: If ``True``, print animal names near the first body-part.
    :param bool show_pose: If ``True``, draw body-part circles.
    :param Optional[float] pre_window: Seconds added before each detected bout.
    :param Optional[float] post_window: Seconds added after each detected bout.
    :param Optional[Union[int, float]] font_size: Override auto font size.
    :param Optional[Union[int, float]] space_size: Override auto vertical text spacing.
    :param Optional[Union[int, float]] text_thickness: Text thickness.
    :param Optional[Union[int, float]] text_opacity: Text background opacity.
    :param Optional[Union[int, float]] circle_size: Pose marker radius.
    :param Optional[str] pose_palette: Color palette name for pose/body-part colors.
    :param Optional[List[str]] clf_names: Classifiers to visualize. If ``None``, all project classifiers are used.
    :param Optional[Literal['seconds', 'hh:mm:ss.ssss']] video_timer: Timer format to render on output frames.
    :param bool overwrite: Overwrite controls for output directory handling.
    :param Optional[Literal['axis-aligned', 'animal-aligned']] bbox: Optional bounding-box style to draw for each animal.
    :param Tuple[int, int, int] text_clr: RGB text color.
    :param Tuple[int, int, int] text_bg_clr: RGB text background color.
    :param bool gpu: If ``True`` and an Nvidia GPU is available, enable GPU path.
    :param bool verbose: If ``True``, print progress messages.
    :param int core_cnt: Number of CPU cores for multiprocessing. Use ``-1`` for all available cores.

    :example:
    >>> plotter = PlotAnnotatedBouts(
    ...     config_path='project_folder/project_config.ini',
    ...     data_paths=['project_folder/csv/targets_inserted/video_1.csv'],
    ...     clf_names=['grooming'],
    ...     pre_window=1.0,
    ...     post_window=1.0,
    ...     show_pose=True,
    ...     animal_names=False,
    ...     core_cnt=4
    ... )
    >>> plotter.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_paths: Optional[Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]]] = None,
                 animal_names: bool = False,
                 show_pose: bool = True,
                 pre_window: Optional[float] = None,
                 post_window: Optional[float] = None,
                 font_size: Optional[Union[int, float]] = None,
                 space_size: Optional[Union[int, float]] = None,
                 text_thickness: Optional[Union[int, float]] = None,
                 text_opacity: Optional[Union[int, float]] = None,
                 circle_size: Optional[Union[int, float]] = None,
                 pose_palette: Optional[str] = 'Set1',
                 clf_names: Optional[List[str]] = None,
                 video_timer: Optional[Literal['seconds', 'hh:mm:ss.ssss']] = 'hh:mm:ss.ssss',
                 overwrite: bool = True,
                 bbox: Optional[Literal['axis-aligned', 'animal-aligned']] = None,
                 text_clr: Tuple[int, int, int] = (255, 255, 255),
                 text_bg_clr: Tuple[int, int, int] = (0, 0, 0),
                 gpu: bool = False,
                 verbose: bool = True,
                 core_cnt: int = -1):

        ConfigReader.__init__(self, config_path=config_path)
        TrainModelMixin.__init__(self)
        PlottingMixin.__init__(self)
        log_event(logger_name=str(__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()))
        for i in [animal_names, show_pose, gpu]:
            check_valid_boolean(value=i, source=self.__class__.__name__, raise_error=True)
        if font_size is not None: check_float(name=f'{self.__class__.__name__} font_size', value=font_size, min_value=0.1)
        if space_size is not None: check_float(name=f'{self.__class__.__name__} space_size', value=space_size, min_value=0.1)
        if text_thickness is not None: check_float(name=f'{self.__class__.__name__} text_thickness', value=text_thickness, min_value=0.1)
        if circle_size is not None: check_float(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=0.1)
        if text_opacity is not None: check_float(name=f'{self.__class__.__name__} text_opacity', value=text_opacity, min_value=0.1)
        if clf_names is not None: check_valid_lst(data=clf_names, source=f'{self.__class__.__name__} clf_names', valid_dtypes=(str,), valid_values=self.clf_names)
        if pre_window is not None: check_float(name=f'{self.__class__.__name__} pre_window', value=pre_window, allow_zero=False, allow_negative=False)
        if post_window is not None: check_float(name=f'{self.__class__.__name__} pre_window', value=post_window, allow_zero=False, allow_negative=False)

        pose_palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        check_str(name=f'{self.__class__.__name__} pose_palette', value=pose_palette, options=pose_palettes)
        if video_timer is not None: check_str(name=f'{self.__class__.__name__} timer', value=video_timer, options=(SECONDS, HHMMSSSSSS,))
        self.clr_lst = create_color_palette(pallete_name=pose_palette, increments=len(self.body_parts_lst)+1)
        check_if_valid_rgb_tuple(data=text_clr, source=f'{self.__class__.__name__} text_clr')
        check_if_valid_rgb_tuple(data=text_bg_clr, source=f'{self.__class__.__name__} text_bg_clr')
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=overwrite, source=f'{self.__class__.__name__} overwrite', raise_error=True)
        if bbox is not None:
            check_str(name=f'{self.__class__.__name__} bbox', value=bbox, options=Options.BBOX_OPTIONS.value, allow_blank=False, raise_error=True)
        self.circle_size, self.font_size, self.animal_names, self.text_opacity = circle_size, font_size, animal_names, text_opacity
        self.text_thickness, self.space_size, self.show_pose, self.pose_palette, self.verbose = text_thickness, space_size, show_pose, pose_palette, verbose
        self.text_color, self.text_bg_color, self.bbox = text_clr, text_bg_clr, bbox
        self.gpu = True if check_nvidea_gpu_available() and gpu else False
        self.pose_threshold = read_config_entry(self.config, ConfigKey.THRESHOLD_SETTINGS.value, ConfigKey.SKLEARN_BP_PROB_THRESH.value, Dtypes.FLOAT.value, 0.00)
        self.overwrite, self.print_timer = overwrite, video_timer
        self.clfs = self.clf_names if clf_names is None else clf_names
        self.pre_window, self.post_window = pre_window, post_window
        if not os.path.exists(self.sklearn_plot_dir):
            os.makedirs(self.sklearn_plot_dir)
        if isinstance(data_paths, str): self.data_paths = [data_paths]
        elif isinstance(data_paths, list): self.data_paths = data_paths
        elif data_paths is None:
            self.data_paths = self.target_file_paths
            if len(self.data_paths) == 0: raise NoDataError(msg=f'Cannot create ANNOTATION videos. No files exist in {self.targets_folder} directory', source=self.__class__.__name__)
        else: raise InvalidInputError(msg=f'data_paths has to be a path of a list of paths. Got {type(data_paths)}', source=self.__class__.__name__)

        self.video_lk = {}
        for data_path in self.data_paths:
            video_name = get_fn_ext(filepath=data_path)[1]
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=True)
            self.video_lk[video_name] = video_path
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.core_cnt = find_core_cnt()[0] if int(core_cnt) == -1 or int(core_cnt) > find_core_cnt()[0] else int(core_cnt)
        if platform.system() == "Darwin": multiprocessing.set_start_method("spawn", force=True)
        self.save_dir = os.path.join(self.annotated_frm_dir, 'videos')
        create_directory(paths=self.save_dir, overwrite=False, verbose=False)

    def __get_print_settings(self):
        results = {}
        optimal_circle_size = self.get_optimal_circle_size(frame_size=(self.video_meta_data["width"], self.video_meta_data["height"]), circle_frame_ratio=100)
        longest_str = str(max(['TIMERS:', 'ANNOTATED BEHAVIOR:'] + self.clf_names, key=len))
        results['text_thickness'] = TextOptions.TEXT_THICKNESS.value if self.text_thickness is None else int(max(self.text_thickness, 1))
        optimal_font_size, _, optimal_spacing_scale = self.get_optimal_font_scales(text=longest_str, accepted_px_width=int(self.video_meta_data["width"] / 2), accepted_px_height=int(self.video_meta_data["height"] / 3), text_thickness=results['text_thickness'])
        results['circle_size'] = optimal_circle_size if self.circle_size is None else int(max(1, self.circle_size))
        results['font_size'] = optimal_font_size if self.font_size is None else self.font_size
        results['space_size'] = optimal_spacing_scale if self.space_size is None else int(max(self.space_size, 1))
        results['text_opacity']= 0.8 if self.text_opacity is None else float(self.text_opacity)
        return results

    def run(self):
        if self.verbose: stdout_information(msg=f'Creating visualization of ANNOTATION bouts using {self.core_cnt} cores...')
        self.pool = get_cpu_pool(core_cnt=self.core_cnt, source=self.__class__.__name__)
        bout_data, print_settings, pose_data = [], {}, {}
        for video_cnt, (video_name, video_path) in enumerate(self.video_lk.items()):
            data_path, pose_data[video_name] = self.data_paths[video_cnt], {}
            if self.verbose: stdout_information(msg=f"Creating ANNOTATION visualization for video {video_name}...")
            self.data_df = read_df(data_path, self.file_type).reset_index(drop=True).fillna(0)
            check_valid_dataframe(df=self.data_df, source=f'{self.__class__.__name__} {data_path}', required_fields=self.clfs)
            check_if_df_field_is_boolean(df=self.data_df, field=self.clfs, raise_error=True, df_name=data_path)
            self.video_meta_data = get_video_meta_data(video_path=video_path)
            clfs_bouts = detect_bouts(data_df=self.data_df, target_lst=self.clfs, fps=self.video_meta_data['fps'])
            self.pre_window_frames = 0 if self.pre_window is None else int(self.pre_window * self.video_meta_data['fps'])
            self.post_window_frames = 0 if self.post_window is None else int(self.post_window * self.video_meta_data['fps'])
            print_settings[video_name] = self.__get_print_settings()
            for clf in self.clfs:
                clf_bouts = (clfs_bouts.loc[clfs_bouts['Event'] == clf, ['Event', 'Start_frame', 'End_frame']].rename(columns={'Start_frame': 'clf_start', 'End_frame': 'clf_end'}))
                if len(clf_bouts) == 0:
                    FrameRangeWarning(msg=f'No ANNOTATIONS of classifier {clf} in video {data_path} detected. Zero videos for this classifier for video {video_name}.', source=self.__class__.__name__)
                    continue
                clf_bouts['bout_start'], clf_bouts['bout_end'] = clf_bouts['clf_start'] - self.pre_window_frames, clf_bouts['clf_end'] + self.post_window_frames
                clf_bouts['bout_start'] = clf_bouts['bout_start'].clip(lower=0)
                clf_bouts['bout_end'] = clf_bouts['bout_end'].clip(upper=self.video_meta_data['frame_count'])
                mask = np.zeros(len(self.data_df), dtype=bool)
                for s, e in zip(clf_bouts['bout_start'], clf_bouts['bout_end']):
                    mask[s:e + 1] = True
                pose_data[video_name][clf] = self.data_df.loc[mask][self.bp_headers]
                clf_bouts['VIDEO_PATH'], clf_bouts['VIDEO_NAME'] = video_path, video_name
                bout_data.append(clf_bouts)
        if len(bout_data) == 0:
            raise FrameRangeError(msg=f"No annotation bouts found for classifiers {self.clfs} in the selected data files: {self.data_paths}", source=self.__class__.__name__,)
        bout_data = pd.concat(bout_data, axis=0).reset_index(drop=True)

        bout_data = [(cnt, x) for (cnt, x) in enumerate(np.array_split(bout_data, self.core_cnt))]
        constants = functools.partial(_multiprocess_annotation_video,
                                      bp_dict=self.animal_bp_dict,
                                      save_dir=self.save_dir,
                                      video_timer=self.print_timer,
                                      show_pose=self.show_pose,
                                      show_animal_names=self.animal_names,
                                      print_settings=print_settings,
                                      text_bg_clr=self.text_bg_color,
                                      text_color=self.text_color,
                                      pose_clr_lst=self.clr_lst,
                                      bbox=self.bbox,
                                      verbose=self.verbose,
                                      pose_data=pose_data)

        for cnt, result in enumerate(self.pool.imap(constants, bout_data, chunksize=self.multiprocess_chunksize)):
            if self.verbose: stdout_information(f"Annotation batch {result} complete ...")
        terminate_cpu_pool(pool=self.pool, force=False, source=self.__class__.__name__)
        self.timer.stop_timer()
        stdout_success(msg=f"ANNOTATION videos saved in {self.save_dir}", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)



# if __name__ == "__main__":
#     x = PlotAnnotatedBouts(config_path=r"E:\troubleshooting\mitra\project_folder\project_config.ini",
#                             data_paths=r"E:\troubleshooting\mitra\project_folder\csv\targets_inserted\grooming\501_MA142_Gi_CNO_0516.csv",
#                             pre_window=3,
#                             post_window=3,
#                             clf_names=['grooming'],
#                             core_cnt=8,
#                             verbose=True)
#     x.run()
#
#
