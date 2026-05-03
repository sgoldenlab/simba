__author__ = "Simon Nilsson; sronilsson@gmail.com"

import functools
import itertools
import multiprocessing
import os
import platform
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.plotting_mixin import PlottingMixin
from simba.roi_tools.ROI_directing_analyzer import DirectingROIAnalyzer
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_string_value_is_valid_video_timestamp,
                                check_if_valid_rgb_tuple, check_int, check_str,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_boolean, check_valid_dict,
                                check_video_and_data_frm_count_align)
from simba.utils.data import (find_frame_numbers_from_time_stamp, get_cpu_pool,
                              slice_roi_dict_for_video, terminate_cpu_pool)
from simba.utils.enums import Formats, TextOptions
from simba.utils.errors import (NoFilesFoundError, NoROIDataError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, get_fn_ext,
                                    get_video_meta_data, read_df,
                                    remove_a_folder, seconds_to_timestamp)

START_TIME, END_TIME = 'start_time', 'end_time'


def _roi_directing_visualizer_mp(frm_range: Tuple[int, np.ndarray],
                                 data_df: pd.DataFrame,
                                 text_locations: dict,
                                 font_size: float,
                                 circle_size: Union[float, int],
                                 save_temp_dir: str,
                                 video_meta_data: dict,
                                 shape_info: dict,
                                 shape_names: list,
                                 video_path: str,
                                 animal_names: list,
                                 roi_dict: dict,
                                 animal_bp_dict: dict,
                                 directing_data: pd.DataFrame,
                                 border_bg_color: tuple,
                                 show_pose: bool,
                                 show_roi_centers: bool,
                                 show_animal_names: bool,
                                 direction_color: Tuple[int, int, int],
                                 direction_thickness: int,
                                 direction_style: str,
                                 verbose: bool,
                                 cumulative_directing: dict,
                                 fps: float):

    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    group_cnt, frm_range = frm_range[0], frm_range[1]
    current_frm, end_frm = frm_range[0], frm_range[-1]
    save_path = os.path.join(save_temp_dir, f"{group_cnt}.mp4")
    writer = cv2.VideoWriter(save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"] * 2, video_meta_data["height"]))
    cap = cv2.VideoCapture(video_path)
    cap.set(1, current_frm)
    directing_lk = set(zip(directing_data["Animal"], directing_data["ROI"], directing_data["Frame"]))
    while current_frm <= end_frm:
        ret, img = cap.read()
        if ret:
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(video_meta_data["width"]), borderType=cv2.BORDER_CONSTANT, value=border_bg_color)
            img = PlottingMixin.roi_dict_onto_img(img=img, roi_dict=roi_dict, circle_size=circle_size, show_center=show_roi_centers)
            if show_pose:
                for animal_name, bp_data in animal_bp_dict.items():
                    for bp_cnt, bp in enumerate(zip(bp_data["X_bps"], bp_data["Y_bps"])):
                        bp_cords = data_df.loc[current_frm, list(bp)].values.astype(np.int64)
                        cv2.circle(img, (bp_cords[0], bp_cords[1]), 0, animal_bp_dict[animal_name]["colors"][bp_cnt], circle_size)
            if show_animal_names:
                for animal_name, bp_data in animal_bp_dict.items():
                    headers = [bp_data["X_bps"][-1], bp_data["Y_bps"][-1]]
                    bp_cords = data_df.loc[current_frm, headers].values.astype(np.int64)
                    cv2.putText(img, animal_name, (bp_cords[0], bp_cords[1]), font, font_size, animal_bp_dict[animal_name]["colors"][0], 1)

            for animal_name, shape_name in itertools.product(animal_names, shape_names):
                is_directing = (animal_name, shape_name, current_frm) in directing_lk
                shape_clr = shape_info[shape_name]["Color BGR"]
                cv2.putText(img, text_locations[animal_name][shape_name]["directing_text"], text_locations[animal_name][shape_name]["directing_text_loc"], font, font_size, shape_clr, 1)
                cv2.putText(img, str(is_directing), text_locations[animal_name][shape_name]["directing_data_loc"], font, font_size, shape_clr, 1)
                cum_key = (animal_name, shape_name)
                cum_frms = cumulative_directing[cum_key][current_frm] if current_frm < len(cumulative_directing[cum_key]) else cumulative_directing[cum_key][-1]
                cum_time = seconds_to_timestamp(seconds=cum_frms / fps, hh_mm_ss_sss=True)
                cv2.putText(img, text_locations[animal_name][shape_name]["total_time_text"], text_locations[animal_name][shape_name]["total_time_text_loc"], font, font_size, shape_clr, 1)
                cv2.putText(img, cum_time, text_locations[animal_name][shape_name]["total_time_data_loc"], font, font_size, shape_clr, 1)
                if is_directing:
                    img = PlottingMixin.insert_directing_line(directing_df=directing_data,
                                                              img=img,
                                                              shape_name=shape_name,
                                                              animal_name=animal_name,
                                                              frame_id=current_frm,
                                                              color=direction_color,
                                                              thickness=direction_thickness,
                                                              style=direction_style)
            writer.write(np.uint8(img))
            if verbose:
                seconds = seconds_to_timestamp(seconds=current_frm / video_meta_data['fps'], hh_mm_ss_sss=True)
                stdout_information(msg=f"Multiprocessing frame: {current_frm}, time-stamp: {seconds} on core {group_cnt}...")
            current_frm += 1
        else:
            break
    cap.release()
    writer.release()
    return group_cnt


class DirectingROIVisualizer(ConfigReader, PlottingMixin):
    """
    Visualize when animals are directing towards ROIs. Draws the ROIs on the video frames, overlays
    pose-estimation body-parts, and draws directing lines (funnel or line style) from the animal eye
    midpoint to the ROI when the animal is directing towards the ROI. A text panel shows the directing
    boolean for each animal-ROI combination per frame. Uses multiprocessing.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param Union[str, os.PathLike] video_path: Path to video file to overlay directing visualization on.
    :param Literal['funnel', 'lines'] direction_style: Style of direction line. Default 'funnel'.
    :param Tuple[int, int, int] direction_color: BGR color of the directing line. Default (0, 0, 255) (red).
    :param Optional[int] direction_thickness: Thickness of the directing line (used for 'lines' style). If None, computed automatically based on video resolution. Default None.
    :param Optional[int] circle_size: Size of the pose-estimation keypoint circles. If None, computed automatically based on video resolution. Default None.
    :param bool show_pose: If True, draw pose-estimation keypoints on the video. Default True.
    :param bool show_roi_centers: If True, draw the center of each ROI. Default True.
    :param bool show_animal_names: If True, display animal names on the video. Default False.
    :param Tuple[int, int, int] border_bg_clr: BGR color for the text panel background. Default (0, 0, 0).
    :param Optional[Dict[str, str]] time_slice: Optional dict with 'start_time' and 'end_time' keys (HH:MM:SS format) to visualize a sub-clip. Default None.
    :param Optional[Union[str, os.PathLike]] roi_coordinates_path: Optional path to ROI definitions file. If None, uses the project default. Default None.
    :param Optional[str] left_ear_name: Optional custom left ear body-part name. Default None.
    :param Optional[str] right_ear_name: Optional custom right ear body-part name. Default None.
    :param Optional[str] nose_name: Optional custom nose body-part name. Default None.
    :param int core_cnt: Number of CPU cores for multiprocessing. -1 uses all available. Default -1.
    :param bool gpu: If True, use GPU for video concatenation when available. Default False.
    :param bool verbose: If True, print progress messages during visualization. Default True.

    .. video:: _static/img/DirectingROIVisualizer.webm
       :width: 1000
       :autoplay:
       :loop:
       :muted:
       :align: center

    :example:
    >>> viz = DirectingROIVisualizer(config_path='/path/to/project_config.ini',
    ...                              video_path='/path/to/video.mp4',
    ...                              direction_style='funnel',
    ...                              show_pose=True,
    ...                              core_cnt=4)
    >>> viz.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 direction_style: Literal['funnel', 'lines'] = 'lines',
                 direction_color: Tuple[int, int, int] = (0, 0, 255),
                 direction_thickness: Optional[int] = None,
                 circle_size: Optional[int] = None,
                 show_pose: bool = True,
                 show_roi_centers: bool = True,
                 show_animal_names: bool = False,
                 border_bg_clr: Tuple[int, int, int] = (0, 0, 0),
                 time_slice: Optional[Dict[str, str]] = None,
                 roi_coordinates_path: Optional[Union[str, os.PathLike]] = None,
                 left_ear_name: Optional[str] = None,
                 right_ear_name: Optional[str] = None,
                 nose_name: Optional[str] = None,
                 core_cnt: int = -1,
                 gpu: bool = False,
                 verbose: bool = True):

        check_file_exist_and_readable(file_path=config_path)
        check_file_exist_and_readable(file_path=video_path)
        check_int(name=f"{self.__class__.__name__} core_cnt", value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_str(name=f"{self.__class__.__name__} direction_style", value=direction_style, options=['funnel', 'lines'])
        check_if_valid_rgb_tuple(data=direction_color, source=f"{self.__class__.__name__} direction_color")
        if direction_thickness is not None:
            check_int(name=f"{self.__class__.__name__} direction_thickness", value=direction_thickness, min_value=1)
        if circle_size is not None:
            check_int(name=f"{self.__class__.__name__} circle_size", value=circle_size, min_value=1)
        check_valid_boolean(value=[show_pose, show_roi_centers, show_animal_names, gpu, verbose], source=self.__class__.__name__)
        check_if_valid_rgb_tuple(data=border_bg_clr, source=f"{self.__class__.__name__} border_bg_clr")
        if time_slice is not None:
            check_valid_dict(x=time_slice, valid_key_dtypes=(str,), valid_values_dtypes=(str,), valid_keys=(START_TIME, END_TIME), required_keys=(START_TIME, END_TIME))
            check_if_string_value_is_valid_video_timestamp(value=time_slice[START_TIME], name='START TIME', raise_error=True)
            check_if_string_value_is_valid_video_timestamp(value=time_slice[END_TIME], name='END TIME', raise_error=True)
            check_that_hhmmss_start_is_before_end(start_time=time_slice[START_TIME], end_time=time_slice[END_TIME], name='TIME SLICE', raise_error=True)

        ConfigReader.__init__(self, config_path=config_path)
        PlottingMixin.__init__(self)
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.gpu = gpu
        self.show_pose, self.show_roi_centers, self.show_animal_names = show_pose, show_roi_centers, show_animal_names
        self.border_bg_clr, self.time_slice = border_bg_clr, time_slice
        self.direction_style, self.direction_color, self.direction_thickness, self.circle_size = direction_style, direction_color, direction_thickness, circle_size
        self.verbose = verbose
        self.video_path = video_path

        if roi_coordinates_path is not None:
            check_file_exist_and_readable(file_path=roi_coordinates_path)
            self.roi_coordinates_path = deepcopy(roi_coordinates_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        self.read_roi_data()

        _, self.video_name, _ = get_fn_ext(video_path)
        self.roi_dict, self.shape_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=self.video_name)
        if len(self.shape_names) == 0:
            raise NoROIDataError(msg=f"No ROIs found for video {self.video_name}. Draw ROIs for this video before creating directing visualizations.", source=self.__class__.__name__)
        self.data_path = os.path.join(self.outlier_corrected_dir, f"{self.video_name}.{self.file_type}")
        if not os.path.isfile(self.data_path):
            raise NoFilesFoundError(msg=f"Could not find the file at path {self.data_path}. Make sure the data file exists to create directing ROI visualizations.", source=self.__class__.__name__)

        self.directing_analyzer = DirectingROIAnalyzer(config_path=config_path,
                                                       data_path=self.data_path,
                                                       left_ear_name=left_ear_name,
                                                       right_ear_name=right_ear_name,
                                                       nose_name=nose_name)
        self.directing_analyzer.run()
        self.directing_df = self.directing_analyzer.results_df
        self.animal_names = list(self.directing_analyzer.direct_bp_dict.keys())

        self.video_meta_data = get_video_meta_data(video_path, fps_as_int=False)
        if direction_thickness is None:
            self.direction_thickness = max(1, self.get_optimal_circle_size(frame_size=(int(self.video_meta_data["height"]), int(self.video_meta_data["width"])), circle_frame_ratio=200))
        self.data_df = read_df(file_path=self.data_path, file_type=self.file_type)
        check_video_and_data_frm_count_align(video=video_path, data=self.data_path, name=video_path, raise_error=False)

        self.save_dir = os.path.join(self.frames_output_dir, "ROI_directing_visualizations")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir, f"{self.video_name}.mp4")
        self.save_temp_dir = os.path.join(self.save_dir, "temp")
        if os.path.exists(self.save_temp_dir):
            remove_a_folder(folder_dir=self.save_temp_dir)
        os.makedirs(self.save_temp_dir)
        self.shape_dicts = self.__create_shape_dicts()
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)

    def __create_shape_dicts(self):
        shape_dicts = {}
        for shape, df in self.roi_dict.items():
            if not df["Name"].is_unique:
                df = df.drop_duplicates(subset=["Name"], keep="first")
            d = df.set_index("Name").to_dict(orient="index")
            shape_dicts = {**shape_dicts, **d}
        return shape_dicts

    def __calc_text_locs(self):
        add_spacer = TextOptions.FIRST_LINE_SPACING.value
        self.loc_dict = {}
        txt_strs = []
        for animal_name in self.animal_names:
            for shape in self.shape_names:
                txt_strs.append(f"{shape} {animal_name} directing")
                txt_strs.append(f"{shape} {animal_name} total time (s)")
        longest_text_str = str(max(txt_strs, key=len)) if len(txt_strs) > 0 else "N/A"
        self.font_size, x_spacer, y_spacer = self.get_optimal_font_scales(text=longest_text_str, accepted_px_width=int(self.video_meta_data["width"] / 2), accepted_px_height=int(self.video_meta_data["height"] / 15), text_thickness=3)
        if self.circle_size is None:
            self.circle_size = self.get_optimal_circle_size(frame_size=(int(self.video_meta_data["height"]), int(self.video_meta_data["height"])), circle_frame_ratio=100)
        for animal_name in self.animal_names:
            self.loc_dict[animal_name] = {}
            for shape in self.shape_names:
                self.loc_dict[animal_name][shape] = {}
                self.loc_dict[animal_name][shape]["directing_text"] = f"{shape} {animal_name} directing"
                self.loc_dict[animal_name][shape]["directing_text_loc"] = ((self.video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + y_spacer * add_spacer))
                self.loc_dict[animal_name][shape]["directing_data_loc"] = (int(self.video_meta_data["width"] + x_spacer + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + y_spacer * add_spacer))
                add_spacer += 1
                self.loc_dict[animal_name][shape]["total_time_text"] = f"{shape} {animal_name} total time (s)"
                self.loc_dict[animal_name][shape]["total_time_text_loc"] = ((self.video_meta_data["width"] + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + y_spacer * add_spacer))
                self.loc_dict[animal_name][shape]["total_time_data_loc"] = (int(self.video_meta_data["width"] + x_spacer + TextOptions.BORDER_BUFFER_X.value), (self.video_meta_data["height"] - (self.video_meta_data["height"] + 10) + y_spacer * add_spacer))
                add_spacer += 1

    def run(self):
        self.timer = SimbaTimer(start=True)
        self.__calc_text_locs()
        data_df = self.data_df.copy()
        if self.time_slice is not None:
            frm_ids = find_frame_numbers_from_time_stamp(start_time=self.time_slice[START_TIME], end_time=self.time_slice[END_TIME], fps=int(self.video_meta_data['fps']))
            data_df = data_df.loc[frm_ids].reset_index(drop=True)
        n_frms = len(data_df)
        cumulative_directing = {}
        directing_lk = set(zip(self.directing_df["Animal"], self.directing_df["ROI"], self.directing_df["Frame"]))
        for animal_name, shape_name in itertools.product(self.animal_names, self.shape_names):
            arr = np.zeros(n_frms, dtype=np.int64)
            for i in range(n_frms):
                arr[i] = 1 if (animal_name, shape_name, i) in directing_lk else 0
            cumulative_directing[(animal_name, shape_name)] = np.cumsum(arr)

        frm_lst = np.arange(0, n_frms)
        frm_lst = np.array_split(frm_lst, self.core_cnt)
        frame_range = [(i, frm_lst[i]) for i in range(len(frm_lst))]
        if self.verbose:
            stdout_information(msg=f"Creating ROI directing visualization for video {self.video_name}, multiprocessing (cores: {self.core_cnt})...")
        pool = get_cpu_pool(core_cnt=self.core_cnt, verbose=self.verbose, source=self.__class__.__name__)
        constants = functools.partial(_roi_directing_visualizer_mp,
                                      data_df=data_df.reset_index(drop=True),
                                      text_locations=self.loc_dict,
                                      font_size=self.font_size,
                                      circle_size=self.circle_size,
                                      video_meta_data=self.video_meta_data,
                                      shape_info=self.shape_dicts,
                                      roi_dict=self.roi_dict,
                                      save_temp_dir=self.save_temp_dir,
                                      directing_data=self.directing_df,
                                      shape_names=self.shape_names,
                                      video_path=self.video_path,
                                      animal_names=self.animal_names,
                                      animal_bp_dict=self.animal_bp_dict,
                                      border_bg_color=self.border_bg_clr,
                                      show_pose=self.show_pose,
                                      show_roi_centers=self.show_roi_centers,
                                      show_animal_names=self.show_animal_names,
                                      direction_color=self.direction_color,
                                      direction_thickness=self.direction_thickness,
                                      direction_style=self.direction_style,
                                      verbose=self.verbose,
                                      cumulative_directing=cumulative_directing,
                                      fps=self.video_meta_data['fps'])
        for cnt, result in enumerate(pool.imap(constants, frame_range, chunksize=self.multiprocess_chunksize)):
            if self.verbose:
                stdout_information(msg=f"Batch core {result + 1}/{self.core_cnt} complete...")
        if self.verbose:
            stdout_information(f"Joining {self.video_name} multi-processed video...")
        concatenate_videos_in_folder(in_folder=self.save_temp_dir, save_path=self.save_path, video_format="mp4", remove_splits=True, gpu=self.gpu)
        self.timer.stop_timer()
        terminate_cpu_pool(pool=pool, force=False, verbose=self.verbose, source=self.__class__.__name__)
        stdout_success(msg=f"ROI directing visualization for video {self.video_name} complete. Video saved at {self.save_path}.", elapsed_time=self.timer.elapsed_time_str)


# if __name__ == '__main__':
#     test = DirectingROIVisualizer(config_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini",
#                                    video_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\videos\Box1_180mISOcontrol_Females.mp4",
#                                    direction_style='funnel',
#                                    show_pose=True,
#                                    time_slice={'start_time': '00:00:00', 'end_time': '00:00:10'},
#                                    core_cnt=4)
#     test.run()
