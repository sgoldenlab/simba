__author__ = "Simon Nilsson; sronilsson@gmail.com"

import itertools
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Union

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from numba import jit, prange

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.enums import ConfigKey
from simba.utils.errors import (CountError, IntegerError, InvalidInputError,
                                NoDataError, NoFilesFoundError)
from simba.utils.read_write import get_fn_ext, seconds_to_timestamp, read_frm_of_video
from simba.utils.warnings import FrameRangeWarning
from simba.utils.lookups import get_img_resize_info,get_monitor_info, get_simba_font_name_and_path

MAX_ID_UI_DISPLAY_RATIO = (0.5, 0.5)  # W, H
MIN_ID_UI_DISPLAY_RATIO = (0.3, 0.3)  # W, H

# FRAMES ADVANCED PER "x" PRESS IN THE "DEFINE ANIMAL IDS" UI. THE LAST STEP IS SHORTENED TO LAND ON THE FINAL
# USABLE FRAME, AFTER WHICH THE NEXT PRESS WRAPS BACK TO FRAME 0. SMALLER = FINER SEARCH, MORE KEYPRESSES TO
# CYCLE A LONG VIDEO. NOTE THIS IS A FRAME COUNT, SO THE TIME IT REPRESENTS DEPENDS ON THE VIDEO FPS.
ID_UI_FRM_STEP = 50

# HEIGHT OF THE INSTRUCTION SIDE-PANEL, AS A FRACTION OF THE DISPLAYED FRAME HEIGHT. THE PANEL IS CONCATENATED
# BELOW THE FRAME, SO THE WINDOW IS ALWAYS (1 + THIS) x THE FRAME HEIGHT - BOTH THE PANEL AND THE WINDOW SIZE
# ARE DERIVED FROM IT SO THEY CANNOT DRIFT APART. LOWER IT TO GIVE MORE OF THE WINDOW TO THE VIDEO.
ID_UI_PANEL_RATIO = 1 / 3

# EVERY STRING PRINTED IN THE "DEFINE ANIMAL IDS" SIDE-PANELS. THE FONT SCALE IS FITTED TO THE LONGEST OF THESE,
# SO KEEPING THEM HERE - RATHER THAN AS LITERALS AT THE PUTTEXT CALLS - IS WHAT STOPS THE FITTED AND PRINTED
# STRINGS FROM DRIFTING APART. SHORTER STRINGS = LARGER FONT.
ID_UI_TXT = {"ASSIGN_Q": "Assign identities from frame {} / {}?",
             "NEW_FRM": 'Press "x" to display new, random, frame',
             "ASSIGN": 'Press "c" to assign identities on this frame',
             "CLICK_ON": "Left mouse click on:",
             "HAPPY_Q": "Are you happy with your assigned identities ?",
             "CONTINUE": 'Press "c" to continue',
             "RESTART": 'Press "x" to re-start assigning identities'}

ID_UI_FONT = 'Poppins Regular'


class PoseImporterMixin(object):
    """
    Methods for importing pose-estimation data.
    """

    def __init__(self):
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        pass

    def initialize_multi_animal_ui(self,
                                   animal_bp_dict: dict,
                                   video_info: dict,
                                   data_df: pd.DataFrame,
                                   video_path: Union[str, os.PathLike],
                                   initial_frame_no: Optional[int] = None):
        """
        :param Optional[int] initial_frame_no: Frame index to start the assignment UI on.
            Useful for jumping directly to a frame where all animals are simultaneously
            detected, so the user doesn't have to press "x" repeatedly to find one.
            If ``None`` (default) the UI starts at frame 0, preserving previous behaviour.
            Clamped to the valid range ``[0, len(data_df) - 1]``; out-of-range values
            emit a :class:`FrameRangeWarning` and fall back to 0.
        """

        start_frame = 0
        if initial_frame_no is not None:
            if not isinstance(initial_frame_no, int) or initial_frame_no < 0 or initial_frame_no >= len(data_df):
                FrameRangeWarning(
                    msg=(f"initial_frame_no={initial_frame_no} is outside the valid range "
                         f"[0, {len(data_df) - 1}] for this video; starting at frame 0 instead."),
                    source=self.__class__.__name__,
                )
            else:
                start_frame = initial_frame_no
        self.video_info, self.data_df, self.frame_no, self.add_spacer = (video_info, data_df, start_frame, 2)
        # OFFSET APPLIED WHEN "x" WRAPS PAST THE LAST FRAME. WITHOUT IT, EVERY LAP WOULD SHOW THE SAME FRAMES THE
        # USER HAS ALREADY REJECTED. HALF A STEP MEANS LAP 1 IS 0, 50, 100 ... AND LAP 2 IS 25, 75, 125 ...
        self.frm_wrap_offset = 0
        self.animal_bp_dict, self.cap = animal_bp_dict, cv2.VideoCapture(video_path)
        _, self.video_name, _ = get_fn_ext(video_path)
        self.monitor_info, (self.display_w, self.display_h) = get_monitor_info()
        self.display_img_w, self.display_img_h, self.display_scale, self.native_scale = get_img_resize_info(img_size=(self.video_info['width'], self.video_info['height']), display_resolution=(self.display_w, self.display_h), max_width_ratio=MAX_ID_UI_DISPLAY_RATIO[0], max_height_ratio=MAX_ID_UI_DISPLAY_RATIO[1], min_width_ratio=MIN_ID_UI_DISPLAY_RATIO[0], min_height_ratio=MIN_ID_UI_DISPLAY_RATIO[1])
        _, self.font_path = get_simba_font_name_and_path(font=ID_UI_FONT)
        self.get_video_scalers(video_info=video_info)

    def get_video_scalers(self, video_info: dict):
        self.scalers = {}
        self.scalers["circle"] = PlottingMixin().get_optimal_circle_size(frame_size=(self.display_img_w, self.display_img_h), circle_frame_ratio=100)
        # ASSIGN_Q is a format template, so the fit needs the WIDEST string it can ever produce for this
        # video (the largest frame index in both slots) - not the raw "{} / {}" template, which is shorter.
        panel_txt = list(ID_UI_TXT.values()) + list(self.animal_bp_dict.keys()) + [ID_UI_TXT["ASSIGN_Q"].format(self._max_id_ui_frm_idx(), self._max_id_ui_frm_idx())]
        self.scalers["font"], _, _ = PlottingMixin.get_optimal_font_size_ttf(text=panel_txt, font_path=self.font_path, accepted_px_width=self.display_img_w, accepted_px_height=int(self.display_img_h / 6))
        self.scalers["space"] = PlottingMixin.get_optimal_font_spacing_ttf(font_path=self.font_path, size_px=self.scalers["font"], text=panel_txt)
        self.scalers["panel_h"] = max(int(self.display_img_h * ID_UI_PANEL_RATIO), int(self.scalers["space"] * (self.add_spacer * 3 + 1)))

    def find_data_files(self, dir: Union[str, os.PathLike], extensions: List[str]) -> List[str]:
        """
        Search for files with specific extensions in a given directory and return their paths.

        :param Union[str, os.PathLike] dir: The directory to search for files. It can be a string or a path-like object.
        :param  List[str] extensions: A list of file extensions to look for.
        :return: A list of paths to the files with the specified extensions found in the directory.
        :raises NoDataError: If no files with the specified extensions are found in the directory.
        """

        data_paths = []
        paths = [f for f in next(os.walk(dir))[2] if not f[0] == "."]
        paths = [os.path.join(dir, f) for f in paths]
        for extension in extensions:
            for path in paths:
                if path.endswith(extension):
                    data_paths.append(path)
        if len(data_paths) == 0:
            raise NoDataError(msg=f"No files with {extensions} extensions found in {dir}.", source=self.__class__.__name__)

        return data_paths






    def link_video_paths_to_data_paths(self,
                                       data_paths: List[str],
                                       video_paths: List[str],
                                       str_splits: Optional[List[str]] = None,
                                       filename_cleaning_func: Optional[object] = None,
                                       raise_error: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Given a list of paths to video files and a separate list of paths to data files, create a dictionary
        pairing each video file to a datafile based on the file names of the video and data file.

        :param List[str] data_paths: List of full paths to data files, e.g., CSV or H5 files.
        :param List[str] video_paths: List of full paths to video files, e.g., MP4 or AVI files.
        :param Optional[List[str]] str_splits: Optional list of substrings that the data_paths would need to be split at in order to find a matching video name. E.g., ['dlc_resnet50'].
        :param Optional[object] filename_cleaning_func: Optional filename cleaning function that the data_paths filenames would have to pass through in order to find a matching video name. E.g., ``simba.utils.read_write.clean_sleap_filename(filepath)``.
        :param bool raise_error: If True, raises an error if a video file representing a data file doesn't exist. If False, return None for the specific key.
        :return dict: Dictionary with the data/file name as keys, and the video and data paths as values.
        """

        results, video_names = {}, []
        for video_path in video_paths:
            _, video_file_name, _ = get_fn_ext(video_path)
            video_names.append(video_file_name.lower())
        for data_path in data_paths:
            _, data_file_name, _ = get_fn_ext(data_path)
            data_file_names = [data_file_name.lower()]
            if str_splits:
                for split_str in str_splits:
                    data_file_names.append(data_file_name.lower().split(split_str)[0])
            data_file_names = list(set(data_file_names))
            if filename_cleaning_func is not None:
                data_file_names = [filename_cleaning_func(x) for x in data_file_names]
            video_idx = [i for i, x in enumerate(video_names) if x in data_file_names]
            if len(video_idx) == 0:
                if raise_error:
                    raise NoFilesFoundError(msg=f"SimBA could not locate a video file in your SimBA project for data file {data_file_name}", source=self.__class__.__name__)
                else:
                    results[data_file_names[0]] = {"DATA": data_path, "VIDEO": None}
            else:
                _, video_name, _ = get_fn_ext(video_paths[video_idx[0]])
                results[video_name] = {"DATA": data_path, "VIDEO": video_paths[video_idx[0]]}
        return results

    def get_x_y_loc_of_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_loc = (x, y)
            self.id_cords[self.cnt] = {}
            self.id_cords[self.cnt]["cord"] = self.click_loc
            self.id_cords[self.cnt]["name"] = self.animal_name

    def insert_all_bodyparts_into_img(self, img: np.ndarray, body_parts: dict):
        for animal, bp_data in body_parts.items():
            for bp_cnt, bp_tuple in enumerate(bp_data):
                try:
                    cv2.circle(img, bp_tuple, self.scalers["circle"], self.animal_bp_dict[animal]["colors"][bp_cnt], -1, lineType=cv2.LINE_AA)
                except Exception as err:
                    if type(err) == OverflowError:
                        raise IntegerError(
                            f"SimBA encountered a pose-estimated body-part located at pixel position {str(bp_tuple)}. "
                            "This value is too large to be converted to an integer. "
                            "Please check your pose-estimation data to make sure that it is accurate.",
                            source=self.__class__.__name__,
                        )

    def insert_animal_names(self):
        for animal_cnt, animal_data in self.id_cords.items():
            self.new_frame = PlottingMixin().put_text(img=self.new_frame, text=animal_data["name"], pos=animal_data["cord"], font_size=self.scalers["font"], font_path=self.font_path, text_color=(255, 255, 0))

    def _set_multianimal_window_id_title(self,
                                         window_name: str):
        self.frm_no_timestamp = seconds_to_timestamp(seconds=self.frame_no / self.video_info['fps'], hh_mm_ss_sss=True)
        # the total is the last DISPLAYABLE frame, not len(data_df) - 1: pose data can be longer than the video,
        # and quoting an index the UI can never reach would contradict the frame counter printed on the panel.
        title = f'{window_name} | {self.video_name} | frame {self.frame_no} / {self._max_id_ui_frm_idx()} | {self.frm_no_timestamp}'
        try:
            cv2.setWindowTitle(window_name, title)
        except cv2.error:
            pass

    def _max_id_ui_frm_idx(self) -> int:
        """
        Highest frame index the "Define animal IDs" UI can display. The pose data and the video can
        disagree in length and BOTH have to be in range - the data is indexed with ``.loc`` (raising
        ``KeyError``) and the video is seeked (returning no frame), so the shorter of the two governs.
        """
        return min(len(self.data_df), int(self.video_info["frame_count"])) - 1

    def _next_id_ui_frm_no(self) -> int:
        """
        Frame index to display on the next ``"x"`` press: step forward by :data:`ID_UI_FRM_STEP`, shortening the
        final step so the last usable frame is reachable, then wrap around to the start. Each wrap advances
        :attr:`frm_wrap_offset` by half a step, so a second lap shows frames BETWEEN those of the first rather
        than repeating the ones the user has already rejected.
        """
        max_frm_idx = self._max_id_ui_frm_idx()
        if self.frame_no >= max_frm_idx:
            self.frm_wrap_offset = (self.frm_wrap_offset + ID_UI_FRM_STEP // 2) % ID_UI_FRM_STEP
            return min(self.frm_wrap_offset, max_frm_idx)
        return min(self.frame_no + ID_UI_FRM_STEP, max_frm_idx)

    def multianimal_identification(self):
        # NOTE: the window is deliberately NOT destroyed here, nor before re-entering this method or
        # choose_animal_ui. cv2 forgets a window's on-screen position when it is destroyed, so tearing it
        # down between screens threw the window back to a window-manager-chosen spot on every keypress.
        # namedWindow on an already-live window is a no-op that keeps the position. The destroys on the way
        # out of choose_animal_ui ARE kept, since they also drop that method's mouse callback.
        interpolation = cv2.INTER_AREA if self.display_scale < 1 else cv2.INTER_CUBIC
        self.img = read_frm_of_video(video_path=self.cap, frame_index=self.frame_no, size=(self.display_img_w, self.display_img_h), interpolation=interpolation)
        self.all_frame_data = self.data_df.loc[self.frame_no, :]
        cv2.namedWindow("Define animal IDs", cv2.WINDOW_NORMAL)
        self._set_multianimal_window_id_title(window_name="Define animal IDs")
        self.img_overlay, self.img_bp_cords = deepcopy(self.img), defaultdict(list)
        for animal_cnt, (animal_name, animal_bps) in enumerate(self.animal_bp_dict.items()):
            self.img_bp_cords[animal_name] = []
            for x_name, y_name in zip(animal_bps["X_bps"], animal_bps["Y_bps"]):
                self.img_bp_cords[animal_name].append(tuple((self.data_df.loc[self.frame_no, [x_name, y_name]].values * self.display_scale).astype(np.int32)))
                #self.img_bp_cords[animal_name].append(tuple(self.data_df.loc[self.frame_no, [x_name, y_name]].values.astype(np.int32)))
        self.insert_all_bodyparts_into_img(img=self.img_overlay, body_parts=self.img_bp_cords)
        side_img = np.ones((self.scalers["panel_h"], self.display_img_w, 3), dtype=np.uint8)
        side_img = PlottingMixin().put_text(img=side_img, text=ID_UI_TXT['ASSIGN_Q'].format(self.frame_no, self._max_id_ui_frm_idx()), pos=(10, int(self.scalers["space"] * self.add_spacer)), font_size=self.scalers["font"], font_path=self.font_path, text_color=(255, 255, 255))
        side_img = PlottingMixin().put_text(img=side_img, text=ID_UI_TXT['NEW_FRM'], pos=(10, int(self.scalers["space"] * (self.add_spacer * 2))), font_size=self.scalers["font"], font_path=self.font_path, text_color=(255, 255, 0))
        side_img = PlottingMixin().put_text(img=side_img, text=ID_UI_TXT['ASSIGN'], pos=(10, int(self.scalers["space"] * (self.add_spacer * 3))), font_size=self.scalers["font"], font_path=self.font_path, text_color=(0, 255, 0))
        self.img_concat = np.uint8(np.concatenate((self.img_overlay, side_img), axis=0))
        cv2.imshow("Define animal IDs", self.img_concat)
        cv2.resizeWindow("Define animal IDs", self.display_img_w, self.display_img_h + self.scalers["panel_h"])
        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(20)
            if (k == ord("x")) or (k == ord("X")):
                cv2.waitKey(1)
                # step forward 50 frames, take a short final hop to the last usable frame, then wrap to 0 -
                # the user must be able to come back round to a frame where the animals are separated.
                self.frame_no = self._next_id_ui_frm_no()
                self.multianimal_identification()
                break
            elif (k == ord("c")) or (k == ord("C")):
                cv2.waitKey(1)
                self.choose_animal_ui()
                break

    def choose_animal_ui(self):
        self.id_cords = {}
        for cnt, animal in enumerate(self.animal_bp_dict.keys()):
            self.animal_name, self.cnt = animal, cnt
            self.new_overlay = deepcopy(self.img_overlay)
            cv2.namedWindow("Define animal IDs", cv2.WINDOW_NORMAL)
            self._set_multianimal_window_id_title(window_name="Define animal IDs")
            self.side_img = np.ones((self.scalers["panel_h"], self.display_img_w, 3), dtype=np.uint8)
            self.side_img = PlottingMixin().put_text(img=self.side_img, text=ID_UI_TXT['CLICK_ON'], pos=(10, int(self.scalers["space"] * self.add_spacer)), font_size=self.scalers["font"], font_path=self.font_path, text_color=(0, 255, 0))
            self.side_img = PlottingMixin().put_text(img=self.side_img, text=animal, pos=(10, int(self.scalers["space"] * (self.add_spacer * 2))), font_size=self.scalers["font"], font_path=self.font_path, text_color=(255, 255, 0))
            for id in self.id_cords.keys():
                self.new_overlay = PlottingMixin().put_text(img=self.new_overlay, text=self.id_cords[id]["name"], pos=self.id_cords[id]["cord"], font_size=self.scalers["font"], font_path=self.font_path, text_color=(255, 255, 255))
            self.new_overlay = np.uint8(np.concatenate((self.new_overlay, self.side_img), axis=0))
            cv2.imshow("Define animal IDs", self.new_overlay)
            cv2.resizeWindow("Define animal IDs", self.display_img_w, self.display_img_h + self.scalers["panel_h"])
            while cnt not in self.id_cords.keys():
                cv2.setMouseCallback("Define animal IDs", self.get_x_y_loc_of_mouse_click)
                cv2.waitKey(200)
        self.confirm_ui()

    def confirm_ui(self):
        cv2.destroyAllWindows()
        cv2.namedWindow("Define animal IDs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Define animal IDs", self.display_img_w, self.display_img_h + self.scalers["panel_h"])
        self.new_frame = deepcopy(self.img)
        self.side_img = np.ones((self.scalers["panel_h"], self.display_img_w, 3), dtype=np.uint8)
        self.side_img = PlottingMixin().put_text(img=self.side_img, text=ID_UI_TXT['HAPPY_Q'], pos=(10, int(self.scalers["space"] * self.add_spacer)), font_size=self.scalers["font"], font_path=self.font_path, text_color=(255, 255, 255))
        self.side_img = PlottingMixin().put_text(img=self.side_img, text=ID_UI_TXT['CONTINUE'], pos=(10, int(self.scalers["space"] * (self.add_spacer * 2))), font_size=self.scalers["font"], font_path=self.font_path, text_color=(255, 255, 0))
        self.side_img = PlottingMixin().put_text(img=self.side_img, text=ID_UI_TXT['RESTART'], pos=(10, int(self.scalers["space"] * (self.add_spacer * 3))), font_size=self.scalers["font"], font_path=self.font_path, text_color=(0, 255, 255))
        self.insert_all_bodyparts_into_img(img=self.new_frame, body_parts=self.img_bp_cords)
        self.insert_animal_names()
        self.img_concat = np.uint8(np.concatenate((self.new_frame, self.side_img), axis=0))
        self._set_multianimal_window_id_title(window_name="Define animal IDs")
        cv2.imshow("Define animal IDs", self.img_concat)
        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(20)
            if (k == ord("x")) or (k == ord("X")):
                cv2.destroyWindow("Define animal IDs")
                cv2.waitKey(1)
                # as in multianimal_identification: advance, hop to the last usable frame, then wrap to 0.
                self.frame_no = self._next_id_ui_frm_no()
                self.multianimal_identification()
                break
            elif (k == ord("c")) or (k == ord("C")):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                self.cap.release()
                self.find_closest_animals()
                break

    def find_closest_animals(self):
        self.animal_order = {}
        for animal_number, animal_click_data in self.id_cords.items():
            animal_name, animal_cord = (animal_click_data["name"], animal_click_data["cord"])
            animal_cord = (animal_cord[0] * self.native_scale, animal_cord[1] * self.native_scale)
            closest_animal = {"animal_name": None, "body_part_name": None, "distance": np.inf}
            for other_animal_name, animal_bps in self.animal_bp_dict.items():
                animal_bp_names_x = self.animal_bp_dict[other_animal_name]["X_bps"]
                animal_bp_names_y = self.animal_bp_dict[other_animal_name]["Y_bps"]
                for x_col, y_col in zip(animal_bp_names_x, animal_bp_names_y):
                    bp_x, bp_y = self.all_frame_data[x_col], self.all_frame_data[y_col]
                    distance = np.sqrt((animal_cord[0] - bp_x) ** 2 + (animal_cord[1] - bp_y) ** 2)
                    if distance < closest_animal["distance"]:
                        closest_animal["animal_name"] = other_animal_name
                        closest_animal["body_part_name"] = (x_col, y_col)
                        closest_animal["distance"] = distance
            if closest_animal["animal_name"] is None:
                raise InvalidInputError(
                    msg=(f'No tracked body-parts found on frame {self.frame_no} for click number '
                         f'{animal_number} ({animal_name}): every body-part of every animal is NaN at '
                         f'this frame. Press "x" to step to a frame where at least one body-part per '
                         f'animal is tracked, then assign identities.'),
                    source=self.__class__.__name__,
                )
            self.animal_order[animal_number] = closest_animal
        self.check_intergity_of_chosen_animal_order()
        self.organize_results()
        self.reinsert_multi_idx_columns()

    def organize_results(self):
        self.out_df = pd.DataFrame()
        for animal_cnt, animal_data in self.animal_order.items():
            closest_animal_dict = self.animal_bp_dict[animal_data["animal_name"]]
            x_cols, y_cols, p_cols = (
                closest_animal_dict["X_bps"],
                closest_animal_dict["Y_bps"],
                closest_animal_dict["P_bps"],
            )
            for x_col, y_col, p_cols in zip(x_cols, y_cols, p_cols):
                df = self.data_df[[x_col, y_col, p_cols]]
                self.out_df = pd.concat([self.out_df, df], axis=1)

    def reinsert_multi_idx_columns(self):
        multi_idx_cols = []
        for col_idx in range(len(self.out_df.columns)):
            multi_idx_cols.append(
                tuple(("IMPORTED_POSE", "IMPORTED_POSE", self.out_df.columns[col_idx]))
            )
        self.out_df.columns = pd.MultiIndex.from_tuples(
            multi_idx_cols, names=("scorer", "bodypart", "coords")
        )

    def insert_multi_idx_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        multi_idx_cols = []
        for col_idx in range(len(df.columns)):
            multi_idx_cols.append(tuple(("IMPORTED_POSE", "IMPORTED_POSE", df.columns[col_idx])))
        df.columns = pd.MultiIndex.from_tuples(multi_idx_cols, names=("scorer", "bodypart", "coords"))

        return df


    def check_intergity_of_chosen_animal_order(self):
        for click_key_combination in itertools.combinations(list(self.animal_order.keys()), 2):
            click_n, click_n1 = click_key_combination[0], click_key_combination[1]
            animal_1, animal_2 = (self.animal_order[click_n]["animal_name"], self.animal_order[click_n1]["animal_name"])
            if animal_1 == animal_2:
                raise InvalidInputError(msg=f"The animal most proximal to click number {str(click_n)} is animal named {animal_1}. The animal most proximal to click number {str(click_n1)} is also animal {animal_2}. Please indicate which animal is which using a video frame where the animals are clearly separated", source=self.__class__.__name__,)

    def intertwine_probability_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame()
        for animal_name, animal_bps in self.animal_bp_dict.items():
            x_cols, y_cols, p_cols = (
                animal_bps["X_bps"],
                animal_bps["Y_bps"],
                animal_bps["P_bps"],
            )
            for x_col, y_col, p_col in zip(x_cols, y_cols, p_cols):
                df = data[[x_col, y_col, p_col]]
                results = pd.concat([results, df], axis=1)
        return results

    def __update_config_animal_cnt(self):
        self.config.set(
            ConfigKey.GENERAL_SETTINGS.value,
            ConfigKey.ANIMAL_CNT.value,
            str(self.animal_cnt),
        )
        with open(self.project_path, "w+") as f:
            self.config.write(f)
        f.close()

    def update_bp_headers_file(self, update_bp_headers: Optional[bool] = False):
        new_headers = []
        for animal_name in self.animal_bp_dict.keys():
            for bp in self.animal_bp_dict[animal_name]["X_bps"]:
                if animal_name not in bp:
                    new_headers.append("{}_{}".format(animal_name, bp[:-2]))
                else:
                    new_headers.append(bp[:-2])
        new_bp_df = pd.DataFrame(new_headers)
        new_bp_df.to_csv(self.body_parts_path, index=False, header=False)
        if update_bp_headers:
            self.bp_headers = new_headers

    def update_config_animal_names(self):
        self.config.set(
            ConfigKey.MULTI_ANIMAL_ID_SETTING.value,
            ConfigKey.MULTI_ANIMAL_IDS.value,
            ",".join(str(x) for x in self.id_lst),
        )
        with open(self.project_path, "w+") as f:
            self.config.write(f)
        f.close()

    @staticmethod
    @jit(nopython=True)
    def transpose_multi_animal_table(data: np.array, idx: np.array, animal_cnt: int) -> np.array:
        unique_tracks = np.unique(idx[:, 0]).flatten()
        unique_tracks = np.sort(unique_tracks)
        unique_tracks = unique_tracks[0:animal_cnt]
        results = np.full((np.max(idx[:, 1]+1), data.shape[1] * animal_cnt), 0.0)
        for i in prange(np.max(idx[:, 1])+1):
            for j in prange(animal_cnt):
                data_idx = np.argwhere((idx[:, 0] == unique_tracks[j]) & (idx[:, 1] == i)).flatten()
                if len(data_idx) == 1:
                    animal_frm_data = data[data_idx[0]]
                else:
                    animal_frm_data = np.full((data.shape[1]), 0.0)
                results[i][j * animal_frm_data.shape[0] : j * animal_frm_data.shape[0] + animal_frm_data.shape[0]] = animal_frm_data
        return results

    def read_apt_trk_file(self, file_path: str):
        print("Reading data using scipy.io...")
        try:
            trk_coordinates = sio.loadmat(file_path)["pTrk"]
            track_cnt = trk_coordinates.shape[3]
            data = [trk_coordinates[..., i] for i in range(track_cnt)]

        except NotImplementedError:
            print("Failed to read data using scipy.io. Reading data using h5py...")
            with h5py.File(file_path, "r") as trk_dict:
                trk_list = list(trk_dict["pTrk"])
                t_second = np.array(trk_list)
                if len(t_second.shape) > 3:
                    t_third = np.swapaxes(t_second, 0, 3)
                    trk_coordinates = np.swapaxes(t_third, 1, 2)
                    track_cnt = trk_coordinates.shape[3]
                    data = [trk_coordinates[..., i] for i in range(track_cnt)]
                else:
                    data = np.swapaxes(t_second, 0, 2)
                    track_cnt = 1

        if track_cnt != self.animal_cnt:
            raise CountError(
                msg=f"There are {str(track_cnt)} tracks in the .trk file {file_path}. But your SimBA project expects {str(self.animal_cnt)} tracks.",
                source=self.__class__.__name__,
            )

        if self.animal_cnt != 1:
            animal_df_lst = []
            for animal in data:
                m, n, r = animal.shape
                out_arr = np.column_stack((np.repeat(np.arange(m), n), animal.reshape(m * n, -1)))
                animal_df_lst.append(pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True))
                self.data_df = pd.concat(animal_df_lst, axis=1).fillna(0).astype(np.int)
        else:
            m, n, r = data.shape
            out_arr = np.column_stack((np.repeat(np.arange(m), n), data.reshape(m * n, -1)))
            self.data_df = (pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True).astype(np.int))

        p_cols = pd.DataFrame(1, index=self.data_df.index, columns=self.data_df.columns[1::2] + 0.5)
        return pd.concat([self.data_df, p_cols], axis=1)  # .sort_index(axis=1)
