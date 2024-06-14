__author__ = "Simon Nilsson"

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

from simba.utils.enums import ConfigKey
from simba.utils.errors import (CountError, IntegerError, InvalidInputError,
                                NoDataError, NoFilesFoundError)
from simba.utils.read_write import get_fn_ext
from simba.utils.warnings import FrameRangeWarning


class PoseImporterMixin(object):
    """
    Methods for importing pose-estimation data.
    """

    def __init__(self):
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        pass

    def initialize_multi_animal_ui(
        self,
        animal_bp_dict: dict,
        video_info: dict,
        data_df: pd.DataFrame,
        video_path: str,
    ):

        self.video_info, self.data_df, self.frame_no, self.add_spacer = (
            video_info,
            data_df,
            0,
            2,
        )
        self.animal_bp_dict, self.cap = animal_bp_dict, cv2.VideoCapture(video_path)
        _, self.video_name, _ = get_fn_ext(video_path)
        self.get_video_scalers(video_info=video_info)

    def get_video_scalers(self, video_info: dict):
        self.scalers = {}
        space_scale, radius_scale, resolution_scale, font_scale = 40, 10, 1500, 1.2
        max_video_dimension = max(video_info["width"], video_info["height"])
        self.scalers["circle"] = int(
            radius_scale / (resolution_scale / max_video_dimension)
        )
        self.scalers["font"] = font_scale / (resolution_scale / max_video_dimension)
        self.scalers["space"] = int(
            space_scale / (resolution_scale / max_video_dimension)
        )

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
                                       filename_cleaning_func: Optional[object] = None) -> Dict[str, str]:
        """
        Given a list of paths to video files and a separate list of paths to data files, create a dictionary
        pairing each video file to a datafile based on the file names of the video and data file.

        :param List[str] data_paths: List of full paths to data files, e.g., CSV or H5 files.
        :param List[str] video_paths: List of full paths to video files, e.g., MP4 or AVI files.
        :param Optional[List[str]] str_splits: Optional list of substrings that the data_paths would need to be split at in order to find a matching video name. E.g., ['dlc_resnet50'].
        :param Optional[object] filename_cleaning_func: Optional filename cleaning function that the data_paths filenames would have to pass through in order to find a matching video name. E.g., ``simba.utils.read_write.clean_sleap_filename(filepath)``.
        :returns dict: Dictionary with the data/file name as keys, and the video and data paths as values.
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
                raise NoFilesFoundError(
                    msg=f"SimBA could not locate a video file in your SimBA project for data file {data_file_name}",
                    source=self.__class__.__name__,
                )
            _, video_name, _ = get_fn_ext(video_paths[video_idx[0]])
            results[video_name] = {
                "DATA": data_path,
                "VIDEO": video_paths[video_idx[0]],
            }
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
                    cv2.circle(
                        img,
                        bp_tuple,
                        self.scalers["circle"],
                        self.animal_bp_dict[animal]["colors"][bp_cnt],
                        -1,
                        lineType=cv2.LINE_AA,
                    )
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
            cv2.putText(
                self.new_frame,
                animal_data["name"],
                animal_data["cord"],
                cv2.FONT_HERSHEY_SIMPLEX,
                self.scalers["font"],
                (255, 255, 255),
                3,
            )

    def multianimal_identification(self):
        cv2.destroyAllWindows()
        self.cap.set(1, self.frame_no)
        self.all_frame_data = self.data_df.loc[self.frame_no, :]
        cv2.namedWindow("Define animal IDs", cv2.WINDOW_NORMAL)
        _, self.img = self.cap.read()
        self.img_overlay, self.img_bp_cords = deepcopy(self.img), defaultdict(list)
        for animal_cnt, (animal_name, animal_bps) in enumerate(
            self.animal_bp_dict.items()
        ):
            self.img_bp_cords[animal_name] = []
            for x_name, y_name in zip(animal_bps["X_bps"], animal_bps["Y_bps"]):
                self.img_bp_cords[animal_name].append(
                    tuple(
                        self.data_df.loc[self.frame_no, [x_name, y_name]].values.astype(
                            int
                        )
                    )
                )
        self.insert_all_bodyparts_into_img(
            img=self.img_overlay, body_parts=self.img_bp_cords
        )
        side_img = np.ones(
            (int(self.video_info["height"] / 2), self.video_info["width"], 3)
        )
        cv2.putText(
            side_img,
            f"Current video: {self.video_name}",
            (10, self.scalers["space"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (255, 255, 255),
            3,
        )
        cv2.putText(
            side_img,
            "Can you assign identities based on the displayed frame ?",
            (10, int(self.scalers["space"] * (self.add_spacer * 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (255, 255, 255),
            2,
        )
        cv2.putText(
            side_img,
            'Press "x" to display new, random, frame',
            (10, int(self.scalers["space"] * (self.add_spacer * 3))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (255, 255, 0),
            3,
        )
        cv2.putText(
            side_img,
            'Press "c" to continue to start assigning identities using this frame',
            (10, int(self.scalers["space"] * (self.add_spacer * 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (0, 255, 0),
            2,
        )
        self.img_concat = np.uint8(np.concatenate((self.img_overlay, side_img), axis=0))
        cv2.imshow("Define animal IDs", self.img_concat)
        cv2.resizeWindow(
            "Define animal IDs", self.video_info["height"], self.video_info["width"]
        )
        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(20)
            if (k == ord("x")) or (k == ord("X")):
                if self.frame_no + 50 > len(self.data_df):
                    FrameRangeWarning(
                        msg=f"Cannot proceed to new frame: the frame {self.frame_no+50} does not exist in video {self.video_name}. The video {self.video_name} has {len(self.data_df)} frames."
                    )
                else:
                    cv2.destroyWindow("Define animal IDs")
                    cv2.waitKey(0)
                    self.frame_no += 50
                    self.multianimal_identification()
                    break
            elif (k == ord("c")) or (k == ord("C")):
                cv2.destroyWindow("Define animal IDs")
                cv2.waitKey(0)
                self.choose_animal_ui()
                break

    def choose_animal_ui(self):
        self.id_cords = {}
        for cnt, animal in enumerate(self.animal_bp_dict.keys()):
            self.animal_name, self.cnt = animal, cnt
            self.new_overlay = deepcopy(self.img_overlay)
            cv2.namedWindow("Define animal IDs", cv2.WINDOW_NORMAL)
            self.side_img = np.ones(
                (int(self.video_info["height"] / 2), self.video_info["width"], 3)
            )
            cv2.putText(
                self.side_img,
                "Left mouse click on:",
                (10, self.scalers["space"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.scalers["font"],
                (255, 255, 255),
                3,
            )
            cv2.putText(
                self.side_img,
                animal,
                (10, int(self.scalers["space"] * (self.add_spacer * 2))),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.scalers["font"],
                (255, 255, 0),
                3,
            )
            for id in self.id_cords.keys():
                cv2.putText(
                    self.new_overlay,
                    self.id_cords[id]["name"],
                    self.id_cords[id]["cord"],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.scalers["font"],
                    (255, 255, 255),
                    3,
                )
            self.new_overlay = np.uint8(
                np.concatenate((self.new_overlay, self.side_img), axis=0)
            )
            cv2.imshow("Define animal IDs", self.new_overlay)
            cv2.resizeWindow(
                "Define animal IDs", self.video_info["height"], self.video_info["width"]
            )
            while cnt not in self.id_cords.keys():
                cv2.setMouseCallback(
                    "Define animal IDs", self.get_x_y_loc_of_mouse_click
                )
                cv2.waitKey(200)
        self.confirm_ui()

    def confirm_ui(self):
        cv2.destroyAllWindows()
        cv2.namedWindow("Define animal IDs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "Define animal IDs", self.video_info["height"], self.video_info["width"]
        )
        self.new_frame = deepcopy(self.img)
        self.side_img = np.ones(
            (int(self.video_info["height"] / 2), self.video_info["width"], 3)
        )
        cv2.putText(
            self.side_img,
            f"Current video: {self.video_name}",
            (10, self.scalers["space"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (255, 255, 255),
            3,
        )
        cv2.putText(
            self.side_img,
            "Are you happy with your assigned identities ?",
            (10, int(self.scalers["space"] * (self.add_spacer * 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.side_img,
            'Press "c" to continue (to finish, or proceed to the next video)',
            (10, int(self.scalers["space"] * (self.add_spacer * 3))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (255, 255, 0),
            2,
        )
        cv2.putText(
            self.side_img,
            'Press "x" to re-start assigning identities',
            (10, int(self.scalers["space"] * (self.add_spacer * 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.scalers["font"],
            (0, 255, 255),
            2,
        )
        self.insert_all_bodyparts_into_img(
            img=self.new_frame, body_parts=self.img_bp_cords
        )
        self.insert_animal_names()
        self.img_concat = np.uint8(
            np.concatenate((self.new_frame, self.side_img), axis=0)
        )
        cv2.imshow("Define animal IDs", self.img_concat)
        cv2.resizeWindow(
            "Define animal IDs", self.video_info["height"], self.video_info["width"]
        )
        keyboard_choice = False
        while not keyboard_choice:
            k = cv2.waitKey(20)
            if (k == ord("x")) or (k == ord("X")):
                if self.frame_no + 50 > len(self.data_df):
                    FrameRangeWarning(
                        msg=f"Cannot proceed to new frame: the frame {self.frame_no+50} does not exist in video {self.video_name}. The video {self.video_name} has {len(self.data_df)} frames."
                    )
                else:
                    cv2.destroyWindow("Define animal IDs")
                    cv2.waitKey(1)
                    self.frame_no += 50
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
            animal_name, animal_cord = (
                animal_click_data["name"],
                animal_click_data["cord"],
            )
            closest_animal = {}
            closest_animal["animal_name"] = None
            closest_animal["body_part_name"] = None
            closest_animal["distance"] = np.inf
            for other_animal_name, animal_bps in self.animal_bp_dict.items():
                animal_bp_names_x = self.animal_bp_dict[other_animal_name]["X_bps"]
                animal_bp_names_y = self.animal_bp_dict[other_animal_name]["Y_bps"]
                for x_col, y_col in zip(animal_bp_names_x, animal_bp_names_y):
                    bp_location = (
                        int(self.all_frame_data[x_col]),
                        int(self.all_frame_data[y_col]),
                    )
                    distance = np.sqrt(
                        (animal_cord[0] - bp_location[0]) ** 2
                        + (animal_cord[1] - bp_location[1]) ** 2
                    )
                    if distance < closest_animal["distance"]:
                        closest_animal["animal_name"] = other_animal_name
                        closest_animal["body_part_name"] = (x_col, y_col)
                        closest_animal["distance"] = distance
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
            multi_idx_cols.append(
                tuple(("IMPORTED_POSE", "IMPORTED_POSE", df.columns[col_idx]))
            )
        df.columns = pd.MultiIndex.from_tuples(
            multi_idx_cols, names=("scorer", "bodypart", "coords")
        )
        return df

    def check_intergity_of_chosen_animal_order(self):
        for click_key_combination in itertools.combinations(
            list(self.animal_order.keys()), 2
        ):
            click_n, click_n1 = click_key_combination[0], click_key_combination[1]
            animal_1, animal_2 = (
                self.animal_order[click_n]["animal_name"],
                self.animal_order[click_n1]["animal_name"],
            )
            if animal_1 == animal_2:
                raise InvalidInputError(
                    msg=f"The animal most proximal to click number {str(click_n)} is animal named {animal_1}. The animal most proximal to click number {str(click_n1)} is also animal {animal_2}."
                    "Please indicate which animal is which using a video frame where the animals are clearly separated",
                    source=self.__class__.__name__,
                )

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
    def transpose_multi_animal_table(
        data: np.array, idx: np.array, animal_cnt: int
    ) -> np.array:
        results = np.full((np.max(idx[:, 1]), data.shape[1] * animal_cnt), 0.0)
        for i in prange(np.max(idx[:, 1])):
            for j in prange(animal_cnt):
                data_idx = np.argwhere((idx[:, 0] == j) & (idx[:, 1] == i)).flatten()
                if len(data_idx) == 1:
                    animal_frm_data = data[data_idx[0]]
                else:
                    animal_frm_data = np.full((data.shape[1]), 0.0)
                results[i][
                    j * animal_frm_data.shape[0] : j * animal_frm_data.shape[0]
                    + animal_frm_data.shape[0]
                ] = animal_frm_data
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
                out_arr = np.column_stack(
                    (np.repeat(np.arange(m), n), animal.reshape(m * n, -1))
                )
                animal_df_lst.append(
                    pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True)
                )
                self.data_df = pd.concat(animal_df_lst, axis=1).fillna(0).astype(np.int)
        else:
            m, n, r = data.shape
            out_arr = np.column_stack(
                (np.repeat(np.arange(m), n), data.reshape(m * n, -1))
            )
            self.data_df = (
                pd.DataFrame(out_arr).T.iloc[1:].reset_index(drop=True).astype(np.int)
            )

        p_cols = pd.DataFrame(
            1, index=self.data_df.index, columns=self.data_df.columns[1::2] + 0.5
        )
        return pd.concat([self.data_df, p_cols], axis=1)  # .sort_index(axis=1)
