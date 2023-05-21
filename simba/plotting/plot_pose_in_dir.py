__author__ = "Simon Nilsson"

import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import pandas as pd

from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.data import create_color_palette
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError
from simba.utils.lookups import get_color_dict
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df)

ACCEPTED_DIRECTORIES = [
    "input_csv",
    "outlier_corrected_movement",
    "outlier_corrected_movement_location",
]


def create_video_from_dir(
    in_directory: str,
    out_directory: str,
    circle_size: int,
    clr_attr: Optional[Dict[str, Tuple[int, int, int]]] = None,
):
    """
    Create pose-estimation visualizations from data within a SimBA project folder.

    :param str in_directory: Path to SimBA project directory containing pose-estimation data in parquet or CSV format.
    :param str out_directory: Directory to where to save the pose-estimation videos.
    :param int Size of the circles denoting the location of the pose-estimated body-parts.
    :param Optional[dict] clr_attr: Python dict where animals are keys and color attributes values. E.g., {'Animal_1':  (255, 107, 198)}. If None,
                                    random palettes will be used.

    :example:
    >>> create_video_from_dir(in_directory='InputDirectory', out_directory='OutputDirectory', circle_size=5, clr_attr=None)

    """

    if (os.path.basename(in_directory)) not in ACCEPTED_DIRECTORIES:
        raise InvalidInputError(
            msg=f"{in_directory} is not an accepted directory for visualization. OPTIONS: {ACCEPTED_DIRECTORIES}"
        )

    files_found = glob.glob(in_directory + "/*." + "csv") + glob.glob(
        in_directory + "/*." + "parquet"
    )
    check_if_filepath_list_is_empty(
        filepaths=files_found,
        error_msg="SIMBA ERROR: 0 files found in {} in CSV or PARQUET file format".format(
            in_directory
        ),
    )
    print("Processing {} videos ...".format(str(len(files_found))))
    timer = SimbaTimer()
    timer.start_timer()
    for video_cnt, file_path in enumerate(files_found):
        dir_name, file_name, ext = get_fn_ext(file_path)
        video_dir = os.path.join(Path(file_path).parents[2], "videos")
        video_file_path = find_video_of_file(video_dir, file_name)
        pose_df = read_df(file_path, ext[1:].lower())
        if os.path.basename(in_directory) == "input_csv":
            pose_df.columns = list(pose_df.loc["coords"])

        pose_df = pose_df.apply(pd.to_numeric, errors="coerce")
        pose_df = pose_df.fillna(0).reset_index(drop=True)
        bp_lst_of_lst = [
            list(pose_df.columns)[i : i + 3] for i in range(0, len(pose_df.columns), 3)
        ]
        color_list = create_color_palette(
            increments=len(bp_lst_of_lst), pallete_name="Set1"
        )
        animal_bp_dict = {}
        if clr_attr:
            clrs = get_color_dict()
            animal_bp_dict = {}
            for animal in range(1, len(clr_attr.keys()) + 1):
                animal_bp_dict["Animal_{}".format(str(animal))] = {}
                animal_bp_dict["Animal_{}".format(str(animal))]["bps"] = []
                animal_bp_dict["Animal_{}".format(str(animal))]["color"] = clrs[
                    clr_attr["Animal_{}".format(str(animal))]
                ]
                for bp in bp_lst_of_lst:
                    if str(animal) in bp[0].split("_")[-2]:
                        animal_bp_dict["Animal_{}".format(str(animal))]["bps"].append(
                            bp
                        )
        video_meta_data = get_video_meta_data(video_path=video_file_path)
        cap = cv2.VideoCapture(video_file_path)
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)

        save_path = os.path.join(out_directory, file_name + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        writer = cv2.VideoWriter(
            save_path,
            fourcc,
            video_meta_data["fps"],
            (video_meta_data["width"], video_meta_data["height"]),
        )

        frm_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if not clr_attr:
                    for cnt, bp in enumerate(bp_lst_of_lst):
                        bp_tuple = (
                            int(pose_df.at[frm_cnt, bp[0]]),
                            int(pose_df.at[frm_cnt, bp[1]]),
                        )
                        cv2.circle(frame, bp_tuple, circle_size, color_list[cnt], -1)
                else:
                    for animal_name, animal_data in animal_bp_dict.items():
                        for bp in animal_data["bps"]:
                            bp_tuple = (
                                int(pose_df.at[frm_cnt, bp[0]]),
                                int(pose_df.at[frm_cnt, bp[1]]),
                            )
                            cv2.circle(
                                frame, bp_tuple, circle_size, animal_data["color"], -1
                            )
                frm_cnt += 1
                writer.write(frame)
                print(
                    "Video: {} / {} Frame: {} / {}".format(
                        str(video_cnt + 1),
                        str(len(files_found)),
                        str(frm_cnt),
                        str(len(pose_df)),
                    )
                )
            else:
                break
        print("{} complete...".format(file_name))
        cap.release()
        writer.release()
    timer.stop_timer()
    stdout_success(
        msg=f"All pose videos complete. Results located in {str(out_directory)} directory",
        elapsed_time=timer.elapsed_time_str,
    )


# create_video_from_dir(in_directory=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/outlier_corrected_movement_location',
#                       out_directory=r'/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/csv/features_extracted/test',
#                       circle_size=5,
#                       clr_attr=None)
