import functools
import multiprocessing
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from simba.data_processors.spontaneous_alternation_calculator import \
    SpontaneousAlternationCalculator
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int, check_str, check_valid_lst)
from simba.utils.data import detect_bouts
from simba.utils.enums import Formats, Paths, TextOptions
from simba.utils.errors import AnimalNumberError, InvalidInputError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    find_core_cnt, find_video_of_file,
                                    get_video_meta_data, read_frm_of_video)

ALTERNATION_COLOR = (0, 255, 0)
ERROR_COLOR = (0, 0, 255)


def spontaneous_alternator_video_mp(
    frm_index: np.ndarray,
    video_path: Union[str, os.PathLike],
    temp_save_dir: Union[str, os.PathLike],
    event_txt: List[List[str]],
    alt_dict: Dict[str, List[int]],
    roi_geometries: Dict[str, Polygon],
    roi_geometry_clrs: Dict[str, Tuple[int]],
    animal_geometries: List[Polygon],
):

    core, frm_index = frm_index[0], frm_index[1:]
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    start_frm, current_frm, end_frm = frm_index[0], frm_index[0], frm_index[-1]
    cap.set(1, start_frm)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    save_path = os.path.join(temp_save_dir, f"{core}.mp4")
    writer = cv2.VideoWriter(
        save_path,
        fourcc,
        video_meta_data["fps"],
        (
            int(video_meta_data["width"] + (video_meta_data["width"] / 2)),
            video_meta_data["height"],
        ),
    )
    while current_frm < end_frm:
        while current_frm < end_frm:
            sequence_lst = event_txt[current_frm]
            border = np.zeros(
                (int(video_meta_data["height"]), int(video_meta_data["width"] / 2), 3),
                dtype=np.uint8,
            )
            ret, img = cap.read()
            for shape_cnt, (k, v) in enumerate(roi_geometries.items()):
                cv2.polylines(
                    img,
                    [np.array(v.exterior.coords).astype(np.int)],
                    True,
                    (roi_geometry_clrs[k]),
                    thickness=2,
                )
            cv2.polylines(
                img,
                [
                    np.array(animal_geometries[current_frm].exterior.coords).astype(
                        np.int
                    )
                ],
                True,
                (178, 102, 255),
                thickness=2,
            )
            if len(list(set(sequence_lst))) == len(list(roi_geometries.keys())) - 1:
                txt_clr = ALTERNATION_COLOR
            else:
                txt_clr = ERROR_COLOR
            error_1 = len(
                [x for x in alt_dict["same_arm_return_errors"] if x <= current_frm]
            )
            error_2 = len(
                [x for x in alt_dict["alt_arm_return_errors"] if x <= current_frm]
            )
            alternations = len(
                [x for x in alt_dict["alternations"] if x <= current_frm]
            )
            cv2.putText(
                border,
                "Sequence:" + ",".join(sequence_lst),
                (10, 50),
                TextOptions.FONT.value,
                1,
                txt_clr,
                2,
            )
            cv2.putText(
                border,
                f"Alternation #: {alternations}",
                (10, 80),
                TextOptions.FONT.value,
                1,
                txt_clr,
                2,
            )
            cv2.putText(
                border,
                f"Errors #: {error_1 + error_2}",
                (10, 110),
                TextOptions.FONT.value,
                1,
                txt_clr,
                2,
            )
            img = np.hstack((img, border))
            writer.write(img)
            print(f"Writing frame {current_frm} (Core: {core})...")
            current_frm += 1
    writer.release()


class SpontaneousAlternationsPlotter(ConfigReader):
    """
    Create plots representing delayed-alternation computations overlayed on video.

    .. image:: _static/img/SpontaneousAlternationsPlotter.gif
       :width: 700
       :align: center

    .. note::
       Uses ``simba.data_processors.spontaneous_alternation_calculator.SpontaneousAlternationCalculator`` to compute alternation statistics.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param List[str] arm_names: List of ROI names representing the arms.
    :param str center_name: Name of the ROI representing the center of the maze
    :param Optional[int] animal_area: Value between 51 and 100, representing the percent of the animal body that has to be situated in a ROI for it to be considered an entry.
    :param Optional[float] threshold: Value between 0.0 and 1.0. Body-parts with detection probabilities below this value will be (if possible) filtered when constructing the animal geometry.
    :param Optional[int] buffer: Millimeters area for which the animal geometry should be increased in size. Useful if the animal geometry does not fully cover the animal.
    :param Optional[int] core_cnt: The number of CPU cores to use when creating the visualization. Defaults to -1 which represents all avaialbale cores.
    :param Optional[Union[str, os.PathLike]] data_path: Path to the file to be analyzed, e.g., CSV file in `project_folder/csv/outlier_corrected_movement_location`` directory.

    :example:
    >>> config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini'
    >>> plotter = SpontaneousAlternationsPlotter(config_path=config_path, arm_names=['A', 'B', 'C'], center_name='Center', threshold=0.0, buffer=1, animal_area=60, data_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/csv/outlier_corrected_movement_location/F1 HAB.csv')
    >>> plotter.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        arm_names: List[str],
        center_name: str,
        animal_area: Optional[int] = 80,
        threshold: Optional[float] = 0.0,
        buffer: Optional[int] = 2,
        core_cnt: Optional[int] = -1,
        verbose: Optional[bool] = False,
        data_path: Optional[Union[str, os.PathLike]] = None,
    ):

        ConfigReader.__init__(self, config_path=config_path)
        if self.animal_cnt != 1:
            raise AnimalNumberError(
                msg=f"Spontaneous alternation can only be calculated in 1 animal projects. Your project has {self.animal_cnt} animals.",
                source=self.__class__.__name__,
            )
        if len(self.body_parts_lst) < 3:
            raise InvalidInputError(
                msg=f"Spontaneous alternation can only be calculated in projects with 3 or more tracked body-parts. Found {len(self.body_parts_lst)}.",
                source=self.__class__.__name__,
            )
        check_valid_lst(
            data=arm_names,
            source=SpontaneousAlternationCalculator.__name__,
            valid_dtypes=(str,),
            min_len=2,
        )
        check_int(name="ANIMAL AREA", value=animal_area, min_value=1, max_value=100)
        check_float(name="THRESHOLD", value=threshold, min_value=0.0, max_value=1.0)
        check_int(name="BUFFER", value=buffer, min_value=1)
        check_file_exist_and_readable(file_path=data_path)
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        check_str(name="CENTER NAME", value=center_name)
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        self.threshold, self.buffer, self.animal_area = threshold, buffer, animal_area
        self.verbose, self.arm_names, self.center_name = verbose, arm_names, center_name
        self.data_path, self.core_cnt = data_path, core_cnt

    def run(self):
        sa_computer = SpontaneousAlternationCalculator(
            config_path=self.config_path,
            arm_names=self.arm_names,
            center_name=self.center_name,
            animal_area=self.animal_area,
            threshold=self.threshold,
            verbose=False,
            buffer=self.buffer,
            data_path=self.data_path,
        )
        sa_computer.run()
        print("Running alternation visualization...")
        video_path = find_video_of_file(
            video_dir=self.video_dir, filename=sa_computer.video_name, raise_error=True
        )
        bout_df = (
            detect_bouts(
                data_df=sa_computer.roi_df,
                target_lst=self.arm_names + [self.center_name],
                fps=1,
            )[["Event", "Start_frame"]]
            .sort_values(["Start_frame"])
            .reset_index(drop=True)
        )
        shifted_ = pd.concat(
            [bout_df, bout_df.shift(-1).add_suffix("_shifted").reset_index(drop=True)],
            axis=1,
        )[["Event", "Event_shifted"]].values
        unique_counts = [len(list(set(list(x)))) for x in shifted_]
        drop_idx = np.argwhere(np.array(unique_counts) == 1) + 1
        bout_df = bout_df.drop(drop_idx.flatten(), axis=0).reset_index(drop=True)
        bout_df = bout_df[bout_df["Event"] != self.center_name]
        frm_index = np.arange(0, len(sa_computer.data_df))
        frm_index = np.array_split(frm_index, self.core_cnt)
        for cnt, i in enumerate(frm_index):
            frm_index[cnt] = np.insert(i, 0, cnt)
        event_txt = []
        for idx in range(len(sa_computer.data_df)):
            preceding_entries = list(
                bout_df["Event"][bout_df["Start_frame"] <= idx].tail(
                    len(self.arm_names)
                )
            )
            event_txt.append(preceding_entries)
        self.temp_folder = os.path.join(
            self.project_path,
            Paths.SPONTANEOUS_ALTERNATION_VIDEOS_DIR.value,
            sa_computer.video_name,
            "temp",
        )
        if os.path.isdir(self.temp_folder):
            shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder)
        alt_dict = {
            "same_arm_return_errors": [
                x
                for xs in sa_computer.video_results["same_arm_returns_dict"].values()
                for x in xs
            ],
            "alt_arm_return_errors": [
                x
                for xs in sa_computer.video_results[
                    "alternate_arm_returns_dict"
                ].values()
                for x in xs
            ],
            "alternations": [
                x
                for xs in sa_computer.video_results["alternations_dict"].values()
                for x in xs
            ],
        }
        save_path = os.path.join(
            self.project_path,
            Paths.SPONTANEOUS_ALTERNATION_VIDEOS_DIR.value,
            f"{sa_computer.video_name}.mp4",
        )
        with multiprocessing.Pool(
            self.core_cnt, maxtasksperchild=self.maxtasksperchild
        ) as pool:
            constants = functools.partial(
                spontaneous_alternator_video_mp,
                video_path=video_path,
                event_txt=event_txt,
                animal_geometries=sa_computer.animal_polygons,
                temp_save_dir=self.temp_folder,
                alt_dict=alt_dict,
                roi_geometries=sa_computer.roi_geos[sa_computer.video_name],
                roi_geometry_clrs=sa_computer.roi_clrs[sa_computer.video_name],
            )
            for cnt, result in enumerate(
                pool.imap(constants, frm_index, chunksize=self.multiprocess_chunksize)
            ):
                print(f"Section {cnt} complete...")
        pool.terminate()
        pool.join()
        print(f"Joining {sa_computer.video_name} multiprocessed video...")
        concatenate_videos_in_folder(in_folder=self.temp_folder, save_path=save_path)
        self.timer.stop_timer()
        stdout_success(
            f"Alternation video saved at {save_path}",
            elapsed_time=self.timer.elapsed_time_str,
        )


#
# x = SpontaneousAlternationsPlotter(
#     config_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini',
#     arm_names=['A', 'B', 'C'],
#     center_name='Center',
#     threshold=0.0,
#     buffer=1,
#     animal_area=60,
#     data_path='/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/csv/outlier_corrected_movement_location/F1 HAB.csv')
# x.run()


#
