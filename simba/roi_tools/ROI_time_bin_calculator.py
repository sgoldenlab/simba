import itertools
import os
from typing import List

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.utils.enums import DirNames
from simba.utils.errors import ROICoordinatesNotFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class ROITimebinCalculator(ConfigReader):
    """
    Calculate how much time and how many entries animals are making into user-defined ROIs
    within user-defined time bins. Results are stored in the ``project_folder/logs`` directory of
    the SimBA project.

    :param str config_path: path to SimBA project config file in Configparser format
    :param int bin_length: length of time bins in seconds.

    Examples
    ----------
    >>> roi_time_bin_calculator = ROITimebinCalculator(config_path='MySimBaConfigPath', bin_length=15, body_parts=['Nose_1'], threshold=0.00)
    >>> roi_time_bin_calculator.run()
    >>> roi_time_bin_calculator.save()
    """

    def __init__(
        self, config_path: str, bin_length: int, body_parts: List[str], threshold: float
    ):
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(
                expected_file_path=self.roi_coordinates_path
            )
        self.read_roi_data()
        self.bin_length, self.body_parts, self.threshold = (
            bin_length,
            body_parts,
            threshold,
        )
        self.save_path_time = os.path.join(
            self.logs_path, f"ROI_time_bins_{bin_length}s_time_data_{self.datetime}.csv"
        )
        self.save_path_entries = os.path.join(
            self.logs_path,
            f"ROI_time_bins_{bin_length}s_entry_data_{self.datetime}.csv",
        )
        settings = {"threshold": threshold, "body_parts": {}}
        for i in body_parts:
            animal_name = self.find_animal_name_from_body_part_name(
                bp_name=i, bp_dict=self.animal_bp_dict
            )
            settings["body_parts"][animal_name] = i
        self.roi_analyzer = ROIAnalyzer(
            ini_path=self.config_path,
            data_path=DirNames.OUTLIER_MOVEMENT_LOCATION.value,
            calculate_distances=False,
            settings=settings,
        )

        self.roi_analyzer.run()
        self.animal_names = list(self.roi_analyzer.bp_dict.keys())
        self.entries_exits_df = self.roi_analyzer.detailed_df

    #
    def run(self):
        self.out_time_lst, self.out_entries_lst = [], []
        print(
            f"Analyzing time-bin data for {len(self.outlier_corrected_paths)} video(s)..."
        )
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            _, _, fps = self.read_video_info(video_name=self.video_name)
            frames_per_bin = int(fps * self.bin_length)
            video_frms = list(
                range(0, len(read_df(file_path=file_path, file_type=self.file_type)))
            )
            frame_bins = [
                video_frms[i * frames_per_bin : (i + 1) * frames_per_bin]
                for i in range((len(video_frms) + frames_per_bin - 1) // frames_per_bin)
            ]
            self.video_data = self.entries_exits_df[
                self.entries_exits_df["VIDEO"] == self.video_name
            ]
            for animal_name, shape_name in list(
                itertools.product(self.animal_names, self.shape_names)
            ):
                results_entries = pd.DataFrame(
                    columns=["VIDEO", "SHAPE", "ANIMAL", "TIME BIN", "ENTRY COUNT"]
                )
                results_time = pd.DataFrame(
                    columns=[
                        "VIDEO",
                        "SHAPE",
                        "ANIMAL",
                        "TIME BIN",
                        "TIME INSIDE SHAPE (S)",
                    ]
                )
                data_df = self.video_data.loc[
                    (self.video_data["SHAPE"] == shape_name)
                    & (self.video_data["ANIMAL"] == animal_name)
                ]
                entry_frms = list(data_df["ENTRY FRAMES"])
                inside_shape_frms = [
                    list(range(x, y))
                    for x, y in zip(
                        list(data_df["ENTRY FRAMES"].astype(int)),
                        list(data_df["EXIT FRAMES"].astype(int) + 1),
                    )
                ]
                inside_shape_frms = [i for s in inside_shape_frms for i in s]
                for bin_cnt, bin_frms in enumerate(frame_bins):
                    frms_inside_roi_in_timebin = [
                        x for x in inside_shape_frms if x in bin_frms
                    ]
                    entry_roi_in_timebin = [x for x in entry_frms if x in bin_frms]
                    results_time.loc[len(results_time)] = [
                        self.video_name,
                        shape_name,
                        animal_name,
                        bin_cnt,
                        len(frms_inside_roi_in_timebin) / fps,
                    ]
                    results_entries.loc[len(results_entries)] = [
                        self.video_name,
                        shape_name,
                        animal_name,
                        bin_cnt,
                        len(entry_roi_in_timebin),
                    ]
                self.out_time_lst.append(results_time)
                self.out_entries_lst.append(results_entries)
            video_timer.stop_timer()
            print(
                f"Video {self.video_name} complete (elapsed time {video_timer.elapsed_time_str}s)"
            )
        self.out_time = pd.concat(self.out_time_lst, axis=0).sort_values(
            by=["VIDEO", "SHAPE", "ANIMAL", "TIME BIN"]
        )
        self.out_entries = pd.concat(self.out_entries_lst, axis=0).sort_values(
            by=["VIDEO", "SHAPE", "ANIMAL", "TIME BIN"]
        )

    def save(self):
        self.out_entries.to_csv(self.save_path_entries)
        self.out_time.to_csv(self.save_path_time)
        self.timer.stop_timer()
        stdout_success(
            msg=f"ROI time bin entry data saved at {self.save_path_entries}",
            elapsed_time=self.timer.elapsed_time_str,
        )
        stdout_success(
            msg=f"ROI time bin time data saved at {self.save_path_time}",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = ROITimebinCalculator(config_path=r"/Users/simon/Desktop/envs/troubleshooting/sleap_5_animals/project_folder/project_config.ini",
#                             bin_length=5, body_parts=['Simon_Nose_1_1', 'JJ_Nose_1_2'], threshold=0.00)
# test.run()
# test.save()
