__author__ = "Simon Nilsson"

import glob
import os

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_that_column_exist)
from simba.utils.printing import stdout_success
from simba.utils.read_write import (get_fn_ext, read_config_file, read_df,
                                    write_df)


class ImportEthovision(ConfigReader):
    """
    Append ETHOVISION human annotations onto featurized pose-estimation data.
    Results are saved within the project_folder/csv/targets_inserted directory of
    the SimBA project (as parquets' or CSVs).

    :param str config_path: path to SimBA project config file in Configparser format
    :param str data_dir: path to folder holding ETHOVISION data files is XLSX or XLS format

    .. note::
       `Third-party import GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.
       `Example of expected ETHOVISION file <https://github.com/sgoldenlab/simba/blob/master/misc/ethovision_example.xlsx>`__.

    Examples
    -----
    >>> ethovision_importer = ImportEthovision(config_path="MyConfigPath", data_dir="MyEthovisionFolderPath")
    >>> ethovision_importer.run()
    """

    def __init__(self, config_path: str, data_dir: str):
        super().__init__(config_path=config_path)

        print("Appending ETHOVISION annotations...")
        self.config = read_config_file(config_path)
        self.files_found = glob.glob(data_dir + "/*.xlsx") + glob.glob(
            data_dir + "/*.xls"
        )
        self.files_found = [x for x in self.files_found if "~$" not in x]
        check_if_filepath_list_is_empty(
            filepaths=self.files_found,
            error_msg="SIMBA ERROR: No ETHOVISION xlsx or xls files found in {}".format(
                str(data_dir)
            ),
        )

    def __read_files(self):
        for file_path in self.files_found:
            ethovision_df = pd.read_excel(file_path, sheet_name=None)
            manual_scoring_sheet_name = list(ethovision_df.keys())[-1]
            ethovision_df = pd.read_excel(
                file_path,
                sheet_name=manual_scoring_sheet_name,
                index_col=0,
                header=None,
            )
            try:
                video_path = ethovision_df.loc["Video file"].values[0]
            except KeyError:
                print(
                    'SIMBA ERROR: "Video file" row does not exist in the sheet named {} in file {}'.format(
                        manual_scoring_sheet_name, file_path
                    )
                )
                raise ValueError(
                    f'SIMBA ERROR: "Video file" does not exist in the sheet named {manual_scoring_sheet_name} in file {file_path}'
                )
            try:
                if np.isnan(video_path):
                    print(
                        'SIMBA ERROR: "Video file" row does not have a value in the sheet named {} in file {}'.format(
                            manual_scoring_sheet_name, file_path
                        )
                    )
                    raise ValueError(
                        f'SIMBA ERROR: "Video file" row does not have a value in the sheet named {manual_scoring_sheet_name} in file {file_path}'
                    )
            except:
                pass
            dir_name, self.video_name, ext = get_fn_ext(video_path)
            self.processed_videos.append(video_path)
            self.features_file_path = os.path.join(
                self.features_dir, self.video_name + "." + self.file_type
            )
            print("Processing annotations for video " + str(self.video_name) + "...")
            _, _, fps = self.read_video_info(video_name=str(self.video_name))
            header_lines_n = (
                int(ethovision_df.loc["Number of header lines:"].values[0]) - 2
            )
            ethovision_df = ethovision_df.iloc[header_lines_n:].reset_index(drop=True)
            ethovision_df.columns = list(ethovision_df.iloc[0])
            ethovision_df = ethovision_df.iloc[2:].reset_index(drop=True)
            self.clf_dict = {}
            check_that_column_exist(
                df=ethovision_df, column_name="Behavior", file_name=file_path
            )
            check_that_column_exist(
                df=ethovision_df, column_name="Recording time", file_name=file_path
            )
            check_that_column_exist(
                df=ethovision_df, column_name="Event", file_name=file_path
            )
            non_clf_behaviors = list(
                set(ethovision_df["Behavior"].unique()) - set(self.clf_names)
            )
            non_clf_behaviors = [x for x in non_clf_behaviors if x.lower() != "start"]
            if len(non_clf_behaviors) > 0:
                print(
                    f"SIMBA WARNING: The ETHOVISION annotation file for video {self.video_name} contains annotations for {str(len(non_clf_behaviors))} behaviors"
                    f" which is NOT defined in the SimBA project: {non_clf_behaviors} and will be SKIPPED."
                )
            for clf in self.clf_names:
                self.clf_dict[clf] = {}
                clf_data = ethovision_df[ethovision_df["Behavior"] == clf]
                if len(clf_data) == 0:
                    print(
                        f"SIMBA WARNING: ZERO ETHOVISION annotations detected for SimBA classifier named {clf} for video {self.video_name}. "
                        f"SimBA will label that the behavior as ABSENT in the entire {self.video_name} video."
                    )
                starts = list(
                    clf_data["Recording time"][clf_data["Event"] == "state start"]
                )
                ends = list(
                    clf_data["Recording time"][clf_data["Event"] == "state stop"]
                )
                self.clf_dict[clf]["start_frames"] = [int(x * fps) for x in starts]
                self.clf_dict[clf]["end_frames"] = [int(x * fps) for x in ends]
                frame_list = []
                for cnt, start in enumerate(self.clf_dict[clf]["start_frames"]):
                    frame_list.extend(
                        list(range(start, self.clf_dict[clf]["end_frames"][cnt]))
                    )
                self.clf_dict[clf]["frames"] = frame_list
            self.__insert_annotations()

    def __insert_annotations(self):
        self.features_df = read_df(self.features_file_path, self.file_type)
        for clf in self.clf_names:
            annotation_mismatch = list(
                set(self.clf_dict[clf]["frames"]) - set(self.features_df.index)
            )
            if len(annotation_mismatch) > 0:
                print(
                    f"SIMBA ETHOVISION WARNING: SimBA found ETHOVISION annotations for behavior {clf} in video "
                    f"{self.video_name} that are annotated to occur at times which is not present in the "
                    f"video data you imported into SIMBA. The video you imported to SimBA has {str(max(self.features_df.index))} frames. "
                    f"However, in ETHOVISION, you have annotated {clf} to happen at frame number {str(annotation_mismatch[0])}. "
                    f"These ambiguous annotations occur in {str(len(annotation_mismatch))} different frames for video {self.video_name} that SimBA will **remove** by default. "
                    f"Please make sure you imported the same video as you annotated in ETHOVISION into SimBA and the video is registered with the correct frame rate."
                )
            self.features_df[clf] = 0
            self.features_df[clf] = np.where(
                self.features_df.index.isin(self.clf_dict[clf]["frames"]), 1, 0
            )
        self.__save_data()

    def __save_data(self):
        save_file_name = os.path.join(
            self.targets_folder, self.video_name + "." + self.file_type
        )
        write_df(self.features_df, self.file_type, save_file_name)
        print("Added Ethovision annotations for video {} ... ".format(self.video_name))

    def run(self):
        self.processed_videos = []
        self.__read_files()
        self.timer.stop_timer()
        stdout_success(
            msg="All Ethovision annotations added. Files with annotation are located in the project_folder/csv/targets_inserted directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = ImportEthovision(config_path= r"/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/project_folder/project_config.ini", data_dir=r'/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/ethovision_data')
# test = ImportEthovision(config_path= r"/Users/simon/Desktop/envs/simba_dev/test/data/test_projects/two_c57/project_folder/project_config.ini", data_dir='/Users/simon/Desktop/envs/simba_dev/test/data/test_projects/two_c57/ethovision_annotations')
# test.run()
