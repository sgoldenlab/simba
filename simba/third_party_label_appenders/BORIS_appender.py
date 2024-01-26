__author__ = "Simon Nilsson / Florian Duclot"

import glob
import os
from copy import deepcopy

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.third_party_label_appenders.tools import is_new_boris_version
from simba.utils.checks import (check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.errors import (ThirdPartyAnnotationEventCountError,
                                ThirdPartyAnnotationOverlapError)
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df
from simba.utils.warnings import (
    ThirdPartyAnnotationsInvalidFileFormatWarning,
    ThirdPartyAnnotationsOutsidePoseEstimationDataWarning)


class BorisAppender(ConfigReader):
    """
    Append BORIS human annotations onto featurized pose-estimation data.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str data_dir: path to folder holding BORIS data files is CSV format

    .. note::
       `Third-party import tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.
       `Example I BORIS input file <https://github.com/sgoldenlab/simba/blob/master/misc/boris_example.csv>`_.
       `Example II BORIS input file <https://github.com/sgoldenlab/simba/blob/master/misc/boris_new_example.csv>`_.

    .. image:: _static/img/boris.png
       :width: 200
       :align: center

    Examples
    ----------
    >>> boris_appender = BorisAppender(config_path='MyProjectConfigPath', data_dir=r'BorisDataFolder')
    >>> boris_appender.create_boris_master_file()
    >>> boris_appender.run()

    References
    ----------

    .. [1] `Behavioral Observation Research Interactive Software (BORIS) user guide <https://boris.readthedocs.io/en/latest/#>`__.
    """

    def __init__(self, config_path: str, data_dir: str):
        super().__init__(config_path=config_path)
        self.boris_dir = data_dir
        self.boris_files_found = glob.glob(self.boris_dir + "/*.csv")
        check_if_dir_exists(data_dir)
        check_if_filepath_list_is_empty(
            filepaths=self.boris_files_found,
            error_msg=f"SIMBA ERROR: 0 BORIS CSV files found in {data_dir} directory",
        )
        print(f"Processing BORIS for {str(len(self.feature_file_paths))} file(s)...")

    def create_boris_master_file(self):
        """
        Method to create concatenated dataframe of BORIS annotations.

        Returns
        -------
        Attribute: pd.Dataframe
            master_boris_df
        """
        self.master_boris_df_list = []
        for file_cnt, file_path in enumerate(self.boris_files_found):
            try:
                _, video_name, _ = get_fn_ext(file_path)
                boris_df = pd.read_csv(file_path)
                if not is_new_boris_version(boris_df):
                    index = boris_df[boris_df["Observation id"] == "Time"].index.values
                    boris_df = pd.read_csv(file_path, skiprows=range(0, int(index + 1)))
                    boris_df = boris_df.loc[
                        :, ~boris_df.columns.str.contains("^Unnamed")
                    ]
                    boris_df.drop(
                        ["Behavioral category", "Comment", "Subject"],
                        axis=1,
                        inplace=True,
                    )
                else:
                    target_cols = [
                        "Time",
                        "Media file path",
                        "Total length",
                        "FPS",
                        "Behavior",
                        "Status",
                    ]
                    boris_df.rename(
                        columns={
                            "Media file name": "Media file path",
                            "Media duration (s)": "Total length",
                            "Behavior type": "Status",
                        },
                        inplace=True,
                    )
                    boris_df = boris_df.reindex(columns=target_cols)
                _, video_base_name, _ = get_fn_ext(boris_df.loc[0, "Media file path"])
                boris_df["Media file path"] = video_base_name
                self.master_boris_df_list.append(boris_df)
            except Exception as e:
                print(e.args)
                ThirdPartyAnnotationsInvalidFileFormatWarning(
                    annotation_app="BORIS", file_path=file_path
                )
        self.master_boris_df = pd.concat(self.master_boris_df_list, axis=0).reset_index(
            drop=True
        )
        print(
            "Found {} annotated behaviors in {} files with {} directory".format(
                str(len(self.master_boris_df["Behavior"].unique())),
                len(self.boris_files_found),
                self.boris_dir,
            )
        )
        print(
            "The following behavior annotations where detected in the boris directory:"
        )
        for behavior in self.master_boris_df["Behavior"].unique():
            print(behavior)

    def __check_non_overlapping_annotations(self):
        shifted_annotations = deepcopy(self.clf_annotations)
        shifted_annotations["START"] = self.clf_annotations["START"].shift(-1)
        shifted_annotations = shifted_annotations.head(-1)
        return shifted_annotations.query("START < STOP")

    def run(self):
        """
        Method to append BORIS annotations created in :meth:`~simba.BorisAppender.create_boris_master_file` to the
        featurized pose-estimation data in the SimBA project. Results (parquets' or CSVs) are saved within the the
        project_folder/csv/targets_inserted directory of the SimBA project.
        """
        for file_cnt, file_path in enumerate(self.feature_file_paths):
            _, self.video_name, _ = get_fn_ext(file_path)
            print("Appending BORIS annotations to {} ...".format(self.video_name))
            data_df = read_df(file_path, self.file_type)
            self.out_df = deepcopy(data_df)
            vid_annotations = self.master_boris_df.loc[
                self.master_boris_df["Media file path"] == self.video_name
            ]
            vid_annotation_starts = vid_annotations[(vid_annotations.Status == "START")]
            vid_annotation_stops = vid_annotations[(vid_annotations.Status == "STOP")]
            vid_annotations = (
                pd.concat(
                    [vid_annotation_starts, vid_annotation_stops],
                    axis=0,
                    join="inner",
                    copy=True,
                )
                .sort_index()
                .reset_index(drop=True)
            )
            vid_annotations = vid_annotations[
                vid_annotations["Behavior"].isin(self.clf_names)
            ]
            if len(vid_annotations) == 0:
                print(
                    "SIMBA WARNING: No BORIS annotations detected for SimBA classifier(s) named {} for video {}".format(
                        str(self.clf_names), self.video_name
                    )
                )
                continue
            video_fps = vid_annotations["FPS"].values[0]
            for clf in self.clf_names:
                self.clf = clf
                clf_annotations = vid_annotations[(vid_annotations["Behavior"] == clf)]
                clf_annotations_start = clf_annotations[
                    clf_annotations["Status"] == "START"
                ].reset_index(drop=True)
                clf_annotations_stop = clf_annotations[
                    clf_annotations["Status"] == "STOP"
                ].reset_index(drop=True)
                if len(clf_annotations_start) != len(clf_annotations_stop):
                    raise ThirdPartyAnnotationEventCountError(
                        video_name=self.video_name,
                        clf_name=self.clf,
                        start_event_cnt=len(clf_annotations_start),
                        stop_event_cnt=len(clf_annotations_stop),
                    )
                self.clf_annotations = (
                    clf_annotations_start["Time"]
                    .to_frame()
                    .rename(columns={"Time": "START"})
                )
                self.clf_annotations["STOP"] = clf_annotations_stop["Time"]
                self.clf_annotations = self.clf_annotations.apply(pd.to_numeric)
                results = self.__check_non_overlapping_annotations()
                if len(results) > 0:
                    raise ThirdPartyAnnotationOverlapError(
                        video_name=self.video_name, clf_name=self.clf
                    )
                self.clf_annotations["START_FRAME"] = (
                    self.clf_annotations["START"] * video_fps
                ).astype(int)
                self.clf_annotations["END_FRAME"] = (
                    self.clf_annotations["STOP"] * video_fps
                ).astype(int)
                if len(self.clf_annotations) == 0:
                    self.out_df[clf] = 0
                    print(
                        f"SIMBA WARNING: No BORIS annotation detected for video {self.video_name} and behavior {clf}. SimBA will set all frame annotations as absent."
                    )
                    continue
                annotations_idx = list(
                    self.clf_annotations.apply(
                        lambda x: list(
                            range(int(x["START_FRAME"]), int(x["END_FRAME"]) + 1)
                        ),
                        1,
                    )
                )
                annotations_idx = [x for xs in annotations_idx for x in xs]
                idx_difference = list(set(annotations_idx) - set(self.out_df.index))
                if len(idx_difference) > 0:
                    ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(
                        video_name=self.video_name,
                        clf_name=clf,
                        frm_cnt=self.out_df.index[-1],
                        first_error_frm=idx_difference[0],
                        ambiguous_cnt=len(idx_difference),
                    )
                    annotations_idx = [
                        x for x in annotations_idx if x not in idx_difference
                    ]
                self.out_df[clf] = 0
                self.out_df.loc[annotations_idx, clf] = 1
            self.__save_boris_annotations()
        self.timer.stop_timer()
        stdout_success(
            msg="BORIS annotations appended to dataset and saved in project_folder/csv/targets_inserted directory",
            elapsed_time=self.timer.elapsed_time_str,
        )

    def __save_boris_annotations(self):
        self.save_path = os.path.join(
            self.targets_folder, self.video_name + "." + self.file_type
        )
        write_df(self.out_df, self.file_type, self.save_path)
        print("Saved BORIS annotations for video {}...".format(self.video_name))


# test = BorisAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      data_dir='/Users/simon/Downloads/FIXED')
# test.create_boris_master_file()
# test.run()

# test = BorisAppender(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', boris_folder=r'/Users/simon/Desktop/troubleshooting/train_model_project/boris_import')
# test.create_boris_master_file()
# test.append_boris()

# test = BorisAppender(config_path='/Users/simon/Desktop/envs/marcel_boris/project_folder/project_config.ini', boris_folder=r'/Users/simon/Desktop/envs/marcel_boris/BORIS_data')
# test.create_boris_master_file()
# test.append_boris()

# test = BorisAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      boris_folder='/Users/simon/Downloads/FIXED')
# test.create_boris_master_file()
# test.append_boris()
