__author__ = "Simon Nilsson"

import glob
import os
from copy import deepcopy

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_that_column_exist)
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class SolomonImporter(ConfigReader):
    """
    Append SOLOMON human annotations onto featurized pose-estimation data.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str data_dir: path to folder holding SOLOMON data files is CSV format

    .. note::
       `Third-party import tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.
       `Example of expected SOLOMON file format <https://github.com/sgoldenlab/simba/blob/master/misc/solomon_example.csv>`__.

    Examples
    ----------
    >>> solomon_imported = SolomonImporter(config_path=r'MySimBAConfigPath', data_dir=r'MySolomonDir')
    >>> solomon_imported.run()

    References
    ----------

    .. [1] `SOLOMON CODER USER-GUIDE (PDF) <https://solomon.andraspeter.com/Solomon%20Intro.pdf>`__.
    """

    def __init__(self, config_path: str, data_dir: str):
        super().__init__(config_path=config_path)
        self.solomon_paths = glob.glob(data_dir + "/*.csv")
        check_if_filepath_list_is_empty(
            filepaths=self.solomon_paths,
            error_msg=f"SIMBA ERROR: No CSV files detected in SOLOMON directory {data_dir}",
        )
        check_if_filepath_list_is_empty(
            filepaths=self.feature_file_paths,
            error_msg=f"SIMBA ERROR: No CSV files detected in feature directory {self.features_dir}",
        )
        if not os.path.exists(self.targets_folder):
            os.mkdir(self.targets_folder)

    def run(self):
        for file_cnt, file_path in enumerate(self.solomon_paths):
            _, file_name, _ = get_fn_ext(file_path)
            feature_file_path = os.path.join(
                self.features_dir, file_name + "." + self.file_type
            )
            _, _, fps = self.read_video_info(video_name=file_name)
            if not os.path.isfile(feature_file_path):
                print(
                    "SIMBA WARNING: Data for video {} does not exist in the features directory. SimBA will SKIP appending annotations for video {}".format(
                        file_name, file_name
                    )
                )
                continue
            save_path = os.path.join(
                self.targets_folder, file_name + "." + self.file_type
            )
            solomon_df = pd.read_csv(file_path)
            check_that_column_exist(
                df=solomon_df, column_name="Behaviour", file_name=file_path
            )
            features_df = read_df(feature_file_path, self.file_type)
            out_df = deepcopy(features_df)
            features_frames = list(features_df.index)
            solomon_df["frame_cnt"] = solomon_df.index
            for clf_name in self.clf_names:
                target_col = list(solomon_df.columns[solomon_df.isin([clf_name]).any()])
                if len(target_col) == 0:
                    print(
                        "SIMBA WARNING: No SOLOMON frames annotated as containing behavior {} in video {}. SimBA will set all frames in video {} as behavior-absent for behavior {}".format(
                            clf_name, file_name, file_name, clf_name
                        )
                    )
                    continue
                target_frm_list = list(
                    solomon_df["frame_cnt"][solomon_df[target_col[0]] == clf_name]
                )
                idx_difference = list(set(target_frm_list) - set(features_frames))
                if len(idx_difference) > 0:
                    if len(idx_difference) > 0:
                        print(
                            f"SIMBA SOLOMON WARNING: SimBA found SOLOMON annotations for behavior {clf_name} in video "
                            f"{file_name} that are annotated to occur at times which is not present in the "
                            f"video data you imported into SIMBA. The video you imported to SimBA has {str(features_frames[-1])} frames. "
                            f"However, in SOLOMON, you have annotated {clf_name} to happen at frame number {str(idx_difference[0])}. "
                            f"These ambiguous annotations occur in {str(len(idx_difference))} different frames for video {file_name} that SimBA will **remove** by default. "
                            f"Please make sure you imported the same video as you annotated in SOLOMON into SimBA and the video is registered with the correct frame rate."
                        )
                    target_frm_list = [
                        x for x in target_frm_list if x not in idx_difference
                    ]
                out_df[clf_name] = 0
                out_df.loc[target_frm_list, clf_name] = 1

            write_df(out_df, self.file_type, save_path)
            print("Solomon annotations appended for video {}...".format(file_name))
        stdout_success(
            msg="All SOLOMON annotations imported. Data saved in the project_folder/csv/targets_inserted directory of the SimBA project"
        )


# test = SolomonImporter(
#     config_path="/Users/simon/Desktop/envs/simba_dev/test/data/test_projects/two_c57/project_folder/project_config.ini",
#     data_dir="/Users/simon/Desktop/envs/simba_dev/test/data/test_projects/two_c57/solomon_annotations",
# )
#
# test.run()
