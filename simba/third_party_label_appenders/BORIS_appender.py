__author__ = "Simon Nilsson / Florian Duclot"

import glob
import os
from copy import deepcopy
from typing import Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.third_party_label_appenders.tools import (
    is_new_boris_version, read_boris_annotation_files)
from simba.utils.checks import (check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.errors import (NoDataError,
                                ThirdPartyAnnotationEventCountError,
                                ThirdPartyAnnotationOverlapError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)
from simba.utils.warnings import (
    ThirdPartyAnnotationsInvalidFileFormatWarning,
    ThirdPartyAnnotationsOutsidePoseEstimationDataWarning)

BEHAVIOR = 'BEHAVIOR'
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

    :example:
    >>> test = BorisAppender(config_path=r"C:\troubleshooting\boris_test\project_folder\project_config.ini", data_dir=r"C:\troubleshooting\boris_test\project_folder\boris_files")
    >>> test.run()

    References
    ----------

    .. [1] `Behavioral Observation Research Interactive Software (BORIS) user guide <https://boris.readthedocs.io/en/latest/#>`__.
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike]):

        super().__init__(config_path=config_path)
        check_if_dir_exists(data_dir)
        self.boris_dir = data_dir
        self.boris_files_found = find_files_of_filetypes_in_directory(directory=self.boris_dir, extensions=['.csv'], raise_error=True)
        print(f"Processing {len(self.boris_files_found)} BORIS annotation file(s) in {data_dir} directory...")
        if len(self.feature_file_paths) == 0:
            raise NoDataError(f'No data files found in the {self.features_dir} directory.', source=self.__class__.__name__)


    def __check_non_overlapping_annotations(self, annotation_df):
        shifted_annotations = deepcopy(annotation_df)
        shifted_annotations["START"] = annotation_df["START"].shift(-1)
        shifted_annotations = shifted_annotations.head(-1)
        error_rows = shifted_annotations.query("START < STOP")
        if len(error_rows) > 0:
            raise ThirdPartyAnnotationOverlapError(video_name=self.file_name, clf_name=self.clf_name)


    def run(self):
        boris_annotation_dict = read_boris_annotation_files(data_paths=self.boris_files_found, video_info_df=self.video_info_df, orient='index')
        print(boris_annotation_dict)
        for file_cnt, file_path in enumerate(self.feature_file_paths):
            self.file_name = get_fn_ext(filepath=file_path)[1]
            self.video_timer = SimbaTimer(start=True)
            print(f'Processing BORIS annotations for feature file {self.file_name}...')
            if self.file_name not in boris_annotation_dict.keys():
                raise NoDataError(msg=f'Your SimBA project has a feature file named {self.file_name}, however no annotations exist for this file in the {self.boris_dir} directory.')
            else:
                video_annot = boris_annotation_dict[self.file_name]
            data_df = read_df(file_path, self.file_type)
            print(data_df)
            video_annot = video_annot.fillna(len(data_df))
            print(data_df)
            for clf_name in self.clf_names:
                self.clf_name = clf_name
                data_df[clf_name] = 0
                if clf_name not in video_annot[BEHAVIOR].unique():
                    print(f"SIMBA WARNING: No BORIS annotation detected for video {self.file_name} and behavior {clf_name}. SimBA will set all frame annotations as absent.")
                    continue
                video_clf_annot = video_annot[video_annot[BEHAVIOR] == clf_name].reset_index(drop=True)
                self.__check_non_overlapping_annotations(video_clf_annot)
                annotations_idx = list(video_clf_annot.apply(lambda x: list(range(int(x["START"]), int(x["STOP"]) + 1)), 1))
                annotations_idx = [x for xs in annotations_idx for x in xs]
                idx_difference = list(set(annotations_idx) - set(data_df.index))
                if len(idx_difference) > 0:
                    ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(video_name=self.file_name, clf_name=clf_name, frm_cnt=data_df.index[-1], first_error_frm=idx_difference[0], ambiguous_cnt=len(idx_difference))
                    annotations_idx = [x for x in annotations_idx if x not in idx_difference]
                data_df.loc[annotations_idx, clf_name] = 1
                print(f'Appended {len(annotations_idx)} BORIS behavior {clf_name} annotations for video {self.file_name}...')
            self.__save_boris_annotations(df=data_df)
        self.timer.stop_timer()
        stdout_success(msg=f"BORIS annotations appended to {len(self.feature_file_paths)} data file(s) and saved in {self.targets_folder}", elapsed_time=self.timer.elapsed_time_str)

    def __save_boris_annotations(self, df):
        self.save_path = os.path.join(self.targets_folder, f"{self.file_name}.{self.file_type}")
        write_df(df, self.file_type, self.save_path)
        self.video_timer.stop_timer()
        print(f"Saved BORIS annotations for video {self.file_name}... (elapsed time: {self.video_timer.elapsed_time_str})")



# test = BorisAppender(config_path=r"C:\troubleshooting\boris_test_2\project_folder\project_config.ini",
#                      data_dir=r"C:\troubleshooting\boris_test_2\project_folder\boris_files")
# test.run()
#
#
# test = BorisAppender(config_path=r"C:\troubleshooting\snake\project_folder\project_config.ini",
#                      data_dir=r"C:\troubleshooting\snake\project_folder\boris")
# test.run()

#
# test = BorisAppender(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini",
#                      data_dir=r"C:\troubleshooting\two_black_animals_14bp\BORIS")
# test.run()