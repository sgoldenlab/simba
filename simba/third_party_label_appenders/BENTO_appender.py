__author__ = "Simon Nilsson"

import glob
import os
from copy import deepcopy
import pandas as pd
from typing import Union, Dict, Optional
import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_if_filepath_list_is_empty, check_if_dir_exists, check_all_file_names_are_represented_in_video_log
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df, find_files_of_filetypes_in_directory, bento_file_reader
from simba.utils.warnings import (ThirdPartyAnnotationsClfMissingWarning, ThirdPartyAnnotationsOutsidePoseEstimationDataWarning)


class BentoAppender(ConfigReader):
    """
    Append BENTO annotation to SimBA featurized datasets.

    .. note::
       `Example BENTO input file <https://github.com/sgoldenlab/simba/blob/master/misc/bento_example.annot>`_.
       'GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`_.

    :param str config_path: path to SimBA project config file in Configparser format
    :param str data_dir: Path to folder containing BENTO data.

    :example:
    >>> bento_dir = 'tests/test_data/bento_example'
    >>> config_path = 'tests/test_data/import_tests/project_folder/project_config.ini'
    >>> bento_appender = BentoAppender(config_path=config_path, data_dir=bento_dir)
    >>> bento_appender.run()

    References
    ----------
    .. [1] Segalin et al., eLife, https://doi.org/10.7554/eLife.63720
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path)
        check_if_dir_exists(in_dir=data_dir)
        self.bento_files = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.annot'], raise_error=False, raise_warning=True)
        check_if_filepath_list_is_empty(filepaths=self.feature_file_paths, error_msg="SIMBA ERROR: No feature files found in project_folder/csv/features_extracted. Extract Features BEFORE appending BENTO annotations")
        check_if_filepath_list_is_empty(filepaths=self.bento_files, error_msg=f"SIMBA ERROR: No BENTO files with .annot extension found in {data_dir}.")
        self.saved_files = []

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.bento_files)
        for file_cnt, bento_file_path in enumerate(self.bento_files):
            _, self.video_name, ext = get_fn_ext(filepath=bento_file_path)
            print(f"Appending BENTO annotation to video {self.video_name}...")
            _, _, fps = self.read_video_info(video_name=self.video_name)
            features_path = os.path.join(self.features_dir, self.video_name + f'.{self.file_type}')
            if not os.path.isfile(features_path):
                raise NoFilesFoundError(msg=f'No features file for annotation file {self.video_name} file in {self.features_dir}. SimBA is expecting a file at path {features_path}')
            self.save_path = os.path.join(self.targets_folder, self.video_name + f'.{self.file_type}')
            feature_df = read_df(file_path=features_path, file_type=self.file_type)
            self.results = deepcopy(feature_df)
            bento_dict = bento_file_reader(file_path=bento_file_path, fps=fps, save_path=None, orient='index')
            for clf_name in self.clf_names:
                self.results[clf_name] = 0
                if clf_name not in bento_dict.keys():
                    ThirdPartyAnnotationsClfMissingWarning(video_name=self.video_name, clf_name=clf_name)
                else:
                    clf_bento_df = bento_dict[clf_name]
                    annotations_idx = [i for s in list(clf_bento_df.apply(lambda x: list(range(int(x["START"]), int(x["STOP"]))), 1)) for i in s]
                    annotations_idx_outside_video = [x for x in annotations_idx if x > len(feature_df)]
                    valid_annotation_ids = [x for x in annotations_idx if x < len(feature_df)]

                    if len(annotations_idx_outside_video) > 0:
                        ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(video_name=self.video_name,
                                                                              clf_name=clf_name,
                                                                              frm_cnt=len(feature_df),
                                                                              first_error_frm=annotations_idx_outside_video[0],
                                                                              ambiguous_cnt=len(annotations_idx_outside_video))
                    if len(valid_annotation_ids) > 0:
                        print(f"Appending {str(len(valid_annotation_ids))} {clf_name} frame annotations to video {self.video_name}...")
                        self.results.loc[valid_annotation_ids, clf_name] = 1
            self.__save()
            stdout_success(msg=f"Annotations for {str(len(self.saved_files))} video(s) and saved in the {self.targets_folder}.")

    def __save(self):
        write_df(df=self.results, file_type=self.file_type, save_path=self.save_path)
        self.saved_files.append(self.save_path)
        print(f"BENTO annotations appended to video {self.video_name} and saved in {self.save_path}")



#
#
#     #
#     #
#     #         annotation_df = pd.read_csv(
#     #             bento_path, delim_whitespace=True, index_col=False, low_memory=False
#     #         )
#     #         start_idx = annotation_df.index[
#     #             annotation_df["Bento"] == "Ch1----------"
#     #         ].values[0]
#     #         sliced_annot = annotation_df.iloc[start_idx + 1 :]
#     #         annotated_behaviors = sliced_annot[sliced_annot["Bento"].str.contains(">")][
#     #             "Bento"
#     #         ].tolist()
#     #         annotated_behavior_names = [x[1:] for x in annotated_behaviors]
#     #         missing_annotation = set(self.clf_names) - set(annotated_behavior_names)
#     #         missing_clf = list(set(annotated_behavior_names) - set(self.clf_names))
#     #         annotation_intersection = [
#     #             x for x in self.clf_names if x in annotated_behavior_names
#     #         ]
#     #         for missing_clf in missing_annotation:
#     #             ThirdPartyAnnotationsClfMissingWarning(
#     #                 video_name=self.video_name, clf_name=missing_clf
#     #             )
#     #             self.results_df[missing_clf] = 0
#     #         if missing_clf:
#     #             ThirdPartyAnnotationsAdditionalClfWarning(
#     #                 video_name=self.video_name, clf_names=missing_clf
#     #             )
#     #
#     #         for clf_name in annotation_intersection:
#     #             self.results_df[clf_name] = 0
#     #             clf_start_idx = sliced_annot.index[
#     #                 sliced_annot["Bento"] == f">{clf_name}"
#     #             ].values[0]
#     #             clf_df = sliced_annot.loc[clf_start_idx + 2 :, :]
#     #             end_idx = (
#     #                 clf_df.isnull()[clf_df.isnull().any(axis=1)].idxmax(axis=1).index
#     #             )
#     #             if end_idx.values:
#     #                 end_idx = end_idx.values[0]
#     #             else:
#     #                 end_idx = max(clf_df.index + 1)
#     #             clf_df = clf_df.loc[: end_idx - 1, :].reset_index(drop=True)
#     #             clf_df.columns = ["start_time", "stop_time", "duration"]
#     #             clf_df["start_frm"] = clf_df["start_time"].astype(float) * fps
#     #             clf_df["end_frm"] = clf_df["stop_time"].astype(float) * fps
#     #             clf_df["start_frm"] = clf_df["start_frm"].astype(int)
#     #             clf_df["end_frm"] = clf_df["end_frm"].astype(int)
#     #             annotations_idx = list(
#     #                 clf_df.apply(
#     #                     lambda x: list(
#     #                         range(int(x["start_frm"]), int(x["end_frm"]) + 1)
#     #                     ),
#     #                     1,
#     #                 )
#     #             )
#     #             annotations_idx = [i for s in annotations_idx for i in s]
#     #             annotations_idx_outside_video = [
#     #                 x for x in annotations_idx if x > video_frm_length
#     #             ]
#     #             valid_annotation_ids = [
#     #                 x for x in annotations_idx if x <= video_frm_length
#     #             ]
#     #             if len(annotations_idx_outside_video):
#     #                 ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(
#     #                     video_name=self.video_name,
#     #                     clf_name=clf_name,
#     #                     frm_cnt=len(feature_df),
#     #                     first_error_frm=annotations_idx_outside_video[0],
#     #                     ambiguous_cnt=len(annotations_idx_outside_video),
#     #                 )
#     #             if len(valid_annotation_ids) > 0:
#     #                 print(
#     #                     f"Appending {str(len(valid_annotation_ids))} {clf_name} frame annotations to video {self.video_name}..."
#     #                 )
#     #                 self.results_df.loc[valid_annotation_ids, clf_name] = 1
#     #         self.__save()
#     #     stdout_success(
#     #         msg=f"Annotations for {str(len(self.saved_files))} video(s) and saved in project_folder/csv/targets_inserted directory."
#     #     )
#     #
#     # def __save(self):
#     #     write_df(df=self.results_df, file_type=self.file_type, save_path=self.save_path)
#     #     self.saved_files.append(self.save_path)
#     #     print(
#     #         f"BENTO annotations appended to video {self.video_name} and saved in {self.save_path}"
#     #     )
#     #
#
# test = BentoAppender(config_path=r"C:\troubleshooting\bento_test\project_folder\project_config.ini",
#                      data_dir=r"C:\troubleshooting\bento_test\bento_files")
# test.run()
