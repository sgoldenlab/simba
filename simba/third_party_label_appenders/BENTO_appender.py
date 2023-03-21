__author__ = "Simon Nilsson"


import os, glob
from simba.feature_extractors.unit_tests import read_video_info
from simba.read_config_unit_tests import check_if_filepath_list_is_empty
from simba.mixins.config_reader import ConfigReader
from simba.misc_tools import get_fn_ext
from simba.rw_dfs import read_df, save_df
from simba.utils.warnings import (ThirdPartyAnnotationsOutsidePoseEstimationDataWarning,
                                  ThirdPartyAnnotationsClfMissingWarning,
                                  ThirdPartyAnnotationsAdditionalClfWarning)
from simba.utils.errors import AnnotationFileNotFoundError
import pandas as pd
from copy import deepcopy

class BentoAppender(ConfigReader):
    """
    Class for appending BENTO annotation to SimBA featurized datasets.

    Notes
    ----------
    `Example BORIS input file <https://github.com/sgoldenlab/simba/blob/master/misc/boris_example.csv`__.
    'GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md`__.

    Examples
    ----------
    >>> bento_dir = 'tests/test_data/bento_example'
    >>> config_path = 'tests/test_data/import_tests/project_folder/project_config.ini'
    >>> bento_appender = BentoAppender(config_path=config_path, bento_dir=bento_dir)
    >>> bento_appender.run()

    References
    ----------

    .. [1] Segalin et al., eLife, https://doi.org/10.7554/eLife.63720

    """

    def __init__(self,
                 config_path: str,
                 bento_dir: str):

        super().__init__(config_path=config_path)
        self.bento_dir = bento_dir
        self.feature_files = glob.glob(self.features_dir + '/*.' + self.file_type)
        self.bento_files = glob.glob(self.bento_dir + '/*.' + 'annot')
        check_if_filepath_list_is_empty(filepaths=self.feature_files, error_msg='SIMBA ERROR: No feature files found in project_folder/csv/features_extracted. Extract Features BEFORE appending BENTO annotations')
        check_if_filepath_list_is_empty(filepaths=self.bento_files, error_msg=f'SIMBA ERROR: No BENTO files with .annot extension found in {self.bento_dir}.')
        self.saved_files = []

    def run(self):
        for file_cnt, file_path in enumerate(self.feature_files):
            _, self.video_name, ext = get_fn_ext(filepath=file_path)
            print(f'Appending BENTO annotation to video {self.video_name}...')
            _, _, fps = read_video_info(vid_info_df=self.video_info_df, video_name=self.video_name)
            bento_path = os.path.join(self.bento_dir, self.video_name + '.annot')
            if bento_path not in self.bento_files:
                raise AnnotationFileNotFoundError(video_name=self.video_name)
            self.save_path = os.path.join(self.targets_folder, self.video_name + '.' + self.file_type)
            feature_df = read_df(file_path=file_path, file_type=self.file_type)
            video_frm_length = len(feature_df)
            self.results_df = deepcopy(feature_df)
            annotation_df = pd.read_csv(bento_path, delim_whitespace=True, index_col=False, low_memory=False)
            start_idx = annotation_df.index[annotation_df['Bento'] == 'Ch1----------'].values[0]
            sliced_annot = annotation_df.iloc[start_idx+1:]
            annotated_behaviors = sliced_annot[sliced_annot['Bento'].str.contains(">")]['Bento'].tolist()
            annotated_behavior_names = [x[1:] for x in annotated_behaviors]
            missing_annotation = set(self.clf_names) - set(annotated_behavior_names)
            missing_clf = list(set(annotated_behavior_names) - set(self.clf_names))
            annotation_intersection = [x for x in self.clf_names if x in annotated_behavior_names]
            for missing_clf in missing_annotation:
                ThirdPartyAnnotationsClfMissingWarning(video_name=self.video_name, clf_name=missing_clf)
                self.results_df[missing_clf] = 0
            if missing_clf:
                ThirdPartyAnnotationsAdditionalClfWarning(video_name=self.video_name, clf_names=missing_clf)

            for clf_name in annotation_intersection:
                self.results_df[clf_name] = 0
                clf_start_idx = sliced_annot.index[sliced_annot['Bento'] == f'>{clf_name}'].values[0]
                clf_df = sliced_annot.loc[clf_start_idx+2:, :]
                end_idx = clf_df.isnull()[clf_df.isnull().any(axis=1)].idxmax(axis=1).index
                if end_idx.values:
                    end_idx = end_idx.values[0]
                else:
                    end_idx = max(clf_df.index+1)
                clf_df = clf_df.loc[:end_idx-1, :].reset_index(drop=True)
                clf_df.columns = ['start_time', 'stop_time', 'duration']
                clf_df['start_frm'] = clf_df['start_time'].astype(float) * fps
                clf_df['end_frm'] = clf_df['stop_time'].astype(float) * fps
                clf_df['start_frm'] = clf_df['start_frm'].astype(int)
                clf_df['end_frm'] = clf_df['end_frm'].astype(int)
                annotations_idx = list(clf_df.apply(lambda x: list(range(int(x['start_frm']), int(x['end_frm']) + 1)), 1))
                annotations_idx = [i for s in annotations_idx for i in s]
                annotations_idx_outside_video = [x for x in annotations_idx if x > video_frm_length]
                valid_annotation_ids = [x for x in annotations_idx if x <= video_frm_length]
                if len(annotations_idx_outside_video):
                    ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(video_name=self.video_name,
                                                                          clf_name=clf_name,
                                                                          frm_cnt=len(feature_df),
                                                                          first_error_frm=annotations_idx_outside_video[0],
                                                                          ambiguous_cnt=len(annotations_idx_outside_video))
                if len(valid_annotation_ids) > 0:
                    print(f'Appending {str(len(valid_annotation_ids))} {clf_name} frame annotations to video {self.video_name}...')
                    self.results_df.loc[valid_annotation_ids, clf_name] = 1
            self.__save()
        print(f'SIMBA COMPLETE: Annotations for {str(len(self.saved_files))} video(s) and saved in project_folder/csv/targets_inserted directory.')

    def __save(self):
        save_df(df=self.results_df, file_type=self.file_type, save_path=self.save_path)
        self.saved_files.append(self.save_path)
        print(f'BENTO annotations appended to video {self.video_name} and saved in {self.save_path}')


# test = BentoAppender(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/project_folder/project_config.ini',
#                      bento_dir='/Users/simon/Desktop/envs/simba_dev/tests/test_data/bento_example')
# test.run()











