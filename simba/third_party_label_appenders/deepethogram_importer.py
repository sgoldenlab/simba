__author__ = "Simon Nilsson"

from simba.rw_dfs import read_df, save_df
import os, glob
from copy import deepcopy
import pandas as pd
from simba.misc_tools import get_fn_ext
from simba.read_config_unit_tests import (check_if_filepath_list_is_empty,
                                          check_if_dir_exists)
from simba.feature_extractors.unit_tests import read_video_info
from simba.mixins.config_reader import ConfigReader

class DeepEthogramImporter(ConfigReader):

    """
    Class for appending DeepEthogram optical flow annotations onto featurized pose-estimation data.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    deep_ethogram_dir: str
        path to folder holding DeepEthogram data files is CSV format

    Notes
    ----------
    `Third-party import tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.
    `Example expected input <https://github.com/sgoldenlab/simba/blob/master/misc/deep_ethogram_labels.csv>`__.

    Examples
    ----------
    >>> deepethogram_importer = DeepEthogramImporter(config_path=r'MySimBAConfigPath', deep_ethogram_dir=r'MyDeepEthogramDir')
    >>> deepethogram_importer.import_deepethogram()

    References
    ----------

    .. [1] `DeepEthogram repo <https://github.com/jbohnslav/deepethogram>`__.
    .. [2] `Example DeepEthogram input file <https://github.com/sgoldenlab/simba/blob/master/misc/deep_ethogram_labels.csv>`__.
    """

    def __init__(self,
                 deep_ethogram_dir: str,
                 config_path: str):

        super().__init__(config_path=config_path)
        self.data_dir = deep_ethogram_dir
        check_if_dir_exists(in_dir=self.data_dir)
        self.deepethogram_files_found = glob.glob(self.data_dir + '/*.csv')
        check_if_filepath_list_is_empty(filepaths=self.deepethogram_files_found,
                                        error_msg=f'SIMBA ERROR: ZERO DeepEthogram CSV files found in {self.data_dir} directory')
        check_if_filepath_list_is_empty(filepaths=self.feature_file_paths,
                                        error_msg='SIMBA ERROR: ZERO files found in the project_folder/csv/features_extracted directory')
        feature_file_names, self.matches_dict = [], {}
        for feature_file_path in self.feature_file_paths:
            _, file_name, ext = get_fn_ext(feature_file_path)
            feature_file_names.append(file_name)
        for file_path in self.deepethogram_files_found:
            _, file_name, ext = get_fn_ext(file_path)
            if file_name in feature_file_names:
                self.matches_dict[file_path] = os.path.join(self.features_dir, file_name + ext)
                pass
            elif file_name.endswith('_labels'):
                short_file_name = file_name[:-7]
                if short_file_name in feature_file_names:
                    self.matches_dict[file_path] = os.path.join(self.features_dir, short_file_name + ext)
                    pass
            else:
                print('SIMBA ERROR: Could not find file in project_folder/csv/features_extracted directory representing {}'.format(file_name))
                raise FileNotFoundError()

    def import_deepethogram(self):
        for cnt, (k, v) in enumerate(self.matches_dict.items()):
            _, video_name, _ = get_fn_ext(filepath=v)
            self.annotations_df = read_df(file_path=k, file_type=self.file_type).reset_index(drop=True)
            self.features_df = read_df(file_path=v, file_type=self.file_type).reset_index(drop=True)
            _, _, self.fps = read_video_info(self.video_info_df, video_name)
            for clf_name in self.clf_names:
                if clf_name not in self.annotations_df.columns:
                    print('SIMBA ERROR: No annotations for behavior {} found in DeepEthogram annotation file for video {}'
                          'Exclude {} from your SimBA project or add DeepEthogram annotations for {} for video {}.'.format(clf_name, video_name, clf_name, clf_name, video_name))
                    raise ValueError()
            if len(self.annotations_df) > len(self.features_df):
                print(f'SIMBA WARNING: The DEEPETHOGRAM annotations for video {video_name} contain data for {str(len(self.annotations_df))} frames. The pose-estimation features for the same video contain data for {str(len(self.features_df))} frames. '
                      'SimBA will use the annotations for the frames present in the pose-estimation data and discard the rest.')
                self.annotations_df = self.annotations_df.head(len(self.features_df))
            if len(self.annotations_df) < len(self.features_df):
                print(f'SIMBA WARNING: The DEEPETHOGRAM annotations for video {video_name} contain data for {str(len(self.annotations_df))} frames. The pose-estimation features for the same video contain data for {str(len(self.features_df))} frames. '
                      'SimBA expects the annotations and pose-estimation data to contain an equal number of frames. SimBA will assume that '
                      'the un-annotated frames have no behaviors present.')
                padding = pd.DataFrame([[0] * (len(self.features_df) - len(self.annotations_df))], columns=self.annotations_df)
                self.annotations_df = self.annotations_df.append(padding, ignore_index=True)

            self.out_data = deepcopy(self.features_df)
            for clf_name in self.clf_names:
                self.out_data[clf_name] = self.annotations_df[clf_name]

            save_path = os.path.join(self.targets_folder, video_name + '.' + self.file_type)
            save_df(df=self.out_data, file_type=self.file_type, save_path=save_path)
            print('DeepEthogram annotation for video {} saved...'.format(video_name))

        print('SIMBA COMPLETE: Annotations for {} behaviors added to {} videos and saved in the project_folder/csv/targets_inserted directory.'.format(len(list(self.clf_names)), len(self.matches_dict.keys())))

# test = DeepEthogramImporter(deep_ethogram_dir='/Users/simon/Desktop/troubleshooting/deepethnogram/deepethnogram',
#                             config_path='/Users/simon/Desktop/troubleshooting/deepethnogram/project_folder/project_config.ini')
# test.import_deepethogram()