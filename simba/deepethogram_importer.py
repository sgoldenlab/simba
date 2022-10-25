from simba.rw_dfs import read_df, save_df
import os, glob
from copy import deepcopy
import pandas as pd
from simba.misc_tools import get_fn_ext
from simba.read_config_unit_tests import read_config_file, read_config_entry
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.train_model_functions import get_all_clf_names

class DeepEthogramImporter(object):

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

        self.config_path, self.data_dir = config_path, deep_ethogram_dir
        if not os.path.isdir(self.data_dir):
            print('SIMBA ERROR: DeepEthogram data location has to be a DIRECTORY')
            raise ValueError()
        self.config = read_config_file(ini_path=config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.features_dir = os.path.join(self.project_path, 'csv', 'features_extracted')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.save_dir = os.path.join(self.project_path, 'csv', 'targets_inserted')
        self.deepethogram_files_found = glob.glob(self.data_dir + '/*.csv')
        self.features_files_found = glob.glob(self.features_dir + '/*.csv')
        self.model_cnt = read_config_entry(self.config, 'SML settings', 'No_targets', data_type='int')
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        if len(self.deepethogram_files_found) == 0:
            print('SIMBA ERROR: ZERO DeepEthogram CSV files found in {}'.format(self.data_dir))
            raise ValueError()
        if len(self.features_files_found) == 0:
            print('SIMBA ERROR: ZERO files found in the project_folder/csv/features_extracted directory')
            raise ValueError()
        feature_file_names, self.matches_dict = [], {}
        for feature_file_path in self.features_files_found:
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
            _, _, self.fps = read_video_info(self.vid_info_df, video_name)
            for clf_name in self.clf_names:
                if clf_name not in self.annotations_df.columns:
                    print('SIMBA ERROR: No annotations for behavior {} found in DeepEthogram annotation file for video {}'
                          'Exclude {} from your SimBA project or add DeepEthogram annotations for {} for video {}.'.format(clf_name, video_name, clf_name, clf_name, video_name))
                    raise ValueError()
            if len(self.annotations_df) > len(self.features_df):
                print('SIMBA WARNING: The annotations contain data for {} frames. The pose-estimation features contain data for {} frames. '
                      'SimBA will use the annotations for the frames present in the pose-estimation data and discard the rest.')
                self.annotations_df = self.annotations_df.head(len(self.features_df))
            if len(self.annotations_df) < len(self.features_df):
                print('SIMBA WARNING: The annotations contain data for {} frames. The pose-estimation features contain data for {} frames. '
                      'SimBA expects the annotations and pose-estimation data to contain an equal number of frames. SimBA will assume that '
                      'the un-annotated frames have no behaviors present.')
                padding = pd.DataFrame([[0] * (len(self.features_df) - len(self.annotations_df))], columns=self.annotations_df)
                self.annotations_df = self.annotations_df.append(padding, ignore_index=True)

            self.out_data = deepcopy(self.features_df)
            for clf_name in self.clf_names:
                self.out_data[clf_name] = self.annotations_df[clf_name]

            save_path = os.path.join(self.save_dir, video_name + '.' + self.file_type)
            save_df(df=self.out_data, file_type=self.file_type, save_path=save_path)
            print('DeepEthogram annotation for video {} saved...'.format(video_name))

        print('SIMBA COMPLETE: Annotations for {} behaviors added to {} videos and saved in the project_folder/csv/targets_inserted directory.'.format(len(list(self.clf_names)), len(self.matches_dict.keys())))

# test = DeepEthogramImporter(deep_ethogram_dir='/Users/simon/Desktop/troubleshooting/deepethnogram/deepethnogram',
#                             config_path='/Users/simon/Desktop/troubleshooting/deepethnogram/project_folder/project_config.ini')
# test.import_deepethogram()