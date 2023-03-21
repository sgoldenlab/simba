__author__ = "Simon Nilsson"

import os, glob
from simba.drop_bp_cords import get_fn_ext
from simba.feature_extractors.unit_tests import read_video_info
from simba.read_config_unit_tests import (check_if_filepath_list_is_empty,
                                          check_that_column_exist)
from simba.rw_dfs import read_df, save_df
from copy import deepcopy
from simba.mixins.config_reader import ConfigReader

class SolomonImporter(ConfigReader):
    """
    Class for appending SOLOMON human annotations onto featurized pose-estimation data.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    solomon_dir: str
        path to folder holding SOLOMON data files is CSV format

    Notes
    ----------
    `Third-party import tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.
    `Example of expected SOLOMON file format <https://github.com/sgoldenlab/simba/blob/master/misc/solomon_example.csv>`__.

    Examples
    ----------
    >>> solomon_imported = SolomonImporter(config_path=r'MySimBAConfigPath', solomon_dir=r'MySolomonDir')
    >>> solomon_imported.import_solomon()

    References
    ----------

    .. [1] `SOLOMON CODER USER-GUIDE (PDF) <https://solomon.andraspeter.com/Solomon%20Intro.pdf>`__.
    """

    def __init__(self,
                 config_path: str,
                 solomon_dir: str):
        super().__init__(config_path=config_path)
        self.solomon_paths = glob.glob(solomon_dir + '/*.csv')
        check_if_filepath_list_is_empty(filepaths=self.solomon_paths,
                                        error_msg=f'SIMBA ERROR: No CSV files detected in SOLOMON directory {solomon_dir}')
        check_if_filepath_list_is_empty(filepaths=self.feature_file_paths,
                                        error_msg=f'SIMBA ERROR: No CSV files detected in feature directory {self.features_dir}')
        if not os.path.exists(self.targets_folder): os.mkdir(self.targets_folder)

    def import_solomon(self):
        for file_cnt, file_path in enumerate(self.solomon_paths):
            _, file_name, _ = get_fn_ext(file_path)
            feature_file_path = os.path.join(self.features_dir, file_name + '.' + self.file_type)
            _, _, fps = read_video_info(self.video_info_df, file_name)
            if not os.path.isfile(feature_file_path):
                print('SIMBA WARNING: Data for video {} does not exist in the features directory. SimBA will SKIP appending annotations for video {}'.format(file_name, file_name))
                continue
            save_path = os.path.join(self.targets_folder, file_name + '.' + self.file_type)
            solomon_df = read_df(file_path, self.file_type).reset_index()
            check_that_column_exist(df=solomon_df, column_name='Behaviour', file_name=file_path)
            features_df = read_df(feature_file_path, self.file_type)
            out_df = deepcopy(features_df)
            features_frames = list(features_df.index)
            solomon_df['frame_cnt'] = solomon_df.index
            for clf_name in self.clf_names:
                target_col = list(solomon_df.columns[solomon_df.isin([clf_name]).any()])
                if len(target_col) == 0:
                    print('SIMBA WARNING: No SOLOMON frames annotated as containing behavior {} in video {}. SimBA will set all frames in video {} as behavior-absent for behavior {}'.format(clf_name, file_name, file_name, clf_name))
                    continue
                target_frm_list = list(solomon_df['frame_cnt'][solomon_df[target_col[0]] == clf_name])
                idx_difference = list(set(target_frm_list) - set(features_frames))
                if len(idx_difference) > 0:
                    if len(idx_difference) > 0:
                        print(f'SIMBA SOLOMON WARNING: SimBA found SOLOMON annotations for behavior {clf_name} in video '
                              f'{file_name} that are annotated to occur at times which is not present in the '
                              f'video data you imported into SIMBA. The video you imported to SimBA has {str(features_frames[-1])} frames. '
                              f'However, in SOLOMON, you have annotated {clf_name} to happen at frame number {str(idx_difference[0])}. '
                              f'These ambiguous annotations occur in {str(len(idx_difference))} different frames for video {file_name} that SimBA will **remove** by default. '
                              f'Please make sure you imported the same video as you annotated in SOLOMON into SimBA and the video is registered with the correct frame rate.')
                    target_frm_list = [x for x in target_frm_list if x not in idx_difference]
                out_df[clf_name] = 0
                out_df.loc[target_frm_list, clf_name] = 1

            save_df(out_df, self.file_type, save_path)
            print('Solomon annotations appended for video {}...'.format(file_name))
        print('SIMBA COMPLETE: All SOLOMON annotations imported. Data saved in the project_folder/csv/targets_inserted directory of the SimBA project')

# test = SolomonImporter(config_path='/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/project_folder/project_config.ini',
#                        solomon_dir='/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/solomon_data')
#
# test.import_solomon()


