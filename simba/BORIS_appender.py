__author__ = "Simon Nilsson", "JJ Choong"

from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          check_if_dir_exists,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.train_model_functions import get_all_clf_names
import os, glob
from simba.enums import ReadConfig, Paths, Dtypes
from simba.rw_dfs import read_df, save_df
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer)
import pandas as pd
from copy import deepcopy

class BorisAppender(object):

    """
    Class for appending BORIS human annotations onto featurized pose-estimation data.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    boris_folder: str
        path to folder holding BORIS data files is CSV format

    Notes
    ----------
    `Example BORIS input file <https://github.com/sgoldenlab/simba/blob/master/misc/boris_example.csv`__.

    Examples
    ----------
    >>> boris_appender = BorisAppender(config_path='MyProjectConfigPath', boris_folder=r'BorisDataFolder')
    >>> boris_appender.create_boris_master_file()
    >>> boris_appender.append_boris()

    References
    ----------

    .. [1] `Behavioral Observation Research Interactive Software (BORIS) user guide <https://boris.readthedocs.io/en/latest/#>`__.
    """

    def __init__(self,
                 config_path: str,
                 boris_folder: str):


        self.config = read_config_file(config_path)
        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.in_features_dir = os.path.join(self.project_path, Paths.FEATURES_EXTRACTED_DIR.value)
        self.out_dir = os.path.join(self.project_path, Paths.TARGETS_INSERTED_DIR.value)
        self.model_cnt = read_config_entry(self.config, ReadConfig.SML_SETTINGS.value, ReadConfig.TARGET_CNT.value, Dtypes.INT.value)
        check_if_dir_exists(boris_folder)
        self.in_boris_dir = boris_folder
        self.clf_names = get_all_clf_names(config=self.config, target_cnt=self.model_cnt)
        self.feature_files_found = glob.glob(self.in_features_dir + '/*.' + self.file_type)
        self.boris_files_found = glob.glob(self.in_boris_dir + '/*.csv')
        check_if_filepath_list_is_empty(filepaths=self.boris_files_found,
                                        error_msg='SIMBA ERROR: 0 BORIS CSV files found in {} directory'.format(self.in_boris_dir))
        print('Processing BORIS for {} file(s)...'.format(str(len(self.feature_files_found))))

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
                index = (boris_df[boris_df['Observation id'] == "Time"].index.values)
                boris_df = pd.read_csv(file_path, skiprows=range(0, int(index + 1)))
                boris_df = boris_df.loc[:, ~boris_df.columns.str.contains('^Unnamed')]
                boris_df.drop(['Behavioral category', 'Comment', 'Subject'], axis=1, inplace=True)
                _, video_base_name, _ = get_fn_ext(boris_df.loc[0, 'Media file path'])
                boris_df['Media file path'] = video_base_name
                self.master_boris_df_list.append(boris_df)
            except Exception as e:
                print('SIMBA WARNING: {} is not a valid BORIS file and is skipped. See the SimBA GitHub repository for expected file format'.format(file_path))
                print(e.args)
        self.master_boris_df = pd.concat(self.master_boris_df_list, axis=0).reset_index(drop=True)
        print('Found {} annotated behaviors in {} files with {} directory'.format(str(len(self.master_boris_df['Behavior'].unique())), len(self.boris_files_found), self.in_boris_dir))
        print('The following behavior annotations where detected in the boris directory:')
        for behavior in self.master_boris_df['Behavior'].unique():
            print(behavior)

    def __check_non_overlapping_annotations(self):
        shifted_annotations = deepcopy(self.clf_annotations)
        shifted_annotations['START'] = self.clf_annotations['START'].shift(-1)
        shifted_annotations = shifted_annotations.head(-1)
        return shifted_annotations.query('START < STOP')

    def append_boris(self):
        """
        Method to append BORIS annotations created in :meth:`~simba.BorisAppender.create_boris_master_file` to the
        featurized pose-estimation data in the SimBA project. Results (parquets' or CSVs) are saved within the the
        project_folder/csv/targets_inserted directory of the SimBA project.
        """
        for file_cnt, file_path in enumerate(self.feature_files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            print('Appending BORIS annotations to {} ...'.format(self.video_name))
            data_df = read_df(file_path, self.file_type)
            self.out_df = deepcopy(data_df)
            vid_annotations = self.master_boris_df.loc[self.master_boris_df['Media file path'] == self.video_name]
            vid_annotation_starts = vid_annotations[(vid_annotations.Status == 'START')]
            vid_annotation_stops = vid_annotations[(vid_annotations.Status == 'STOP')]
            vid_annotations = pd.concat([vid_annotation_starts, vid_annotation_stops], axis=0, join='inner', copy=True).sort_index().reset_index(drop=True)
            vid_annotations = vid_annotations[vid_annotations['Behavior'].isin(self.clf_names)]
            if (len(vid_annotations) == 0):
                print('SIMBA WARNING: No BORIS annotations detected for SimBA classifier(s) named {} for video {}'.format(str(self.clf_names), self.video_name))
                continue
            video_fps = vid_annotations['FPS'].values[0]
            for clf in self.clf_names:
                self.clf = clf
                clf_annotations = vid_annotations[(vid_annotations['Behavior'] == clf)]
                clf_annotations_start = clf_annotations[clf_annotations['Status'] == 'START'].reset_index(drop=True)
                clf_annotations_stop = clf_annotations[clf_annotations['Status'] == 'STOP'].reset_index(drop=True)
                if len(clf_annotations_start) != len(clf_annotations_stop):
                    print('SIMBA BORIS ERROR: The BORIS annotations for behavior {} in video {} contains {} start events and {} stop events.'
                          'SimBA requires the number of stop and start event counts to be equal'.format(clf, self.video_name, str(clf_annotations_start), str(clf_annotations_stop)))
                    raise ValueError()
                self.clf_annotations = clf_annotations_start['Time'].to_frame().rename(columns={'Time': "START"})
                self.clf_annotations['STOP'] = clf_annotations_stop['Time']
                self.clf_annotations = self.clf_annotations.apply(pd.to_numeric)
                results = self.__check_non_overlapping_annotations()
                if len(results) > 0:
                    print('SIMBA BORIS ERROR: The BORIS annotations for behavior {} in video {} contains '
                          'behavior start events that are initiated PRIOR '
                          'to the PRECEDING behavior event ending.'
                          'SimBA requires a specific behavior event to end before another behavior event can start.'.format(clf, self.video_name))
                    raise ValueError()
                self.clf_annotations['START_FRAME'] = (self.clf_annotations['START'] * video_fps).astype(int)
                self.clf_annotations['END_FRAME'] = (self.clf_annotations['STOP'] * video_fps).astype(int)
                if len(self.clf_annotations) == 0:
                    self.out_df[clf] = 0
                    print(f'SIMBA WARNING: No BORIS annotation detected for video {self.video_name} and behavior {clf}. SimBA will set all frame annotations as absent.')
                    continue
                annotations_idx = list(self.clf_annotations.apply(lambda x: list(range(int(x['START_FRAME']), int(x['END_FRAME']) + 1)), 1))
                annotations_idx = [x for xs in annotations_idx for x in xs]
                idx_difference = list(set(annotations_idx) - set(self.out_df.index))
                if len(idx_difference) > 0:
                    print(f'SIMBA BORIS WARNING: SimBA found BORIS annotations for behavior {clf} in video '
                          f'{self.video_name} that are annotated to occur at times which is not present in the '
                          f'video data you imported into SIMBA. The video you imported to SimBA has {str(self.out_df.index[-1])} frames. '
                          f'However, in BORIS, you have annotated {clf} to happen at frame number {str(idx_difference[0])}. '
                          f'These ambiguous annotations occur in {str(len(idx_difference))} different frames for video {self.video_name} that SimBA will **remove** by default. '
                          f'Please make sure you imported the same video as you annotated in BORIS into SimBA and the video is registered with the correct frame rate.')
                    annotations_idx = [x for x in annotations_idx if x not in idx_difference]
                self.out_df[clf] = 0
                self.out_df.loc[annotations_idx, clf] = 1
            self.__save_boris_annotations()

        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: BORIS annotations appended to dataset and saved in project_folder/csv/targets_inserted directory (elapsed time: {self.timer.elapsed_time_str}s).')

    def __save_boris_annotations(self):
        save_path = os.path.join(self.out_dir, self.video_name + '.' + self.file_type)
        save_df(self.out_df, self.file_type, save_path)
        print('Saved BORIS annotations for video {}...'.format(self.video_name))

# test = BorisAppender(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                      boris_folder='/Users/simon/Downloads/FIXED')
# test.create_boris_master_file()
# test.append_boris()

# test = BorisAppender(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini', boris_folder=r'/Users/simon/Desktop/troubleshooting/train_model_project/boris_import')
# test.create_boris_master_file()
# test.append_boris()

# test = BorisAppender(config_path='/Users/simon/Desktop/envs/marcel_boris/project_folder/project_config.ini', boris_folder=r'/Users/simon/Desktop/envs/marcel_boris/BORIS_data')
# test.create_boris_master_file()
# test.append_boris()
