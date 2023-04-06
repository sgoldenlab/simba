from simba.mixins.config_reader import ConfigReader
from simba.read_config_unit_tests import check_if_filepath_list_is_empty
import os, glob
import numpy as np
from simba.misc_tools import (get_fn_ext,
                              SimbaTimer)
from simba.rw_dfs import read_df, save_df
from simba.utils.errors import (ColumnNotFoundError,
                                AnnotationFileNotFoundError,
                                ThirdPartyAnnotationEventCountError,
                                ThirdPartyAnnotationOverlapError)
from simba.utils.warnings import (ThirdPartyAnnotationsClfMissingWarning,
                                  ThirdPartyAnnotationsOutsidePoseEstimationDataWarning)
from simba.feature_extractors.unit_tests import read_video_info
import pandas as pd
from copy import deepcopy

TIME_FIELD = 'Time_Relative_hmsf'
VIDEO_NAME_FIELD = 'Observation'
BEHAVIOR_FIELD = 'Behavior'
EVENT_TYPE_FIELD = 'Event_Type'
POINT_EVENT = 'Point'
START = 'State start'
STOP = 'State stop'
EXPECTED_FIELDS = [TIME_FIELD, VIDEO_NAME_FIELD, BEHAVIOR_FIELD, EVENT_TYPE_FIELD]

class NoldusObserverImporter(ConfigReader):
    """
    Class for appending Noldu Observer human annotations onto featurized pose-estimation data.
    Results are saved within the project_folder/csv/targets_inserted directory of
    the SimBA project (as parquets' or CSVs).

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    data_dir: str
        path to folder holding Observer data files is XLSX or XLS format

    Notes
    -----
    `Third-party import GitHub tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/third_party_annot.md>`__.
    `Expected input example 1 <https://github.com/sgoldenlab/simba/blob/master/misc/Observer_example_1.xlsx>`__.
    `Expected input example 2 <https://github.com/sgoldenlab/simba/blob/master/misc/Observer_example_2.xlsx>`__.

    Examples
    -----
    >>> importer = NoldusObserverImporter(config_path='MyConfigPath', data_dir='MyNoldusObserverDataDir')
    >>> importer.run()
    """

    def __init__(self,
                 config_path: str,
                 data_dir: str):

        super().__init__(config_path=config_path)
        self.observer_files_found = glob.glob(data_dir + '/*.xlsx') + glob.glob(data_dir + '/*.xls')
        self.observer_files_found = [x for x in self.observer_files_found if '~$' not in x]
        check_if_filepath_list_is_empty(filepaths=self.observer_files_found,
                                        error_msg=f'SIMBA ERROR: The {data_dir} directory contains ZERO xlsx/xls files')
        check_if_filepath_list_is_empty(filepaths=self.feature_file_paths,
                                        error_msg=f'SIMBA ERROR: The {self.features_dir} directory contains ZERO files')
        self.__read_data()

    def __check_column_names(self, df: pd.DataFrame, file_path: str):
        remain = list(set(EXPECTED_FIELDS) - set(list(df.columns)))
        if remain:
            raise ColumnNotFoundError(file_name=file_path, column_name=remain[0])

    def check_timestamps(self, timestamps=list):
        corrected_ts = []
        for timestamp in timestamps:
            h, m, s = timestamp.split(':', 3)
            missing_fractions = 9 - len(s)
            if missing_fractions == 0:
                corrected_ts.append(timestamp)
            else:
                corrected_ts.append(f'{h}:{m}:{s}.{"0" * missing_fractions}')
        return corrected_ts


    def __read_data(self):
        print(f'Reading Noldus Observer annotation files ({str(len(self.observer_files_found))} files)...')
        self.annotation = {}
        for file_path in self.observer_files_found:
            try:
                df = pd.read_excel(file_path, sheet_name=None,usecols=EXPECTED_FIELDS).popitem(last=False)[1]
            except KeyError:
                raise ColumnNotFoundError(file_name=file_path, column_name=', '.join(EXPECTED_FIELDS))
            for video_name in df[VIDEO_NAME_FIELD].unique():
                video_df = df[df[VIDEO_NAME_FIELD] == video_name].reset_index(drop=True)
                self.__check_column_names(df=video_df, file_path=file_path)
                video_df = video_df[video_df[EVENT_TYPE_FIELD] != POINT_EVENT]
                video_df[TIME_FIELD] = self.check_timestamps(timestamps=list(video_df[TIME_FIELD].astype(str)))
                video_df[TIME_FIELD] = pd.to_timedelta(video_df[TIME_FIELD])
                video_df[EVENT_TYPE_FIELD] = video_df[EVENT_TYPE_FIELD].replace({START: 'START', STOP: 'STOP'})
                _, _, fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
                video_df['FRAME'] = video_df[TIME_FIELD].dt.total_seconds() * fps
                video_df['FRAME'] = video_df['FRAME'].apply(np.floor)
                video_df = video_df.drop([TIME_FIELD, VIDEO_NAME_FIELD], axis=1)
                if video_name in list(self.annotation.keys()):
                    self.annotation[video_name] = pd.concat([self.annotation[video_name], video_df], axis=0).reset_index(drop=True)
                else:
                    self.annotation[video_name] = video_df
        for k, v in self.annotation.items():
            self.annotation[k] = v.sort_values(by='FRAME').reset_index(drop=True)
        print(f'Annotations for {str(len(list(self.annotation.keys())))} video names found in Ethovision files...')

    def run(self):
        for file_path in self.feature_file_paths:
            video_timer = SimbaTimer()
            video_timer.start_timer()
            _, file_name, _ = get_fn_ext(filepath=file_path)
            if file_name not in list(self.annotation.keys()):
                raise AnnotationFileNotFoundError(video_name=file_name)
            data_df = read_df(file_path=file_path, file_type=self.file_type)
            output_df = deepcopy(data_df)
            for clf_name in self.clf_names:
                clf_df = self.annotation[file_name][[EVENT_TYPE_FIELD, 'FRAME']][self.annotation[file_name][BEHAVIOR_FIELD] == clf_name].reset_index(drop=True)
                start_events, stop_events = len(clf_df[clf_df[EVENT_TYPE_FIELD] == 'START']), len(clf_df[clf_df[EVENT_TYPE_FIELD] == 'STOP'])
                if start_events != stop_events:
                    raise ThirdPartyAnnotationEventCountError(video_name=file_name, clf_name=clf_name,start_event_cnt=start_events, stop_event_cnt=stop_events)
                start_df, stop_df = clf_df[clf_df[EVENT_TYPE_FIELD] == 'START'].reset_index(drop=True), clf_df[clf_df[EVENT_TYPE_FIELD] == 'STOP'].reset_index(drop=True)
                start_df, stop_df = start_df['FRAME'].rename(columns={'FRAME': 'START'}), stop_df['FRAME'].rename(columns={'FRAME': 'STOP'})
                clf_df = pd.concat([start_df, stop_df], axis=1).reset_index(drop=True)
                clf_df.columns = ['START', 'STOP']
                if len(clf_df.query('START > STOP')) > 0:
                    raise ThirdPartyAnnotationOverlapError(video_name=file_name, clf_name=clf_name)
                if len(clf_df) == 0:
                    ThirdPartyAnnotationsClfMissingWarning(video_name=file_name, clf_name=clf_name)
                    output_df[clf_name] = 0
                    continue
                annot_idx = list(clf_df.apply(lambda x: list(range(int(x['START']), int(x['STOP']) + 1)), 1))
                annot_idx = [x for xs in annot_idx for x in xs]
                idx_diff = list(set(annot_idx) - set(data_df.index))
                if len(idx_diff) > 0:
                    ThirdPartyAnnotationsOutsidePoseEstimationDataWarning(video_name=file_name,
                                                                          clf_name=clf_name,
                                                                          frm_cnt=data_df.index[-1],
                                                                          first_error_frm=idx_diff[0],
                                                                          ambiguous_cnt=len(idx_diff))
                    annot_idx = [x for x in annot_idx if x not in idx_diff]
                output_df[clf_name] = 0
                output_df.loc[annot_idx, clf_name] = 1
            self.__save(df=output_df, path= os.path.join(self.targets_folder, file_name + '.' + self.file_type))
            video_timer.stop_timer()
            print(f'Imported Noldus Observer annotations for video {file_name} (elapsed time {video_timer.elapsed_time_str}s)...')
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: Imported annotations saved in project/folder/csv/targets_inserted directory (elapsed time {self.timer.elapsed_time_str}s).')

    def __save(self, df: pd.DataFrame, path: str):
        save_df(df=df, file_type=self.file_type, save_path=path)

# test = NoldusObserverImporter(config_path='/Users/simon/Desktop/envs/troubleshooting/Gosia/project_folder/project_config.ini',
#                               data_dir='/Users/simon/Desktop/envs/troubleshooting/Gosia/source/behaviours/Exp_38')
# test.run()


# for k, v in test.annotation.items():
#     print(v[BEHAVIOR_FIELD].unique())
#
#
#
# # test.run()



