import pandas as pd
import numpy as np
from simba.misc_tools import get_fn_ext, detect_bouts
from simba.feature_extractors.unit_tests import read_video_info, read_video_info_csv
from simba.enums import Methods
from simba.utils.warnings import ThirdPartyAnnotationsInvalidFileFormatWarning
from simba.utils.errors import (InvalidFileTypeError, ColumnNotFoundError)

def observer_timestamp_corrector(timestamps=list):
    corrected_ts = []
    for timestamp in timestamps:
        h, m, s = timestamp.split(':', 3)
        missing_fractions = 9 - len(s)
        if missing_fractions == 0:
            corrected_ts.append(timestamp)
        else:
            corrected_ts.append(f'{h}:{m}:{s}.{"0" * missing_fractions}')
    return corrected_ts


def read_boris_annotation_files(data_paths: list,
                                error_setting: str,
                                video_info_df: pd.DataFrame,
                                log_setting: bool=False):

    MEDIA_FILE_PATH = 'Media file path'
    OBSERVATION_ID = 'Observation id'
    TIME = 'Time'
    BEHAVIOR = 'Behavior'
    STATUS = 'Status'
    EXPECTED_HEADERS = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]

    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(file_path)
        boris_df = pd.read_csv(file_path)
        try:
            start_idx = (boris_df[boris_df[OBSERVATION_ID] == TIME].index.values)
            df = pd.read_csv(file_path, skiprows=range(0, int(start_idx + 1)))[EXPECTED_HEADERS]
            _, video_base_name, _ = get_fn_ext(df.loc[0, MEDIA_FILE_PATH])
            df.drop(MEDIA_FILE_PATH, axis=1, inplace=True)
            df.columns = ['TIME', 'BEHAVIOR', 'EVENT']
            df['TIME'] = df['TIME'].astype(float)
            dfs[video_base_name] = df.sort_values(by='TIME')
        except Exception as e:
            print(e)
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app='BORIS',
                                                              file_path=file_path,
                                                              log_status=log_setting)
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file. See the docs for expected file format.')
            else:
                pass
    for video_name, video_df in dfs.items():
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        video_df['FRAME'] = (video_df['TIME'] * fps).astype(int)
        video_df.drop('TIME', axis=1, inplace=True)
    return dfs


# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_boris_annotation_files(data_paths=['/Users/simon/Downloads/FIXED/c_oxt27_190816_122013_s_trimmcropped.csv'],
#                                 error_setting='WARNING',
#                                  log_setting=False,
#                                  video_info_df=video_info_df)


def read_ethovision_files(data_paths: list,
                          error_setting: str,
                          video_info_df: pd.DataFrame,
                          log_setting: bool=False):

    VIDEO_FILE = 'Video file'
    HEADER_LINES = 'Number of header lines:'
    RECORDING_TIME = 'Recording time'
    BEHAVIOR = 'Behavior'
    EVENT = 'Event'
    POINT_EVENT = 'point event'
    STATE_START = 'state start'
    STATE_STOP = 'state stop'
    START = 'START'
    STOP = 'STOP'


    EXPECTED_FIELDS = [RECORDING_TIME, BEHAVIOR, EVENT]

    dfs = {}
    data_paths = [x for x in data_paths if '~$' not in x]
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Reading ETHOVISION annotation file ({str(file_cnt + 1)} / {str(len(data_paths))}) ...')
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            sheet_name = list(df.keys())[-1]
            df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0, header=None)
            video_path = df.loc[VIDEO_FILE].values[0]
            _, video_name, ext = get_fn_ext(video_path)
            header_n = int(df.loc[HEADER_LINES].values[0]) - 2
            df = df.iloc[header_n:].reset_index(drop=True)
            df.columns = list(df.iloc[0])
            df = df.iloc[2:].reset_index(drop=True)[EXPECTED_FIELDS]
            df.columns = ['TIME', 'BEHAVIOR', 'EVENT']
            df = df[df['EVENT'] != POINT_EVENT].reset_index(drop=True)
            df['EVENT'] = df['EVENT'].replace({STATE_START: START, STATE_STOP: STOP})
            dfs[video_name] = df

        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app='ETHOVISION', file_path=file_path, log_status=log_setting)
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid ETHOVISION file. See the docs for expected file format.')
            else:
                pass

    for video_name, video_df in dfs.items():
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        video_df['FRAME'] = (video_df['TIME'] * fps).astype(int)
        video_df.drop('TIME', axis=1, inplace=True)

    return dfs
# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_ethovision_files(data_paths=['/Users/simon/Desktop/envs/simba_dev/tests/test_data/import_tests/ethovision_data/correct.xlsx'],
#                                 error_setting='WARNING',
#                                  log_setting=False,
#                                  video_info_df=video_info_df)
#


def read_observer_files(data_paths: list,
                        error_setting: str,
                        video_info_df: pd.DataFrame,
                        log_setting: bool=False):

    TIME_FIELD = 'Time_Relative_hmsf'
    VIDEO_NAME_FIELD = 'Observation'
    BEHAVIOR_FIELD = 'Behavior'
    EVENT_TYPE_FIELD = 'Event_Type'
    POINT_EVENT = 'Point'
    START = 'State start'
    STOP = 'State stop'
    EXPECTED_FIELDS = [TIME_FIELD, VIDEO_NAME_FIELD, BEHAVIOR_FIELD, EVENT_TYPE_FIELD]

    dfs = {}
    for file_path in data_paths:
        try:
            df = pd.read_excel(file_path, sheet_name=None,usecols=EXPECTED_FIELDS).popitem(last=False)[1]
        except KeyError:
            raise ColumnNotFoundError(file_name=file_path, column_name=', '.join(EXPECTED_FIELDS))
        try:
            for video_name in df[VIDEO_NAME_FIELD].unique():
                video_df = df[df[VIDEO_NAME_FIELD] == video_name].reset_index(drop=True)
                video_df = video_df[video_df[EVENT_TYPE_FIELD] != POINT_EVENT]
                video_name = video_df[VIDEO_NAME_FIELD].iloc[0]
                video_df.drop(VIDEO_NAME_FIELD, axis=1, inplace=True)
                video_df[TIME_FIELD] = observer_timestamp_corrector(timestamps=list(video_df[TIME_FIELD].astype(str)))
                video_df[TIME_FIELD] = pd.to_timedelta(video_df[TIME_FIELD])
                video_df[EVENT_TYPE_FIELD] = video_df[EVENT_TYPE_FIELD].replace({START: 'START', STOP: 'STOP'})
                video_df.columns = ['TIME', 'BEHAVIOR', 'EVENT']
                if video_name in list(dfs.keys()):
                    dfs[video_name] = pd.concat([dfs[video_name], video_df], axis=0).reset_index(drop=True)
                else:
                    dfs[video_name] = video_df

        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app='OBSERVER', file_path=file_path, log_status=log_setting)
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid OBSERVER file. See the docs for expected file format.')
            else:
                pass

    for video_name, video_df in dfs.items():
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        video_df['FRAME'] = video_df['TIME'].dt.total_seconds() * fps
        video_df['FRAME'] = video_df['FRAME'].apply(np.floor).astype(int)
        video_df.drop('TIME', axis=1, inplace=True)

    return dfs
# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_observer_files(data_paths=['/Users/simon/Desktop/envs/troubleshooting/Gosia/source/behaviours/Exp_38/03+11WT_20171010-120856_4_no_dupl_no_audio_fps4_grey-simba_crop_frame_n.xlsx'],
#                          error_setting='WARNING',
#                          log_setting=False,
#                         video_info_df=video_info_df)


def read_solomon_files(data_paths: list,
                        error_setting: str,
                        video_info_df: pd.DataFrame,
                        log_setting: bool=False):

    BEHAVIOR = 'Behaviour'
    TIME = 'Time'
    EXPECTED_COLUMNS = [TIME, BEHAVIOR]

    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, file_name, _ = get_fn_ext(file_path)
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=file_name)
        try:
            df = pd.read_csv(file_path)[EXPECTED_COLUMNS]
            df = df[~df.isnull().any(axis=1)].reset_index(drop=True)
            df['FRAME'] = df[TIME] * fps
            df['FRAME'] = df['FRAME'].apply(np.floor).astype(int)
            video_df = pd.DataFrame()
            for behavior in df[BEHAVIOR].unique():
                behavior_arr = df['FRAME'][df[BEHAVIOR] == behavior].reset_index(drop=True).values
                new_arr = np.full((np.max(behavior_arr)+2), 0)
                for i in behavior_arr: new_arr[i] = 1
                bouts = detect_bouts(data_df=pd.DataFrame(new_arr, columns=[behavior]), target_lst=[behavior], fps=1)[['Event', 'Start_frame', 'End_frame']].values
                results = []
                for obs in bouts:
                    results.append([obs[0], 'START', obs[1]])
                    results.append([obs[0], 'STOP', obs[2]])
                video_df = pd.concat([video_df, pd.DataFrame(results, columns=['BEHAVIOR', 'EVENT', 'FRAME']).sort_values(by=['FRAME'])], axis=0)
            dfs[file_name] = video_df.reset_index(drop=True)

        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app='SOLOMON', file_path=file_path, log_status=log_setting)
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid SOLOMON file. See the docs for expected file format.')
            else:
                pass

    return dfs

# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_solomon_files(data_paths=['/Users/simon/Desktop/envs/simba_dev/tests/test_data/solomon_import/solomon_import/Together_1.csv'],
#                          error_setting='WARNING',
#                          log_setting=False,
#                          video_info_df=video_info_df)



def read_bento_files(data_paths: list,
                     error_setting: str,
                     video_info_df: pd.DataFrame,
                     log_setting: bool=False):
    BENTO = 'Bento'
    CHANNEL = 'Ch1----------'

    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, ext = get_fn_ext(filepath=file_path)
        _, _, fps = read_video_info(vid_info_df=video_info_df, video_name=video_name)
        try:
            data_df = pd.read_csv(file_path, delim_whitespace=True, index_col=False, low_memory=False)
            start_idx = data_df.index[data_df[BENTO] == CHANNEL].values[0]
            sliced_annot = data_df.iloc[start_idx + 1:]
            clfs = sliced_annot[sliced_annot[BENTO].str.contains(">")]['Bento'].tolist()
            video_events = []
            for clf_name in clfs:
                start_idx = sliced_annot.index[sliced_annot[BENTO] == f'{clf_name}'].values[0]
                clf_df = sliced_annot.loc[start_idx + 2:, :]
                end_idx = clf_df.isnull()[clf_df.isnull().any(axis=1)].idxmax(axis=1).index
                if end_idx.values: end_idx = end_idx.values[0]
                else: end_idx = max(clf_df.index + 1)
                clf_df = clf_df.loc[:end_idx - 1, :].reset_index(drop=True).drop('file', axis=1).astype(float)
                clf_df.columns = ['START', 'STOP']
                clf_df = clf_df * fps
                for obs in clf_df.values:
                    video_events.append([clf_name, 'START', obs[0]])
                    video_events.append([clf_name, 'STOP', obs[1]])
            video_df = pd.DataFrame(video_events, columns=['BEHAVIOR', 'EVENT', 'FRAME'])
            video_df['FRAME'] = video_df['FRAME'].astype(int)
            video_df['BEHAVIOR'] = video_df['BEHAVIOR'].str[1:]
            dfs[video_name] = video_df
        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app='BENTO', file_path=file_path, log_status=log_setting)
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BENTO file. See the docs for expected file format.')
            else:
                pass
    return dfs

# video_info_df = read_video_info_csv(file_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/logs/video_info.csv')
#
# df = read_bento_files(data_paths=['/Users/simon/Desktop/envs/simba_dev/tests/test_data/bento_example/Together_1.annot'],
#                          error_setting='WARNING',
#                          log_setting=False,
#                          video_info_df=video_info_df)





def read_deepethogram_files(data_paths: list,
                             error_setting: str,
                             log_setting: bool=False):

    BACKGROUND = 'background'
    dfs = {}
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(file_path)
        try:
            data_df = pd.read_csv(file_path, index_col=0)
            data_df.drop(BACKGROUND, axis=1, inplace=True)
            bouts = detect_bouts(data_df=data_df, target_lst=list(data_df.columns), fps=1)[['Event', 'Start_frame', 'End_frame']].values
            results = []
            for obs in bouts:
                results.append([obs[0], 'START', obs[1]])
                results.append([obs[0], 'STOP', obs[2]])
            dfs[video_name] = pd.DataFrame(results, columns=['BEHAVIOR', 'EVENT', 'FRAME']).sort_values(by=['FRAME']).reset_index(drop=True)
        except Exception as e:
            if error_setting == Methods.WARNING.value:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app='DEEPETHOGRAM',
                                                              file_path=file_path,
                                                              log_status=log_setting)
            elif error_setting == Methods.ERROR.value:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file. See the docs for expected file format.')
            else:
                pass


    return dfs

def fix_uneven_start_stop_count(data: pd.DataFrame):
    starts = data['FRAME'][data['EVENT'] == 'START'].values
    stops = data['FRAME'][data['EVENT'] == 'STOP'].values

    if starts.shape[0] < stops.shape[0]:
        sorted_stops = np.sort(stops)
        for start in starts:
            stop_idx = np.argwhere(sorted_stops>start)[0][0]
            sorted_stops = np.delete(sorted_stops, stop_idx)
        for remove_val in sorted_stops:
            remove_idx = np.argwhere(stops==remove_val)[0][0]
            stops = np.delete(stops, remove_idx)

    if stops.shape[0] < starts.shape[0]:
        sorted_starts = np.sort(starts)
        for stop in stops:
            start_idx = np.argwhere(sorted_starts<stop)[-1][0]
            sorted_starts = np.delete(sorted_starts, start_idx)
        for remove_val in sorted_starts:
            remove_idx = np.argwhere(starts==remove_val)[0][0]
            starts = np.delete(starts, remove_idx)


    return pd.DataFrame({'START': starts, 'STOP': stops})

def check_stop_events_prior_to_start_events(df: pd.DataFrame):
    overlaps_idx = []
    for obs_cnt, obs in enumerate(df.values):
        if obs[0] > obs[1]:
            overlaps_idx.append(obs_cnt)
    return overlaps_idx