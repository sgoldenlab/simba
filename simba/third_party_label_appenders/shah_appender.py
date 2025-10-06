import os
from typing import List, Union

import pandas as pd

from simba.utils.checks import check_if_dir_exists, check_valid_lst
from simba.utils.errors import FrameRangeError, NoDataError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    read_df, write_df)

VIDEO_FILE_SET, FULL_LOG, BEHAVIOR = 'Video file set', 'FULL LOG', 'BEHAVIOR'
EVENT, FRAME = 'EVENT', 'FRAME'


def shah_appender(labels_dir: Union[str, os.PathLike],
                  features_dir: Union[str, os.PathLike],
                  targets_dir: Union[str, os.PathLike],
                  clf_names: List[str]):

    """
    Appends behavioral annotations from Shah-formatted .txt files to featurized pose estimation data.
    
    Parses third-party annotation files in Shah format (containing FULL LOG sections with start/stop events),
    matches them with corresponding feature-extracted CSV files by video name, and creates binary target columns
    for each specified behavior. The merged data is saved to the targets directory.

    .. note::
       For expected input files, see `simba.tests.data.shah_examples.zip`. Data from Alyssa Hall, Broad Institute.

    :param Union[str, os.PathLike] labels_dir: Path to directory containing Shah-formatted .txt annotation files with start/stop event logs.
    :param Union[str, os.PathLike] features_dir: Path to directory containing featurized pose estimation CSV files.
    :param Union[str, os.PathLike] targets_dir: Path to directory where merged feature and annotation CSV files will be saved.
    :param List[str] clf_names: List of behavior/classifier names to extract from the annotation files and append as binary target columns.
    :return: None

    :example:
    >>> LABELS_DIR = r'/Users/simon/Desktop/envs/simba/troubleshooting/shah_data'                                           #PATH TO A DIRECTORY CONTAINING .TXT FILES
    >>> FEATURES_DIR = r'/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/features_extracted'       #PATH TO A DIRECTORY CONTAINING FEATURIZED POSE ESTIMATION
    >>> TARGETS_DIR = r'/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/targets_inserted'          #PATH TO A DIRECTORY WHERE FEATURIZED POSE ESTIMATION AND ALIGNED ANNOTATIONS ARE TO BE SAVED
    >>> CLF_NAMES = ['Rearing', 'Grooming']                                                                                 #NAMES OF THE ANNOTATED BEHAVIORS TO BUILD CLASSIFIERS FROM
    >>> shah_appender(labels_dir=LABELS_DIR, features_dir=FEATURES_DIR, targets_dir=TARGETS_DIR, clf_names=CLF_NAMES)
    """

    def get_video_name(data: str, file_path: str):
        data = data.split('\n')
        for line_cnt, line in enumerate(data):
            if VIDEO_FILE_SET in line:
                video_name = data[line_cnt + 1].strip().split(' ', 2)[0].split('.', 2)[0]
                return video_name
        raise NoDataError(f'Could not find video name in file {file_path}', source=shah_appender.__name__)

    def get_full_log(data: str, file_path: str):
        data = data.split('\n')
        full_log_start, full_log_end = None, None
        for idx, line in enumerate(data):
            if FULL_LOG in line: full_log_start = idx
        for idx in range(full_log_start + 4, len(data)):
            if data[idx].startswith('---'):
                full_log_end = idx;
                break
        results = pd.DataFrame(columns=[FRAME, 'START_TIME', BEHAVIOR, 'GROUP', EVENT])
        for i in range(full_log_start + 4, full_log_end):
            line = data[i].strip()
            parts = [part.strip() for part in line.split('    ')]
            results.loc[len(results)] = parts
        results[FRAME] = results[FRAME].astype(int)
        return results[[FRAME, BEHAVIOR, EVENT]]

    lbls_paths = find_files_of_filetypes_in_directory(directory=labels_dir, extensions=['.txt'], as_dict=True,
                                                      raise_error=True)
    x_paths = find_files_of_filetypes_in_directory(directory=features_dir, extensions=['.csv'], as_dict=True,
                                                   raise_error=True)
    check_if_dir_exists(in_dir=targets_dir, source=shah_appender.__name__, raise_error=True)
    check_valid_lst(data=clf_names, valid_dtypes=(str,), raise_error=True)

    for file_cnt, (file_name, file_path) in enumerate(lbls_paths.items()):
        print(f'Processing {file_path}...')
        video_timer = SimbaTimer(start=True)
        with open(file_path, 'r') as f:
            file_data = f.read()
        video_name = get_video_name(data=file_data, file_path=file_path)
        if video_name not in x_paths.keys():
            raise NoDataError(f'Could not find file for video {video_name} in directory {features_dir}',
                              source=shah_appender.__name__)
        x_df = read_df(x_paths[video_name])
        save_path = os.path.join(targets_dir, f'{video_name}.csv')
        video_data = get_full_log(data=file_data, file_path=file_path)
        for clf in clf_names:
            x_df[clf] = 0
            clf_data = video_data[video_data[BEHAVIOR] == clf].sort_values(FRAME, ascending=True)
            start_frames = clf_data[clf_data[EVENT] == 'start'][FRAME].values
            stop_frames = clf_data[clf_data[EVENT] == 'stop'][FRAME].values
            if len(start_frames) != len(stop_frames):
                raise FrameRangeError(
                    msg=f'File {file_path} has {len(start_frames)} start events and {len(stop_frames)} stop events for behavior {clf}')
            behavior_frames = {behavior: [frame for start, stop in zip(
                clf_data[clf_data[BEHAVIOR] == behavior].sort_values(FRAME)[clf_data[EVENT] == 'start'][FRAME].values,
                clf_data[clf_data[BEHAVIOR] == behavior].sort_values(FRAME)[clf_data[EVENT] == 'stop'][FRAME].values)
                                          for frame in range(start, stop + 1)] for behavior in
                               clf_data[BEHAVIOR].unique()}[clf]
            x_df.loc[x_df.index.isin(behavior_frames), clf] = 1
        write_df(df=x_df, file_type='csv', save_path=save_path)
        video_timer.stop_timer()
        print(f'Completed annotation file {save_path} (elapsed time {video_timer.elapsed_time_str}s)')

    stdout_success(msg=f'{len(lbls_paths.keys())} annotation file(s) saved in directory {targets_dir}', source=shah_appender.__name__)

# LABELS_DIR = r'/Users/simon/Desktop/envs/simba/troubleshooting/shah_data'                                           #PATH TO A DIRECTORY CONTAINING .TXT FILES
# FEATURES_DIR = r'/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/features_extracted'       #PATH TO A DIRECTORY CONTAINING FEATURIZED POSE ESTIMATION
# TARGETS_DIR = r'/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/csv/targets_inserted'          #PATH TO A DIRECTORY WHERE FEATURIZED POSE ESTIMATION AND ALIGNED ANNOTATIONS ARE TO BE SAVED
# CLF_NAMES = ['Rearing', 'Grooming']                                                                                 #NAMES OF THE ANNOTATED BEHAVIORS TO BUILD CLASSIFIERS FROM
#
# shah_appender(labels_dir=LABELS_DIR, features_dir=FEATURES_DIR, targets_dir=TARGETS_DIR, clf_names=CLF_NAMES)
#
