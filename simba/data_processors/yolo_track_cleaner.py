import os
import random
import warnings
from typing import Union

import numpy as np
import pandas as pd

from simba.utils.checks import (check_file_exist_and_readable, check_if_dir_exists, check_int, check_str, check_valid_boolean, check_valid_dataframe)
from simba.utils.errors import InvalidFilepathError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,  get_fn_ext, recursive_file_search)

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

FRAME = 'FRAME'
CLASS_ID = 'CLASS_ID'
CONFIDENCE = 'CONFIDENCE'
CLASS_NAME = 'CLASS_NAME'
TRACK = 'TRACK'
BOX_CORD_FIELDS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
EXPECTED_COLS = [FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE, TRACK, 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']

class YoloTrackCleaner:
    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 bp_loc: str,
                 max_frame_gap: int = 50,
                 max_pixel_gap: int = 100,
                 recursive: bool = False,
                 randomize_order: bool = False,
                 overwrite: bool = False):

        """
        Merge fragmented YOLO tracks that likely belong to the same animal based on spatial and temporal proximity.
        Uses a union-find algorithm to group track fragments while preventing temporal overlap conflicts.

        :param Union[str, os.PathLike] data_path: Path to a single CSV file or directory containing YOLO tracking data.
        :param Union[str, os.PathLike] save_dir: Directory where cleaned track files will be saved.
        :param str bp_loc: Body part location name to use for distance calculations (e.g., 'NOSE', 'CENTER').
        :param int max_frame_gap: Maximum allowed frame gap between track end and start to consider merging. Default: 50.
        :param int max_pixel_gap: Maximum allowed pixel distance between track end and start positions to consider merging. Default: 100.
        :param bool recursive: If True and data_path is a directory, search recursively for CSV files. Default: False.
        :param bool randomize_order: If True, randomize the order in which files are processed. Default: False.
        :param bool overwrite: If True, overwrite existing output files. If False, skip files that already exist. Default: False.

        :example:
        >>> x = YoloTrackCleaner(data_path=r"E:\netholabs_videos\two_tracks_102725\csvs\cage_1_date_2025_09_13_hour_00_minute_23.csv", save_dir=r'E:\netholabs_videos\two_tracks_102725\tracks_cleaned', bp_loc='NOSE')
        >>> x.run()
        """

        check_valid_boolean(value=recursive, raise_error=True)
        if os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.data_paths = {get_fn_ext(filepath=data_path)[1]: data_path}
        elif os.path.isdir(data_path):
            if not recursive:
                self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=('.csv',), as_dict=True, raise_error=True)
            else:
                self.data_paths = recursive_file_search(directory=data_path, extensions=['.csv', ], as_dict=True, raise_error=True)
        else:
            raise InvalidFilepathError(msg=f'{data_path} is not a valid data path or directory', source=self.__class__.__name__)

        check_if_dir_exists(in_dir=save_dir)
        check_int(name=f'{self.__class__.__name__} max_frame_gap', value=max_frame_gap, min_value=1, raise_error=True)
        check_int(name=f'{self.__class__.__name__} max_pixel_gap', value=max_pixel_gap, min_value=0, raise_error=True)
        check_str(name=f'{self.__class__.__name__} bp_loc', value=bp_loc, raise_error=True, allow_blank=False)
        check_valid_boolean(value=randomize_order, raise_error=True)
        check_valid_boolean(value=overwrite, raise_error=True)
        self.bp_cols = [f'{bp_loc}_X', f'{bp_loc}_Y']
        self.max_frame_gap, self.max_pixel_gap, self.save_dir, self.overwrite = max_frame_gap, max_pixel_gap, save_dir, overwrite
        if randomize_order: self.data_paths = dict(random.sample(list(self.data_paths.items()), len(self.data_paths)))

    def get_track_info(self, video_tracks: list):
        track_info = {}
        for track_id in video_tracks:
            if track_id not in [0, -1]:
                track = self.data_df[self.data_df[TRACK] == track_id].sort_values(by=[FRAME])
                start_frm, end_frm = track.iloc[0][FRAME], track.iloc[-1][FRAME]
                start_loc, end_loc = track.iloc[0][self.bp_cols].astype(np.int32).values, track.iloc[-1][
                    self.bp_cols].astype(np.int32).values
                track_info[track_id] = {'start_frame': start_frm,
                                        'start_pos': start_loc,
                                        'end_frame': end_frm,
                                        'end_pos': end_loc}

        return track_info

    def _get_group(self, tid, track_groups):
        root = tid
        while track_groups[root] != root:
            root = track_groups[root]
        temp = tid
        while track_groups[temp] != root:
            next_temp = track_groups[temp]
            track_groups[temp] = root
            temp = next_temp
        return root

    def _check_temporal_overlap(self, group1_tids, group2_tids, track_info):
        for tid1 in group1_tids:
            t1_start = track_info[tid1]['start_frame']
            t1_end = track_info[tid1]['end_frame']
            for tid2 in group2_tids:
                t2_start = track_info[tid2]['start_frame']
                t2_end = track_info[tid2]['end_frame']
                if not (t1_end < t2_start or t2_end < t1_start):
                    return True
        return False

    def get_track_groups(self, track_info: dict, track_groups: dict):
        for end_tid, end_info in track_info.items():
            end_frame = end_info['end_frame']
            end_pos = end_info['end_pos']
            end_group = self._get_group(end_tid, track_groups)
            best_candidate = None
            best_distance = float('inf')
            for start_tid, start_info in track_info.items():
                if end_tid == start_tid:
                    continue

                start_group = self._get_group(start_tid, track_groups)
                if end_group == start_group:
                    continue

                start_frame = start_info['start_frame']
                start_pos = start_info['start_pos']

                gap = start_frame - end_frame
                if gap < 1 or gap > self.max_frame_gap:
                    continue

                distance = np.linalg.norm(end_pos - start_pos)
                if distance > self.max_pixel_gap:
                    continue

                end_group_members = [tid for tid in track_info.keys() if
                                     self._get_group(tid, track_groups) == end_group]
                start_group_members = [tid for tid in track_info.keys() if
                                       self._get_group(tid, track_groups) == start_group]
                if self._check_temporal_overlap(end_group_members, start_group_members, track_info):
                    continue

                if distance < best_distance:
                    best_distance = distance
                    best_candidate = start_tid

            if best_candidate is not None:
                end_root = self._get_group(end_tid, track_groups)
                start_root = self._get_group(best_candidate, track_groups)

                if end_root != start_root:
                    if end_root < start_root:
                        track_groups[start_root] = end_root
                        for tid in track_info.keys():
                            if self._get_group(tid, track_groups) == start_root:
                                track_groups[self._get_group(tid, track_groups)] = end_root
                    else:
                        track_groups[end_root] = start_root
                        for tid in track_info.keys():
                            if self._get_group(tid, track_groups) == end_root:
                                track_groups[self._get_group(tid, track_groups)] = start_root
        final_groups = {}
        for tid in track_info.keys():
            final_groups[tid] = self._get_group(tid, track_groups)
        return final_groups

    def run(self):
        timer = SimbaTimer(start=True)
        for video_cnt, (video_name, data_path) in enumerate(self.data_paths.items()):
            save_path = os.path.join(self.save_dir, f'{video_name}.csv')
            if os.path.isfile(save_path) and not self.overwrite:
                print(f'Skipping video {video_name} (file {save_path} exist and overwrite is set to False)')
                continue
            print(f'Fixing tracks in video {video_name} ({video_cnt + 1}/{len(self.data_paths.keys())})...')
            video_timer = SimbaTimer(start=True)
            self.data_df = pd.read_csv(data_path, index_col=0)
            check_valid_dataframe(df=self.data_df, source=self.__class__.__name__, required_fields=EXPECTED_COLS)
            video_classes, video_tracks = np.unique(self.data_df[CLASS_NAME].values), [int(x) for x in np.unique(self.data_df[TRACK].values)]
            missing_tracks = self.data_df[self.data_df[TRACK] == -1]
            fill_ids = list(range(max(video_tracks) + 1, max(video_tracks) + 1 + len(missing_tracks)))
            missing_tracks[TRACK] = fill_ids
            self.data_df.update(missing_tracks)
            video_classes, video_tracks = np.unique(self.data_df[CLASS_NAME].values), [int(x) for x in np.unique(self.data_df[TRACK].values)]
            track_info = self.get_track_info(video_tracks=video_tracks)
            track_groups = {tid: tid for tid in track_info.keys()}
            final_groups = self.get_track_groups(track_info=track_info, track_groups=track_groups)
            unique_groups = sorted(set(track_groups.values()))
            group_to_new_track = {group_id: new_id for new_id, group_id in enumerate(unique_groups, start=1)}
            track_mapping = {}
            for old_track_id, group_id in track_groups.items():
                track_mapping[old_track_id] = group_to_new_track[group_id]
            cleaned_df = self.data_df.copy()
            cleaned_df[TRACK] = cleaned_df[TRACK].apply(
                lambda x: track_mapping.get(int(x), x) if int(x) not in [0, -1] else x)
            try:
                cleaned_df.to_csv(save_path)
            except PermissionError:
                raise InvalidInputError(msg=f'Cannot save file at {save_path}. Is the file open in another program?', source=self.__class__.__name__)
            video_timer.stop_timer()
            print(f"Merged {len(video_tracks)} tracks into {len(set(final_groups.values()))} final tracks for video {save_path} (elapsed time: {video_timer.elapsed_time_str}s)")
        timer.stop_timer()
        stdout_success(msg=f'Track cleaning complete for {len(self.data_paths.keys())} data file(s). Results saved in {self.save_dir}', elapsed_time=timer.elapsed_time_str)

#
# x = YoloTrackCleaner(data_path=r"E:\netholabs_videos\two_tracks_102725\csvs\cage_1_date_2025_09_13_hour_00_minute_23.csv", save_dir=r'E:\netholabs_videos\two_tracks_102725\tracks_cleaned', bp_loc='NOSE')
# x.run()
# x = YoloTrackCleaner(data_path=r"E:\netholabs_videos\two_tracks_102725\csvs", save_dir=r'E:\netholabs_videos\two_tracks_102725\tracks_cleaned', bp_loc='NOSE')
# x.run()
# x = YoloTrackCleaner(data_path=r"E:\netholabs_videos\two_tracks_102725\csvs\cage_1_date_2025_09_13_hour_03_minute_46.csv", save_dir=r'E:\netholabs_videos\two_tracks_102725\tracks_cleaned', bp_loc='NOSE')
# x.run()