import os
from typing import Union

import numpy as np
import pandas as pd

from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_if_dir_exists, check_int, check_str,
    check_valid_dataframe)
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, read_video_info,
                                    read_video_info_csv)


def mitra_timebins(data_dir: Union[str, os.PathLike],
                   frm_path: Union[str, os.PathLike],
                   clf_name: str,
                   video_video_path: Union[str, os.PathLike],
                   save_path: Union[str, os.PathLike],
                   window_size: int):

    check_file_exist_and_readable(file_path=video_video_path)
    check_file_exist_and_readable(file_path=frm_path)
    check_if_dir_exists(in_dir=data_dir)
    check_int(name='window_size', value=window_size, min_value=1)
    check_str(name='clf', value=clf_name)
    data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], raise_error=True)
    frm_df = pd.read_csv(frm_path)
    video_info = read_video_info_csv(file_path=video_video_path)
    check_all_file_names_are_represented_in_video_log(video_info_df=video_info, data_paths=data_paths)

    results = pd.DataFrame(columns=['VIDEO', 'TIME-BIN', f'BEHAVIOR {clf_name} (S)', f'BEHAVIOR {clf_name} (FRAMES)'])

    for file_cnt, file_path in enumerate(data_paths):
        video_name = get_fn_ext(filepath=file_path)[1]
        _, _, fps = read_video_info(vid_info_df=video_info, video_name=video_name)
        two_min_frames = (fps * 60) * 2
        data_df = read_df(file_path=file_path, file_type='csv', usecols=[clf_name]).reset_index(drop=True)
        print(video_name)
        start_frm = frm_df[frm_df['VIDEO'] == video_name].iloc[0]['CNO onset (frame)']
        end_frm = np.ceil(((fps * 60) * 10)) + start_frm
        frm_win_size = np.ceil(fps * window_size)
        pre_bins_cnt = int(120 / window_size)
        video_bins_pre = np.arange(start_frm, two_min_frames, -frm_win_size)[:pre_bins_cnt]
        video_bins_pre = np.append(video_bins_pre, two_min_frames).astype(np.int32)
        print(video_bins_pre)
        video_bins_post = np.arange(start_frm, end_frm+frm_win_size, frm_win_size).astype(np.int32)
        print(video_bins_post, end_frm, start_frm)
        for epoch_idx in range(video_bins_pre.shape[0]-1):
            stop, start = video_bins_pre[epoch_idx], video_bins_pre[epoch_idx+1]
            epoch_arr = data_df.loc[start:stop][clf_name].values
            epoch_s = np.sum(epoch_arr) / fps
            results.loc[len(results)] = [video_name, -(epoch_idx)-1, round(epoch_s, 4), np.sum(epoch_arr)]
        for epoch_idx in range(video_bins_post.shape[0] - 1):
            start, stop  = video_bins_post[epoch_idx], video_bins_post[epoch_idx + 1]
            epoch_arr = data_df.loc[start:stop][clf_name].values
            epoch_s = np.sum(epoch_arr) / fps
            results.loc[len(results)] = [video_name, epoch_idx, round(epoch_s, 4), np.sum(epoch_arr)]

    results.sort_values(['VIDEO', 'TIME-BIN']).to_csv(save_path, index=False)

frm_path = r"D:\troubleshooting\mitra\Start_annotations_Simon_Hallie.csv"
data_dir = r"D:\troubleshooting\mitra\project_folder\logs\rearing_data"
window_size = 120
video_video_path = r"D:\troubleshooting\mitra\project_folder\logs\video_info.csv"
save_path = r"D:\troubleshooting\mitra\project_folder\logs\rearing_timebins_120s.csv"
mitra_timebins(data_dir=data_dir, frm_path=frm_path, window_size=window_size, video_video_path=video_video_path, clf_name='rearing', save_path=save_path)