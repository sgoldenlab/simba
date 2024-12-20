import os
from typing import Optional, Union

import matplotlib
import numpy as np

matplotlib.use('Agg')  # For non-GUI environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float,
    check_if_dir_exists)
from simba.utils.data import detect_bouts
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, read_video_info,
                                    read_video_info_csv)

GROUP_COLORS = {'CNO': 'red', 'SALINE': 'blue'}

def frequency_grapher(data_dir: Union[str, os.PathLike],
                      video_info_path: Union[str, os.PathLike],
                      start_times_path: Union[str, os.PathLike],
                      save_dir: Union[str, os.PathLike],
                      min_bout: float,
                      clf: str,
                      bin_size: Optional[int] = 55) -> None:

    """
    :param Union[str, os.PathLike] data_dir: Path to directory holding machine learning results.
    :param Union[str, os.PathLike] video_info_path: Path to CSV holding video sample rate (fps).
    :param Union[str, os.PathLike] start_times_path: Path to CSV holding the CNO onset times.
    :param Union[str, os.PathLike] save_path: Where to save th image.
    :param float min_bout: The minimum bout to plot in seconds.
    :param str clf: The name of the classifier.
    :param Optional[int] bin_size: The size of each plotted bar in frames.

    """

    plt.close('all')
    data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
    video_info_df = read_video_info_csv(file_path=video_info_path)
    check_all_file_names_are_represented_in_video_log(video_info_df=video_info_df, data_paths=data_paths)
    start_times = pd.read_csv(start_times_path, index_col=0)
    check_float(name='min_bout', value=min_bout, min_value=10e-7)
    check_if_dir_exists(in_dir=os.path.dirname(save_dir))
    df_save_path = os.path.join(save_dir, f'{clf}.csv')
    img_save_path = os.path.join(save_dir, f'{clf}.png')
    results, fps_dict = [], {}
    for file_cnt, file_path in enumerate(data_paths):
        video_name = get_fn_ext(filepath=file_path)[1]
        print(f'Analyzing {video_name}...')
        group = 'SALINE'
        if 'CNO' in video_name:
            group = 'CNO'
        df = read_df(file_path=file_path, file_type='csv', usecols=[clf])
        _, _, fps = read_video_info(video_name=video_name, video_info_df=video_info_df)
        fps_dict[video_name] = fps
        start_frm_number = start_times[start_times['VIDEO'] == video_name]['CNO onset (frame)'].values[0]
        start_frm = max(0, start_frm_number - int(fps * 120))
        end_frm = start_frm_number + int((fps * 60) * 10)
        df = df.loc[start_frm:end_frm, :].reset_index(drop=True)
        bouts = detect_bouts(data_df=df, target_lst=[clf], fps=fps)
        bouts = bouts[bouts['Bout_time'] >= min_bout]
        bouts = list(bouts['Start_frame'])
        video_results = pd.DataFrame()
        video_results['start_frame'] = bouts
        video_results['start_time'] = video_results['start_frame'] / fps
        video_results['start_time'] = video_results['start_time'] - 120
        video_results['duration'] = bin_size
        video_results['group'] = group
        video_results['event'] = clf
        video_results['length'] = len(df) / fps
        video_results['group'] = group
        video_results['video_name'] = video_name
        results.append(video_results)

    results = pd.concat(results, axis=0).sort_values(by=['start_time']).reset_index(drop=True)
    results['length'] = round(results['length'], 2)
    results['start_time'] = round(results['start_time'], 2)
    sns.set_style("white")
    tick_positions = np.arange(-120, results['length'].max(), 60)
    tick_labels = [str(int(i)) for i in tick_positions]
    plt.xticks(ticks=tick_positions, labels=tick_labels)
    plt.xlim(-120, 601)

    for idx, row in results.iterrows():
        plt.barh(y=row['event'], width=row['duration'], left=row['start_time'] + 0.1 * idx, color=GROUP_COLORS[row['group']], edgecolor=None, height=0.8, alpha=0.4)

    plt.xlabel('time (s)')
    plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS['CNO']), plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS['SALINE'])], labels=['CNO', 'SALINE'], title='Groups')
    plt.savefig(img_save_path, format='png', dpi=1200, bbox_inches='tight')
    results.to_csv(df_save_path)



frequency_grapher(data_dir=r"D:\troubleshooting\mitra\project_folder\logs\straub_tail_data",
                  clf='straub_tail',
                  video_info_path=r"D:\troubleshooting\mitra\project_folder\logs\video_info.csv",
                  start_times_path=r"D:\troubleshooting\mitra\Start_annotations_Simon_Hallie.csv",
                  min_bout=2.5,
                  bin_size=2,
                  save_dir=r'C:\Users\sroni\OneDrive\Desktop\mitra')