import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists,
    check_str, check_valid_boolean, check_valid_dataframe)
from simba.utils.data import create_color_palette, detect_bouts
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, read_video_info)


def plot_clf_cumcount(config_path: Union[str, os.PathLike],
                      clf: str,
                      data_dir: Optional[Union[str, os.PathLike]] = None,
                      save_path: Optional[Union[str, os.PathLike]] = None,
                      bouts: Optional[bool] = False,
                      seconds: Optional[bool] = False) -> None:

    """

    Generates and saves a cumulative count plot of a specified classifier's occurrences over video frames or time.

    .. image:: _static/img/plot_clf_cumcount.webp
       :width: 500
       :align: center


    :param Union[str, os.PathLike] config_path: Path to the configuration file, which includes settings and paths for data processing and storage.
    :param str clf: The classifier name (e.g., 'CIRCLING') for which to calculate cumulative counts.
    :param Optional[Union[str, os.PathLike]] data_dir: Directory containing the log files to analyze.  If not provided, the default path in the configuration is used.
    :param Optional[Union[str, os.PathLike]] save_path: Destination path to save the plot image. If None,  saves to the logs path in the configuration.
    :param Optional[bool] bouts: If True, calculates the cumulative count in terms of detected bouts instead  of time or frames.
    :param Optional[bool] seconds: If True, calculates time in seconds rather than frames.
    :return: None.

    :example:
    >>> plot_clf_cumcount(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini", clf='CIRCLING', data_dir=r'D:\troubleshooting\mitra\project_folder\logs\test', seconds=True, bouts=True)
    """

    config = ConfigReader(config_path=config_path, read_video_info=True, create_logger=False)
    if data_dir is not None:
        check_if_dir_exists(in_dir=data_dir, source=f'{plot_clf_cumcount.__name__} data_dir')
    else:
        data_dir = config.machine_results_dir
    if save_path is None:
        save_path = os.path.join(config.logs_path, f'cumcount_{config.datetime}.png')
    data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=[f'.{config.file_type}'], raise_error=True)
    check_valid_boolean(value=[bouts, seconds], source=plot_clf_cumcount.__name__, raise_error=True)
    check_str(name=f'{plot_clf_cumcount.__name__} clf', value=clf)
    x_name = 'VIDEO TIME (FRAMES)'
    y_name = f'{clf} TIME (FRAMES)'
    if seconds:
        check_all_file_names_are_represented_in_video_log(video_info_df=config.video_info_df, data_paths=data_paths)
        x_name = f'VIDEO TIME (S)'
    if bouts:
        y_name = f'{clf} (BOUT COUNT)'

    clrs = create_color_palette(pallete_name='Set2', increments=len(data_paths), as_rgb_ratio=True)
    for file_cnt, file_path in enumerate(data_paths):
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Analysing video {video_name} ({file_cnt+1}/{len(data_paths)})...')
        df = read_df(file_path=file_path, file_type=config.file_type)
        check_valid_dataframe(df=df, source=f'{plot_clf_cumcount.__name__} {file_path}', required_fields=[clf])
        if not bouts and not seconds:
            clf_sum = list(df[clf].cumsum().ffill())
            time = list(df.index)
        elif not bouts and seconds:
            _, _, fps = read_video_info(vid_info_df=config.video_info_df, video_name=video_name)
            clf_sum = np.round(np.array(df[clf].cumsum().ffill() / fps), 2)
            time = list(df.index / fps)
        else:
            bout_starts = detect_bouts(data_df=df, target_lst=[clf], fps=1)['Start_frame'].values
            bouts_arr = np.full(len(df), fill_value=np.nan, dtype=np.float32)
            bouts_arr[0] = 0
            for bout_cnt in range(bout_starts.shape[0]): bouts_arr[bout_starts[bout_cnt]] = bout_cnt+1
            clf_sum = pd.DataFrame(bouts_arr, columns=[clf]).ffill().values.reshape(-1)
            if seconds:
                _, _, fps = read_video_info(vid_info_df=config.video_info_df, video_name=video_name)
                time = list(df.index / fps)
            else:
                time = list(df.index)
        video_results = pd.DataFrame(data=clf_sum, columns=[y_name])
        video_results['VIDEO'] = video_name
        video_results[x_name] = time
        sns.lineplot(data=video_results, x=x_name, y=y_name, hue="VIDEO", palette=[clrs[file_cnt]])

    plt.savefig(save_path)

    config.timer.stop_timer()
    stdout_success(msg=f"Graph saved at {save_path}", elapsed_time=config.timer.elapsed_time_str)
    #
    #
    #
    #
    #









plot_clf_cumcount(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini",
                  clf='CIRCLING',
                  data_dir=r'D:\troubleshooting\mitra\project_folder\logs\test',
                  seconds=True,
                  bouts=True)