import glob
import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import typed

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.utils.checks import (check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df


def velocity_aggregator(config_path: Union[str, os.PathLike],
                        data_dir: Union[str, os.PathLike],
                        body_part: str,
                        ts_plot: Optional[bool] = True):

    """
    Aggregate velocity data from multiple pose-estimation files.

    :param Union[str, os.PathLike] config_path: Path to SimBA configuration file.
    :param Union[str, os.PathLike] data_dir: Directory containing data files.
    :param str data_dir body_part: Body part to use when calculating velocity.
    :param Optional[bool] data_dir ts_plot: Whether to generate a time series plot of velocities for each data file. Defaults to True.
    :example:
    >>> config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini'
    >>> data_dir = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location'
    >>> body_part = 'Nose_1'
    >>> velocity_aggregator(config_path=config_path, data_dir=data_dir, body_part=body_part)
    """

    timer = SimbaTimer(start=True)
    check_if_dir_exists(in_dir=data_dir)
    config = ConfigReader(config_path=config_path, create_logger=False)
    file_paths = glob.glob(data_dir + f'/*.{config.file_type}')
    save_dir = os.path.join(config.logs_path, f'rolling_velocities_{config.datetime}')
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    check_if_filepath_list_is_empty(filepaths=file_paths, error_msg=f'No data in {config.file_type} format found in {data_dir}')
    bp_cols = [f'{body_part}_x', f'{body_part}_y']
    mean_velocities = {}
    for file_cnt, file_path in enumerate(file_paths):
        rolling_results = pd.DataFrame()
        _, video_name, _ = get_fn_ext(filepath=file_path)
        print(f'Analyzing {video_name}...')
        data_df = read_df(file_path=file_path, file_type=config.file_type, usecols=bp_cols).astype(int)
        _, px_per_mm, fps = config.read_video_info(video_name=video_name)
        shifted_df = FeatureExtractionMixin.create_shifted_df(df=data_df).drop(bp_cols, axis=1)
        frm_dist = FeatureExtractionMixin().framewise_euclidean_distance(location_1=data_df.values, location_2=shifted_df.values, px_per_mm=px_per_mm, centimeter= True).astype(np.float32)
        rolling = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=frm_dist, window_sizes=np.array([1.0]), sample_rate=int(fps), statistics=typed.List(['sum'])).flatten()
        rolling_results[f'Rolling velocity (cm/s) - {video_name}'] = rolling
        mean_velocities[video_name] = np.mean(rolling)
        rolling_results.to_csv(os.path.join(save_dir, f'{video_name}.csv'))
        if ts_plot:
            sns.set(style="whitegrid")  # Set the style
            plt.figure(figsize=(10, 6))  # Set the figure size
            sns.lineplot(data=rolling_results, palette="tab10", linewidth=2.5)
            plt.savefig(os.path.join(save_dir, f'rolling_velocities_{video_name}_{config.datetime}.png'))
            plt.close('all')
    mean_velocities = pd.DataFrame.from_dict(mean_velocities, orient='index', columns=['MEAN VELOCITY (CM/S)'])
    mean_velocities.to_csv(os.path.join(config.logs_path, f'mean_velocities_{config.datetime}.csv'))
    timer.stop_timer()
    stdout_success(msg=f'Velocity aggregator for {len(file_paths)} files complete. Data saved in {config.logs_path} directory ', source=velocity_aggregator.__name__)


#DEFINE THE PATH TO YOUR CONFIG, THE DIRECTORY WITH YOUR CSV DATA, AND THE NAME OF YOUR BODY-PART YOU AWANT TO COMPUTE VELOCITY
config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini'
data_dir = '/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location'
body_part = 'Nose_1'

#RUN THE ANALYSIS
velocity_aggregator(config_path=config_path,
                    data_dir=data_dir,
                    body_part=body_part,
                    ts_plot=True)



