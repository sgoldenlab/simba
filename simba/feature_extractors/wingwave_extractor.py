import argparse
import os
import sys
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd

from simba.feature_extractors.perimeter_jit import get_hull_sizes
from simba.mixins.abstract_classes import AbstractFeatureExtraction
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.statistics_mixin import Statistics
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_instance,
    check_valid_array, check_valid_dataframe)
from simba.utils.enums import Formats
from simba.utils.printing import stdout_information, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    read_df, read_video_info, write_df)

WING_LEFT, WING_RIGHT = 'wing_L', 'wing_R'
HEAD, THORAX, ABDOMEN = 'head', 'thorax', 'abdomen'
WINDOW_SIZES = np.array([0.5, 1.0, 2.0, 3.0, 6.0])

class WingWaveFeatureExtractor(ConfigReader, FeatureExtractionMixin, AbstractFeatureExtraction):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=True)
        self.data_dir = self.outlier_corrected_dir if data_dir is None else data_dir
        self.data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions='.csv', as_dict=True, raise_error=True, sort_alphabetically=True)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=list(self.data_paths.values()))
        for file_cnt, (file_name, file_path) in enumerate(self.data_paths.items()):
            data_df = read_df(file_path=file_path, file_type='csv', verbose=False)
            out_df = deepcopy(data_df)
            _, px_per_mm, fps = read_video_info(video_info_df=self.video_info_df, video_name=file_name)
            save_path = os.path.join(self.features_dir, f'{file_name}.csv')
            stdout_information(msg=f'Analyzing {file_name}...')
            for id in range(1, self.animal_cnt + 1):
                print(id)
                animal_df = data_df[[x for x in data_df.columns if f'_{id}_' in x]]
                wing_l, wing_r = animal_df[[f'{WING_LEFT}_{id}_x', f'{WING_LEFT}_{id}_y']].values, animal_df[[f'{WING_RIGHT}_{id}_x', f'{WING_RIGHT}_{id}_y']].values
                head, thorax = animal_df[[f'{HEAD}_{id}_x', f'{HEAD}_{id}_y']].values, animal_df[[f'{THORAX}_{id}_x', f'{THORAX}_{id}_y']].values
                abdomen = animal_df[[f'{ABDOMEN}_{id}_x', f'{ABDOMEN}_{id}_y']].values

                p_vals = animal_df[[f'{WING_LEFT}_{id}_p', f'{WING_RIGHT}_{id}_p', f'{HEAD}_{id}_p', f'{ABDOMEN}_{id}_x', f'{THORAX}_{id}_x']].values

                out_df[f'wing_l_thorax_wing_r_angle_animal_{id}'] = FeatureExtractionMixin.three_point_angle(bp_1=wing_l, bp_2=thorax, bp_3=wing_r)
                out_df[f'wing_l_head_wing_r_angle_animal_{id}'] = FeatureExtractionMixin.three_point_angle(bp_1=wing_l, bp_2=head, bp_3=wing_r)
                out_df[f'wing_l_abdomen_wing_r_angle_animal_{id}'] = FeatureExtractionMixin.three_point_angle(bp_1=wing_l, bp_2=head, bp_3=wing_r)
                wing_l_thorax_wing_r = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'wing_l_thorax_wing_r_angle_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['mean'])[0], columns=[f'wing_l_thorax_wing_r_mean_500_animal_{id}', f'wing_l_thorax_wing_r_mean_1000_animal_{id}', f'wing_l_thorax_wing_r_mean_2000_animal_{id}', f'wing_l_thorax_wing_r_mean_3000_animal_{id}', f'wing_l_thorax_wing_r_mean_6000_animal_{id}'])
                wing_l_head_wing_r = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'wing_l_head_wing_r_angle_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['mean'])[0], columns=[f'wing_l_head_wing_r_mean_500_animal_{id}', f'wing_l_head_wing_r_mean_1000_animal_{id}', f'wing_l_head_wing_r_mean_2000_animal_{id}', f'wing_l_head_wing_r_mean_3000_animal_{id}', f'wing_l_head_wing_r_mean_6000_animal_{id}'])
                wing_l_abdomen_wing_r = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'wing_l_head_wing_r_angle_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['mean'])[0], columns=[f'wing_l_abdomen_wing_r_mean_500_animal_{id}', f'wing_l_abdomen_wing_r_mean_1000_animal_{id}', f'wing_l_abdomen_wing_r_mean_2000_animal_{id}', f'wing_l_abdomen_wing_r_mean_3000_animal_{id}', f'wing_l_abdomen_wing_r_mean_6000_animal_{id}'])
                out_df = pd.concat([out_df, wing_l_thorax_wing_r, wing_l_head_wing_r, wing_l_abdomen_wing_r], axis=1)

                out_df[f'framewise_wing_distance_animal_{id}'] = FeatureExtractionMixin.bodypart_distance(bp1_coords=wing_l, bp2_coords=wing_r, px_per_mm=px_per_mm, in_centimeters=False)
                wing_dist_absenergy = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_distance_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['absenergy'])[0], columns=[f'wing_distance_energy_500_animal_{id}', f'wing_distance_energy_1000_animal_{id}', f'wing_distance_energy_2000_animal_{id}', f'wing_distance_energy_3000_animal_{id}', f'wing_distance_energy_6000_animal_{id}'])
                wing_dist_variance = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_distance_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['var'])[0], columns=[f'wing_distance_var_500_animal_{id}', f'wing_distance_var_1000_animal_{id}', f'wing_distance_var_2000_animal_{id}', f'wing_distance_var_3000_animal_{id}', f'wing_distance_var_6000_animal_{id}'])
                out_df = pd.concat([out_df, wing_dist_absenergy, wing_dist_variance], axis=1)

                out_df[f'framewise_wing_l_head_distance_animal_{id}'] =  FeatureExtractionMixin.framewise_euclidean_distance(location_1=wing_l.astype(np.float64), location_2=head.astype(np.float64), px_per_mm=np.float64(px_per_mm), centimeter=False)
                out_df[f'framewise_wing_r_head_distance_animal_{id}'] =  FeatureExtractionMixin.framewise_euclidean_distance(location_1=wing_r.astype(np.float64), location_2=head.astype(np.float64), px_per_mm=np.float64(px_per_mm), centimeter=False)
                out_df[f'framewise_wing_l_abdomen_distance_animal_{id}'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=wing_l.astype(np.float64), location_2=abdomen.astype(np.float64),px_per_mm=np.float64(px_per_mm), centimeter=False)
                out_df[f'framewise_wing_r_abdomen_distance_animal_{id}'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=wing_r.astype(np.float64), location_2=abdomen.astype(np.float64),px_per_mm=np.float64(px_per_mm), centimeter=False)
                out_df[f'framewise_wing_l_thorax_distance_animal_{id}'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=wing_l.astype(np.float64), location_2=thorax.astype(np.float64),px_per_mm=np.float64(px_per_mm), centimeter=False)
                out_df[f'framewise_wing_r_thorax_distance_animal_{id}'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=wing_r.astype(np.float64), location_2=thorax.astype(np.float64),px_per_mm=np.float64(px_per_mm), centimeter=False)

                wing_l_head_distance_mean = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_l_head_distance_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['mean'])[0], columns=[f'wing_l_head_mean_500_animal_{id}', f'wing_l_head_mean_1000_animal_{id}', f'wing_l_head_mean_2000_animal_{id}', f'wing_l_head_mean_3000_animal_{id}', f'wing_l_head_mean_6000_animal_{id}'])
                wing_r_head_distance_mean = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_r_head_distance_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['mean'])[0], columns=[f'wing_r_head_mean_500_animal_{id}', f'wing_r_head_mean_1000_animal_{id}', f'wing_r_head_mean_2000_animal_{id}', f'wing_r_head_mean_3000_animal_{id}', f'wing_r_head_mean_6000_animal_{id}'])
                wing_r_head_distance_std = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_r_head_distance_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['std'])[0], columns=[f'wing_r_head_std_500_animal_{id}', f'wing_r_head_std_1000_animal_{id}', f'wing_r_head_std_2000_animal_{id}', f'wing_r_head_std_3000_animal_{id}', f'wing_r_head_std_6000_animal_{id}'])
                wing_l_head_distance_std = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_l_head_distance_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['std'])[0], columns=[f'wing_l_head_std_500_animal_{id}', f'wing_l_head_std_1000_animal_{id}', f'wing_l_head_std_2000_animal_{id}', f'wing_lr_head_std_3000_animal_{id}', f'wing_l_head_std_6000_animal_{id}'])
                out_df = pd.concat([out_df, wing_l_head_distance_mean, wing_r_head_distance_mean, wing_r_head_distance_std, wing_l_head_distance_std], axis=1)

                out_df[f'framewise_wing_l_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=wing_l, px_per_mm=px_per_mm, centimeter=False)
                out_df[f'framewise_wing_r_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=wing_r, px_per_mm=px_per_mm, centimeter=False)
                out_df[f'framewise_head_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=head, px_per_mm=px_per_mm, centimeter=False)
                out_df[f'framewise_thorax_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=thorax, px_per_mm=px_per_mm, centimeter=False)
                out_df[f'framewise_abdomen_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=abdomen, px_per_mm=px_per_mm, centimeter=False)
                wing_l_sum = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_l_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'wing_l_sum_500_animal_{id}', f'wing_l_sum_1000_animal_{id}', f'wing_l_sum_2000_animal_{id}', f'wing_l_sum_3000_animal_{id}', f'wing_l_sum_6000_animal_{id}'])
                wing_r_sum = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_r_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'wing_r_sum_500_animal_{id}', f'wing_r_sum_1000_animal_{id}', f'wing_r_sum_2000_animal_{id}', f'wing_r_sum_3000_animal_{id}', f'wing_r_sum_6000_animal_{id}'])
                head_sum = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_head_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'head_sum_500_animal_{id}', f'head_sum_1000_animal_{id}', f'head_sum_2000_animal_{id}', f'head_sum_3000_animal_{id}', f'head_sum_6000_animal_{id}'])
                thorax_sum = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_thorax_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'thorax_sum_500_animal_{id}', f'thorax_sum_1000_animal_{id}', f'thorax_sum_2000_animal_{id}', f'thorax_sum_3000_animal_{id}', f'thorax_sum_6000_animal_{id}'])
                abdomen_sum = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_abdomen_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'abdomen_sum_500_animal_{id}', f'abdomen_sum_1000_animal_{id}', f'abdomen_sum_2000_animal_{id}', f'abdomen_sum_3000_animal_{id}', f'abdomen_sum_6000_animal_{id}'])
                out_df = pd.concat([out_df, wing_l_sum, wing_r_sum, head_sum, thorax_sum, abdomen_sum], axis=1)

                head_wing_l_pearson = pd.DataFrame(
                    Statistics.sliding_pearsons_r(
                        sample_1=out_df[f'framewise_head_movement_animal_{id}'].values.astype(np.float32),
                        sample_2=out_df[f'framewise_wing_l_movement_animal_{id}'].values.astype(np.float32),
                        time_windows=WINDOW_SIZES,
                        fps=fps,
                    ),
                    columns=[
                        f'head_wing_l_movement_pearson_500_animal_{id}',
                        f'head_wing_l_movement_pearson_1000_animal_{id}',
                        f'head_wing_l_movement_pearson_2000_animal_{id}',
                        f'head_wing_l_movement_pearson_3000_animal_{id}',
                        f'head_wing_l_movement_pearson_6000_animal_{id}',
                    ],
                )
                head_wing_r_pearson = pd.DataFrame(
                    Statistics.sliding_pearsons_r(
                        sample_1=out_df[f'framewise_head_movement_animal_{id}'].values.astype(np.float32),
                        sample_2=out_df[f'framewise_wing_r_movement_animal_{id}'].values.astype(np.float32),
                        time_windows=WINDOW_SIZES,
                        fps=fps,
                    ),
                    columns=[
                        f'head_wing_r_movement_pearson_500_animal_{id}',
                        f'head_wing_r_movement_pearson_1000_animal_{id}',
                        f'head_wing_r_movement_pearson_2000_animal_{id}',
                        f'head_wing_r_movement_pearson_3000_animal_{id}',
                        f'head_wing_r_movement_pearson_6000_animal_{id}',
                    ],
                )
                out_df = pd.concat([out_df, head_wing_l_pearson, head_wing_r_pearson], axis=1)

                out_df[f'framewise_head_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=head, px_per_mm=px_per_mm, centimeter=False)
                out_df[f'framewise_wing_l_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=wing_l, px_per_mm=px_per_mm, centimeter=False)
                out_df[f'framewise_wing_r_movement_animal_{id}'] = FeatureExtractionMixin.framewise_bodypart_movement(data=wing_r, px_per_mm=px_per_mm, centimeter=False)
                out_df[f'framewise_wing_l_abdomen_distance_animal_{id}'] = FeatureExtractionMixin.bodypart_distance(bp1_coords=wing_l, bp2_coords=abdomen, px_per_mm=px_per_mm, in_centimeters=False)
                out_df[f'framewise_wing_r_abdomen_distance_animal_{id}'] = FeatureExtractionMixin.bodypart_distance(bp1_coords=wing_r, bp2_coords=abdomen, px_per_mm=px_per_mm, in_centimeters=False)

                wing_l_head_wing_r_angle_var = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'wing_l_head_wing_r_angle_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['var'])[0], columns=[f'wing_head_angle_var_125_animal_{id}', f'wing_head_angle_var_250_animal_{id}', f'wing_head_angle_var_500_animal_{id}', f'wing_head_angle_var_1000_animal_{id}', f'wing_head_angle_var_2000_animal_{id}'])
                wing_l_head_wing_r_angle_energy = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'wing_l_head_wing_r_angle_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['absenergy'])[0], columns=[f'wing_head_angle_energy_125_animal_{id}', f'wing_head_angle_energy_250_animal_{id}', f'wing_head_angle_energy_500_animal_{id}', f'wing_head_angle_energy_1000_animal_{id}', f'wing_head_angle_energy_2000_animal_{id}'])
                out_df = pd.concat([out_df, wing_l_head_wing_r_angle_var, wing_l_head_wing_r_angle_energy], axis=1)


                head_movement = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_head_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'head_movement_sum_125_animal_{id}',f'head_movement_sum_250_animal_{id}', f'head_movement_sum_500_animal_{id}', f'head_movement_sum_1000_animal_{id}', f'head_movement_sum_2000_animal_{id}'])
                wing_r_movement = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_r_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'wing_r_movement_sum_125_animal_{id}', f'wing_r_movement_sum_250_animal_{id}', f'wing_r_movement_sum_500_animal_{id}', f'wing_r_movement_sum_1000_animal_animal_{id}', f'wing_r_movement_sum_2000_animal_animal_{id}'])
                wing_l_movement = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_wing_l_movement_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['sum'])[0], columns=[f'wing_l_movement_sum_125_animal_{id}', f'wing_l_movement_sum_250_animal_{id}', f'wing_l_movement_sum_500_animal_{id}', f'wing_l_movement_sum_1000_animal_animal_{id}', f'wing_l_movement_sum_2000_animal_animal_{id}'])
                out_df = pd.concat([out_df, head_movement, wing_l_movement, wing_r_movement], axis=1)

                animal_3d = np.stack([wing_l, wing_r, head, thorax, abdomen], axis=1)
                out_df[f'framewise_animal_size_animal_{id}'] = get_hull_sizes(points=animal_3d, pixels_per_mm=px_per_mm)
                size_std = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_animal_size_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['std'])[0], columns=[f'size_std_500_animal_{id}', f'size_std_1000_animal_{id}', f'size_std_2000_animal_{id}', f'size_std_3000_animal_{id}', f'size_std_6000_animal_{id}'])
                size_mean = pd.DataFrame(TimeseriesFeatureMixin.sliding_window_stats(data=out_df[f'framewise_animal_size_animal_{id}'].values, window_sizes=WINDOW_SIZES, sample_rate=fps, statistics=['mean'])[0], columns=[f'size_mean_500_animal_{id}', f'size_mean_1000_animal_{id}', f'size_mean_2000_animal_{id}', f'size_mean_3000_animal_{id}', f'size_mean_6000_animal_{id}'])
                out_df = pd.concat([out_df, size_std, size_mean], axis=1)

                p_val_cnts = pd.DataFrame(FeatureExtractionMixin.count_values_in_range(data=p_vals, ranges=np.array([[0.0, 0.25], [0.25, 0.5], [0.5, 0.75]])), columns=[f'p_vals_0.25_animal_{id}', f'p_vals_0.50_animal_{id}', f'p_vals_0.75_animal_{id}'])
                out_df = pd.concat([out_df, p_val_cnts], axis=1)

                out_df[f'rolling_t_sizes_500_animal_{id}'] = Statistics.rolling_independent_sample_t(data=out_df[f'framewise_animal_size_animal_{id}'].values.astype(np.float32), time_window=0.5, fps=fps)
                out_df[f'rolling_t_wing_l_movement_500_animal_{id}'] = Statistics.rolling_independent_sample_t(data=out_df[f'framewise_wing_l_movement_animal_{id}'].values.astype(np.float32), time_window=0.5, fps=fps)
                out_df[f'rolling_t_wing_r_movement_500_animal_{id}'] = Statistics.rolling_independent_sample_t(data=out_df[f'framewise_wing_r_movement_animal_{id}'].values.astype(np.float32), time_window=0.5, fps=fps)
                out_df[f'rolling_t_thorax_movement_500_animal_{id}'] = Statistics.rolling_independent_sample_t(data=out_df[f'framewise_thorax_movement_animal_{id}'].values.astype(np.float32), time_window=0.5, fps=fps)

                out_df[f'wing_l_movement_autocorreleation_1000_animal_{id}'] = Statistics.sliding_autocorrelation(data=out_df[f'framewise_wing_l_movement_animal_{id}'].values.astype(np.float32), max_lag=4.0, time_window=1.0, fps=fps)
                out_df[f'wing_r_movement_autocorreleation_1000_animal_{id}'] = Statistics.sliding_autocorrelation(data=out_df[f'framewise_wing_r_movement_animal_{id}'].values.astype(np.float32), max_lag=4.0, time_window=1.0, fps=fps)
                out_df[f'head_movement_autocorreleation_1000_animal_{id}'] = Statistics.sliding_autocorrelation(data=out_df[f'framewise_head_movement_animal_{id}'].values.astype(np.float32), max_lag=4.0, time_window=1.0, fps=fps)


                x = pd.DataFrame(Statistics.sliding_dominant_frequencies(data=out_df[f'framewise_wing_l_movement_animal_{id}'].values.astype(np.float32), fps=fps, k=1, time_windows=np.array([1.0, 3.0, 5.0]), window_function='Hamming'), columns=[f'dominant_wing_l_frequency_1000_animal_{id}', f'dominant_wing_l_frequency_3000_animal_{id}', f'dominant_wing_l_frequency_5000_animal_{id}'])
                y = pd.DataFrame(Statistics.sliding_dominant_frequencies(data=out_df[f'framewise_wing_r_movement_animal_{id}'].values.astype(np.float32), fps=fps, k=1, time_windows=np.array([1.0, 3.0, 5.0]), window_function='Hamming'), columns=[f'dominant_wing_r_frequency_1000_animal_{id}', f'dominant_wing_r_frequency_3000_animal_{id}', f'dominant_wing_r_frequency_5000_animal_{id}'])
                z = pd.DataFrame(Statistics.sliding_dominant_frequencies(data=out_df[f'framewise_head_movement_animal_{id}'].values.astype(np.float32), fps=fps, k=1, time_windows=np.array([1.0, 3.0, 5.0]), window_function='Hamming'), columns=[f'dominant_head_frequency_1000_animal_{id}', f'dominant_head_frequency_3000_animal_{id}', f'dominant_head_frequency_5000_animal_{id}'])
                out_df = pd.concat([out_df, x, y, z], axis=1)
            self.save(data=out_df, save_path=save_path)
        stdout_success(f'Feature extraction complete. Data saved in {self.features_dir}.')

    def save(self, data: pd.DataFrame, save_path: Union[str, os.PathLike]):
        write_df(df=data, file_type='csv', save_path=save_path)


if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Extract features for aggression classification in resident intruder setups.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to SimBA Project config.')
    args = parser.parse_args()
    runner = WingWaveFeatureExtractor(config_path=args.config_path)
    runner.run()

#
#
# CONFIG_PATH = r"F:\troubleshooting\sophiaa\project_folder\project_config.ini"
#
#
# r = WingWaveFeatureExtractor(config_path=CONFIG_PATH)
# r.run()
#
#



