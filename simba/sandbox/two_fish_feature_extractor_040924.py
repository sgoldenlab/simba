from __future__ import division

import os.path
from copy import deepcopy
import numpy as np
import pandas as pd
from numba import typed

from simba.data_processors.interpolation_smoothing import Interpolate
from simba.utils.checks import check_if_filepath_list_is_empty, check_all_file_names_are_represented_in_video_log
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import FeatureExtractionSupplemental
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.mixins.circular_statistics import CircularStatisticsMixin
from simba.utils.printing import SimbaTimer, stdout_success
from simba.mixins.statistics_mixin import Statistics
from simba.utils.read_write import get_fn_ext, read_df, read_video_info, write_df
from simba.utils.lookups import cardinality_to_integer_lookup
from simba.feature_extractors.perimeter_jit import jitted_hull


ANIMAL_NAMES = ['Cleaner', 'Client']
MID_BODYPARTS = ['BodyMid_1', 'BodyMid_2']
MOUTH_BODYPARTS = ['HeadTerminalMouth_1', 'HeadTerminalMouth_2']
HEAD_MID = ['HeadMid_1', 'HeadMid_2']
ROLL_WINDOWS_VALUES_S = np.array([10, 5, 2, 1, 0.5, 0.25])

class TwoFishFeatureExtractor(ConfigReader, FeatureExtractionMixin):
    """

    :example:
    >>> feature_extractor = TwoFishFeatureExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_fish/project_folder/project_config.ini')
    >>> feature_extractor.run()
    """

    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=True)
        check_if_filepath_list_is_empty(filepaths=self.outlier_corrected_paths, error_msg=f'No data files found in {self.outlier_corrected_dir} directory')
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.outlier_corrected_paths)
        _ = Interpolate(input_path=self.outlier_corrected_dir, config_path=self.config_path, method='Body-parts: Quadratic')
        print(f"Extracting features from {len(self.outlier_corrected_paths)} file(s)...")

    def run(self):
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            self.video_timer = SimbaTimer(start=True)
            _, self.file_name, _ = get_fn_ext(file_path)
            self.data = read_df(file_path=file_path, file_type=self.file_type)
            self.results = deepcopy(self.data)
            video_info, self.px_per_mm, self.fps = read_video_info(self.video_info_df, self.file_name)
            self.angular_dispersion_windows = []
            for i in ROLL_WINDOWS_VALUES_S: self.angular_dispersion_windows.append(int(self.fps * i))
            self.data.columns = self.bp_headers
            self.save_path = os.path.join(self.features_dir, f'{self.file_name}.{self.file_type}')

            print('Compute X relative to Y movement...')
            for animal_name, body_part_name in zip(ANIMAL_NAMES, MID_BODYPARTS):
                bp_arr = self.data[[f'{animal_name}_{body_part_name}_x', f'{animal_name}_{body_part_name}_y']].values
                d = FeatureExtractionSupplemental.rolling_horizontal_vs_vertical_movement(data=bp_arr, pixels_per_mm=self.px_per_mm, time_windows=ROLL_WINDOWS_VALUES_S, fps=self.fps)
                df = pd.DataFrame(d, columns=[f'X_relative_to_Y_animal_{animal_name}_{x}' for x in ROLL_WINDOWS_VALUES_S])
                self.results = pd.concat([self.results, df], axis=1)

            print('Compute velocity and acceleration...')
            for animal_name, body_part_name in zip(ANIMAL_NAMES, MID_BODYPARTS):
                bp_df = self.data[[f'{animal_name}_{body_part_name}_x', f'{animal_name}_{body_part_name}_y']]
                shifted_ = self.create_shifted_df(df=bp_df)
                self.results[f'framewise_metric_movement_{animal_name}'] = self.framewise_euclidean_distance(location_1=shifted_.values[:, [0, 1]], location_2=shifted_.values[:, [2, 3]], px_per_mm=self.px_per_mm).astype(np.float32)
                velocity = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=self.results[f'framewise_metric_movement_{animal_name}'].values.astype(np.float32), window_sizes=ROLL_WINDOWS_VALUES_S, sample_rate=int(self.fps), statistics=typed.List(['sum'])).astype(np.float32)
                df = pd.DataFrame(velocity[0], columns=[f'sliding_sum_movement_{animal_name}_{x}' for x in ROLL_WINDOWS_VALUES_S])
                self.results = pd.concat([self.results, df], axis=1)
                self.results[f'acceleration_{animal_name}'] = TimeseriesFeatureMixin.acceleration(data=self.results[f'framewise_metric_movement_{animal_name}'].values.astype(np.float32), pixels_per_mm=self.px_per_mm, fps=int(self.fps))
                #SLOW self.results[f'distance_moved_autocorrelation_{animal_name}'] = Statistics.sliding_autocorrelation(data=self.results[f'framewise_metric_movement_{animal_name}'].values.astype(np.float32), max_lag=1.0, time_window=float(self.fps*2), fps=float(self.fps))

            acceleration_correlation = Statistics.sliding_spearman_rank_correlation(sample_1=self.results[f'acceleration_{ANIMAL_NAMES[0]}'].values.astype(np.float32), sample_2=self.results[f'acceleration_{ANIMAL_NAMES[1]}'].values.astype(np.float32), time_windows=ROLL_WINDOWS_VALUES_S, fps=int(self.fps))
            df = pd.DataFrame(acceleration_correlation, columns=[f'animal_acceleration_correlations_{x}' for x in ROLL_WINDOWS_VALUES_S])
            self.results = pd.concat([self.results, df], axis=1)
            movement_autocorrelation = TimeseriesFeatureMixin.sliding_two_signal_crosscorrelation(x=self.results[f'framewise_metric_movement_{ANIMAL_NAMES[0]}'].values.astype(np.float64), y=self.results[f'framewise_metric_movement_{ANIMAL_NAMES[1]}'].values.astype(np.float64), windows=ROLL_WINDOWS_VALUES_S, sample_rate=float(self.fps), normalize=True, lag=self.fps)
            df = pd.DataFrame(movement_autocorrelation, columns=[f'sliding_movement_autocorrelation_{x}' for x in ROLL_WINDOWS_VALUES_S])
            self.results = pd.concat([self.results, df], axis=1)

            print('Compute circular statistics...')
            for (animal, bp1, bp2) in zip(ANIMAL_NAMES, MOUTH_BODYPARTS, HEAD_MID):
                bp_1 = self.data[[f'{animal}_{bp1}_x', f'{animal}_{bp1}_y']].values
                bp_2 = self.data[[f'{animal}_{bp2}_x', f'{animal}_{bp2}_y']].values
                angle = CircularStatisticsMixin.direction_two_bps(anterior_loc=bp_1, posterior_loc=bp_2).astype(np.float32)
                self.results[f'direction_degrees_{animal}'] = angle
                compass_direction = [cardinality_to_integer_lookup()[d] for d in CircularStatisticsMixin.degrees_to_cardinal(data=angle)]
                self.results[f'compass_direction_{animal}'] = compass_direction
                sliding_unique = TimeseriesFeatureMixin.sliding_unique(x=np.array(compass_direction).astype(np.int64), time_windows=ROLL_WINDOWS_VALUES_S, fps=int(self.fps))
                df = pd.DataFrame(sliding_unique, columns=[f'sliding_unique_compass_directions_{animal}_{x}' for x in ROLL_WINDOWS_VALUES_S])
                self.results = pd.concat([self.results, df], axis=1)
                self.results[f'instantaneous_angular_velocity_{animal}'] = CircularStatisticsMixin.instantaneous_angular_velocity(data=angle, bin_size=1)
                self.results[f'instantaneous_rotational_direction_{animal}'] = CircularStatisticsMixin.rotational_direction(data=angle, stride=1)
            sliding_circular_correlation = CircularStatisticsMixin.sliding_circular_correlation(sample_1=self.results[f'direction_degrees_{ANIMAL_NAMES[0]}'].values.astype(np.float32), sample_2=self.results[f'direction_degrees_{ANIMAL_NAMES[1]}'].values.astype(np.float32), time_windows=ROLL_WINDOWS_VALUES_S, fps=float(self.fps))
            df = pd.DataFrame(sliding_circular_correlation, columns=[f'sliding_circular_correlation_{x}' for x in ROLL_WINDOWS_VALUES_S])
            self.results = pd.concat([self.results, df], axis=1)
            for s in ROLL_WINDOWS_VALUES_S:
                window = int(self.fps * s)
                for animal in ANIMAL_NAMES:
                    c = f"animal_{animal}_instantaneous_rotational_direction_rolling_{window}_mean"
                    self.results[c] = self.results[f'instantaneous_rotational_direction_{animal}'].rolling(window, min_periods=1).median()
                    c = f"animal_{animal}_instantaneous_angular_velocity_{window}_mean"
                    self.results[c] = self.results[f'instantaneous_angular_velocity_{animal}'].rolling(window, min_periods=1).median()

            print('Computing pose probability scores...')
            print('Computing animal sizes...')
            for animal_name, animal_bps in self.animal_bp_dict.items():
                x_y = [v for p in zip(animal_bps['X_bps'], animal_bps['Y_bps']) for v in p]
                p = self.data[animal_bps['P_bps']].values
                ranges = FeatureExtractionMixin.count_values_in_range(data=p, ranges=np.array([[0.0, 0.1],[0.000000000, 0.5],[0.000000000, 0.75],[0.000000000, 0.95],[0.000000000, 0.99]]))
                df = pd.DataFrame(ranges, columns=[f'pose_probability_score_{animal_name}_below_{x}' for x in [0.1, 0.5, 0.75, 0.95, 0.99]])
                self.results = pd.concat([self.results, df], axis=1)
                animal_data = self.data[x_y].values.astype(np.float32).reshape(-1, len(animal_bps['Y_bps']), 2)
                self.results[f'animal_{animal_name}_area'] = jitted_hull(points=animal_data, target='area') / self.px_per_mm
            for s in ROLL_WINDOWS_VALUES_S:
                window = int(self.fps * s)
                for animal in ANIMAL_NAMES:
                    c = f"animal_{animal}_area_median_rolling_{window}"
                    self.results[c] = self.results[f'animal_{animal}_area'].rolling(window, min_periods=1).median()

            print('Computing animal distances...')
            distances = pd.DataFrame()
            animal_1_bps = [v for p in zip(self.animal_bp_dict[ANIMAL_NAMES[0]]['X_bps'], self.animal_bp_dict[ANIMAL_NAMES[0]]['Y_bps']) for v in p]
            animal_2_bps = [v for p in zip(self.animal_bp_dict[ANIMAL_NAMES[1]]['X_bps'], self.animal_bp_dict[ANIMAL_NAMES[1]]['Y_bps']) for v in p]
            for i in range(1, len(animal_1_bps), 2):
                bp_1_name = animal_1_bps[i][:-2]
                animal_1_data = self.results[animal_1_bps[i-1:i+1]].values
                for j in range(1, len(animal_2_bps), 2):
                    bp_2_name = animal_2_bps[i][:-2]
                    animal_2_data = self.results[animal_2_bps[j - 1:j + 1]].values
                    animal_dist = self.framewise_euclidean_distance(location_1=animal_1_data, location_2=animal_2_data, px_per_mm=self.px_per_mm).astype(np.float32)
                    distances[f'{bp_1_name}_{bp_2_name}'] = animal_dist
            self.results['animal_minimum_body_part_distance'] = distances.min(axis=1)
            self.results['animal_maximum_body_part_distance'] = distances.max(axis=1)
            self.results['animal_variance_body_part_distance'] = distances.var(axis=1)
            self.results['animal_median_body_part_distance'] = distances.median(axis=1)
            for s in ROLL_WINDOWS_VALUES_S:
                window = int(self.fps * s)
                c = f"animal_minimum_body_part_distance_rolling_{window}"
                self.results[c] = self.results["animal_minimum_body_part_distance"].rolling(window, min_periods=1).min()
                c = f"animal_median_body_part_distance_rolling_{window}"
                self.results[c] = self.results["animal_median_body_part_distance"].rolling(window, min_periods=1).median()
            self.save()

        self.timer.stop_timer()
        stdout_success(msg=f'Feature extraction complete for {len(self.outlier_corrected_paths)} files.', elapsed_time=self.timer.elapsed_time_str)

    def save(self):
        _ = write_df(df=self.results.fillna(-1.0), file_type=self.file_type, save_path=self.save_path)
        self.video_timer.stop_timer()
        print(f'Feature extraction complete video {self.file_name} (elapsed time: {self.video_timer.elapsed_time_str})')

# feature_extractor = TwoFishFeatureExtractor(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_fish/project_folder/project_config.ini')
# feature_extractor.run()
