import argparse
import itertools
import os
import sys
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from numba.typed import List

from simba.feature_extractors.perimeter_jit import get_hull_sizes
from simba.mixins.abstract_classes import AbstractFeatureExtraction
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.statistics_mixin import Statistics
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_if_dir_exists)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    read_df, write_df)

BPS = ['NOSE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SIDE', 'CENTER', 'RIGHT_SIDE', 'TAIL_BASE']
ANIMAL_NAMES = ['resident', 'intruder']
TIME_WINDOWS = np.array([0.1, 0.25, 0.5, 1.0])

class AgressionFeatureExtractor(ConfigReader, AbstractFeatureExtraction):
    """
    Extracts behavioral features from pose estimation data for aggression analysis between two animals (resident and intruder).

    .. note::
       Custom feature extraction class inhereting from ``AbstractFeatureExtraction`` which is not used by default in any of the standard SimBA entry-points.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] data_dir: Directory containing input CSV files with pose estimation data. If None, uses the project's outlier_corrected_dir from the config.
    :param Union[str, os.PathLike] save_dir: Directory where featurized CSV files will be saved. If None, uses the project's features_dir from the config.

    :example I:
    >>> extractor = AgressionFeatureExtractor(config_path='MyProjectConfig')
    >>> extractor.run()

    :example II:
    >>> f = AgressionFeatureExtractor(config_path=r"E:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
    >>> f.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Union[str, os.PathLike] = None,
                 save_dir: Union[str, os.PathLike] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        data_dir = self.outlier_corrected_dir if data_dir is None else data_dir
        check_if_dir_exists(in_dir=data_dir, raise_error=True)
        self.save_dir = self.features_dir if save_dir is None else save_dir
        check_if_dir_exists(in_dir=self.save_dir, raise_error=True)
        self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions='.csv', raise_error=True, as_dict=True)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=list(self.data_paths.values()))
        for video_cnt, (video_name, data_path) in enumerate(self.data_paths.items()):
            video_timer = SimbaTimer(start=True)
            print(f'Analysing FEATURES video {video_name} ({video_cnt+1}/{len(list(self.data_paths.keys()))})....')
            df = read_df(file_path=data_path, file_type='csv')
            df_shifted = df.shift(periods=1).combine_first(df)
            self.save_path = os.path.join(self.save_dir, f'{video_name}.csv')
            self.results = deepcopy(df)
            _, px_per_mm, fps = self.read_video_info(video_name=video_name)

            # INDIVIDUAL ANIMAL FEATURES
            framewise_movement_cols = {ANIMAL_NAMES[0]: [], ANIMAL_NAMES[1]: []}
            for animal_name, animal_bps in self.animal_bp_dict.items():
                for bp in animal_bps['X_bps']:
                    bp_name = bp[:-2]
                    bp_arr = df[[f'{bp_name}_x', f'{bp_name}_y']].values.astype(np.float32)
                    bp_arr_shifted = df_shifted[[f'{bp_name}_x', f'{bp_name}_y']].values.astype(np.float32)
                    x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=bp_arr, window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=List(['sum']))
                    for i in range(x.shape[0]):
                        v = pd.DataFrame(x[i], columns=[f'MOVEMENT_SUM_{animal_name}_{bp_name}_100', f'MOVEMENT_SUM_{animal_name}_{bp_name}_250', f'MOVEMENT_SUM_{animal_name}_{bp_name}_500', f'MOVEMENT_SUM_{animal_name}_{bp_name}_1000'])
                        self.results = pd.concat([self.results, v], axis=1)
                    out_col_name = f'MOVEMENT_FRAMEWISE_{bp_name}'
                    framewise_movement_cols[animal_name].append(out_col_name)
                    self.results[out_col_name] = FeatureExtractionMixin.bodypart_distance(bp1_coords=bp_arr, bp2_coords=bp_arr_shifted, px_per_mm=np.float64(px_per_mm), in_centimeters=False).astype(np.int32)


                bps_x , bps_y = [f'{bp[:-2]}_x' for bp in animal_bps['X_bps']], [f'{bp[:-2]}_y' for bp in animal_bps['X_bps']]
                animal_arr = df[[x for pair in zip(bps_x, bps_y) for x in pair]].values.reshape(len(df), -1, 2)
                self.results[f'HULL_SIZE_{animal_name}'] = get_hull_sizes(points=animal_arr, target='perimeter', pixels_per_mm=px_per_mm)
                x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=self.results[f'HULL_SIZE_{animal_name}'].values.astype(np.float32), window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=List(['mean']))
                for i in range(x.shape[0]):
                    v = pd.DataFrame(x[i], columns=[f'HULL_SIZE_MEAN_{animal_name}_100', f'HULL_SIZE_MEAN_{animal_name}_250', f'HULL_SIZE_MEAN_{animal_name}_500', f'HULL_SIZE_MEAN_{animal_name}_1000'])
                    self.results = pd.concat([self.results, v], axis=1)
                x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=self.results[f'HULL_SIZE_{animal_name}'].values.astype(np.float32), window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=List(['var']))
                for i in range(x.shape[0]):
                    v = pd.DataFrame(x[i], columns=[f'HULL_SIZE_VARIANCE_{animal_name}_100', f'HULL_SIZE_VARIANCE_{animal_name}_250', f'HULL_SIZE_VARIANCE_{animal_name}_500', f'HULL_SIZE_VARIANCE_{animal_name}_1000'])
                    self.results = pd.concat([self.results, v], axis=1)

            pairs = list(itertools.product(framewise_movement_cols[ANIMAL_NAMES[0]], framewise_movement_cols[ANIMAL_NAMES[1]]))
            correlation_cols = []
            for col_1, col_2 in pairs:
                x, y = self.results[col_1].values.astype(np.float32), self.results[col_2].values.astype(np.float32)
                corr = np.abs(Statistics.sliding_pearsons_r(sample_1=x, sample_2=y, time_windows=TIME_WINDOWS, fps=int(fps)))
                v = pd.DataFrame(corr, columns=[f'{col_1}_{col_2}_correlation_250', f'{col_1}_{col_2}_correlation_500', f'{col_1}_{col_2}_correlation_1000', f'{col_1}_{col_2}_correlation_2000'])
                correlation_cols.extend((list(v.columns)))
                self.results = pd.concat([self.results, v], axis=1)
            self.results['MIN_ANIMAL_BP_MOVEMENT_CORRELATION'] = self.results[correlation_cols].min(axis=1).astype(np.int32)
            self.results['MAX_ANIMAL_BP_MOVEMENT_CORRELATION'] = self.results[correlation_cols].max(axis=1).astype(np.int32)
            self.results['STD_ANIMAL_BP_MOVEMENT_CORRELATION'] = self.results[correlation_cols].std(axis=1).astype(np.int32)

            # ANIMAL DISTANCES
            distance_cols = []
            bp_distance_pairs = list(itertools.product([(ANIMAL_NAMES[0], bp) for bp in BPS],  [(ANIMAL_NAMES[1], bp) for bp in BPS]))
            for bp_1, bp_2 in bp_distance_pairs:
                bp_1_col_name, bp_2_col_name = f'{bp_1[0]}_{bp_1[1]}', f'{bp_2[0]}_{bp_2[1]}'
                bp_1_data = df[[f'{bp_1_col_name}_x', f'{bp_1_col_name}_y']].values
                bp_2_data = df[[f'{bp_2_col_name}_x', f'{bp_2_col_name}_y']].values
                out_col = f'DISTANCE_{bp_1_col_name}_{bp_2_col_name}'
                self.results[out_col] = FeatureExtractionMixin.keypoint_distances(a=bp_1_data, b=bp_2_data, px_per_mm=px_per_mm, in_centimeters=False).astype(np.int32)
                distance_cols.append(out_col)
            self.results['MIN_ANIMAL_BP_DISTANCES'] = self.results[distance_cols].min(axis=1).astype(np.int32)
            self.results['MAX_ANIMAL_BP_DISTANCES'] = self.results[distance_cols].max(axis=1).astype(np.int32)
            self.results['STD_ANIMAL_BP_DISTANCES'] = self.results[distance_cols].std(axis=1).astype(np.int32)

            #INCLUDE POSE PROBABILITY SCORES!!!


            bp_data = pd.DataFrame()
            for animal_name, animal_bps in self.animal_bp_dict.items():
                bps_p = [f'{bp[:-2]}_p' for bp in animal_bps['X_bps']]
                animal_p_data = df[bps_p].astype(np.float32)
                bp_data = pd.concat([bp_data, animal_p_data], axis=1)
                self.results[f'{animal_name}_FRAME_MEAN_BP_CONFIDENCE'] = np.mean(animal_p_data, axis=1)
                bp_prob_bins = FeatureExtractionMixin.count_values_in_range(data=animal_p_data.values, ranges=np.array([[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]]))
                p_df = pd.DataFrame(data=bp_prob_bins, columns=[f'{animal_name}_0_0.25_BP_PROB_COUNT', f'{animal_name}_0.25_0.5_BP_PROB_COUNT', f'{animal_name}_0.25_1.0_BP_PROB_COUNT'])
                self.results = pd.concat([self.results, p_df], axis=1)
                x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=self.results[f'{animal_name}_FRAME_MEAN_BP_CONFIDENCE'].values.astype(np.float32), window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=List(['mean']))
                for i in range(x.shape[0]):
                    v = pd.DataFrame(x[i], columns=[f'BP_CONFIDENCE_MEAN_{animal_name}_100', f'BP_CONFIDENCE_MEAN_{animal_name}_250', f'BP_CONFIDENCE_MEAN_{animal_name}_500', f'BP_CONFIDENCE_MEAN_{animal_name}_1000'])
                    self.results = pd.concat([self.results, v], axis=1)
                x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=self.results[f'{animal_name}_FRAME_MEAN_BP_CONFIDENCE'].values.astype(np.float32), window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=List(['var']))
                for i in range(x.shape[0]):
                    v = pd.DataFrame(x[i], columns=[f'BP_CONFIDENCE_VARIANCE_{animal_name}_100', f'BP_CONFIDENCE_VARIANCE_{animal_name}_250', f'BP_CONFIDENCE_VARIANCE_{animal_name}_500', f'BP_CONFIDENCE_VARIANCE_{animal_name}_1000'])
                    self.results = pd.concat([self.results, v], axis=1)
            self.results[f'ANIMALS_FRAME_MEAN_BP_CONFIDENCE'] = np.mean(bp_data, axis=1)
            x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=self.results[f'ANIMALS_FRAME_MEAN_BP_CONFIDENCE'].values.astype(np.float32), window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=List(['mean']))
            for i in range(x.shape[0]):
                v = pd.DataFrame(x[i], columns=['ANIMALS_FRAME_MEAN_BP_CONFIDENCE_MEAN_100', 'ANIMALS_FRAME_MEAN_BP_CONFIDENCE_MEAN_250', f'ANIMALS_FRAME_MEAN_BP_CONFIDENCE_MEAN_500', f'ANIMALS_FRAME_MEAN_BP_CONFIDENCE_MEAN_1000'])
                self.results = pd.concat([self.results, v], axis=1)
            x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=self.results[f'ANIMALS_FRAME_MEAN_BP_CONFIDENCE'].values.astype(np.float32), window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=List(['var']))
            for i in range(x.shape[0]):
                v = pd.DataFrame(x[i], columns=['ANIMALS_FRAME_MEAN_BP_CONFIDENCE_VARIANCE_100', 'ANIMALS_FRAME_MEAN_BP_CONFIDENCE_VARIANCE_250', f'ANIMALS_FRAME_MEAN_BP_CONFIDENCE_VARIANCE_500', f'ANIMALS_FRAME_MEAN_BP_CONFIDENCE_VARIANCE_1000'])
                self.results = pd.concat([self.results, v], axis=1)
            self.save(data=self.results, save_path=self.save_path)
            video_timer.stop_timer()
            print(f'Video complete {self.save_path} ({video_cnt+1}/{len(list(self.data_paths.keys()))}) (elapsed time: {video_timer.elapsed_time_str}s)....')

        self.timer.stop_timer()
        stdout_success(msg=f'Featurized data for {len(list(self.data_paths.keys()))} data file(s) saved in {self.save_dir}', elapsed_time=self.timer.elapsed_time_str)

    def save(self, data: pd.DataFrame, save_path: Union[str, os.PathLike]):
        write_df(df=self.results, file_type='csv', save_path=self.save_path)


if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Extract features for aggression classification in resident intruder setups.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to SimBA Project config.')
    args = parser.parse_args()
    runner = AgressionFeatureExtractor(config_path=args.config_path)
    runner.run()





# f = AgressionFeatureExtractor(config_path=r"E:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
# f.run()






