import os
import numpy as np
import pandas as pd
from numba.typed import List
from itertools import product
import argparse
from typing import Union

from simba.mixins.abstract_classes import AbstractFeatureExtraction
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.mixins.statistics_mixin import Statistics
from simba.mixins.circular_statistics import CircularStatisticsMixin
from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.utils.checks import check_if_filepath_list_is_empty, check_all_file_names_are_represented_in_video_log
from simba.utils.read_write import read_df, get_fn_ext, read_frm_of_video
from simba.utils.read_write import SimbaTimer, stdout_success, write_df

NOSE = 'nose'
LEFT_SIDE = 'left_side'
RIGHT_SIDE = 'right_side'
LEFT_EAR = 'left_ear'
RIGHT_EAR = 'right_ear'
CENTER = 'center'
TAIL_BASE = 'tail_base'
TAIL_CENTER = 'tail_center'
TAIL_TIP = 'tail_tip'

TIME_WINDOWS = np.array([0.25, 0.5, 1.0, 2.0])

class MitraFeatureExtractor(ConfigReader,
                            AbstractFeatureExtraction):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        check_if_filepath_list_is_empty(filepaths=self.outlier_corrected_paths, error_msg=f'No data files found in {self.outlier_corrected_dir} directory.')
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.outlier_corrected_paths)

    def run(self):
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            df = read_df(file_path=file_path, file_type=self.file_type)
            results = pd.DataFrame()
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            save_path = os.path.join(self.features_dir, video_name + f'.{self.file_type}')
            print(f'Featurizing video {video_name} ...(Video {file_cnt+1}/{len(self.outlier_corrected_paths)})')
            _, px_per_mm, fps = self.read_video_info(video_name=video_name)
            shifted_ =  df.shift(periods=1).combine_first(df)
            nose_arr = df[[f'{NOSE}_x', f'{NOSE}_y']].values.astype(np.float32)
            p_arr = df[self.animal_bp_dict['Animal_1']['P_bps']].values.astype(np.float32)
            tailbase_arr = df[[f'{TAIL_BASE}_x', f'{TAIL_BASE}_y']].values.astype(np.float32)
            left_ear_arr = df[[f'{LEFT_EAR}_x', f'{LEFT_EAR}_y']].values.astype(np.float32)
            right_ear_arr = df[[f'{RIGHT_EAR}_x', f'{RIGHT_EAR}_y']].values.astype(np.float32)
            center_arr = df[[f'{CENTER}_x', f'{CENTER}_y']].values.astype(np.float32)
            lat_left_arr = df[[f'{LEFT_SIDE}_x', f'{LEFT_SIDE}_y']].values.astype(np.float32)
            lat_right_arr = df[[f'{RIGHT_SIDE}_x', f'{RIGHT_SIDE}_y']].values.astype(np.float32)
            tail_center_arr = df[[f'{TAIL_CENTER}_x', f'{TAIL_CENTER}_y']].values.astype(np.float32)
            tail_tip_arr = df[[f'{TAIL_TIP}_x', f'{TAIL_TIP}_y']].values.astype(np.float32)
            animal_hull_arr = df[[f'{LEFT_EAR}_x', f'{LEFT_EAR}_y', f'{RIGHT_EAR}_x', f'{RIGHT_EAR}_y', f'{NOSE}_x', f'{NOSE}_y', f'{LEFT_SIDE}_x', f'{LEFT_SIDE}_y', f'{RIGHT_SIDE}_x', f'{RIGHT_SIDE}_y', f'{TAIL_BASE}_x', f'{TAIL_BASE}_y']].values.astype(np.float32).reshape(len(df), 6, 2)
            animal_head_arr = df[[f'{LEFT_EAR}_x', f'{LEFT_EAR}_y', f'{RIGHT_EAR}_x', f'{RIGHT_EAR}_y', f'{NOSE}_x', f'{NOSE}_y']].values.astype(np.float32).reshape(len(df), 3, 2)
            animal_body_arr = df[[f'{LEFT_EAR}_x', f'{LEFT_EAR}_y', f'{RIGHT_EAR}_x', f'{RIGHT_EAR}_y', f'{LEFT_SIDE}_x', f'{LEFT_SIDE}_y', f'{RIGHT_SIDE}_x', f'{RIGHT_SIDE}_y', f'{TAIL_BASE}_x', f'{TAIL_BASE}_y']].values.astype(np.float32).reshape(len(df), 5, 2)
            animal_lower_body_arr = df[[f'{LEFT_SIDE}_x', f'{LEFT_SIDE}_y', f'{RIGHT_SIDE}_x', f'{RIGHT_SIDE}_y', f'{TAIL_BASE}_x', f'{TAIL_BASE}_y']].values.astype(np.float32).reshape(len(df), 3, 2)
            animal_upper_body_arr = df[[f'{LEFT_EAR}_x', f'{LEFT_EAR}_y', f'{RIGHT_EAR}_x', f'{RIGHT_EAR}_y', f'{NOSE}_x', f'{NOSE}_y', f'{LEFT_SIDE}_x', f'{LEFT_SIDE}_y', f'{RIGHT_SIDE}_x', f'{RIGHT_SIDE}_y']].values.astype(np.float32).reshape(len(df), 5, 2)
            left_body_arr = df[[f'{LEFT_EAR}_x', f'{LEFT_EAR}_y', f'{NOSE}_x', f'{NOSE}_y', f'{LEFT_SIDE}_x', f'{LEFT_SIDE}_y', f'{TAIL_BASE}_x', f'{TAIL_BASE}_y', f'{CENTER}_x', f'{CENTER}_y']].values.astype(np.float32).reshape(len(df), 5, 2)
            right_body_arr = df[[f'{RIGHT_EAR}_x', f'{RIGHT_EAR}_y', f'{NOSE}_x', f'{NOSE}_y', f'{RIGHT_SIDE}_x', f'{RIGHT_SIDE}_y', f'{TAIL_BASE}_x', f'{TAIL_BASE}_y', f'{CENTER}_x', f'{CENTER}_y']].values.astype(np.float32).reshape(len(df), 5, 2)
            direction_degrees = CircularStatisticsMixin().direction_three_bps(nose_loc=nose_arr, left_ear_loc=left_ear_arr, right_ear_loc=right_ear_arr).astype(np.float32)

            # GEOMETRY FEATURES
            print('Compute geometry features...')
            results['GEOMETRY_FRAME_HULL_LENGTH'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=nose_arr, location_2=tailbase_arr, px_per_mm=px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_HULL_WIDTH'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=lat_left_arr, location_2=lat_right_arr, px_per_mm=px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_HULL_AREA'] = (jitted_hull(points=animal_hull_arr, target='area') / px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_BODY_AREA'] = (jitted_hull(points=animal_body_arr, target='area') / px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_LOWER_BODY_AREA'] = (jitted_hull(points=animal_lower_body_arr, target='area') / px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_UPPER_BODY_AREA'] = (jitted_hull(points=animal_upper_body_arr, target='area') / px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_HEAD_AREA'] = (jitted_hull(points=animal_head_arr, target='area') / px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_LEFT_BODY_AREA'] = (jitted_hull(points=left_body_arr, target='area') / px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_RIGHT_BODY_AREA'] = (jitted_hull(points=right_body_arr, target='area') / px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_TAIL_LENGTH'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=tailbase_arr, location_2=tail_tip_arr, px_per_mm=px_per_mm).astype(np.int32)
            results['GEOMETRY_FRAME_EAR_DISTANCE'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=left_ear_arr, location_2=right_ear_arr, px_per_mm=px_per_mm).astype(np.int32)

            for time, feature in product(TIME_WINDOWS, ['HULL_LENGTH', 'HULL_WIDTH', 'HULL_AREA', 'BODY_AREA', 'LOWER_BODY_AREA', 'UPPER_BODY_AREA', 'HEAD_AREA', 'LEFT_BODY_AREA', 'RIGHT_BODY_AREA', 'TAIL_LENGTH', 'EAR_DISTANCE']):
                results[f'GEOMETRY_MEAN_{feature}_{time}'] = results[f'GEOMETRY_FRAME_{feature}'].rolling(int(time * fps), min_periods=1).mean().fillna(0).astype(np.int32)
                results[f'GEOMETRY_VAR_{feature}_{time}'] = results[f'GEOMETRY_FRAME_{feature}'].rolling(int(time * fps), min_periods=1).var().fillna(0).astype(np.float32)
                results[f'GEOMETRY_SUM_{feature}_{time}'] = results[f'GEOMETRY_FRAME_{feature}'].rolling(int(time * fps), min_periods=1).sum().fillna(0).astype(np.int32)

            for feature in ['HULL_LENGTH', 'HULL_WIDTH', 'HULL_AREA', 'BODY_AREA', 'LOWER_BODY_AREA', 'UPPER_BODY_AREA', 'HEAD_AREA', 'LEFT_BODY_AREA', 'RIGHT_BODY_AREA', 'TAIL_LENGTH', 'EAR_DISTANCE']:
                sliding_skew = pd.DataFrame(Statistics.sliding_z_scores(data=results[f'GEOMETRY_FRAME_{feature}'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=int(fps)), columns=[f'GEOMETRY_{feature}_SLIDING_Z_SCORE_250', f'GEOMETRY_{feature}_SLIDING_Z_SCORE_500', f'GEOMETRY_{feature}_SLIDING_Z_SCORE_1000', f'GEOMETRY_{feature}_SLIDING_Z_SCORE_2000'])
                sliding_mad_median = pd.DataFrame(Statistics.sliding_mad_median_rule(data=results[f'GEOMETRY_FRAME_{feature}'].values.astype(np.float32), k=0.5, time_windows=TIME_WINDOWS, fps=fps), columns=[f'GEOMETRY_{feature}_SLIDING_MAD_MEDIAN_0.5_250', f'GEOMETRY_{feature}_SLIDING_MAD_MEDIAN_0.5_500', f'GEOMETRY_{feature}_SLIDING_MAD_MEDIAN_0.5_1000', f'GEOMETRY_{feature}_SLIDING_MAD_MEDIAN_0.5_2000'])
                results = pd.concat([results, sliding_skew, sliding_mad_median], axis=1)

            for feature in ['HULL_LENGTH', 'HULL_WIDTH', 'HULL_AREA', 'BODY_AREA', 'LOWER_BODY_AREA', 'UPPER_BODY_AREA', 'HEAD_AREA', 'LEFT_BODY_AREA', 'RIGHT_BODY_AREA', 'TAIL_LENGTH', 'EAR_DISTANCE']:
                statistics = List(['mac', 'rms'])
                x = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=results[f'GEOMETRY_FRAME_{feature}'].values.astype(np.float32), window_sizes=TIME_WINDOWS, sample_rate=int(fps), statistics=statistics)
                for i in range(x.shape[0]):
                    v = pd.DataFrame(x[i], columns=[f'GEOMETRY_{feature}_{statistics[i]}_250', f'GEOMETRY_{feature}_{statistics[i]}_500', f'GEOMETRY_{feature}_{statistics[i]}_1000', f'GEOMETRY_{feature}_{statistics[i]}_2000'])
                    results = pd.concat([results, v], axis=1)

            upper_lower_body_size_correlations = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['GEOMETRY_FRAME_UPPER_BODY_AREA'].values.astype(np.float32), sample_2=results['GEOMETRY_FRAME_LOWER_BODY_AREA'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['GEOMETRY_UPPER_LOWER_BODY_SIZE_SPEARMAN_250', 'GEOMETRY_UPPER_LOWER_BODY_SIZE_SPEARMAN_500', 'GEOMETRY_UPPER_LOWER_BODY_SIZE_SPEARMAN_1000', 'GEOMETRY_UPPER_LOWER_BODY_SIZE_SPEARMAN_2000']).astype(np.float32)
            hull_head_correlations = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['GEOMETRY_FRAME_HULL_AREA'].values.astype(np.float32), sample_2=results['GEOMETRY_FRAME_HEAD_AREA'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['GEOMETRY_HULL_HEAD_SIZE_SPEARMAN_250', 'GEOMETRY_HULL_HEAD_BODY_SIZE_SPEARMAN_500', 'GEOMETRY_HULL_HEAD_SIZE_SPEARMAN_1000', 'GEOMETRY_HULL_HEAD_SIZE_SPEARMAN_2000']).astype(np.float32)
            hull_tail_length_correlations = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['GEOMETRY_FRAME_HULL_LENGTH'].values.astype(np.float32), sample_2=results['GEOMETRY_FRAME_TAIL_LENGTH'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['GEOMETRY_HULL_TAIL_LENGTH_SPEARMAN_250', 'GEOMETRY_HULL_TAIL_LENGTH_SPEARMAN_500', 'GEOMETRY_HULL_TAIL_LENGTH_SPEARMAN_1000', 'GEOMETRY_HULL_TAIL_LENGTH_SPEARMAN_2000']).astype(np.float32)
            left_body_right_body_correlations = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['GEOMETRY_FRAME_LEFT_BODY_AREA'].values.astype(np.float32), sample_2=results['GEOMETRY_FRAME_RIGHT_BODY_AREA'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['GEOMETRY_LEFT_RIGHT_BODY_SPEARMAN_250', 'GEOMETRY_LEFT_RIGHT_BODY_SPEARMAN_500', 'GEOMETRY_LEFT_RIGHT_BODY_SPEARMAN_1000', 'GEOMETRY_LEFT_RIGHT_BODY_SPEARMAN_2000']).astype(np.float32)
            results = pd.concat([results, upper_lower_body_size_correlations, hull_head_correlations, hull_tail_length_correlations, left_body_right_body_correlations], axis=1)

            # CIRCULAR FEATURES
            print('Compute circular features...')
            results['CIRCULAR_FRAME_HULL_3POINT_ANGLE'] = FeatureExtractionMixin.angle3pt_serialized(data=np.hstack([nose_arr, center_arr, tailbase_arr]))
            results['CIRCULAR_FRAME_TAIL_3POINT_ANGLE'] = FeatureExtractionMixin.angle3pt_serialized(data=np.hstack([tailbase_arr, tail_center_arr, tail_tip_arr]))
            results['CIRCULAR_FRAME_HEAD_3POINT_ANGLE'] = FeatureExtractionMixin.angle3pt_serialized(data=np.hstack([left_ear_arr, nose_arr, right_ear_arr]))
            results['CIRCULAR_INSTANTANEOUS_ANGULAR_VELOCITY'] = CircularStatisticsMixin.instantaneous_angular_velocity(data=direction_degrees, bin_size=1)
            angular_difference = pd.DataFrame(CircularStatisticsMixin.sliding_angular_diff(data=direction_degrees, time_windows=TIME_WINDOWS, fps=int(fps)), columns=['CIRCULAR_HEAD_DIRECTION_ANGULAR_DIFFERENCE_250', 'CIRCULAR_HEAD_DIRECTION_ANGULAR_DIFFERENCE_500', 'CIRCULAR_HEAD_DIRECTION_ANGULAR_DIFFERENCE_1000',  'CIRCULAR_HEAD_DIRECTION_ANGULAR_DIFFERENCE_2000'])
            rao_spacing = pd.DataFrame(CircularStatisticsMixin.sliding_rao_spacing(data=direction_degrees, time_windows=TIME_WINDOWS, fps=int(fps)), columns=['CIRCULAR_HEAD_DIRECTION_RAO_SPACING_250', 'CIRCULAR_HEAD_DIRECTION_RAO_SPACING_500', 'CIRCULAR_HEAD_DIRECTION_RAO_SPACING_1000', 'CIRCULAR_HEAD_DIRECTION_RAO_SPACING_2000'])
            circular_range = pd.DataFrame(CircularStatisticsMixin.sliding_circular_range(data=direction_degrees, time_windows=TIME_WINDOWS, fps=int(fps)), columns=['CIRCULAR_HEAD_DIRECTION_RANGE_250', 'CIRCULAR_HEAD_DIRECTION_RANGE_500', 'CIRCULAR_HEAD_DIRECTION_RANGE_1000', 'CIRCULAR_HEAD_DIRECTION_RANGE_2000'])
            circular_std = pd.DataFrame(CircularStatisticsMixin.sliding_circular_std(data=direction_degrees, time_windows=TIME_WINDOWS, fps=int(fps)), columns=['CIRCULAR_HEAD_DIRECTION_STD_250', 'CIRCULAR_HEAD_DIRECTION_STD_500', 'CIRCULAR_HEAD_DIRECTION_STD_1000', 'CIRCULAR_HEAD_DIRECTION_STD_2000'])
            head_hull_angular_corr = pd.DataFrame(CircularStatisticsMixin.sliding_circular_correlation(sample_1=results['CIRCULAR_FRAME_HULL_3POINT_ANGLE'].values.astype(np.float32), sample_2=results['CIRCULAR_FRAME_HEAD_3POINT_ANGLE'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['CIRCULAR_HULL_HEAD_3POINT_ANGLE_CORRELATION_250', 'CIRCULAR_HULL_HEAD_3POINT_ANGLE_CORRELATION_500', 'CIRCULAR_HULL_HEAD_3POINT_ANGLE_CORRELATION_1000', 'CIRCULAR_HULL_HEAD_3POINT_ANGLE_CORRELATION_2000'])
            hull_tail_angular_corr = pd.DataFrame(CircularStatisticsMixin.sliding_circular_correlation(sample_1=results['CIRCULAR_FRAME_HULL_3POINT_ANGLE'].values.astype(np.float32), sample_2=results['CIRCULAR_FRAME_TAIL_3POINT_ANGLE'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['CIRCULAR_HULL_TAIL_3POINT_ANGLE_CORRELATION_250', 'CIRCULAR_HULL_TAIL_3POINT_ANGLE_CORRELATION_500', 'CIRCULAR_HULL_TAIL_3POINT_ANGLE_CORRELATION_1000', 'CIRCULAR_HULL_TAIL_3POINT_ANGLE_CORRELATION_2000'])
            mean_resultant_vector_length = pd.DataFrame(CircularStatisticsMixin.sliding_mean_resultant_vector_length(data=direction_degrees, fps=int(fps), time_windows=TIME_WINDOWS), columns=['CIRCULAR_MEAN_RESULTANT_LENGTH_250', 'CIRCULAR_MEAN_RESULTANT_LENGTH_500', 'CIRCULAR_MEAN_RESULTANT_LENGTH_1000', 'CIRCULAR_MEAN_RESULTANT_LENGTH_2000'])
            results = pd.concat([results, angular_difference, rao_spacing, circular_range, circular_std, head_hull_angular_corr, hull_tail_angular_corr, mean_resultant_vector_length], axis=1)

            # MOVEMENT FEATURES
            print('Compute movement features...')
            results['MOVEMENT_FRAME_NOSE'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=nose_arr, location_2=shifted_[[f'{NOSE}_x', f'{NOSE}_y']].values.astype(np.float32), px_per_mm=px_per_mm).astype(np.int32)
            results['MOVEMENT_FRAME_CENTER'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=center_arr, location_2=shifted_[[f'{CENTER}_x', f'{CENTER}_y']].values.astype(np.float32), px_per_mm=px_per_mm).astype(np.int32)
            results['MOVEMENT_FRAME_TAILBASE'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=tailbase_arr, location_2=shifted_[[f'{TAIL_BASE}_x', f'{TAIL_BASE}_y']].values.astype(np.float32), px_per_mm=px_per_mm).astype(np.int32)
            results['MOVEMENT_FRAME_TAILTIP'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=tail_tip_arr, location_2=shifted_[[f'{TAIL_TIP}_x', f'{TAIL_TIP}_y']].values.astype(np.float32), px_per_mm=px_per_mm).astype(np.int32)
            results['MOVEMENT_FRAME_TAILCENTER'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=tail_tip_arr, location_2=shifted_[[f'{TAIL_CENTER}_x', f'{TAIL_CENTER}_y']].values.astype(np.float32), px_per_mm=px_per_mm).astype(np.int32)
            results['MOVEMENT_FRAME_LEFT_EAR'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=left_ear_arr, location_2=shifted_[[f'{LEFT_EAR}_x', f'{LEFT_EAR}_y']].values.astype(np.float32), px_per_mm=px_per_mm).astype(np.int32)
            results['MOVEMENT_FRAME_RIGHT_EAR'] = FeatureExtractionMixin.framewise_euclidean_distance(location_1=right_ear_arr, location_2=shifted_[[f'{RIGHT_EAR}_x', f'{RIGHT_EAR}_y']].values.astype(np.float32), px_per_mm=px_per_mm).astype(np.int32)
            results['MOVEMENT_FRAME_SUMMED'] = results['MOVEMENT_FRAME_NOSE'] + results['MOVEMENT_FRAME_CENTER'] + results['MOVEMENT_FRAME_TAILBASE'] + results['MOVEMENT_FRAME_TAILTIP'] + results['MOVEMENT_FRAME_TAILCENTER'] + results['MOVEMENT_FRAME_LEFT_EAR'] + results['MOVEMENT_FRAME_RIGHT_EAR']
            results['MOVEMENT_NOSE_ACCELERATION_MM_S'] = TimeseriesFeatureMixin.acceleration(data=results['MOVEMENT_FRAME_NOSE'].values.astype(np.float32), pixels_per_mm=px_per_mm, fps=fps)
            results['MOVEMENT_CENTER_ACCELERATION_MM_S'] = TimeseriesFeatureMixin.acceleration(data=results['MOVEMENT_FRAME_CENTER'].values.astype(np.float32), pixels_per_mm=px_per_mm, fps=fps)
            results['MOVEMENT_TAILBASE_ACCELERATION_MM_S'] = TimeseriesFeatureMixin.acceleration(data=results['MOVEMENT_FRAME_TAILBASE'].values.astype(np.float32), pixels_per_mm=px_per_mm, fps=fps)
            results['MOVEMENT_TAILTIP_ACCELERATION_MM_S'] = TimeseriesFeatureMixin.acceleration(data=results['MOVEMENT_FRAME_TAILTIP'].values.astype(np.float32), pixels_per_mm=px_per_mm, fps=fps)
            results['MOVEMENT_TAILCENTER_ACCELERATION_MM_S'] = TimeseriesFeatureMixin.acceleration(data=results['MOVEMENT_FRAME_TAILCENTER'].values.astype(np.float32), pixels_per_mm=px_per_mm, fps=fps)
            nose_center_acceleration_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_NOSE_ACCELERATION_MM_S'].values.astype(np.float32), sample_2=results['MOVEMENT_CENTER_ACCELERATION_MM_S'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_NOSE_CENTER_ACCELERATION_SPEARMAN_CORRELATION_250', 'MOVEMENT_NOSE_CENTER_ACCELERATION_SPEARMAN_CORRELATION_500', 'MOVEMENT_NOSE_CENTER_ACCELERATION_SPEARMAN_CORRELATION_1000',  'MOVEMENT_NOSE_CENTER_ACCELERATION_SPEARMAN_CORRELATION_2000'])
            nose_tailbase_acceleration_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_NOSE_ACCELERATION_MM_S'].values.astype(np.float32), sample_2=results['MOVEMENT_TAILBASE_ACCELERATION_MM_S'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_NOSE_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_250', 'MOVEMENT_NOSE_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_500', 'MOVEMENT_NOSE_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_1000',  'MOVEMENT_NOSE_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_2000'])
            center_tailbase_acceleration_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_CENTER_ACCELERATION_MM_S'].values.astype(np.float32), sample_2=results['MOVEMENT_TAILBASE_ACCELERATION_MM_S'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_CENTER_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_250', 'MOVEMENT_CENTER_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_500', 'MOVEMENT_CENTER_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_1000',  'MOVEMENT_CENTER_TAILBASE_ACCELERATION_SPEARMAN_CORRELATION_2000'])
            tailtip_tailbase_acceleration_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_TAILBASE_ACCELERATION_MM_S'].values.astype(np.float32), sample_2=results['MOVEMENT_TAILTIP_ACCELERATION_MM_S'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_TAILBASE_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_250', 'MOVEMENT_TAILBASE_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_500', 'MOVEMENT_TAILBASE_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_1000',  'MOVEMENT_TAILBASE_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_2000'])
            tailcenter_tailend_acceleration_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_TAILCENTER_ACCELERATION_MM_S'].values.astype(np.float32), sample_2=results['MOVEMENT_TAILTIP_ACCELERATION_MM_S'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_TAILCENTER_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_250', 'MOVEMENT_TAILCENTER_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_500', 'MOVEMENT_TAILCENTER_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_1000',  'MOVEMENT_TAILCENTER_TAILEND_ACCELERATION_SPEARMAN_CORRELATION_2000'])
            nose_center_movement_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_FRAME_NOSE'].values.astype(np.float32), sample_2=results['MOVEMENT_FRAME_CENTER'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_NOSE_CENTER_MOVEMENT_SPEARMAN_CORRELATION_250', 'MOVEMENT_NOSE_CENTER_MOVEMENT_SPEARMAN_CORRELATION_500', 'MOVEMENT_NOSE_CENTER_MOVEMENT_SPEARMAN_CORRELATION_1000',  'MOVEMENT_NOSE_CENTER_MOVEMENT_SPEARMAN_CORRELATION_2000'])
            nose_tailbase_movement_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_FRAME_NOSE'].values.astype(np.float32), sample_2=results['MOVEMENT_FRAME_TAILBASE'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_NOSE_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_250', 'MOVEMENT_NOSE_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_500', 'MOVEMENT_NOSE_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_1000',  'MOVEMENT_NOSE_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_2000'])
            center_tailbase_movement_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_FRAME_CENTER'].values.astype(np.float32), sample_2=results['MOVEMENT_FRAME_TAILBASE'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_CENTER_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_250', 'MOVEMENT_CENTER_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_500', 'MOVEMENT_CENTER_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_1000',  'MOVEMENT_CENTER_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_2000'])
            tailbase_tailend_movement_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_FRAME_TAILBASE'].values.astype(np.float32), sample_2=results['MOVEMENT_FRAME_TAILTIP'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_TAILTIP_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_250', 'MOVEMENT_TAILTIP_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_500', 'MOVEMENT_TAILTIP_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_1000',  'MOVEMENT_TAILTIP_TAILBASE_MOVEMENT_SPEARMAN_CORRELATION_2000'])
            tailcenter_tailend_movement_spearman = pd.DataFrame(Statistics.sliding_spearman_rank_correlation(sample_1=results['MOVEMENT_FRAME_TAILCENTER'].values.astype(np.float32), sample_2=results['MOVEMENT_FRAME_TAILTIP'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=fps), columns=['MOVEMENT_TAILTIP_TAILCENTER_MOVEMENT_SPEARMAN_CORRELATION_250', 'MOVEMENT_TAILTIP_TAILCENTER_MOVEMENT_SPEARMAN_CORRELATION_500', 'MOVEMENT_TAILTIP_TAILCENTER_MOVEMENT_SPEARMAN_CORRELATION_1000',  'MOVEMENT_TAILTIP_TAILCENTER_MOVEMENT_SPEARMAN_CORRELATION_2000'])
            results = pd.concat([results, nose_center_acceleration_spearman, nose_tailbase_acceleration_spearman, center_tailbase_acceleration_spearman, tailtip_tailbase_acceleration_spearman, tailcenter_tailend_acceleration_spearman, nose_center_movement_spearman, nose_tailbase_movement_spearman, center_tailbase_movement_spearman, tailbase_tailend_movement_spearman, tailcenter_tailend_movement_spearman], axis=1)

            dominant_f_nose = pd.DataFrame(Statistics.sliding_dominant_frequencies(data=results['MOVEMENT_FRAME_NOSE'].values.astype(np.float32), fps=fps, k=2, time_windows=TIME_WINDOWS), columns=['MOVEMENT_NOSE_MOVEMENT_DOMINANT_FREQUENCY_250', 'MOVEMENT_NOSE_MOVEMENT_DOMINANT_FREQUENCY_500', 'MOVEMENT_NOSE_MOVEMENT_DOMINANT_FREQUENCY_1000', 'MOVEMENT_NOSE_MOVEMENT_DOMINANT_FREQUENCY_2000'])
            dominant_f_center = pd.DataFrame(Statistics.sliding_dominant_frequencies(data=results['MOVEMENT_FRAME_CENTER'].values.astype(np.float32), fps=fps, k=2, time_windows=TIME_WINDOWS), columns=['MOVEMENT_CENTER_MOVEMENT_DOMINANT_FREQUENCY_250', 'MOVEMENT_CENTER_MOVEMENT_DOMINANT_FREQUENCY_500', 'MOVEMENT_CENTER_MOVEMENT_DOMINANT_FREQUENCY_1000', 'MOVEMENT_CENTER_MOVEMENT_DOMINANT_FREQUENCY_2000'])
            results = pd.concat([results, dominant_f_nose, dominant_f_center], axis=1)
            results['MOVEMENT_NOSE_AUTOCORRELATION_500'] = Statistics.sliding_autocorrelation(data=results['MOVEMENT_FRAME_NOSE'].values.astype(np.float32), max_lag=0.5, time_window=1.0, fps=fps)

            for time, bp in product(TIME_WINDOWS, [NOSE, CENTER, 'TAILTIP', 'TAILCENTER', 'SUMMED']):
                results[f'MOVEMENT_MEAN_{time}_{bp.upper()}'] = results[f'MOVEMENT_FRAME_{bp.upper()}'].rolling(int(time * fps), min_periods=1).mean()
                results[f'MOVEMENT_VAR_{time}_{bp.upper()}'] = results[f'MOVEMENT_FRAME_{bp.upper()}'].rolling(int(time * fps), min_periods=1).var()
                results[f'MOVEMENT_SUM_{time}_{bp.upper()}'] = results[f'MOVEMENT_FRAME_{bp.upper()}'].rolling(int(time * fps), min_periods=1).sum()

            # POSE CONFIDENCE FEATURES
            print('Compute probability features...')
            p_df = pd.DataFrame(FeatureExtractionMixin.count_values_in_range(data=p_arr, ranges=np.array([[0.0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]])), columns=['PROBABILITIES_LOW_COUNT', 'PROBABILITIES_MEDIUM_LOW_COUNT', 'PROBABILITIES_MEDIUM_HIGHT', 'PROBABILITIES_HIGH_COUNT']).astype(np.int32)
            sliding_z_p_low = pd.DataFrame(Statistics.sliding_z_scores(data=p_df['PROBABILITIES_LOW_COUNT'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=int(fps)), columns=[f'PROBABILITIES_LOW_COUNT_SLIDING_Z_SCORE_250', f'PROBABILITIES_LOW_COUNT_SLIDING_Z_SCORE_500', f'PROBABILITIES_LOW_COUNT_SLIDING_Z_SCORE_1000', f'PROBABILITIES_LOW_COUNT_SLIDING_Z_SCORE_2000'])
            sliding_z_p_high = pd.DataFrame(Statistics.sliding_z_scores(data=p_df['PROBABILITIES_HIGH_COUNT'].values.astype(np.float32), time_windows=TIME_WINDOWS, fps=int(fps)), columns=[f'PROBABILITIES_HIGH_COUNT_SLIDING_Z_SCORE_250', f'PROBABILITIES_HIGH_COUNT_SLIDING_Z_SCORE_500', f'PROBABILITIES_HIGH_COUNT_SLIDING_Z_SCORE_1000', f'PROBABILITIES_HIGH_COUNT_SLIDING_Z_SCORE_2000'])
            results = pd.concat([df, results, p_df, sliding_z_p_low, sliding_z_p_high], axis=1).fillna(-1)
            self.save(df=results, save_path=save_path)
            video_timer.stop_timer()
            print(f'Video {video_name} complete (elapsed time: {video_timer.elapsed_time_str}s)...')

        self.timer.stop_timer()
        stdout_success(msg=f'Features extracted for {len(self.outlier_corrected_paths)} files(s)', elapsed_time=self.timer.elapsed_time_str)

    def save(self, df: pd.DataFrame, save_path: os.PathLike):
        write_df(df=df.astype(np.float32), file_type=self.file_type, save_path=save_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='SimBA Custom Feature Extractor')
#     parser.add_argument('--config_path', type=str, help='SimBA project config path')
#     args = parser.parse_args()
#     feature_extractor = MitraFeatureExtractor(config_path=args.config_path)
#     feature_extractor.run()
#



# feature_extractor = MitraFeatureExtractor(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")
# feature_extractor.run()








