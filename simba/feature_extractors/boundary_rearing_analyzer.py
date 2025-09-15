import os
from typing import Union, Optional
import itertools
from copy import deepcopy
try:
    from typing import Literal
except:
    from typing_extensions import Literal
import pandas as pd
import numpy as np
from simba.mixins.abstract_classes import AbstractFeatureExtraction
from shapely.geometry import LineString
from simba.utils.checks import check_all_file_names_are_represented_in_video_log, check_valid_dataframe, check_that_column_exist, check_if_dir_exists
from simba.utils.read_write import read_df, get_fn_ext, write_df, find_files_of_filetypes_in_directory, find_core_cnt
from simba.utils.lookups import get_current_time
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.enums import Formats
from simba.feature_extractors.perimeter_jit import get_hull_sizes

####################################################################

TOP_LEFT_CORNER = 'top_left'
TOP_RIGHT_CORNER = 'top_right'
BOTTOM_LEFT_CORNER = 'bottom_left'
BOTTOM_RIGHT_CORNER = 'bottom_right'
SNOUT = 'snout'
TAILBASE = 'tailbase'
TAIL1 = 'tail1'
TAIL2 = 'tail2'
TAILTIP = 'tailtip'
LEFT_HIP = 'lefthip'
RIGHT_HIP = 'righthip'
WINDOW_SIZES = [0.5, 1.0, 2.0, 4.0]

####################################################################

LEFT, RIGHT = 'left', 'right'
TOP, BOTTOM = 'top', 'bottom'


class BoundaryRearingFeaturizer(ConfigReader,
                                AbstractFeatureExtraction):

    """
    :example:
    >>> x = BoundaryRearingFeaturizer(config_path=r"C:\troubleshooting\open_field_rearing\project_folder\project_config.ini")
    >>> x.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        if data_dir is None:
            self.data_dir = deepcopy(self.outlier_corrected_dir)
        else:
            check_if_dir_exists(in_dir=data_dir)
            self.data_dir = deepcopy(self.data_dir)
        self.data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=True)
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        self.core_count = find_core_cnt()[1]
        self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y = f'{TOP_LEFT_CORNER}_x', f'{TOP_LEFT_CORNER}_y', f'{TOP_RIGHT_CORNER}_x', f'{TOP_RIGHT_CORNER}_y'
        self.bottom_left_x, self.bottom_left_y, self.bottom_right_x, self.bottom_right_y = f'{BOTTOM_LEFT_CORNER}_x', f'{BOTTOM_LEFT_CORNER}_y', f'{BOTTOM_RIGHT_CORNER}_x', f'{BOTTOM_RIGHT_CORNER}_y'
        self.tail_base_x, self.tail_base_y, self.snout_x, self.snout_y =  f'{TAILBASE}_x', f'{TAILBASE}_y', f'{SNOUT}_x', f'{SNOUT}_y'
        self.lefthip_x, self.lefthip_y, self.righthip_x, self.righthip_y = f'{LEFT_HIP}_x', f'{LEFT_HIP}_y', f'{RIGHT_HIP}_x', f'{RIGHT_HIP}_y'
        self.corner_cols = [self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y, self.bottom_left_x, self.bottom_left_y, self.bottom_right_x, self.bottom_right_y]
        self.hull_cols = [x for x in self.bp_headers if not any(key in x for key in (TAIL1, TAIL2, TAILTIP, TOP_LEFT_CORNER, TOP_RIGHT_CORNER, BOTTOM_LEFT_CORNER, BOTTOM_RIGHT_CORNER)) and not x.endswith('_p')]
        self.p_cols = [x for x in self.bp_headers if x.endswith('_p')]

    def run(self):
        print(f'Processing features for {len(self.data_paths)} files (using cores: {self.core_count})...')
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            save_path = os.path.join(self.features_dir, f'{video_name}.{self.file_type}')
            print(f'Processing {video_name} ({file_cnt+1}/{len(self.data_paths)})... file start time: {get_current_time()}')
            _, pixels_per_mm, fps = self.read_video_info(video_name=video_name)
            df = read_df(file_path=file_path, file_type=self.file_type)
            check_valid_dataframe(df=df, source=f'{self.__class__.__name__} {file_path}', valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.bp_col_names)
            check_that_column_exist(df=df, column_name=[self.tail_base_x, self.tail_base_y, self.snout_x, self.snout_y], file_name=file_path, raise_error=True)
            tl_x, tl_y = df[self.top_left_x].values, df[self.top_left_y].values
            br_x, br_y = df[self.bottom_right_x].values, df[self.bottom_right_y].values
            bl_x, bl_y = df[self.bottom_left_x].values, df[self.bottom_left_y].values
            tr_x, tr_y  = df[self.top_right_x].values, df[self.top_right_y].values
            lines = {}
            lines[LEFT] = [LineString(pts) for pts in np.stack([np.column_stack([tl_x, tl_y]), np.column_stack([bl_x, bl_y])], axis=1)]
            lines[RIGHT] = [LineString(pts) for pts in np.stack([np.column_stack([tr_x, tr_y]), np.column_stack([br_x, br_y])], axis=1)]
            lines[TOP] = [LineString(pts) for pts in np.stack([np.column_stack([tl_x, tl_y]), np.column_stack([tr_x, tr_y])], axis=1)]
            lines[BOTTOM] = [LineString(pts) for pts in np.stack([np.column_stack([bl_x, bl_y]), np.column_stack([br_x, br_y])], axis=1)]

            self.results, side_col_names = deepcopy(df), []
            wall_dists = pd.DataFrame()
            for (bp, line) in list(itertools.product([SNOUT, TAILBASE], [TOP, BOTTOM, LEFT, RIGHT])):
                bp_data = GeometryMixin.bodyparts_to_points(data=df[[f'{bp}_x', f'{bp}_y']].values.astype(np.int32))
                wall_dists[f'{bp}->{line}_mm'] = GeometryMixin().multiframe_shape_distance(shapes_a=bp_data, shapes_b=lines[line], core_cnt=self.core_count, verbose=True, shape_names=f'{video_name}, {bp}->{line}', pixels_per_mm=pixels_per_mm)
                side_col_names.append(f'{bp}->{line}_mm')
            self.results["min_wall_distance_mm"] = wall_dists[side_col_names].min(axis=1)
            self.results["max_wall_distance_mm"] = wall_dists[side_col_names].max(axis=1)
            self.results["mean_wall_distance_mm"] = wall_dists[side_col_names].mean(axis=1)
            self.results["skew_wall_distance_mm"] = wall_dists[side_col_names].skew(axis=1)
            self.results["std_wall_distance_mm"] = wall_dists[side_col_names].std(axis=1)

            print('Computing movement sliding windows...')
            for l in range(0, len(self.hull_cols), 2):
                bp_data = self.results[self.hull_cols[l:l+2]]
                shifted = FeatureExtractionMixin.create_shifted_df(df=bp_data, periods=1).values[:, -2:]
                frame_movement = FeatureExtractionMixin().keypoint_distances(a=bp_data.values, b=shifted, px_per_mm=pixels_per_mm, in_centimeters=False).astype(np.int32)
                bp_name = self.hull_cols[l][:-2]
                sum_arr = TimeseriesFeatureMixin.sliding_window_stats(data=frame_movement, window_sizes=WINDOW_SIZES, statistics=['sum'], sample_rate=fps)
                self.results = pd.concat([self.results, pd.DataFrame(sum_arr[0], columns=[f'{bp_name}_sum_movement_{WINDOW_SIZES[0]}s', f'{bp_name}_sum_movement_{WINDOW_SIZES[1]}s', f'{bp_name}_sum_movement_{WINDOW_SIZES[2]}s', f'{bp_name}_sum_movement_{WINDOW_SIZES[3]}s'])], axis=1)

            print('Computing hull sizes...')
            self.results['hull_perimeter_mm'] = get_hull_sizes(points=df[self.hull_cols].values.reshape(len(df), -1, 2), target='perimeter', pixels_per_mm=pixels_per_mm)
            self.results['nose_2_tail_distance_mm'] = FeatureExtractionMixin().keypoint_distances(a=self.results[[self.snout_x, self.snout_y]].values, b=self.results[[self.tail_base_x, self.tail_base_y]].values, px_per_mm=pixels_per_mm, in_centimeters=False).astype(np.int32)
            self.results['left_2_right_hip_distance_mm'] = FeatureExtractionMixin().keypoint_distances(a=self.results[[self.lefthip_x, self.lefthip_y]].values, b=self.results[[self.righthip_x, self.righthip_y]].values, px_per_mm=pixels_per_mm, in_centimeters=False).astype(np.int32)

            print('Computing pose confidence distributions...')
            p = FeatureExtractionMixin.count_values_in_range(data=self.results[self.p_cols].values, ranges=np.array([[0.0, 0.20], [0.20, 0.40]]))
            p = pd.DataFrame(data=p, columns=['low_conf_detections_0_2', 'low_conf_detections_2_4'])
            self.results = pd.concat([self.results, p], axis=1)

            print('Computing hull sizes sliding windows...')
            window_cols = ['hull_perimeter_mm', 'nose_2_tail_distance_mm', 'left_2_right_hip_distance_mm', 'low_conf_detections_0_2', 'low_conf_detections_2_4']
            for measure in window_cols:
                min_arr = TimeseriesFeatureMixin.sliding_window_stats(data=self.results[measure].values.flatten().astype(np.float32), window_sizes=WINDOW_SIZES, statistics=['min'], sample_rate=fps)
                max_arr = TimeseriesFeatureMixin.sliding_window_stats(data=self.results[measure].values.flatten().astype(np.float32), window_sizes=WINDOW_SIZES, statistics=['max'], sample_rate=fps)
                mean_arr = TimeseriesFeatureMixin.sliding_window_stats(data=self.results[measure].values.flatten().astype(np.float32), window_sizes=WINDOW_SIZES, statistics=['mean'], sample_rate=fps)
                self.results = pd.concat([self.results, pd.DataFrame(min_arr[0], columns=[f'{measure}_min_{WINDOW_SIZES[0]}s', f'{measure}_min_{WINDOW_SIZES[1]}s', f'{measure}_min_{WINDOW_SIZES[2]}s', f'{measure}_min_{WINDOW_SIZES[3]}s'])], axis=1)
                self.results = pd.concat([self.results, pd.DataFrame(max_arr[0], columns=[f'{measure}_max_{WINDOW_SIZES[0]}s', f'{measure}_max_{WINDOW_SIZES[1]}s', f'{measure}_max_{WINDOW_SIZES[2]}s', f'{measure}_max_{WINDOW_SIZES[3]}s'])], axis=1)
                self.results = pd.concat([self.results, pd.DataFrame(mean_arr[0], columns=[f'{measure}_mean_{WINDOW_SIZES[0]}s', f'{measure}_mean_{WINDOW_SIZES[1]}s', f'{measure}_mean_{WINDOW_SIZES[2]}s', f'{measure}_mean_{WINDOW_SIZES[3]}s'])], axis=1)


            self.save(data=self.results, save_path=save_path)
            video_timer.stop_timer()
            stdout_success(msg=f'{video_name} complete!', elapsed_time=video_timer.elapsed_time_str)

        self.timer.stop_timer()
        stdout_success(msg=f'{len(self.data_paths)} data files saved in {self.features_dir}', elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)

    def save(self,
             data: pd.DataFrame,
             save_path: str):

        write_df(df=data, file_type=self.file_type, save_path=save_path)

# x = BoundaryRearingFeaturizer(config_path=r"C:\troubleshooting\open_field_rearing\project_folder\project_config.ini")
# x.run()