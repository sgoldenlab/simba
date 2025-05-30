import os
from typing import Iterable, List, Optional, Union

import pandas as pd
from numba import typed

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import functools
import multiprocessing
from copy import deepcopy

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.geometry_mixin import GeometryMixin
from simba.mixins.image_mixin import ImageMixin
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin
from simba.plotting.geometry_plotter import GeometryPlotter
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float, check_int,
    check_valid_array, check_valid_dataframe)
from simba.utils.enums import Defaults, Formats
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext, read_df,
                                    read_frm_of_video, read_video_info,
                                    write_df)
from simba.video_processors.video_processing import video_bg_subtraction_mp


def _multiframe_img_histocomparison(data: List[int],
                                    imgs: np.ndarray,
                                    lag: int,
                                    method: str):

    results = []
    for current_frm_idx in range(data.shape[0]):




        frm_range = frm_index[frm_range_idx]
        print(f"Analyzing frame {frm_range[1]}...")

def multiframe_img_histocomparison(imgs: np.ndarray,
                                   lag: Optional[Union[float, int]] = 1,
                                   fps: Optional[Union[float, int]] = 1,
                                   core_cnt: Optional[int] = -1,
                                   method: Optional[Literal["simple", "none", "l2", "kcos"]] = "simple",
                                   canny: Optional[bool] = True):

    check_valid_array(data=imgs, source=multiframe_img_histocomparison.__name__)
    check_float(name=f'{multiframe_img_histocomparison.__name__} lag', value=lag, min_value=10e-16)
    check_float(name=f'{multiframe_img_histocomparison.__name__} fps', value=fps, min_value=10e-16)
    check_int(name=f'{multiframe_img_histocomparison.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
    if core_cnt <= 0:
        core_cnt = find_core_cnt()[0]

    frm_idx = np.arange(0, imgs.shape[0])
    print(frm_idx)
    results = []
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_multiframe_img_histocomparison,
                                      data=imgs,
                                      canny=canny,
                                      lag=lag,
                                      method=method)
        for cnt, result in enumerate(pool.imap(constants, imgs, chunksize=1)):
            results.append(result)

    return [item for sublist in results for item in sublist]


class MitraTailAnalyzer(ConfigReader):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 anchor_points: Iterable[str],
                 body_parts: Iterable[str],
                 save_dir: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 video_dir: Optional[Union[str, os.PathLike]] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True, create_logger=False)
        if data_dir is None:
            self.data_paths = find_files_of_filetypes_in_directory(directory=self.outlier_corrected_dir, extensions=['.csv'])
        else:
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'])
        if video_dir is not None:
            self.video_dir = video_dir
        self.paths = {}
        for data_path in self.data_paths:
            video = find_video_of_file(video_dir=self.video_dir, filename=get_fn_ext(filepath=data_path)[1])
            self.paths[data_path] = video
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        self.tail_cols, self.bp_cols = [], []
        for bp in anchor_points:
            self.tail_cols.append(f'{bp}_x'.lower()); self.tail_cols.append(f'{bp}_y'.lower())
        for bp in body_parts:
            self.bp_cols.append(f'{bp}_x'.lower()); self.bp_cols.append(f'{bp}_y'.lower())
        self.required_cols = self.tail_cols + self.bp_cols
        self.save_dir = save_dir

    def run(self):
        for file_cnt, (file_path, video_path) in enumerate(self.paths.items()):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            _, px_per_mm, fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
            print(f'Analyzing {video_name} ({file_cnt+1}/{len(self.data_paths)})...')
            save_path = os.path.join(self.save_dir, f'{video_name}.csv')
            print(video_path, save_path)
            if not os.path.isfile(save_path) and (video_path is not None) and os.path.isfile(video_path):
                df = read_df(file_path=file_path, file_type=self.file_type)
                out_df = deepcopy(df)
                df.columns = [str(x).lower() for x in df.columns]
                check_valid_dataframe(df=df, valid_dtypes=Formats.NUMERIC_DTYPES.value, required_fields=self.required_cols)

                tail_geometry_df = df[self.tail_cols].values.reshape(len(df), int(len(self.tail_cols) /2), 2).astype(np.int64)
                tail_geometries = GeometryMixin().bodyparts_to_polygon(data=tail_geometry_df, parallel_offset=35, pixels_per_mm=px_per_mm)

                hull_geometry_df = df[self.bp_cols].values.reshape(len(df), int(len(self.bp_cols) / 2), 2).astype(np.int64)
                hull_geometries = GeometryMixin().bodyparts_to_polygon(data=hull_geometry_df, parallel_offset=40, pixels_per_mm=px_per_mm)










                out_df['tail_area'] = GeometryMixin().multiframe_area(shapes=tail_geometries, pixels_per_mm=px_per_mm)

                tail_area_std = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=out_df['tail_area'].values.astype(np.float32), window_sizes=np.array([0.5, 1.0, 2.0]), sample_rate=fps, statistics=typed.List(['std']))
                out_df = pd.concat([out_df, pd.DataFrame(tail_area_std[0], columns=['tail_area_std_05', 'tail_area_std_1', 'tail_area_std_2'])], axis=1)

                tail_area_mean = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=out_df['tail_area'].values.astype(np.float32), window_sizes=np.array([0.5, 1.0, 2.0]), sample_rate=fps, statistics=typed.List(['mean']))
                out_df = pd.concat([out_df, pd.DataFrame(tail_area_mean[0], columns=['tail_area_mean_05', 'tail_area_mean_1', 'tail_area_mean_2'])], axis=1)

                tail_area_mad = TimeseriesFeatureMixin.sliding_descriptive_statistics(data=out_df['tail_area'].values.astype(np.float32), window_sizes=np.array([0.5, 1.0, 2.0]), sample_rate=fps, statistics=typed.List(['mad']))
                out_df = pd.concat([out_df, pd.DataFrame(tail_area_mad[0], columns=['tail_area_mad_05', 'tail_area_mad_1', 'tail_area_mad_2'])], axis=1)

                out_df['tail_hausdorf_05'] = GeometryMixin().multiframe_hausdorff_distance(geometries=tail_geometries, lag=0.5, sample_rate=fps)
                out_df['tail_hausdorf_1'] = GeometryMixin().multiframe_hausdorff_distance(geometries=tail_geometries, lag=1, sample_rate=fps)
                out_df['tail_hausdorf_2'] = GeometryMixin().multiframe_hausdorff_distance(geometries=tail_geometries, lag=2, sample_rate=fps)

                out_df['hull_hausdorf_05'] = GeometryMixin().multiframe_hausdorff_distance(geometries=hull_geometries, lag=0.5, sample_rate=fps)
                out_df['hull_hausdorf_1'] = GeometryMixin().multiframe_hausdorff_distance(geometries=hull_geometries, lag=1, sample_rate=fps)
                out_df['hull_hausdorf_2'] = GeometryMixin().multiframe_hausdorff_distance(geometries=hull_geometries, lag=2, sample_rate=fps)

                if video_path is not None:
                    tail_imgs = ImageMixin().slice_shapes_in_imgs(imgs=video_path, shapes=tail_geometries, bg_color=(255, 255, 255), core_cnt=8, verbose=True)
                    tail_imgs = ImageMixin.pad_img_stack(image_dict=tail_imgs)
                    tail_imgs = np.stack(list(tail_imgs.values()))
                    out_df['tail_mse_05'] = ImageMixin.img_sliding_mse(imgs=tail_imgs, slide_length=0.5, sample_rate=float(fps))
                    out_df['tail_mse_1'] = ImageMixin.img_sliding_mse(imgs=tail_imgs, slide_length=1.0, sample_rate=float(fps))
                    out_df['tail_mse_2'] = ImageMixin.img_sliding_mse(imgs=tail_imgs, slide_length=2.0, sample_rate=float(fps))
                    video_timer.stop_timer()
                    write_df(df=out_df, file_type='csv', save_path=save_path)
                    print(video_timer.elapsed_time_str)

# runner = MitraTailAnalyzer(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                            data_dir=r'C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed\rotated',
#                            video_dir=r'C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed\rotated',
#                            save_dir=r'C:\troubleshooting\mitra\project_folder\videos\additional\bg_removed\rotated\tail_features_additional',
#                            anchor_points=('tail_base', 'tail_center', 'tail_tip'),
#                            body_parts=('nose', 'left_ear', 'right_ear', 'right_side', 'left_side', 'tail_base'))
# runner.run()