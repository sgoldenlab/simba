import argparse
import os
import sys
from typing import Union

import numpy as np
import pandas as pd

from simba.data_processors.find_animal_blob_location import (
    get_blob_vertices_from_video, get_left_right_points,
    get_nose_tail_from_vertices, stabilize_body_parts)
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import (check_if_dir_exists, check_instance, check_int,
                                check_nvidea_gpu_available, check_valid_dict)
from simba.utils.data import resample_geometry_vertices, savgol_smoother
from simba.utils.errors import SimBAGPUError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt, get_video_meta_data,
                                    read_pickle, remove_files, write_df)
from simba.video_processors.video_processing import (video_bg_subtraction,
                                                     video_bg_subtraction_mp)

CENTER_X = 'center_x'
CENTER_Y = 'center_y'
NOSE_X = 'nose_x'
NOSE_Y = 'nose_y'
TAIL_X = 'tail_x'
TAIL_Y = 'tail_y'
LEFT_X = 'left_x'
LEFT_Y = 'left_y'
RIGHT_X = 'right_x'
RIGHT_Y = 'right_y'
VERTICES = 'vertices'

IN_DIR = 'input_dir'
OUT_DIR = 'output_dir'
VIDEO_DATA = 'video_data'
VERTICE_CNT = 'vertice_cnt'
REFERENCE = 'reference'
CORE_CNT = 'core_cnt'
CLOSING_KERNEL = 'close_kernel'
CLOSING_ITS = 'close_iterations'
OPENING_ITS = 'open_iterations'
OPENING_KERNEL = 'open_kernel'
SAVE_BG_VIDEOS = 'save_bg_videos'
GPU = 'gpu'

REQUIRED_KEYS = (IN_DIR, OUT_DIR, GPU, CORE_CNT, VIDEO_DATA, VERTICE_CNT, SAVE_BG_VIDEOS, CLOSING_ITS, OPENING_ITS)

METHOD = 'method'
THRESHOLD = 'threshold'


class BlobTrackingExecutor():

    """
    Perform animal blob tracking from video data using background subtraction, blob location detection,
    and geometry-based processing. The class handles the processing of video frames in parallel, tracks blob locations
    across frames, and saves the results to CSV files.

    :param Union[dict, str, os.PathLike] data: Path to a configuration file or a dictionary containing the necessary parameters for blob tracking. The configuration should include keys for the video data, input/output directories, and other processing settings.
    :param int vertice_cnt: The number of vertices to use when sampling the blob geometry. Default: 10.
    :param int batch_size: The batch size for blob location detection. Default: 4k.

    :example:
    >>> tracker = BlobTrackingExecutor(data=r"C:\troubleshooting\mitra\test\.temp\blob_definitions.h5", vertice_cnt=10)
    >>> tracker.run()
    """


    def __init__(self,
                 data: Union[dict, str, os.PathLike],
                 batch_size: int = 5000,
                 rostrocaudal : bool = True,
                 mediolateral: bool = True,
                 center: bool = True):

        self.timer = SimbaTimer(start=True)
        check_instance(source=f'{self.__class__.__name__} data', instance=data, accepted_types=(str, os.PathLike, dict), raise_error=True)
        if isinstance(data, (str, os.PathLike)):
            data = read_pickle(data_path=data)
        check_valid_dict(x=data, valid_key_dtypes=(str,), required_keys=REQUIRED_KEYS)
        check_if_dir_exists(in_dir=data[IN_DIR], source=self.__class__.__name__, raise_error=True)
        check_if_dir_exists(in_dir=data[OUT_DIR], source=self.__class__.__name__, raise_error=True)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=data[CORE_CNT], min_value=1, max_value=find_core_cnt()[0], raise_error=True)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1, raise_error=True)
        self.data, self.gpu, self.core_cnt, self.batch_size, self.vertice_cnt, self.closing_iterations = data, data[GPU], data[CORE_CNT], batch_size, data[VERTICE_CNT], data[CLOSING_ITS]
        self.save_bg_videos, self.save_dir, self.opening_iterations = data[SAVE_BG_VIDEOS], data[OUT_DIR], data[OPENING_ITS]
        self.rostrocaudal, self.mediolateral, self.center = rostrocaudal, mediolateral, center
        if self.gpu and not check_nvidea_gpu_available():
            raise SimBAGPUError(msg='GPU is set to True, but SImBA could not find a GPU on the machine', source=self.__class__.__name__)
        self.vertice_col_names = []
        self.video_cnt = len(list(self.data[VIDEO_DATA].keys()))
        for i in range(self.vertice_cnt):
            self.vertice_col_names.append(f"vertice_{i}_x"); self.vertice_col_names.append(f"vertice_{i}_y")

    def run(self):
        self._remove_bgs()
        self._find_blobs()
        self.timer.stop_timer()
        stdout_success(msg=f'Animal tracking complete. Results save din directory {self.data[OUT_DIR]}', elapsed_time=self.timer.elapsed_time_str)

    def _interpolate_vertices(self, arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr.reshape(arr.shape[0], -1))
        df = df.fillna(0).clip(lower=0).astype(np.int32)
        for c in df.columns:
            bp_df = df[[c]].astype(np.float32)
            missing_idx = bp_df[bp_df[c] <= 0.0].index
            bp_df.loc[missing_idx, c] = np.nan
            if bp_df[c].notna().sum() >= 2:
                bp_df[c] = bp_df[c].interpolate(method='nearest', axis=0).ffill().bfill()
            else:
                bp_df[c] = bp_df[c].fillna(0)
            df.update(bp_df)
        return df.values.reshape(arr.shape).astype(np.int32)

    def _remove_bgs(self):
        bg_timer = SimbaTimer(start=True)
        self.bg_video_paths = []
        for video_cnt, (video_name, video_data) in enumerate(self.data[VIDEO_DATA].items()):
            video_meta = get_video_meta_data(video_path=video_data['video_path'])
            bg_video_path = os.path.join(self.save_dir, f'{video_name}.mp4')
            self.bg_video_paths.append(bg_video_path)
            close_kernel_size = None if video_data[CLOSING_KERNEL] is None else tuple(video_data[CLOSING_KERNEL])
            opening_kernel_size = None if video_data[OPENING_KERNEL] is None else tuple(video_data[OPENING_KERNEL])
            print(f'Starting background subtraction on video {video_name} (video {video_cnt+1}/{self.video_cnt}, frame count: {video_meta["frame_count"]}, cores: {self.core_cnt}, gpu: {self.gpu})')
            if (self.core_cnt == 1):
                video_bg_subtraction(video_path=video_data['video_path'],
                                     bg_video_path=video_data[REFERENCE],
                                     threshold=video_data[THRESHOLD],
                                     save_path=bg_video_path,
                                     verbose=True,
                                     fg_color=(255, 255, 255),
                                     closing_kernel_size=close_kernel_size,
                                     closing_iterations=self.closing_iterations,
                                     opening_kernel_size=opening_kernel_size,
                                     opening_iterations=self.opening_iterations)
            else:
                video_bg_subtraction_mp(video_path=video_data['video_path'],
                                        bg_video_path=video_data[REFERENCE],
                                        threshold=video_data[THRESHOLD],
                                        save_path=bg_video_path,
                                        verbose=True,
                                        gpu=self.gpu,
                                        core_cnt=self.core_cnt,
                                        fg_color=(255, 255, 255),
                                        closing_kernel_size=close_kernel_size,
                                        closing_iterations=self.closing_iterations,
                                        opening_kernel_size=opening_kernel_size,
                                        opening_iterations=self.opening_iterations)

        bg_timer.stop_timer()
        print(f'Background subtraction COMPLETE: videos saved at {self.save_dir}, (elapsed time: {bg_timer.elapsed_time_str}s)')



    def _find_blobs(self):
        blob_timer = SimbaTimer(start=True)
        for video_cnt, (video_name, video_data) in enumerate(self.data[VIDEO_DATA].items()):
            video_timer, geometries = SimbaTimer(start=True), None
            video_meta = get_video_meta_data(video_path=video_data['video_path'])
            temp_video_path = os.path.join(self.save_dir, f'{video_name}.mp4')
            save_path = os.path.join(self.data[OUT_DIR], f'{video_meta["video_name"]}.csv')
            inclusion_zone = None if 'inclusion_zones' not in video_data.keys() else video_data['inclusion_zones']
            window_size = None if 'window_size' not in video_data.keys() else video_data['window_size']
            vertices = get_blob_vertices_from_video(video_path=temp_video_path,
                                                    gpu=self.gpu,
                                                    verbose=True,
                                                    core_cnt=self.core_cnt,
                                                    batch_size=None,
                                                    inclusion_zone=inclusion_zone,
                                                    window_size=window_size,
                                                    convex_hull=False,
                                                    vertice_cnt=self.vertice_cnt)

            vertices = self._interpolate_vertices(arr=vertices)
            results = pd.DataFrame()
            if video_data['buffer_size'] is not None:
                geometries = GeometryMixin.bodyparts_to_polygon(data=vertices, parallel_offset=video_data['buffer_size'], convex_hull=False, simplify_tolerance=0.1)
                vertices = [np.array(x.exterior.coords).astype(np.int32) for x in geometries]
                vertices = resample_geometry_vertices(vertices=vertices, vertice_cnt=self.vertice_cnt)
            if video_data['smoothing_time'] is not None:
                vertices = savgol_smoother(data=vertices.reshape(vertices.shape[0], -1), fps=video_meta['fps'], time_window=video_data['smoothing_time'], source=self.__class__.__name__).astype(np.int32)
                vertices = vertices.reshape(vertices.shape[0], self.vertice_cnt, 2)
            if self.center or self.mediolateral or self.rostrocaudal:
                geometries = geometries if geometries != None else GeometryMixin.bodyparts_to_polygon(data=vertices, convex_hull=False, simplify_tolerance=0.1)
                centers = GeometryMixin.get_center(shape=geometries).astype(np.int32)
                results[CENTER_X], results[CENTER_Y] = centers[:, 0], centers[:, 1]
            if self.rostrocaudal or self.mediolateral:
                nose, tail = get_nose_tail_from_vertices(vertices=vertices, fps=video_meta['fps'], smooth_factor=1.0)
                #nose, tail = stabilize_body_parts(bp_1=nose, bp_2=tail, center_positions=centers)
                results[NOSE_X], results[NOSE_Y] = nose[:, 0], nose[:, 1]
                results[TAIL_X], results[TAIL_Y] = tail[:, 0], tail[:, 1]
            if self.mediolateral:
                left, right = get_left_right_points(hull_vertices=vertices, anterior=nose, center=centers, posterior=tail)
                results[LEFT_X], results[LEFT_Y] = left[:, 0], left[:, 1]
                results[RIGHT_X], results[RIGHT_Y] = right[:, 0], right[:, 1]

            vertices = pd.DataFrame(vertices.reshape(vertices.shape[0], (self.vertice_cnt*2)).astype(np.int32), columns=self.vertice_col_names)
            results = pd.concat([results, vertices], axis=1).reset_index(drop=True).fillna(0)
            write_df(df=results, file_type='csv', save_path=save_path)
            video_timer.stop_timer()
            print(f'Animal blob tracking data for video {video_meta["video_name"]} saved at {save_path}, (elapsed time: {video_timer.elapsed_time_str}s).')
        blob_timer.stop_timer()
        if not self.save_bg_videos:
            remove_files(file_paths=self.bg_video_paths, raise_error=False)
        print(f'Blob tracking COMPLETE: data saved at {self.save_dir}, (elapsed time: {blob_timer.elapsed_time_str}s')

if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Execute Blob tracking in SimBA.")
    parser.add_argument('--data', type=str, required=True, help='Path to the pickle holding the parameters for performing blob tracking')
    args = parser.parse_args()
    tracker = BlobTrackingExecutor(data=args.data)
    tracker.run()


# DATA_PATH = r"C:\troubleshooting\blob_track_tester\results\blob_definitions.pickle"
# tracker = BlobTrackingExecutor(data=DATA_PATH)
# tracker.run()



# #DATA_PATH = r"/mnt/d/open_field_3/sample/.temp/blob_definitions.h5"
# #DATA_PATH = r"D:\open_field_3\sample\blob_data\blob_definitions.json"
# # DATA_PATH = r"D:\EPM\sampled\.temp\blob_definitions.h5"
# #DATA_PATH = r"D:\EPM\sample_2\.temp\blob_definitions.h5"
#
# #
# DATA_PATH = r"D:\open_field_2\sample\clipped_10min\data\blob_definitions.json"
#
#
# #DATA_PATH = r"D:\open_field_4\data\blob_definitions.json"
# DATA_PATH = r"D:\open_field\data\blob_definitions.json"
#
# DATA_PATH = r"D:\EPM_3\out\blob_definitions.json"
# # # # # # # # # #
# # # # # # # # # #
# tracker = BlobTrackingExecutor(data=DATA_PATH)
# tracker.run()
# # # #
# # #
# #

# data = {'input_dir': r'C:\\troubleshooting\\mitra\\test',
#         'output_dir': r'C:\\troubleshooting\\mitra\\test\\blob_data',
#         'gpu': False,
#         'core_cnt': 32,
#         'video_data': {r'C:\\troubleshooting\\mitra\\test\\501_MA142_Gi_Saline_0515.mp4': {'threshold': 50, 'method': 'absolute', 'window_size': 'None', 'window_weight': 'None', 'reference': r'C:/troubleshooting/mitra/test\background_dir\501_MA142_Gi_Saline_0515.mp4', 'visualize': False, 'exclusion_zones': None,
#                                                                                            'inclusion_zones': {'inclusion_zone': {'Video': '501_MA142_Gi_Saline_0515', 'Shape_type': 'rectangle', 'Name': 'inclusion_zone', 'Color name': 'Red', 'Color BGR': (0, 0, 255), 'Thickness': 5, 'Center_X': 365, 'Center_Y': 293, 'topLeftX': 204, 'topLeftY': 131, 'Bottom_right_X': 526, 'Bottom_right_Y': 455, 'width': 322, 'height': 324, 'width_cm': 32.2, 'height_cm': 32.4, 'area_cm': 1043.28, 'Tags': {'Center tag': (365, 293), 'Top left tag': (204, 131), 'Bottom right tag': (526, 455), 'Top right tag': (526, 131), 'Bottom left tag': (204, 455), 'Top tag': (365, 131), 'Right tag': (526, 293), 'Left tag': (204, 293), 'Bottom tag': (365, 455)}, 'Ear_tag_size': 15}}}}}
#



# data = {'input_dir': r'C:\\troubleshooting\\mitra\\test',
#         'output_dir': r'C:\\troubleshooting\\mitra\\test\\blob_data',
#         'gpu': False,
#         'core_cnt': 32,
#         'video_data': {r'C:\\troubleshooting\\mitra\\test\\501_MA142_Gi_Saline_0515': {'threshold': 30, 'method': 'absolute', 'window_size': 'None', 'window_weight': 'None', 'reference': 'C:/troubleshooting/mitra/test\\501_MA142_Gi_Saline_0515.mp4', 'visualize': False, 'exclusion_zones': None, 'inclusion_zones': None},
#                        r'C:\\troubleshooting\\mitra\\test\\502_MA141_Gi_CNO_0514': {'threshold': 30, 'method': 'absolute', 'window_size': 'None', 'window_weight': 'None', 'reference': 'C:/troubleshooting/mitra/test\\502_MA141_Gi_CNO_0514.mp4', 'visualize': False, 'exclusion_zones': None, 'inclusion_zones': None},
#                        r'C:\\troubleshooting\\mitra\\test\\503_MA109_Gi_CNO_0521': {'threshold': 30, 'method': 'absolute', 'window_size': 'None', 'window_weight': 'None', 'reference': 'C:/troubleshooting/mitra/test\\503_MA109_Gi_CNO_0521.mp4', 'visualize': False, 'exclusion_zones': None, 'inclusion_zones': None}}}
#



# data = read_pickle(data_path=r"C:\troubleshooting\mitra\blob_data.pickle")
# # data['gpu'] = True
# # data['core_cnt'] = 16
# tracker = BlobTrackingExecutor(data=data)
# tracker.run()