import functools
import multiprocessing
import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_tuple,
                                check_video_and_data_frm_count_align)
from simba.utils.data import create_color_palette
from simba.utils.enums import Defaults, Formats, Options
from simba.utils.errors import (CountError, FrameRangeError,
                                InvalidFilepathError)
from simba.utils.lookups import get_random_color_palette
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory,
                                    find_all_videos_in_directory,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data,
                                    read_frm_of_video, remove_a_folder)
from simba.utils.warnings import FrameRangeWarning, MissingFileWarning

FRAME = 'FRAME'
CLASS_ID = 'CLASS_ID'
CONFIDENCE = 'CONFIDENCE'
CLASS_NAME = 'CLASS_NAME'
TRACK = 'TRACK'
BOX_CORD_FIELDS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']

EXPECTED_COLS = [FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE, TRACK, 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']

def _yolo_keypoint_track_visualizer(frm_ids: np.ndarray,
                              data: pd.DataFrame,
                              threshold: float,
                              video_path: str,
                              save_dir: str,
                              circle_size: int,
                              thickness: int,
                              palettes: dict,
                              show_bbox: bool):

    batch_id, frame_rng = frm_ids[0], frm_ids[1]
    start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    fourcc, font = cv2.VideoWriter_fourcc(*f"{Formats.MP4_CODEC.value}"), cv2.FONT_HERSHEY_DUPLEX
    video_save_path = os.path.join(save_dir, f'{batch_id}.mp4')
    video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    while current_frm <= end_frm:
        print(f'Processing frame {current_frm}/{video_meta_data["frame_count"]} (batch: {batch_id})...')
        img = read_frm_of_video(video_path=video_path, frame_index=current_frm)
        frm_data = data.loc[data[FRAME] == current_frm]
        frm_data = frm_data[frm_data[CONFIDENCE] > threshold]
        for cnt, (row, row_data) in enumerate(frm_data.iterrows()):
            clrs = np.array(palettes[int(row_data[TRACK])]).astype(np.int32)
            bbox_cords = row_data[BOX_CORD_FIELDS].values.astype(np.int32).reshape(-1, 2)
            kp_coords = row_data.drop(EXPECTED_COLS).values.astype(np.int32).reshape(-1, 3)[:, :-1]
            clr = tuple(int(c) for c in clrs[0])
            if show_bbox:
                img = cv2.polylines(img, [bbox_cords], True, clr, thickness=thickness, lineType=cv2.LINE_AA)
            for kp_cnt, kp in enumerate(kp_coords):
                clr = tuple(int(c) for c in clrs[kp_cnt+1])
                img = cv2.circle(img, (tuple(kp)), circle_size, clr, -1)
        video_writer.write(img)
        current_frm += 1
    cap.release()
    video_writer.release()
    return batch_id


class YOLOPoseTrackVisualizer():
    """
    Visualizes YOLO-based keypoint pose estimation data on video frames and creates an annotated output video.

    This class takes keypoint data (CSV) and overlays it onto the corresponding video using color-coded keypoints
    and optional filtering. The result is saved as a new annotated video, and supports multicore parallel rendering
    for efficient processing of long videos.

    .. seelalso::
       To create YOLO pose data, see `:func:~simba.bounding_box_tools.yolo.yolo_pose_inference.YOLOPoseInference`

    :param Union[str, os.PathLike] data_path: Path to the CSV file containing keypoint data (output from YOLO pose inference).
    :param Union[str, os.PathLike] video_path: Path to the original input video to overlay keypoints on.
    :param Union[str, os.PathLike] save_dir: Directory to save the resulting annotated video.
    :param Optional[Union[str, Tuple[str, ...]]] palettes: Name of the color palette(s) to use for drawing keypoints. Can be a string or a tuple of strings (e.g., 'Set1', ('Set1', 'Dark2')). Defaults to 'Set1'.
    :param Optional[int] core_cnt: Number of CPU cores to use for parallel rendering. Defaults to -1 (use all available cores).
    :param float threshold: Confidence threshold for visualizing keypoints. Only keypoints with confidence >= threshold are drawn. Defaults to 0.0.
    :param Optional[int] thickness: Thickness of lines connecting keypoints. If None, determined automatically. Defaults to None.
    :param Optional[int] circle_size: Radius of the circles drawn for keypoints. If None, determined automatically based on frame size. Defaults to None.
    :param Optional[bool] verbose: If True, enables logging and progress messages. Defaults to False.


    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
    >>> data_path = "/mnt/c/troubleshooting/mitra/yolo_pose/501_MA142_Gi_CNO_0521.csv"
    >>> kp_vis = YOLOPoseVisualizer(data_path=data_path,
    >>>                            video_path=video_path,
    >>>                            save_dir='/mnt/c/troubleshooting/mitra/yolo_pose/',
    >>>                            core_cnt=18)
    >>> kp_vis.run()
    """



    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 palettes: Optional[Union[str, Tuple[str, ...]]] = None,
                 core_cnt: Optional[int] = -1,
                 threshold: float = 0.0,
                 thickness: Optional[int] = None,
                 circle_size: Optional[int] = None,
                 verbose: Optional[bool] = False,
                 show_bbox: Optional[bool] = False):

        if not os.path.isdir(data_path) and not os.path.isfile(data_path):
            raise InvalidFilepathError(msg=f'data_path {data_path} is not a valid directory of file path.', source=self.__class__.__name__)
        if not os.path.isdir(video_path) and not os.path.isfile(video_path):
            raise InvalidFilepathError(msg=f'video_path {video_path} is not a valid directory of file path.', source=self.__class__.__name__)
        if os.path.isdir(data_path) and not os.path.isdir(video_path):
            raise InvalidFilepathError(msg=f'If data_path ({data_path}) is a directory, video_path ({video_path}) also needs to be a directory.', source=self.__class__.__name__)
        elif os.path.isdir(video_path) and not os.path.isdir(data_path):
            raise InvalidFilepathError(msg=f'If data_path ({data_path}) is a directory, video_path ({video_path}) also needs to be a directory.', source=self.__class__.__name__)

        if os.path.isfile(video_path):
            check_file_exist_and_readable(file_path=data_path)
            self.data_paths, self.video_paths = {get_fn_ext(filepath=data_path)[1]: data_path}, {get_fn_ext(filepath=data_path)[1]: video_path}
        else:
            self.data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=('.csv',), as_dict=True, raise_error=True, raise_warning=False)
            self.video_paths = find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True)
            missing_videos = [x for x in self.data_paths.keys() if x not in self.video_paths.keys()]
            if len(missing_videos) > 0:
                MissingFileWarning(msg=f'Data files {missing_videos} is missing a video file in the {video_path} directory', source=self.__class__.__name__)
                self.data_paths = {k: v for k, v in self.data_paths.items() if k not in missing_videos}

        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_if_dir_exists(in_dir=save_dir)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=[show_bbox], source=f'{self.__class__.__name__} show_bbox', raise_error=True)
        if core_cnt == -1 or core_cnt > find_core_cnt()[0]: core_cnt = find_core_cnt()[0]
        if circle_size is not None: check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        if thickness is not None: check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        self.save_dir, self.verbose, self.palettes, self.thickness, self.core_cnt = save_dir, verbose, palettes, thickness, core_cnt
        self.threshold, self.circle_size, self.thickness, self.show_bbox = threshold, circle_size, thickness, show_bbox

    def run(self):
        for video_cnt, (video_name, data_path) in enumerate(self.data_paths.items()):
            print(f'Visualizing YOLO pose tracks in video {video_name} ({video_cnt+1}/{len(self.data_paths.keys())}) ...')
            video_timer = SimbaTimer(start=True)
            video_temp_dir = os.path.join(self.save_dir, video_name)
            save_path = os.path.join(self.save_dir, f'{video_name}.mp4')
            if os.path.isdir(video_temp_dir): remove_a_folder(folder_dir=video_temp_dir)
            create_directory(paths=video_temp_dir)
            self.video_meta_data = get_video_meta_data(video_path=self.video_paths[video_name])
            self.data_df = pd.read_csv(data_path, index_col=0)
            check_valid_dataframe(df=self.data_df, source=self.__class__.__name__, required_fields=EXPECTED_COLS)
            df_frm_cnt = np.unique(self.data_df[FRAME].values).shape[0]
            if df_frm_cnt != self.video_meta_data['frame_count']:
                FrameRangeWarning(msg=f'The data file {data_path} contains data for {df_frm_cnt} frames, but the video {self.video_paths[video_name]} contains {self.video_meta_data["frame_count"]} frames', source=self.__class__.__name__)
            video_circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(self.video_meta_data['width'], self.video_meta_data['height']), circle_frame_ratio=100) if self.circle_size is None else self.circle_size
            video_thickness = video_circle_size if self.thickness is None else self.thickness
            video_classes, video_tracks = np.unique(self.data_df[CLASS_NAME].values), [int(x) for x in np.unique(self.data_df[TRACK].values)]
            video_palettes = {}
            if self.palettes is None:
                for track_cnt, track_id in enumerate(video_tracks):
                    video_palettes[track_id] = get_random_color_palette(n_colors=len(self.data_df.columns) - len(EXPECTED_COLS))
            else:
                video_palettes = self.palettes

            frm_batches = np.array_split(np.array(list(range(0, df_frm_cnt))), self.core_cnt)
            frm_batches = [(i, j) for i, j in enumerate(frm_batches)]
            with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
                constants = functools.partial(_yolo_keypoint_track_visualizer,
                                              data=self.data_df,
                                              threshold=self.threshold,
                                              video_path=self.video_paths[video_name],
                                              save_dir=video_temp_dir,
                                              circle_size=video_circle_size,
                                              thickness=video_thickness,
                                              palettes=video_palettes,
                                              show_bbox=self.show_bbox)
                for cnt, result in enumerate(pool.imap(constants, frm_batches, chunksize=1)):
                    print(f'Video batch {result+1}/{self.core_cnt} complete...')
            pool.terminate()
            pool.join()
            video_timer.stop_timer()
            concatenate_videos_in_folder(in_folder=video_temp_dir, save_path=save_path, gpu=True)
            stdout_success(msg=f'YOLO track pose video saved at {save_path}', source=self.__class__.__name__, elapsed_time=video_timer.elapsed_time_str)


# video_path = r"/mnt/d/ares/data/termite_2/videos/termite.mp4"
# data_path = "/mnt/d/ares/data/termite_2/yolo/results/termite.csv"
# kp_vis = YOLOPoseTrackVisualizer(data_path=data_path, video_path=video_path, save_dir='/mnt/c/troubleshooting/mitra/yolo_pose/', core_cnt=18)
# kp_vis.run()

# VIDEO_PATH = r"/mnt/d/ares/data/ant/sleap_video/ant.mp4"
# DATA_PATH = "/mnt/d/ares/data/ant/yolo/results/ant.csv"
# SAVE_DIR = "/mnt/d/ares/data/ant/yolo/results"
# kp_vis = YOLOPoseTrackVisualizer(data_path=DATA_PATH, video_path=VIDEO_PATH, save_dir=SAVE_DIR, core_cnt=18)
# #kp_vis.run()


# if __name__ == "__main__":
#     VIDEO_PATH = r"E:\netholabs_videos\two_tracks_102725\videos\cage_1_date_2025_09_13_hour_03_minute_46.avi"
#     DATA_PATH = r"E:\netholabs_videos\two_tracks_102725\tracks_cleaned\cage_1_date_2025_09_13_hour_03_minute_46.csv"
#     SAVE_DIR = r"E:\netholabs_videos\two_tracks_102725\out_videos"
#     kp_vis = YOLOPoseTrackVisualizer(data_path=DATA_PATH,
#                                      video_path=VIDEO_PATH,
#                                      save_dir=SAVE_DIR,
#                                      core_cnt=8,
#                                      show_bbox=True)
#     kp_vis.run()


if __name__ == "__main__":
    VIDEO_PATH = r"E:\netholabs_videos\two_tracks_102725\videos"
    DATA_PATH = r"E:\netholabs_videos\two_tracks_102725\tracks_cleaned"
    SAVE_DIR = r"E:\netholabs_videos\two_tracks_102725\out_videos"
    kp_vis = YOLOPoseTrackVisualizer(data_path=DATA_PATH,
                                     video_path=VIDEO_PATH,
                                     save_dir=SAVE_DIR,
                                     core_cnt=8,
                                     show_bbox=True)
    kp_vis.run()