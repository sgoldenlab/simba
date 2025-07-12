import functools
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst, check_valid_tuple)
from simba.utils.data import create_color_palette
from simba.utils.enums import Defaults, Options
from simba.utils.errors import CountError, DataHeaderError, FrameRangeError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory, find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data,
                                    read_frm_of_video)

FRAME = 'FRAME'
CLASS_ID = 'CLASS_ID'
CONFIDENCE = 'CONFIDENCE'
CLASS_NAME = 'CLASS_NAME'
BOX_CORD_FIELDS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
EXPECTED_COLS = [FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE, 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']



def _yolo_keypoint_visualizer(frm_ids: np.ndarray,
                              data: pd.DataFrame,
                              threshold: float,
                              video_path: str,
                              save_dir: str,
                              circle_size: int,
                              thickness: int,
                              palettes: dict,
                              bbox: bool,
                              skeleton: list):

    batch_id, frame_rng = frm_ids[0], frm_ids[1]
    start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    fourcc, font = cv2.VideoWriter_fourcc(*"mp4v"), cv2.FONT_HERSHEY_DUPLEX
    video_save_path = os.path.join(save_dir, f'{batch_id}.mp4')
    video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    while current_frm <= end_frm:
        print(f'Processing frame {current_frm}/{video_meta_data["frame_count"]} (batch: {batch_id}, video: {video_meta_data["video_name"]})...')
        img = read_frm_of_video(video_path=video_path, frame_index=current_frm)
        frm_data = data.loc[data[FRAME] == current_frm]
        frm_data = frm_data[frm_data[CONFIDENCE] > threshold]
        for cnt, (row, row_data) in enumerate(frm_data.iterrows()):
            clrs = np.array(palettes[int(row_data[CLASS_ID])]).astype(np.int32)
            bbox_cords = row_data[BOX_CORD_FIELDS].values.astype(np.int32).reshape(-1, 2)
            kp_coords = row_data.drop(EXPECTED_COLS).values.astype(np.int32).reshape(-1, 3)[:, :-1]
            p_cords = row_data.drop(EXPECTED_COLS).values.astype(np.float64).reshape(-1, 3)[:, -1:].flatten()
            clr = tuple(int(c) for c in clrs[0])
            if bbox:
                img = cv2.polylines(img, [bbox_cords], True, clr, thickness=thickness, lineType=cv2.LINE_AA)
            for kp_cnt, kp in enumerate(kp_coords):
                if p_cords[kp_cnt] > threshold:
                    clr = tuple(int(c) for c in clrs[kp_cnt+1])
                    img = cv2.circle(img, (tuple(kp)), circle_size, clr, -1)
            if skeleton is not None:
                for (kp1, kp2) in skeleton:
                    if (row_data[f'{kp1}_P'] > threshold) and (row_data[f'{kp2}_P'] > threshold):
                        pos_1 = np.array([row_data[[f'{kp1}_X', f'{kp1}_Y']].values.astype(np.int32)])
                        pos_2 = np.array([row_data[[f'{kp2}_X', f'{kp2}_Y']].values.astype(np.int32)])
                        img = PlottingMixin().draw_lines_on_img(img=img, start_positions=pos_1, end_positions=pos_2, color=(105,105,105), highlight_endpoint=False, thickness=int(max(1, int(circle_size/2))))
        video_writer.write(img)
        current_frm += 1
    cap.release()
    video_writer.release()
    return batch_id


class YOLOPoseVisualizer():
    """
    Visualizes YOLO-based keypoint pose estimation data on video frames and creates an annotated output video.

    This class takes keypoint data (CSV) and overlays it onto the corresponding video using color-coded keypoints
    and optional filtering. The result is saved as a new annotated video, and supports multicore parallel rendering
    for efficient processing of long videos.

    .. seealso::
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
                 palettes: Optional[Union[str, Tuple[str, ...]]] = 'Set1',
                 core_cnt: Optional[int] = -1,
                 threshold: float = 0.0,
                 thickness: Optional[int] = None,
                 circle_size: Optional[int] = None,
                 verbose: Optional[bool] = False,
                 bbox: Optional[bool] = True,
                 skeleton: List[Tuple[str, str]] = None):

        check_file_exist_and_readable(file_path=data_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.data_path, self.video_path = data_path, video_path
        self.video_name = get_fn_ext(filepath=data_path)[1]
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        if circle_size is None:
            circle_size = PlottingMixin().get_optimal_circle_size(frame_size=(self.video_meta_data['width'], self.video_meta_data['height']), circle_frame_ratio=80)
        else:
            check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        if thickness is None:
            thickness = circle_size
        else:
            check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        self.core_cnt = core_cnt
        if core_cnt == -1 or core_cnt > find_core_cnt()[0]: self.core_cnt = find_core_cnt()[0]
        check_if_dir_exists(in_dir=save_dir)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=[bbox], source=f'{self.__class__.__name__} bbox', raise_error=True)
        self.data_df = pd.read_csv(self.data_path, index_col=0)
        check_valid_dataframe(df=self.data_df, source=self.__class__.__name__, required_fields=EXPECTED_COLS)
        self.df_frm_cnt = np.unique(self.data_df[FRAME].values).shape[0]
        self.classes = np.unique(self.data_df[CLASS_NAME].values)
        if isinstance(palettes, str):
            palettes = (palettes,)
        else:
            check_valid_tuple(x=palettes, source=f'{self.__class__.__name__} palettes', minimum_length=1, valid_dtypes=(str,), accepted_values=Options.PALETTE_OPTIONS_CATEGORICAL.value)
        if len(self.classes) != len(palettes):
            raise CountError(msg=f'{len(self.classes)} classes detected in {self.data_path}, but {len(palettes)} color palette names passed: {palettes}', source=self.__class__.__name__)
        self.palettes = {}
        for cnt, palette in enumerate(palettes):
            self.palettes[cnt] = create_color_palette(pallete_name=palette, increments=len(self.data_df.columns)-len(EXPECTED_COLS))
        if skeleton is not None:
            check_valid_lst(data=skeleton, source=f'{self.__class__.__name__} skeleton', valid_dtypes=(list, tuple,), min_len=1, raise_error=True)
            for i in skeleton:
                check_valid_tuple(x=i, source=f'{self.__class__.__name__} {i}', accepted_lengths=(2,), valid_dtypes=(str,))
            required_s_cols = [f'{x}_{suffix}' for xs in skeleton for x in xs for suffix in ('X', 'Y', 'P')]
            missing = [x for x in required_s_cols if x not in self.data_df.columns]
            if len(missing) > 0:
                raise DataHeaderError(msg=f'Columns {missing} missing in file {data_path} as passe by skeleton.', source=self.__class__.__name__)

        self.save_dir, self.verbose, self.palette, self.thickness = save_dir, verbose, palettes, thickness
        self.threshold, self.circle_size, self.thickness, self.bbox = threshold, circle_size, thickness, bbox
        self.video_temp_dir = os.path.join(self.save_dir, self.video_name, "temp")
        self.save_path, self.skeleton = os.path.join(self.save_dir, f'{self.video_name}.mp4'), skeleton
        create_directory(paths=self.video_temp_dir)


    def run(self):
        video_timer = SimbaTimer(start=True)
        if self.video_meta_data['frame_count'] != self.df_frm_cnt:
            raise FrameRangeError(msg=f'The bounding boxes contain data for {self.df_frm_cnt} frames, while the video is {self.video_meta_data["frame_count"]} frames', source=self.__class__.__name__)
        frm_batches = np.array_split(np.array(list(range(0, self.df_frm_cnt))), self.core_cnt)
        frm_batches = [(i, j) for i, j in enumerate(frm_batches)]
        if self.verbose: print(f'Visualizing video {self.video_meta_data["video_name"]} (frame count: {self.video_meta_data["frame_count"]})...')
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(_yolo_keypoint_visualizer,
                                          data=self.data_df,
                                          threshold=self.threshold,
                                          video_path=self.video_path,
                                          save_dir=self.video_temp_dir,
                                          circle_size=self.circle_size,
                                          thickness=self.thickness,
                                          palettes=self.palettes,
                                          bbox=self.bbox,
                                          skeleton=self.skeleton)
            for cnt, result in enumerate(pool.imap(constants, frm_batches, chunksize=1)):
                print(f'Video batch {result+1}/{self.core_cnt} complete...')
        pool.terminate()
        pool.join()
        video_timer.stop_timer()
        concatenate_videos_in_folder(in_folder=self.video_temp_dir, save_path=self.save_path, gpu=True)

        stdout_success(msg=f'YOLO pose video saved at {self.save_path}', source=self.__class__.__name__, elapsed_time=video_timer.elapsed_time_str)


if __name__ == "__main__":
    #video_path = r"D:\cvat_annotations\videos\mp4_20250624155703\s34-drinking.mp4"
    #data_path = r"D:\cvat_annotations\yolo_07032025\out_data\s34-drinking.csv"
    #save_dir = r"D:\cvat_annotations\yolo_07032025\out_video"

    video_path = r"D:\cvat_annotations\videos\mp4_20250624155703\s34-drinking.mp4"
    data_path = r"D:\cvat_annotations\yolo_mdl_07122025\out_data\s34-drinking.csv"
    save_dir = r'D:\cvat_annotations\yolo_mdl_07122025\out_video'
    video_dir = r"D:\cvat_annotations\videos\mp4_20250624155703"
    data_dir = r"D:\cvat_annotations\yolo_mdl_07122025\out_data"

    video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=['.mp4'], as_dict=True)
    data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], as_dict=True)


    skeleton = [('NOSE', 'LEFT_EAR'),
                ('NOSE', 'RIGHT_EAR'),
                ('RIGHT_EAR', 'LEFT_EAR'),
                ('LEFT_EAR', 'LEFT_SIDE'),
                ('RIGHT_EAR', 'RIGHT_SIDE'),
                ('LEFT_SIDE', 'CENTER'),
                ('LEFT_EAR', 'CENTER'),
                ('RIGHT_EAR', 'CENTER'),
                ('CENTER', 'RIGHT_SIDE'),
                ('CENTER', 'TAIL_BASE'),
                ('LEFT_SIDE', 'TAIL_BASE'),
                ('RIGHT_SIDE', 'TAIL_BASE'),
                ('TAIL_BASE', 'TAIL_CENTER'),
                ('TAIL_CENTER', 'TAIL_TIP')]

    for video_name, video_path in video_paths.items():
        data_path = data_paths[video_name]
        #kp_vis = YOLOPoseVisualizer(data_path=data_path, video_path=video_path)

        kp_vis = YOLOPoseVisualizer(data_path= data_path,
                                    video_path=video_path,
                                    save_dir=save_dir,
                                    core_cnt=28,
                                    bbox=True,
                                    verbose=True,
                                    skeleton=skeleton,
                                    threshold=0.5)

        kp_vis.run()

    # video_dir = r'D:\cvat_annotations\videos\mp4_20250624155703'
    # data_dir = r'D:\cvat_annotations\yolo_mdl_07102025\out_csv'
    # save_dir = r'D:\cvat_annotations\yolo_mdl_07102025\out_video'
    #
    # video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=['.mp4'], as_dict=True)
    # data_paths = find_files_of_filetypes_in_directory(directory=data_dir, extensions=['.csv'], as_dict=True)
    #
    # for video_name, data_path in data_paths.items():
    #     kp_vis = YOLOPoseVisualizer(data_path= data_path,
    #                                 video_path=video_paths[video_name],
    #                                 save_dir=save_dir,
    #                                 core_cnt=28,
    #                                 bbox=True,
    #                                 verbose=True)
    #
    #     kp_vis.run()


# if __name__ == "__main__":
#     from simba.utils.read_write import find_files_of_filetypes_in_directory
#
#     video_paths = find_files_of_filetypes_in_directory(directory=r'D:\cvat_annotations\videos', extensions=['.mp4'], as_dict=True)
#     data_paths = find_files_of_filetypes_in_directory(directory=r'D:\cvat_annotations\frames\data_results_yolo11m', extensions=['.csv'], as_dict=True)
#     save_dir = r'D:\cvat_annotations\frames\video_results_yolo11m'
#
#     for video_name, video_path in video_paths.items():
#         data_path = data_paths[video_name]
#         kp_vis = YOLOPoseVisualizer(data_path=data_path,
#                                     video_path=video_path,
#                                     save_dir=save_dir,
#                                     core_cnt=18)
#         kp_vis.run()


# video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
# video_path = "/mnt/c/troubleshooting/mitra/project_folder/videos/502_MA141_Gi_CNO_0514.mp4"
# data_path = r"/mnt/c/troubleshooting/mitra/yolo/results/502_MA141_Gi_CNO_0514.csv"
# save_dir = r'/mnt/c/troubleshooting/mitra/yolo/results/'
# kp_vis = YOLOPoseVisualizer(data_path=data_path,
#                             video_path=video_path,
#                             save_dir=save_dir,
#                             core_cnt=18)
#
#
# kp_vis.run()



