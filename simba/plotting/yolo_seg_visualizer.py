import functools
import multiprocessing
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_valid_boolean,
                                check_valid_dataframe, check_valid_lst,
                                check_valid_tuple)
from simba.utils.data import create_color_palette
from simba.utils.enums import Defaults, Options
from simba.utils.errors import CountError, DataHeaderError, FrameRangeError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (concatenate_videos_in_folder,
                                    create_directory, find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data, read_df,
                                    read_frm_of_video)

FRAME = 'FRAME'
VERTICE = 'VERTICE'
CLASS_ID = 'ID'


def _yolo_seg_visualizer(frm_range: tuple,
                         data: pd.DataFrame,
                         video_path: str,
                         save_dir: str,
                         color: tuple,
                         shape_opacity: 0.5):
    batch_id, frame_rng = frm_range[0], frm_range[1]
    start_frm, end_frm, current_frm = frame_rng[0], frame_rng[-1], frame_rng[0]
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    fourcc, font = cv2.VideoWriter_fourcc(*"mp4v"), cv2.FONT_HERSHEY_DUPLEX
    video_save_path = os.path.join(save_dir, f'{batch_id}.mp4')
    video_writer = cv2.VideoWriter(video_save_path, fourcc, video_meta_data["fps"], (video_meta_data["width"], video_meta_data["height"]))
    while current_frm <= end_frm:
        print(f'Processing frame {current_frm}/{video_meta_data["frame_count"]} (batch: {batch_id}, video: {video_meta_data["video_name"]})...')
        img = read_frm_of_video(video_path=video_path, frame_index=current_frm)
        img_cpy = img.copy()
        if current_frm in data[FRAME].values:
            frm_data = data.loc[data[FRAME] == current_frm]
            frm_data = frm_data.drop([FRAME, CLASS_ID], axis=1).values.astype(np.int32).reshape(-1, 2)
            frm_data = frm_data.reshape(-1, len(frm_data), 2)
            img = cv2.polylines(img, frm_data, True, color, thickness=2, lineType=-1)
            img_cpy = cv2.fillPoly(img_cpy, frm_data, color=color)
        if shape_opacity is not None:
            img = cv2.addWeighted(img_cpy, shape_opacity, img, 1 - shape_opacity, 0, img)
        else:
            img = np.copy(img_cpy)

        video_writer.write(img.astype(np.uint8))
        current_frm += 1
    cap.release()
    video_writer.release()
    return batch_id


class YOLOSegmentationVisualizer():
    """
    Visualizes polygon-based YOLO segmentation results overlaid on video frames.

    .. seealso::
       To run segmentation inference, see :func:`simba.bounding_box_tools.yolo.yolo_seg_inference.YOLOSegmentationInference`
       To **create** YOLO segmentation dataset for fitting, use :func:`simba.third_party_label_appenders.transform.labelme_to_yolo_seg.LabelmeKeypoints2YoloSeg`.
       To fit YOLO model, see `:func:~simba.bounding_box_tools.yolo.yolo_fit.FitYolo`

    :param Union[str, os.PathLike] data_path (): Path to the CSV file with YOLO segmentation output. Must include columns "FRAME", "ID", and at least six "VERTICE" columns.
    :param Union[str, os.PathLike] video_path: Path to the source video file.
    :param Union[str, os.PathLike] save_dir: Directory where output videos (temp and final) are saved.
    :param Tuple[int, int, int] color: RGB color for drawing polygons. Defaults to (255, 255, 0).
    :param Optional[int] core_cnt: Number of parallel processes to use. Defaults to -1 (auto all available cores).
    :param float threshold: Confidence threshold (not currently used in rendering). Defaults to 0.0.
    :param bool verbose: Whether to print detailed progress messages. Defaults to False.
    :param float shape_opacity: Alpha blending factor for filled polygon overlay (range 0.0–1.0). If `None`, solid fill is used. Defaults to 0.5.

    :example:
    runner = YOLOSegmentationVisualizer(data_path=r"D:\platea\platea_videos\videos\yolo_results\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7.csv", video_path=r"D:\platea\platea_videos\videos\clipped\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7.mp4", save_dir=r'D:\platea\platea_videos\videos\yolo_seg_videos', verbose=True)
    runner.run()
    """



    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 color: Tuple[int, int, int] = (255, 255, 0),
                 core_cnt: Optional[int] = -1,
                 threshold: float = 0.0,
                 verbose: Optional[bool] = False,
                 shape_opacity: float = 0.5):


        check_file_exist_and_readable(file_path=data_path)
        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.data_path, self.video_path = data_path, video_path
        self.video_name = get_fn_ext(filepath=data_path)[1]
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} shape_opacity', value=shape_opacity, min_value=0.0, max_value=1.0)
        self.core_cnt = core_cnt
        if core_cnt == -1 or core_cnt > find_core_cnt()[0]: self.core_cnt = find_core_cnt()[0]
        check_if_dir_exists(in_dir=save_dir)
        check_if_valid_rgb_tuple(data=color)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        self.data_df = read_df(file_path=self.data_path, file_type='csv')
        check_valid_dataframe(df=self.data_df, source=self.__class__.__name__, required_fields=[FRAME, CLASS_ID])
        self.vertice_cols = [x for x in self.data_df.columns if VERTICE in x]
        if len(self.vertice_cols) < 6:
            raise CountError(msg=f'The data {data_path} contains less than 6 vertice columns ({len(self.vertice_cols)})', source=self.__class__.__name__)

        self.save_dir, self.verbose, self.shape_opacity = save_dir, verbose, shape_opacity
        self.threshold, self.color = threshold, color
        self.video_temp_dir = os.path.join(self.save_dir, self.video_name, "temp")
        self.save_path = os.path.join(self.save_dir, f'{self.video_name}.mp4')
        create_directory(paths=self.video_temp_dir)

    def run(self):
        video_timer = SimbaTimer(start=True)
        video_frms = list(range(0, self.video_meta_data['frame_count']))
        missing_frms = [x for x in list(self.data_df['FRAME']) if x not in video_frms]
        if len(missing_frms) > 0:
            raise FrameRangeError(msg=f'The data {self.data_path} contains frames that are outside the scope of the video: {missing_frms}', source=self.__class__.__name__)
        frm_batches = np.array_split(video_frms, self.core_cnt)
        frm_batches = [(i, j) for i, j in enumerate(frm_batches)]
        if self.verbose: print(f'Visualizing video {self.video_meta_data["video_name"]} (frame count: {self.video_meta_data["frame_count"]})...')
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(_yolo_seg_visualizer,
                                          data=self.data_df,
                                          video_path=self.video_path,
                                          save_dir=self.video_temp_dir,
                                          color=self.color,
                                          shape_opacity=self.shape_opacity)
            for cnt, result in enumerate(pool.imap(constants, frm_batches, chunksize=1)):
                print(f'Video batch {result+1}/{self.core_cnt} complete...')
        pool.terminate()
        pool.join()
        video_timer.stop_timer()
        concatenate_videos_in_folder(in_folder=self.video_temp_dir, save_path=self.save_path, gpu=True)
        stdout_success(msg=f'YOLO pose video saved at {self.save_path}', source=self.__class__.__name__, elapsed_time=video_timer.elapsed_time_str)


# if __name__ == "__main__":
#
#     runner = YOLOSegmentationVisualizer(data_path=r"D:\platea\platea_videos\videos\yolo_results\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7.csv",
#                                         video_path=r"D:\platea\platea_videos\videos\clipped\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7.mp4",
#                                         save_dir=r'D:\platea\platea_videos\videos\yolo_seg_videos',
#                                         verbose=True)
#     runner.run()