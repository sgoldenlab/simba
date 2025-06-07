
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from simba.bounding_box_tools.yolo.utils import load_yolo_model
from simba.third_party_label_appenders.converters import \
    yolo_obb_data_to_bounding_box
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_array, check_valid_boolean,
                                check_valid_lst, check_valid_tuple, get_fn_ext)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import find_core_cnt, get_video_meta_data

COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
SMOOTHING_METHODS = ('savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential')
NEAREST = 'nearest'

class YoloInference():
    """
    Performs object detection inference on a video using a YOLO model.

    This function runs YOLO-based object detection on a given video file, optionally utilizing GPU acceleration for
    inference, and either returns the results or saves them to a specified directory. The function outputs bounding box
    coordinates and class confidence scores for detected objects in each frame of the video.

    :param Union[str, os.PathLike] weights_path: Path to the YOLO model weights file.
    :param Union[str, os.PathLike] video_path: Path to the input video file for performing inference.
    :param Optional[bool] verbose: If True, outputs progress information and timing. Defaults to False.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the inference results as CSV files. If not provided, results are returned as a dictionary. Defaults to None.
    :param Optional[bool] gpu: If True, performs inference on the GPU. Defaults to False.
    :param Optional[bool] stream: If True, iterate over a generater. Default to True. Recommended on longer videos.
    :param Optional[bool] batch_size: Number of frames to process in parallel. Default to 4.
    :param Optional[bool] interpolate: If True, interpolates missing bounding boxes using ``nearest`` method.

    :example:
    >>> video_path = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
    >>> i = YoloInference(weights_path=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt", video_path=video_path, save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results", verbose=True, gpu=True)
    >>> i.run()
    """
    def __init__(self,
                 weights_path: Union[str, os.PathLike],
                 video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                 verbose: Optional[bool] = False,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 gpu: Optional[bool] = False,
                 half_precision: Optional[bool] = True,
                 batch_size: Optional[int] = 4,
                 core_cnt: int = 8,
                 smoothing_method: Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] = None,
                 smoothing_time_window: Optional[int] = None,
                 interpolate: bool = False,
                 stream: Optional[bool] = True) -> Union[None, Dict[str, pd.DataFrame]]:

        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif isinstance(video_path, str):
            check_file_exist_and_readable(file_path=video_path)
            video_path = [video_path]
        for i in video_path:
            _ = get_video_meta_data(video_path=i)
        check_file_exist_and_readable(file_path=weights_path)
        check_valid_boolean(value=[half_precision, gpu, verbose, stream, interpolate], source=self.__class__.__name__)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        if smoothing_method is not None:
            check_str(name=f'{self.__class__.__name__} smoothing', value=smoothing_method, options=SMOOTHING_METHODS)
            check_float(name=f'{self.__class__.__name__} smoothing_time_window', value=smoothing_time_window, min_value=10e-6)
        torch.set_num_threads(core_cnt)
        self.model = load_yolo_model(weights_path=weights_path, verbose=verbose)
        self.video_path, self.half_precision, self.stream, self.batch_size = video_path, half_precision, stream, batch_size
        self.interpolate, self.smoothing_method, self.smoothing_time_window = interpolate, smoothing_method, smoothing_time_window
        self.save_dir, self.verbose = save_dir, verbose

    def run(self):
        class_dict = self.model.names
        results = {}
        timer = SimbaTimer(start=True)
        for path in self.video_path:
            _, video_name, _ = get_fn_ext(filepath=path)
            video_meta_data = get_video_meta_data(video_path=path)
            video_out = []
            video_predictions = self.model.predict(source=path, half=self.half_precision, batch=self.batch_size, stream=self.stream)
            for frm_cnt, video_prediction in enumerate(video_predictions):
                if video_prediction.obb is not None:
                    boxes = np.array(video_prediction.obb.data.cpu()).astype(np.float32)
                else:
                    boxes = np.array(video_prediction.boxes.data.cpu()).astype(np.float32)
                for c in list(class_dict.keys()):
                    cls_data = boxes[np.argwhere(boxes[:, -1] == c)]
                    if cls_data.shape[0] == 0:
                        video_out.append(np.array([frm_cnt, c, class_dict[c], -1, -1, -1, -1, -1, -1, -1, -1, -1]))
                    else:
                        cls_data = cls_data.reshape(-1, boxes.shape[1])[np.argmax(boxes[:, -2].flatten())]
                        if video_prediction.obb is not None:
                            box = yolo_obb_data_to_bounding_box(center_x=cls_data[0], center_y=cls_data[1], width=cls_data[2], height=cls_data[3], angle=cls_data[4]).flatten()
                        else:
                            box = np.array([cls_data[0], cls_data[1], cls_data[2], cls_data[1], cls_data[2], cls_data[3], cls_data[0], cls_data[3]]).astype(np.int32)
                        video_out.append([frm_cnt, cls_data[-1], class_dict[cls_data[-1]], cls_data[-2]] + list(box))
            results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
            if self.interpolate:
                for cord_col in COORD_COLS:
                    results[video_name][cord_col] = results[video_name][cord_col].astype(np.int32).replace(to_replace=-1, value=np.nan)
                    results[video_name][cord_col] = results[video_name][cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()
            if self.smoothing_method is not None:
                if self.smoothing_method != 'savitzky-golay':
                    smoothened = df_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__, method=self.smoothing_method)
                else:
                    smoothened = savgol_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__)
                results[video_name].update(smoothened)
        timer.stop_timer()
        if not self.save_dir:
            if self.verbose:
                stdout_success(f'YOLO results created', elapsed_time=timer.elapsed_time_str)
            return results
        else:
            for k, v in results.items():
                save_path = os.path.join(self.save_dir, f'{k}.csv')
                v.to_csv(save_path)
            if self.verbose:
                stdout_success(f'YOLO results saved in {self.save_dir} directory', elapsed_time=timer.elapsed_time_str)

#video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
# #video_path = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
# i = YoloInference(weights_path=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt",
#                   video_path=video_path, save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results",
#                   verbose=True,
#                   gpu=True)
# i.run()