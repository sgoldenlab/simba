
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

from simba.bounding_box_tools.yolo.utils import (_get_undetected_obs,
                                                 check_valid_device,
                                                 load_yolo_model, yolo_predict)
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

    This class performs YOLO-based object detection on one or more video files. It supports GPU acceleration,
    batch processing, streaming, and optional result saving. The model returns bounding box coordinates and
    class confidence scores for each frame. Results can be smoothed or interpolated to handle detection gaps.

    .. seealso::
       To perform bounding box and keypoint (pose) detection, see :func:`~simba.bounding_box_tools.yolo.yolo_pose_inference.YOLOPoseInference`

    :param Union[str, os.PathLike] weights_path: Path to the YOLO model weights file.
    :param Union[str, os.PathLike] or List[Union[str, os.PathLike]] video_path: Path(s) to the input video file(s) for performing inference.
    :param Optional[bool] verbose: If True, outputs progress information and timing. Defaults to False.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the inference results (e.g., as CSV). If None, results are returned. Defaults to None.
    :param Optional[bool] half_precision: If True, uses half precision (fp16) for inference. Reduces memory usage and speeds up inference on supported devices. Defaults to True.
    :param Union[Literal['cpu'], int] device: Device to use for inference. Use 'cpu' for CPU or GPU index (e.g., 0 for CUDA:0). Defaults to 0.
    :param Optional[int] batch_size: Number of frames to process in parallel. Defaults to 4.
    :param int core_cnt: Number of CPU cores to use for preprocessing or parallel work. Defaults to 8.
    :param float threshold: Confidence threshold for object detections. Detections below this value are discarded. Defaults to 0.25.
    :param int max_detections: Maximum number of detections per frame. Defaults to 300.
    :param Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] smoothing_method: Optional smoothing method to apply to bounding box sequences over time to reduce jitter.
    :param Optional[int] smoothing_time_window: Number of frames to consider for temporal smoothing. Effective only if smoothing_method is specified.
    :param bool interpolate: If True, interpolates missing bounding boxes using the 'nearest' method. Useful for filling detection gaps.
    :param int imgsz: Input image size for inference. Must be a square value. Defaults to 640.
    :param Optional[bool] stream: If True, performs inference using a generator that yields predictions iteratively, reducing memory usage. Defaults to True.
    :return: If `save_dir` is None, returns a dictionary of results keyed by video file names. Otherwise, results are saved to `save_dir`.
    :rtype: Union[None, Dict[str, pd.DataFrame]]

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
                 half_precision: Optional[bool] = True,
                 device: Union[Literal['cpu'], int] = 0,
                 batch_size: Optional[int] = 4,
                 core_cnt: int = 8,
                 threshold: float = 0.25,
                 max_detections: int = 300,
                 smoothing_method: Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] = None,
                 smoothing_time_window: Optional[int] = None,
                 interpolate: bool = False,
                 imgsz: int = 640,
                 stream: Optional[bool] = True) -> Union[None, Dict[str, pd.DataFrame]]:

        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif isinstance(video_path, str):
            check_file_exist_and_readable(file_path=video_path)
            video_path = [video_path]
        for i in video_path:
            _ = get_video_meta_data(video_path=i)
        check_file_exist_and_readable(file_path=weights_path)
        check_valid_boolean(value=[half_precision, verbose, stream, interpolate], source=self.__class__.__name__)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} max_detections', value=max_detections, min_value=1)
        check_valid_device(device=device)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        if smoothing_method is not None:
            check_str(name=f'{self.__class__.__name__} smoothing', value=smoothing_method, options=SMOOTHING_METHODS)
            check_float(name=f'{self.__class__.__name__} smoothing_time_window', value=smoothing_time_window, min_value=10e-6)
        torch.set_num_threads(core_cnt)
        self.model = load_yolo_model(weights_path=weights_path, verbose=verbose, device=device)
        self.video_path, self.half_precision, self.stream, self.batch_size = video_path, half_precision, stream, batch_size
        self.interpolate, self.smoothing_method, self.smoothing_time_window = interpolate, smoothing_method, smoothing_time_window
        self.save_dir, self.verbose, self.imgsz, self.core_cnt, self.device = save_dir, verbose, imgsz, core_cnt,device
        self.threshold, self.max_detections = threshold, max_detections

    def run(self):
        class_dict = self.model.names
        results = {}
        timer = SimbaTimer(start=True)
        for path in self.video_path:
            _, video_name, _ = get_fn_ext(filepath=path)
            video_meta_data = get_video_meta_data(video_path=path)
            video_out = []
            video_predictions = yolo_predict(model=self.model, source=path, half=self.half_precision, batch_size=self.batch_size, stream=self.stream, imgsz=self.imgsz, device=self.device, threshold=self.threshold, max_detections=self.max_detections, verbose=self.verbose)
            for frm_cnt, video_prediction in enumerate(video_predictions):
                boxes = video_prediction.obb.data if video_prediction.obb is not None else video_prediction.boxes.data
                boxes = boxes.cpu().numpy().astype(np.float32)
                detected_classes = np.unique(boxes[:, -1]).astype(int) if boxes.size > 0 else []
                for class_id, class_name in class_dict.items():
                    if class_id not in detected_classes:
                        video_out.append(_get_undetected_obs(frm_id=frm_cnt, class_id=class_id, class_name=class_name, value_cnt=9))
                        continue
                    cls_data = boxes[np.argwhere(boxes[:, -1] == class_id)]
                    cls_data = cls_data.reshape(-1, boxes.shape[1])[np.argmax(boxes[:, -2].flatten())]
                    if video_prediction.obb is not None:
                        box = yolo_obb_data_to_bounding_box(center_x=cls_data[0], center_y=cls_data[1], width=cls_data[2], height=cls_data[3], angle=cls_data[4]).flatten()
                    else:
                        box = np.array([cls_data[0], cls_data[1], cls_data[2], cls_data[1], cls_data[2], cls_data[3], cls_data[0], cls_data[3]]).astype(np.int32)
                    video_out.append([frm_cnt, cls_data[-1], class_dict[cls_data[-1]], cls_data[-2]] + list(box))
            results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
            if self.interpolate:
                for cord_col in COORD_COLS:
                    results[video_name][cord_col] = results[video_name][cord_col].astype(np.float32).astype(np.int32).replace(to_replace=-1, value=np.nan)
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

# video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
# # video_path = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
# # video_path = r"D:\cvat_annotations\videos\s24-eating.mp4"
# weights_path = r"D:\cvat_annotations\yolo_07032025\bbox_annot\mdl\train10\weights\best.pt"
# save_dir = r"D:\cvat_annotations\yolo_07032025\out_data"
# i = YoloInference(weights_path=weights_path,
#                   video_path=video_path,
#                   save_dir=save_dir,
#                   stream=True,
#                   verbose=True)
# i.run()