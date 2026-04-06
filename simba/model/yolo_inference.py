__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal
try:
    import torch
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None
    torch = None

import numpy as np
import pandas as pd

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.third_party_label_appenders.converters import \
    yolo_obb_data_to_bounding_box
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_instance, check_int,
                                check_str, check_valid_boolean,
                                check_valid_lst, check_valid_tuple, get_fn_ext)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.enums import Formats, Options
from simba.utils.errors import (InvalidVideoFileError, SimBAGPUError,
                                SimBAPAckageVersionError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_video_meta_data)
from simba.utils.yolo import (_get_undetected_obs, apply_fixed_bbox_size,
                              check_valid_device, load_yolo_model,
                              yolo_predict)

COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
SMOOTHING_METHODS = ('savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential')
NEAREST = 'nearest'
SAVITZKY_GOLAY, CONFIDENCE = 'savitzky-golay', 'CONFIDENCE'

class YoloInference():
    """
    Performs object detection inference on a video using a YOLO model.

    YOLO-based object detection (bounding-box) on one or more video files. It supports GPU acceleration,
    batch processing, streaming, and optional result saving. The model returns bounding box coordinates and
    class confidence scores for each frame. Results can be smoothed or interpolated to handle detection gaps.

    .. seealso::
       To perform bounding box and **keypoint (pose) detection**, see :func:`~simba.bounding_box_tools.yolo.yolo_pose_inference.YOLOPoseInference`.
       To perform keypoint (pose) detection with tracking, see :func:`~simba.model.yolo_pose_track_inference.YOLOPoseTrackInference`
       To visualize bounding boxes only, see :func:`~simba.plotting.yolo_visualize.YOLOVisualizer`

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../docs/tables/YoloInference.csv
       :widths: 10, 10, 40, 40
       :align: center
       :header-rows: 1

    .. video:: _static/img/YoloInference_1.webm
       :width: 500
       :loop:
       :autoplay:
       :muted:
       :align: center

    .. video:: _static/img/YoloInference_2.webm
       :width: 500
       :loop:
       :autoplay:
       :muted:
       :align: center

    :param Union[str, os.PathLike, YOLO] weights: Path to YOLO model weights or a preloaded ``ultralytics.YOLO`` model instance.
    :param Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]] video_path: Input video path, list of paths, or directory containing videos.
    :param Optional[bool] verbose: If True, print progress information.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save output CSV files. If None, results are returned in-memory.
    :param Optional[bool] half_precision: If True, run inference in fp16 where supported.
    :param Union[Literal['cpu'], int] device: Inference device ('cpu' or CUDA index).
    :param Optional[int] batch_size: Number of frames per prediction batch.
    :param int core_cnt: CPU thread count used by torch.
    :param float threshold: Detection confidence threshold in [0.0, 1.0].
    :param int max_detections: Maximum detections per frame.
    :param Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] smoothing_method: Optional temporal smoothing method for bbox coordinates.
    :param Optional[int] smoothing_time_window: Smoothing window in milliseconds. Used only when ``smoothing_method`` is not None.
    :param bool interpolate: If True, interpolate missing bbox coordinates (nearest, per class).
    :param int imgsz: Model inference image size.
    :param Optional[Tuple[int, int]] bbox_size: Optional fixed bbox size ``(height, width)`` in pixels applied to detected boxes.
    :param Optional[bool] stream: If True, use streaming predictions.
    :return: If ``save_dir`` is None, returns a dict mapping video name to result dataframe. Otherwise saves CSVs and returns None.
    :rtype: Union[None, Dict[str, pd.DataFrame]]

    :example:
    >>> video_path = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
    >>> i = YoloInference(
    ...     weights=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt",
    ...     video_path=video_path,
    ...     save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results",
    ...     verbose=True,
    ...     device=0,
    ...     interpolate=True,
    ...     bbox_size=(128, 128)
    ... )
    >>> i.run()
    """
    def __init__(self,
                 weights: Union[str, os.PathLike, YOLO],
                 video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                 verbose: Optional[bool] = False,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 half_precision: Optional[bool] = True,
                 device: Union[Literal['cpu'], int] = 0,
                 batch_size: Optional[int] = 400,
                 core_cnt: int = 8,
                 threshold: float = 0.25,
                 max_detections: int = 300,
                 smoothing_method: Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] = None,
                 smoothing_time_window: Optional[int] = None,
                 interpolate: bool = False,
                 imgsz: int = 320,
                 bbox_size: Optional[Tuple[int, int]] = None,
                 stream: Optional[bool] = True) -> Union[None, Dict[str, pd.DataFrame]]:

        if not _is_cuda_available()[0]:
            raise SimBAGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        if YOLO is None:
            raise SimBAPAckageVersionError(msg='ultralytics.YOLO package not detected.', source=self.__class__.__name__)
        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif os.path.isfile(video_path):
            check_file_exist_and_readable(file_path=video_path)
            video_path = [video_path]
        elif os.path.isdir(video_path):
            video_path = find_files_of_filetypes_in_directory(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_warning=False, raise_error=True, as_dict=False)
        else:
            raise InvalidVideoFileError(msg=f'{video_path} is not a valid video path or directory path', source=self.__class__.__name__)
        for i in video_path:
            _ = get_video_meta_data(video_path=i)
        check_instance(source=f'{self.__class__.__name__} weights', instance=weights, accepted_types=(str, os.PathLike, YOLO))
        if not isinstance(weights, YOLO):
            check_file_exist_and_readable(file_path=weights)
            self.model = load_yolo_model(weights_path=weights, verbose=verbose, device=device)
        else:
            self.model = weights
        check_valid_boolean(value=[half_precision, verbose, stream, interpolate], source=self.__class__.__name__)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1)
        
        check_valid_device(device=device)
        if bbox_size is not None:
            check_valid_tuple(x=bbox_size, source=f'{self.__class__.__name__} bbox_size', accepted_lengths=(2,), valid_dtypes=Formats.INTEGER_DTYPES.value, min_integer=1, raise_error=True)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        if smoothing_method is not None:
            check_str(name=f'{self.__class__.__name__} smoothing', value=smoothing_method, options=SMOOTHING_METHODS)
            check_float(name=f'{self.__class__.__name__} smoothing_time_window', value=smoothing_time_window, min_value=10e-6)
        torch.set_num_threads(core_cnt)
        self.video_path, self.half_precision, self.stream, self.batch_size = video_path, half_precision, stream, batch_size
        self.interpolate, self.smoothing_method, self.smoothing_time_window = interpolate, smoothing_method, smoothing_time_window
        self.save_dir, self.verbose, self.imgsz, self.core_cnt, self.device = save_dir, verbose, imgsz, core_cnt,device
        self.threshold, self.max_detections, self.bbox_size = threshold, max_detections, bbox_size

    def run(self):
        class_dict = self.model.names
        results, fps_speeds = {}, []
        timer = SimbaTimer(start=True)
        for path in self.video_path:
            _, video_name, _ = get_fn_ext(filepath=path)
            video_meta_data = get_video_meta_data(video_path=path)
            video_out, video_timer = [], SimbaTimer(start=True)
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
                    cls_data = cls_data.reshape(-1, boxes.shape[1])[np.argmax(cls_data.reshape(-1, boxes.shape[1])[:, -2].flatten())]
                    if video_prediction.obb is not None:
                        box = yolo_obb_data_to_bounding_box(center_x=cls_data[0], center_y=cls_data[1], width=cls_data[2], height=cls_data[3], angle=cls_data[4]).flatten()
                    else:
                        box = np.array([cls_data[0], cls_data[1], cls_data[2], cls_data[1], cls_data[2], cls_data[3], cls_data[0], cls_data[3]]).astype(np.int32)
                    video_out.append([frm_cnt, cls_data[-1], class_dict[cls_data[-1]], cls_data[-2]] + list(box))
            results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
            results[video_name]["CLASS_ID"] = (pd.to_numeric(results[video_name]["CLASS_ID"], errors="coerce").fillna(-1).astype(np.int32))
            if self.interpolate:
                for class_id in class_dict.keys():
                    class_df = results[video_name][results[video_name]["CLASS_ID"] == int(class_id)].copy()
                    if class_df.empty: continue
                    for cord_col in COORD_COLS:
                        class_df[cord_col] = pd.to_numeric(class_df[cord_col], errors="coerce").astype(np.float32)
                        class_df[cord_col] = class_df[cord_col].replace(-1, np.nan)
                        class_df[cord_col] = (class_df[cord_col].interpolate(method='linear', axis=0).ffill().bfill().replace([np.inf, -np.inf], np.nan).round().fillna(-1).astype(np.int32))
                    results[video_name].update(class_df)
                results[video_name][CONFIDENCE] = 0
            if self.smoothing_method is not None:
                if self.smoothing_method != SAVITZKY_GOLAY:
                    smoothened = df_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__, method=self.smoothing_method)
                else:
                    smoothened = savgol_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__)
                results[video_name].update(smoothened)
            if self.bbox_size is not None:
                results[video_name] = apply_fixed_bbox_size(data=results[video_name], video_name=video_name, img_w=int(video_meta_data["width"]), img_h=int(video_meta_data["height"]), bbox_size=self.bbox_size)
            video_timer.stop_timer()
            fps_speeds.append(len(results[video_name]) / video_timer.elapsed_time)

        if self.save_dir:
            for k, v in results.items():
                save_path = os.path.join(self.save_dir, f'{k}.csv')
                v.to_csv(save_path)
        timer.stop_timer()
        if self.verbose:
            if self.save_dir:
                stdout_success(f'YOLO results for {len(self.video_path)} video(s) saved in {self.save_dir} directory', elapsed_time=timer.elapsed_time_str)
            else:
                stdout_success(f'YOLO results created for {len(self.video_path)} video(s)', elapsed_time=timer.elapsed_time_str)
        if not self.save_dir:
            return results







# VIDEO_PATH = r"E:\open_video\open_field_2\sample\clips"
# WEIGHTS_PATH = r'E:\open_video\open_field_2\yolo_bbox_project\mdl\train2\weights\best.pt'
# SAVE_DIR = r'E:\open_video\open_field_2\yolo_bbox_project\results'
# i = YoloInference(weights=WEIGHTS_PATH,
#                   video_path=VIDEO_PATH,
#                   save_dir=SAVE_DIR,
#                   stream=True,
#                   threshold=0.10,
#                   verbose=True,
#                   core_cnt=18,
#                   imgsz=256,
#                   bbox_size=None,#(512, 512),
#                   interpolate=True,
#                   batch_size=8)
# i.run()
#


# video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
# video_path = "/mnt/d/netholabs/yolo_videos/input/mp4_20250606083508/2025-05-28_19-50-23.mp4"
# video_path = r"E:\maplight_videos\yolo_runtime_test\videos"
# weights_path = r"E:\maplight_videos\yolo_mdl\mdl\train\weights\best.pt"
# save_dir = r"E:\maplight_videos\yolo_mdl\mdl\results"
# i = YoloInference(weights=weights_path,
#                   video_path=video_path,
#                   save_dir=save_dir,
#                   stream=True,
#                   verbose=True,
#                   core_cnt=18,
#                   imgsz=256,
#                   batch_size=500)
# i.run()