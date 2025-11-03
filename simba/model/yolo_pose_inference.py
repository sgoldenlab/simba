import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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

import random
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_instance, check_int,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple, get_fn_ext)
from simba.utils.enums import Options
from simba.utils.errors import (CountError, InvalidFilepathError,
                                InvalidFileTypeError, SimBAGPUError,
                                SimBAPAckageVersionError)
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_video_meta_data, recursive_file_search)
from simba.utils.warnings import FileExistWarning, NoDataFoundWarning
from simba.utils.yolo import (_get_undetected_obs, filter_yolo_keypoint_data,
                              load_yolo_model, yolo_predict)

OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
NEAREST, CLASS_ID, CONFIDENCE, FRAME  = 'nearest', 'CLASS_ID', 'CONFIDENCE', 'FRAME'


class YOLOPoseInference():

    """
    YOLOPoseInference performs pose estimation on videos using a YOLO-based keypoint detection model.

    This class runs YOLO-based keypoint detection on a given video or list of videos. It supports GPU acceleration,
    batch or stream-based inference, result interpolation, and saving results to disk. The model returns detected
    keypoints and their confidence scores for each frame, and optionally tracks poses over time.

    .. seealso::
       For bounding box inference only (no pose), see :func:`~simba.bounding_box_tools.yolo.yolo_inference.YoloInference`.
       For segmentation inference, see :func:`~simba.bounding_box_tools.yolo.yolo_seg_inference.YOLOSegmentationInference`.

    :param Union[str, os.PathLike] weights: Path to the trained YOLO model weights (e.g., 'best.pt').
    :param Union[str, os.PathLike] or List[Union[str, os.PathLike]] video_path: Path to a single video, list of videos, or directory containing video files.
    :param Tuple[str, ...] keypoint_names: Tuple containing the names of keypoints to be tracked (e.g., ('nose', 'left_ear', ...)). If None, ('BP_0', 'BP_1', ...) will be used.
    :param Optional[bool] verbose: If True, outputs progress information and timing. Defaults to True.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the inference results. If None, results are returned in memory. Defaults to None.
    :param Union[Literal['cpu'], int] device: Device to use for inference. Use 'cpu' for CPU or GPU index (e.g., 0 for CUDA:0). Defaults to 0.
    :param Optional[str] format: Optional export format for the model. Supported values: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite". Defaults to None.
    :param Optional[int] batch_size: Number of frames to process in parallel. Defaults to 4.
    :param int torch_threads: Number of PyTorch threads to use. Defaults to 8.
    :param bool half_precision: If True, uses half-precision (FP16) inference. Defaults to True.
    :param bool stream: If True, processes frames one-by-one in a generator style. Recommended for long videos. Defaults to False.
    :param float box_threshold: Confidence threshold bounding box detection. All detections (bounding boxes AND keypoints) below this value are ignored. Defaults to 0.5.
    :param Optional[int] max_tracks: Maximum number (total sum) of pose tracks to keep. If None, all tracks are retained.
    :param Optional[int] max_per_class: Maximum number pose tracks per class. E.g., if one 'resident' and one 'intruder' is expecte, set this to 1. Defaults to None meaning all detected instances of each class are retained.
    :param bool interpolate: If True, interpolates missing keypoints across frames using the 'nearest' method. Defaults to False.
    :param bool smoothing: If not None, then the time in milliseconds for Gaussian-applied body-part smoothing.
    :param bool overwrite: If True, overwrites the data at the ``save_dir``. If False, skips the file if it exists.
    :param bool raise_error: If True, raise error if the input video metadata can't be read. If False, then skips the video file.
    :param bool randomize_order: If True, analyzes the input data in a random order. If False, then in fixed order.
    :param bool recursive: If True, analyzes all video files found recursively in the video_path directory. If False, only looks in the top directory.
    :param int imgsz: Input image size for inference. Must be square. Defaults to 640.
    """

    def __init__(self,
                 weights: Union[str, os.PathLike, YOLO],
                 video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                 keypoint_names: Optional[Tuple[str, ...]] = None,
                 verbose: Optional[bool] = True,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 device: Union[Literal['cpu'], int] = 0,
                 format: Optional[str] = None,
                 batch_size: Optional[int] = 4,
                 torch_threads: int = 8,
                 half_precision: bool = True,
                 stream: bool = False,
                 box_threshold: float = 0.5,
                 max_tracks: Optional[int] = None,
                 max_per_class: Optional[int] = None,
                 interpolate: bool = False,
                 smoothing: Optional[int] = None,
                 imgsz: int = 640,
                 iou: float = 0.5,
                 overwrite: bool = True,
                 raise_error: bool = True,
                 randomize_order: bool = False,
                 recursive: bool = False):

        gpu_available, gpus = _is_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg='No GPU detected.', source=self.__class__.__name__)
        else:
            print(f'GPUS AVAILABLE: {gpus}')
        if YOLO is None:
            raise SimBAPAckageVersionError(msg='ultralytics.YOLO package not detected.', source=self.__class__.__name__)
        check_valid_boolean(value=raise_error, source=f'{self.__class__.__name__} raise_error')
        check_valid_boolean(value=recursive, source=f'{self.__class__.__name__} recursive')
        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif os.path.isfile(video_path):
            video_path = [video_path]
        elif os.path.isdir(video_path):
            if not recursive:
                video_path = find_files_of_filetypes_in_directory(directory=video_path, extensions=list(Options.ALL_VIDEO_FORMAT_OPTIONS.value), as_dict=False)
            else:
                video_path = recursive_file_search(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=False)
        for i in video_path:
            _ = get_video_meta_data(video_path=i, raise_error=raise_error)
        check_instance(source=f'{self.__class__.__name__} weights', instance=weights, accepted_types=(str, os.PathLike, YOLO))
        if not isinstance(weights, YOLO):
            check_file_exist_and_readable(file_path=weights)
            self.model = load_yolo_model(weights_path=weights, verbose=verbose, device=device, format=format)
        else:
            self.model = weights
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=interpolate, source=f'{self.__class__.__name__} interpolate')
        check_valid_boolean(value=overwrite, source=f'{self.__class__.__name__} overwrite')
        check_valid_boolean(value=randomize_order, source=f'{self.__class__.__name__} randomize_order')
        if randomize_order:
            random.shuffle(video_path)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        if max_per_class is not None:
            check_int(name=f'{self.__class__.__name__} max_per_class', value=max_per_class, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=box_threshold, min_value=10e-6, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} iou', value=iou, min_value=10e-6, max_value=1.0)
        if keypoint_names is not None:
            check_valid_tuple(x=keypoint_names, source=f'{self.__class__.__name__} keypoint_names', min_integer=1, valid_dtypes=(str,))
            if self.model.model.kpt_shape[0] != len(keypoint_names):
                raise CountError(msg=f'The YOLO model expects {self.model.model.model.head.kpt_shape[0]} keypoints but you passed {len(keypoint_names)}: {keypoint_names}', source=self.__class__.__name__)
        else:
            keypoint_names = [f'BP_{x+1}' for x in range(self.model.model.kpt_shape[0])]
        if max_tracks is not None:
            check_int(name=f'{self.__class__.__name__} max_tracks', value=max_tracks, min_value=1)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        if smoothing is not None and smoothing is not False:
            check_int(name=f'{self.__class__.__name__} smoothing', value=smoothing, min_value=0, raise_error=True)
            smoothing = smoothing if smoothing > 0 else False
        self.keypoint_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y', 'p']]
        self.keypoint_cord_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y']]
        OUT_COLS.extend(self.keypoint_col_names)
        COORD_COLS.extend(self.keypoint_cord_col_names)
        torch.set_num_threads(torch_threads)
        self.half_precision, self.stream, self.video_path, self.raise_error, self.smoothing = half_precision, stream, video_path, raise_error, smoothing
        self.device, self.batch_size, self.threshold, self.max_tracks, self.iou = device, batch_size, box_threshold, max_tracks, iou
        self.verbose, self.save_dir, self.imgsz, self.interpolate, self.overwrite, self.max_per_class = verbose, save_dir, imgsz, interpolate, overwrite, max_per_class
        if self.model.model.task != 'pose':
            raise InvalidFileTypeError(msg=f'The model {weights} is not a pose model. It is a {self.model.model.task} model', source=self.__class__.__name__)

    def run(self):
        results = {}
        class_dict = self.model.names
        timer = SimbaTimer(start=True)
        for video_cnt, path in enumerate(self.video_path):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=path)
            if self.save_dir:
                save_path = os.path.join(self.save_dir, f'{video_name}.csv')
                if not self.overwrite and os.path.isfile(save_path):
                    FileExistWarning(msg=f'Skipping video {video_name} (already exist and overwrite equals False) ....')
                    continue
            video_meta = get_video_meta_data(video_path=path, raise_error=self.raise_error)
            if video_meta is None and not self.raise_error:
                NoDataFoundWarning(msg=f'Skipping video {video_name} (could not read video meta data)....')
                continue
            video_out = []
            video_predictions = yolo_predict(model=self.model,
                                             source=path,
                                             half=self.half_precision,
                                             batch_size=self.batch_size,
                                             stream=self.stream,
                                             imgsz=self.imgsz,
                                             device=self.device,
                                             threshold=self.threshold,
                                             max_detections=self.max_tracks,
                                             verbose=self.verbose,
                                             iou=self.iou)
            for frm_cnt, video_prediction in enumerate(video_predictions):
                boxes = video_prediction.obb.data if video_prediction.obb is not None else video_prediction.boxes.data
                boxes = boxes.cpu().numpy().astype(np.float32)
                keypoints = video_prediction.keypoints.data.cpu().numpy().astype(np.float32)
                detected_classes = np.unique(boxes[:, -1]).astype(np.int32) if boxes.size > 0 else []
                for class_id, class_name in class_dict.items():
                    if class_id not in detected_classes:
                        video_out.append(_get_undetected_obs(frm_id=frm_cnt, class_id=class_id, class_name=class_name, value_cnt=(9 + (len(self.keypoint_col_names)))))
                        continue
                    cls_boxes, cls_keypoints = filter_yolo_keypoint_data(bbox_data=boxes, keypoint_data=keypoints, class_id=class_id, confidence=None, class_idx=-1, confidence_idx=None)
                    if self.max_per_class is not None:
                        cls_boxes = cls_boxes[:self.max_per_class, :]
                    for i in range(cls_boxes.shape[0]):
                        box = np.array([cls_boxes[i][0], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][3], cls_boxes[i][0], cls_boxes[i][3]]).astype(np.int32)
                        bbox = np.array([frm_cnt, cls_boxes[i][-1], class_dict[cls_boxes[i][-1]], cls_boxes[i][-2]] + list(box))
                        bbox = np.append(bbox, cls_keypoints[i].flatten())
                        video_out.append(bbox)

            results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
            results[video_name][FRAME] = results[video_name][FRAME].astype(np.int64)
            results[video_name].loc[:, CLASS_ID] = (pd.to_numeric(results[video_name][CLASS_ID], errors='coerce').fillna(0).astype(np.int32))
            if self.interpolate:
                for class_id in class_dict.keys():
                    class_df = results[video_name][results[video_name][CLASS_ID] == int(class_id)]
                    for cord_col in COORD_COLS:
                        class_df[cord_col] = class_df[cord_col].astype(np.float32).replace([-1, 0], np.nan)
                        class_df[cord_col] = class_df[cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()
                    class_df[CONFIDENCE] = class_df[CONFIDENCE].astype(np.float32).replace([-1, 0], np.nan)
                    class_df[CONFIDENCE] = class_df[CONFIDENCE].interpolate(method=NEAREST, axis=0).ffill().bfill()
                    results[video_name].update(class_df)
            if self.smoothing:
                frms_in_smoothing_window = int(self.smoothing / (1000 / video_meta['fps']))
                if frms_in_smoothing_window > 1:
                    for class_id in class_dict.keys():
                        class_df = results[video_name][results[video_name][CLASS_ID] == int(class_id)]
                        for cord_col in COORD_COLS:
                            class_df[cord_col] = class_df[cord_col].rolling(window=frms_in_smoothing_window, win_type='gaussian', center=True).mean(std=5).fillna(results[video_name][cord_col]).abs()
                        results[video_name].update(class_df)

            results[video_name] = results[video_name].replace([-1, -1.0, '-1'], 0).reset_index(drop=True)
            if self.save_dir:
                save_path = os.path.join(self.save_dir, f'{video_name}.csv')
                try:
                    results[video_name].to_csv(save_path)
                except PermissionError:
                    raise InvalidFilepathError(msg=f'Permission error: Cannot save file {save_path}. Is the file open in another program?', source=self.__class__.__name__)
                del results[video_name]
            video_timer.stop_timer()
            if self.verbose:
                print(f'Video {video_name} complete (elapsed time: {video_timer.elapsed_time_str}s, video {video_cnt+1}/{len(self.video_path)})')

        timer.stop_timer()
        if not self.save_dir:
            if self.verbose:
                print(f'YOLO results created for {len(self.video_path)} videos', timer.elapsed_time_str)
            return results
        else:
            if self.verbose:
                print(f'YOLO results for {len(self.video_path)} videos saved in {self.save_dir} directory', timer.elapsed_time_str)
            return None


if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Perform YOLO-based keypoint pose estimation inference on videos.")
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the trained YOLO model weights (e.g., "best.pt").')
    parser.add_argument('--video_path', type=str, nargs='+', required=True, help='One or more paths to video files to process. Can be a directory of a file path.')
    parser.add_argument('--keypoint_names', type=str, nargs='+', required=True, help='List of keypoint names, e.g., nose left_ear right_ear.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save output CSV files. If omitted, results are returned in memory.')
    parser.add_argument('--device', type=str, default='0', help="Device to use: 'cpu' or GPU index as string (e.g., '0').")
    parser.add_argument('--format', type=str, default=None, help='Optional export format: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite".')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for inference.')
    parser.add_argument('--torch_threads', type=int, default=8, help='Number of PyTorch threads to use.')
    parser.add_argument('--half_precision', action='store_true', help='Use half-precision (FP16) inference.')
    parser.add_argument('--stream', action='store_true', default=True, help='Process frames in stream (one-by-one) mode.')
    parser.add_argument('--box_threshold', type=float, default=0.1, help='Confidence threshold for detections (0.0 - 1.0).')
    parser.add_argument('--max_tracks', type=int, default=100, help='Maximum number of pose tracks to retain.')
    parser.add_argument('--interpolate', action='store_true', default=True, help='Interpolate missing keypoints across frames.')
    parser.add_argument('--smoothing', type=int, default=100, help='Time in milliseconds to perform smoothing')
    parser.add_argument('--max_per_class', type=int, default=2, help='Maximum number pose tracks per class.')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (square). Default is 640.')
    parser.add_argument('--recursive', action='store_true', help='Use half-precision (FP16) inference.')
    args = parser.parse_args()

    keypoints_tuple = tuple(args.keypoint_names[0].split(","))
    device_val = args.device if args.device == 'cpu' else int(args.device)
    video_paths = args.video_path if len(args.video_path) > 1 else args.video_path[0]

    inference = YOLOPoseInference(weights=args.weights_path,
                                  video_path=video_paths,
                                  keypoint_names=keypoints_tuple,
                                  verbose=args.verbose,
                                  save_dir=args.save_dir,
                                  device=device_val,
                                  format=args.format,
                                  batch_size=args.batch_size,
                                  torch_threads=args.torch_threads,
                                  half_precision=args.half_precision,
                                  stream=args.stream,
                                  recursive=args.recursive,
                                  box_threshold=args.box_threshold,
                                  max_tracks=args.max_tracks,
                                  max_per_class=args.max_per_class,
                                  interpolate=args.interpolate,
                                  smoothing=args.smoothing,
                                  imgsz=args.imgsz)
    inference.run()


# VIDEO_DIR = r'E:\netholabs_videos\mosaics\subset'
# SAVE_DIR = r'E:\netholabs_videos\mosaics\yolo_mdl\results_csvs'
# WEIGHTS_PATH = r"E:\netholabs_videos\mosaics\yolo_mdl\mdl\train2\weights\best.pt"
# KEYPOINT_NAMES = ('Nose', 'Left_ear', 'Right_ear', 'Left_side', 'Center', 'Right_side', 'Tail_base', 'Tail_center', 'Tail_tip')

#
# VIDEO_DIR = r"E:\netholabs_videos\mosaics\subset"
# WEIGHTS_PATH = r"E:\netholabs_videos\mosaics\yolo_mdl_wo_tail\mdl\train2\weights\best.pt"
# SAVE_DIR = r"E:\netholabs_videos\mosaics\yolo_mdl_wo_tail\results_csv"
#
# # VIDEO_DIR = r"E:\maplight_videos"
# # SAVE_DIR = r"E:\maplight_videos\yolo_mdl\mdl\results"
# # WEIGHTS_PATH = r"E:\maplight_videos\yolo_mdl\mdl\train\weights\best.pt"
#
# KEYPOINT_NAMES = ('Nose', 'Left_ear', 'Right_ear', 'Left_side', 'Center', 'Right_side', 'Tail_base',)
# #
# i = YOLOPoseInference(weights=WEIGHTS_PATH,
#                         video_path=VIDEO_DIR,
#                         save_dir=SAVE_DIR,
#                         verbose=True,
#                         device=0,
#                         format=None,
#                         stream=True,
#                         keypoint_names=KEYPOINT_NAMES,
#                         batch_size=250,
#                         imgsz=640,
#                         interpolate=False,
#                         box_threshold=0.5,
#                         max_tracks=3,
#                         max_per_class=None,
#                         overwrite=True,
#                         randomize_order=True,
#                         raise_error=False,
#                         smoothing=None,
#                         recursive=False)
# i.run()

#
#
#
# # #
# VIDEO_PATH = r"E:\netholabs_videos\two_tracks\videos"
# WEIGHTS_PASS = r"E:\netholabs_videos\mosaics\yolo_mdl_w_tail\mdl\train2\weights\best.pt"
# SAVE_DIR = r"E:\netholabs_videos\two_tracks\csv_no_track_025"
# # #
# KEYPOINT_NAMES = ('Nose', 'Left_ear', 'Right_ear', 'Left_side', 'Center', 'Right_side', 'Tail_base', 'Tail_center', 'Tail_tip')
# # # #
# # # #
# i = YOLOPoseInference(weights=WEIGHTS_PASS,
#                       video_path=VIDEO_PATH,
#                       save_dir=SAVE_DIR,
#                       verbose=True,
#                       device=0,
#                       format=None,
#                       stream=True,
#                       keypoint_names=KEYPOINT_NAMES,
#                       batch_size=100,
#                       imgsz=640,
#                       interpolate=False,
#                       box_threshold=0.25,
#                       max_tracks=4)
# i.run()
#
# #
# # video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
# # video_path = "/mnt/d/netholabs/yolo_videos/2025-05-28_19-46-56.mp4"
# # video_path = "/mnt/d/netholabs/yolo_videos/2025-05-28_19-50-23.mp4"
# #
# #save_dir = r'/mnt/d/netholabs/yolo_videos_out'
#
#
#
# # video_path = "/mnt/d/platea/platea_videos/videos/2C_Mouse_5-choice_MustTouchTrainingNEWFINAL_a8.mp4"
# # save_dir = r"/mnt/d/TS_DLC/yolo_kpt/results"
# # weights_path = "/mnt/d/TS_DLC/yolo_kpt/mdl/train2/weights/best.pt"
# #
# # keypoint_names = ('right_ear', 'left_body', 'nose', 'center', 'tail', 'right_body', 'left_ear')
#
#
#
# weights_path = r"D:\platea\yolo_071525\mdl\train4\weights\best.pt"
# video_path = r"D:\platea\platea_videos\videos\clipped\10B_Mouse_5-choice_MustTouchTrainingNEWFINAL_a7.mp4"
# save_dir=r"D:\platea\platea_videos\videos\yolo_results"
# #
# #
# #keypoint_names = ('Left_ear', 'Right_ear', 'Nose', 'Left_side', 'Right_side', 'Tail_base', 'Center', 'Tail_center', 'Tail_tip')
# keypoint_names = ('Left_ear', 'Right_ear', 'Nose', 'Left_side', 'Right_side', 'Center', 'Tail_base')
# #
# #
# i = YOLOPoseInference(weights_path=weights_path,
#                         video_path=video_path,
#                         save_dir=save_dir,
#                         verbose=True,
#                         device=0,
#                         format=None,
#                         stream=True,
#                         keypoint_names=keypoint_names,
#                         batch_size=500,
#                         imgsz=320,
#                         interpolate=True,
#                         threshold=0.8,
#                         retina_msk=True)
# i.run()


# WEIGHTS_PATH = r"E:\netholabs_videos\3d\yolo_mdl\mdl\train6\weights\best.pt"
# VIDEO_PATH = r"E:\netholabs_videos\3d_videos_batch_2\Cage_3_Simon_vid_examples_for_annotatino-20251019T183855Z-1-001\Cage_3_Simon_vid_examples_for_annotatino\rot\test_inference"
# SAVE_DIR = r"E:\netholabs_videos\3d_videos_batch_2\Cage_3_Simon_vid_examples_for_annotatino-20251019T183855Z-1-001\Cage_3_Simon_vid_examples_for_annotatino\predictions"
# KEYPOINT_NAMES = ('NOSE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_FRONT_LEG', 'LEFT_FRONT_PAW',
#                   'RIGHT_FRONT_LEG', 'RIGHT_FRONT_PAW', 'CENTER', 'LEFT_BACK_LEG',
#                   'LEFT_BACK_PAW', 'RIGHT_BACK_LEG', 'RIGHT_BACK_PAW', 'TAIL_BASE',
#                   'MIDDLE_TAIL', 'TAIL_END')
#
# i = YOLOPoseInference(weights=WEIGHTS_PATH,
#                       video_path=VIDEO_PATH,
#                       save_dir=SAVE_DIR,
#                       verbose=True,
#                       device=0,
#                       format=None,
#                       stream=True,
#                       keypoint_names=KEYPOINT_NAMES,
#                       batch_size=100,
#                       imgsz=640,
#                       interpolate=False,
#                       box_threshold=0.25,
#                       max_tracks=4)
# i.run()
# #