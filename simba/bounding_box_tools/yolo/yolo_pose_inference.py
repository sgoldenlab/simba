import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
try:
    from typing import Literal
except:
    from typing_extensions import Literal

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from simba.bounding_box_tools.yolo.utils import (_get_undetected_obs,
                                                 filter_yolo_keypoint_data,
                                                 load_yolo_model, yolo_predict)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple, get_fn_ext)
from simba.utils.enums import Options
from simba.utils.errors import CountError, InvalidFileTypeError
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_video_meta_data)

OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
NEAREST = 'nearest'


class YOLOPoseInference():

    """
    YOLOPoseInference performs pose estimation on videos using a YOLO-based keypoint detection model.

    This class runs YOLO-based keypoint detection on a given video or list of videos. It supports GPU acceleration,
    batch or stream-based inference, result interpolation, and saving results to disk. The model returns detected
    keypoints and their confidence scores for each frame, and optionally tracks poses over time.

    .. seealso::
       For bounding box inference only (no pose), see :func:`~simba.bounding_box_tools.yolo.yolo_inference.YoloInference`.
       For segmentation inference, see :func:`~simba.bounding_box_tools.yolo.yolo_seg_inference.YOLOSegmentationInference`.

    :param Union[str, os.PathLike] weights_path: Path to the trained YOLO model weights (e.g., 'best.pt').
    :param Union[str, os.PathLike] or List[Union[str, os.PathLike]] video_path: Path to a single video, list of videos, or directory containing video files.
    :param Tuple[str, ...] keypoint_names: Tuple containing the names of keypoints to be tracked (e.g., ('nose', 'left_ear', ...)).
    :param Optional[bool] verbose: If True, outputs progress information and timing. Defaults to True.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the inference results. If None, results are returned in memory. Defaults to None.
    :param Union[Literal['cpu'], int] device: Device to use for inference. Use 'cpu' for CPU or GPU index (e.g., 0 for CUDA:0). Defaults to 0.
    :param Optional[str] format: Optional export format for the model. Supported values: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite". Defaults to None.
    :param Optional[int] batch_size: Number of frames to process in parallel. Defaults to 4.
    :param int torch_threads: Number of PyTorch threads to use. Defaults to 8.
    :param bool half_precision: If True, uses half-precision (FP16) inference. Defaults to True.
    :param bool stream: If True, processes frames one-by-one in a generator style. Recommended for long videos. Defaults to False.
    :param float threshold: Confidence threshold for keypoint detection. Detections below this value are ignored. Defaults to 0.5.
    :param Optional[int] max_tracks: Maximum number of pose tracks to keep. If None, all tracks are retained.
    :param bool interpolate: If True, interpolates missing keypoints across frames using the 'nearest' method. Defaults to False.
    :param int imgsz: Input image size for inference. Must be square. Defaults to 640.
    """

    def __init__(self,
                 weights_path: Union[str, os.PathLike],
                 video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                 keypoint_names: Tuple[str, ...],
                 verbose: Optional[bool] = True,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 device: Union[Literal['cpu'], int] = 0,
                 format: Optional[str] = None,
                 batch_size: Optional[int] = 4,
                 torch_threads: int = 8,
                 half_precision: bool = True,
                 stream: bool = False,
                 threshold: float = 0.5,
                 max_tracks: Optional[int] = None,
                 interpolate: bool = False,
                 imgsz: int = 640,
                 iou: float = 0.5):

        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif os.path.isfile(video_path):
            video_path = [video_path]
        elif os.path.isdir(video_path):
            video_path = find_files_of_filetypes_in_directory(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=False)
        for i in video_path:
            _ = get_video_meta_data(video_path=i)
        check_file_exist_and_readable(file_path=weights_path)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=interpolate, source=f'{self.__class__.__name__} interpolate')
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=10e-6, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} iou', value=iou, min_value=10e-6, max_value=1.0)
        check_valid_tuple(x=keypoint_names, source=f'{self.__class__.__name__} keypoint_names', min_integer=1, valid_dtypes=(str,))
        if max_tracks is not None:
            check_int(name=f'{self.__class__.__name__} max_tracks', value=max_tracks, min_value=1)
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        self.keypoint_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y', 'p']]
        self.keypoint_cord_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y']]
        OUT_COLS.extend(self.keypoint_col_names)
        COORD_COLS.extend(self.keypoint_cord_col_names)
        torch.set_num_threads(torch_threads)
        self.model = load_yolo_model(weights_path=weights_path, device=device, format=format)
        self.half_precision, self.stream, self.video_path = half_precision, stream, video_path
        self.device, self.batch_size, self.threshold, self.max_tracks, self.iou = device, batch_size, threshold, max_tracks, iou
        self.verbose, self.save_dir, self.imgsz, self.interpolate = verbose, save_dir, imgsz, interpolate
        if self.model.model.task != 'pose':
            raise InvalidFileTypeError(msg=f'The model {weights_path} is not a pose model. It is a {self.model.model.task} model', source=self.__class__.__name__)
        if self.model.model.kpt_shape[0] != len(keypoint_names):
            raise CountError(msg=f'The YOLO model expects {self.model.model.model.head.kpt_shape[0]} keypoints but you passed {len(keypoint_names)}: {keypoint_names}', source=self.__class__.__name__)

    def run(self):
        results = {}
        class_dict = self.model.names
        timer = SimbaTimer(start=True)
        for path in self.video_path:
            _, video_name, _ = get_fn_ext(filepath=path)
            _ = get_video_meta_data(video_path=path)
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
                detected_classes = np.unique(boxes[:, -1]).astype(int) if boxes.size > 0 else []
                for class_id, class_name in class_dict.items():
                    if class_id not in detected_classes:
                        video_out.append(_get_undetected_obs(frm_id=frm_cnt, class_id=class_id, class_name=class_name, value_cnt=(9 + (len(self.keypoint_col_names)))))
                        continue
                    cls_boxes, cls_keypoints = filter_yolo_keypoint_data(bbox_data=boxes, keypoint_data=keypoints, class_id=class_id, confidence=None, class_idx=-1, confidence_idx=None)

                    for i in range(cls_boxes.shape[0]):
                        box = np.array([cls_boxes[i][0], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][3], cls_boxes[i][0], cls_boxes[i][3]]).astype(np.int32)
                        bbox = np.array([frm_cnt, cls_boxes[i][-1], class_dict[cls_boxes[i][-1]], cls_boxes[i][-2]] + list(box))
                        bbox = np.append(bbox, cls_keypoints[i].flatten())
                        video_out.append(bbox)

            results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
            if self.interpolate:
                for cord_col in COORD_COLS:
                    results[video_name][cord_col] = results[video_name][cord_col].astype(np.float32).astype(np.int32).replace(to_replace=-1, value=np.nan)
                    results[video_name][cord_col] = results[video_name][cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()
            if self.save_dir:
                save_path = os.path.join(self.save_dir, f'{video_name}.csv')
                results[video_name].to_csv(save_path)
                del results[video_name]

        timer.stop_timer()
        if not self.save_dir:
            if self.verbose:
                print(f'YOLO results created', timer.elapsed_time_str)
            return results
        else:
            if self.verbose:
                print(f'YOLO results saved in {self.save_dir} directory', timer.elapsed_time_str)
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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference.')
    parser.add_argument('--torch_threads', type=int, default=8, help='Number of PyTorch threads to use.')
    parser.add_argument('--half_precision', action='store_true', help='Use half-precision (FP16) inference.')
    parser.add_argument('--stream', action='store_true', help='Process frames in stream (one-by-one) mode.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detections (0.0 - 1.0).')
    parser.add_argument('--max_tracks', type=int, default=None, help='Maximum number of pose tracks to retain.')
    parser.add_argument('--interpolate', action='store_true', help='Interpolate missing keypoints across frames.')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (square). Default is 640.')
    args = parser.parse_args()

    keypoints_tuple = tuple(args.keypoint_names)
    device_val = args.device if args.device == 'cpu' else int(args.device)
    video_paths = args.video_path if len(args.video_path) > 1 else args.video_path[0]

    inference = YOLOPoseInference(weights_path=args.weights_path,
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
                                  threshold=args.threshold,
                                  max_tracks=args.max_tracks,
                                  interpolate=args.interpolate,
                                  imgsz=args.imgsz)
    inference.run()

# #
# # video_paths = r"D:\cvat_annotations\videos\mp4_20250624155703"
# # weights_path = r"D:\cvat_annotations\yolo_mdl_07122025\weights\best.pt"
# # save_dir = r"D:\cvat_annotations\yolo_mdl_07122025\out_data"
# #
# # keypoint_names = ('Nose', 'Left_ear', 'Right_ear', 'Left_side', 'Center', 'Right_side', 'Tail_base', 'Tail_center', 'Tail_tip')
# # # # #
# # # # #
# # i = YOLOPoseInference(weights_path=weights_path,
# #                         video_path=video_paths,
# #                         save_dir=save_dir,
# #                         verbose=True,
# #                         device=0,
# #                         format=None,
# #                         stream=True,
# #                         keypoint_names=keypoint_names,
# #                         batch_size=100,
# #                         imgsz=640,
# #                         interpolate=False,
# #                         threshold=0.5,
# #                         max_tracks=4)
# # i.run()
#
#
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