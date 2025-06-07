import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import List, Optional, Tuple, Union
from typing import Literal
import numpy as np
import pandas as pd
import torch
from simba.utils.checks import (check_file_exist_and_readable, check_float, check_if_dir_exists, check_int, check_valid_boolean, check_valid_lst, get_fn_ext, check_valid_tuple, check_str, check_valid_array)
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import get_video_meta_data
from simba.bounding_box_tools.yolo.utils import filter_yolo_keypoint_data, load_yolo_model
from simba.utils.errors import CountError

OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
NEAREST = 'nearest'


class YOLOPoseInference():

    """
    YOLOPoseInference performs pose estimation on videos using a YOLO-based keypoint detection model.

    :param weights_path : str or os.PathLike Path to the trained YOLO model weights (e.g., 'best.pt').
    :param video_path : str, os.PathLike, or list of str or os.PathLike. Path to a single video or a list of video paths to process.
    :param keypoint_names : tuple of str. Tuple containing the names of keypoints to be tracked (e.g., ('nose', 'left_ear', ...)).
    :param verbose : bool, optional (default=False) If True, enables verbose logging and progress information.
    :param save_dir : str or os.PathLike, optional (default=None) Directory to save output results. If None, results will not be saved.
    :param device : {'cpu', int}, optional (default=0) Compute device for inference. Set to 'cpu' or the GPU index (e.g., 0 for CUDA:0).
    :param format : str, optional (default=None) Optional format for the model. Alternatives: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite"
    :param batch_size : int, optional (default=4) Number of frames to process simultaneously during inference.
    :param torch_threads : int, optional (default=8) Number of PyTorch threads to use during inference.
    :param half_precision : bool, optional (default=True) If True, uses half-precision (FP16) inference for better performance on compatible GPUs.
    :param stream : bool, optional (default=False) If True, enables streaming mode where frames are processed one-by-one instead of in batches.
    :param threshold : float, optional (default=0.5) Confidence threshold for detecting keypoints. Detections below this value are ignored.
    :param max_tracks : int, optional (default=None) Maximum number of distinct pose tracks to retain. If None, all detected tracks are kept.
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
                 imgsz: int = 640):

        if isinstance(video_path, list):
            check_valid_lst(data=video_path, source=f'{self.__class__.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
        elif isinstance(video_path, str):
            check_file_exist_and_readable(file_path=video_path)
            video_path = [video_path]
        for i in video_path:
            _ = get_video_meta_data(video_path=i)
        check_file_exist_and_readable(file_path=weights_path)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        check_valid_boolean(value=interpolate, source=f'{self.__class__.__name__} interpolate')
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} imgsz', value=imgsz, min_value=1)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=10e-6, max_value=1.0)
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
        self.batch_size, self.threshold, self.max_tracks = batch_size, threshold, max_tracks
        self.verbose, self.save_dir, self.imgsz, self.interpolate = max_tracks, save_dir, imgsz, interpolate
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
            video_predictions = self.model.predict(source=path, half=self.half_precision, batch=self.batch_size, stream=self.stream, imgsz=self.imgsz)
            for frm_cnt, video_prediction in enumerate(video_predictions):
                if video_prediction.obb is not None:
                    boxes = np.array(video_prediction.obb.data.cpu()).astype(np.float32)
                else:
                    boxes = np.array(video_prediction.boxes.data.cpu()).astype(np.float32)
                keypoints = np.array(video_prediction.keypoints.data.cpu()).astype(np.float32)
                for c in list(class_dict.keys()):
                    if boxes.shape[0] == 0:
                        bbox = np.array([frm_cnt, c, class_dict[c], -1, -1, -1, -1, -1, -1, -1, -1, -1])
                        bbox = np.append(bbox, [-1] * len(self.keypoint_col_names))
                        video_out.append(bbox)
                    else:
                        cls_boxes, cls_keypoints = filter_yolo_keypoint_data(bbox_data=boxes, keypoint_data=keypoints, class_id=c, confidence=self.threshold,  class_idx=-1, confidence_idx=4)
                        if cls_boxes.shape[0] == 0:
                            bbox = np.array([frm_cnt, c, class_dict[c], -1, -1, -1, -1, -1, -1, -1, -1, -1])
                            bbox = np.append(bbox, [-1] * len(self.keypoint_col_names))
                            video_out.append(bbox)
                        else:
                            if self.max_tracks is not None:
                                cls_idx = np.argsort(cls_boxes[:, 4])[::-1]
                                cls_boxes = cls_boxes[cls_idx][:self.max_tracks, :]
                                cls_keypoints = cls_keypoints[cls_idx][:self.max_tracks, :]
                            for i in range(cls_boxes.shape[0]):
                                box = np.array([cls_boxes[i][0], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][3], cls_boxes[i][0], cls_boxes[i][3]]).astype(np.int32)
                                bbox = np.array([frm_cnt, cls_boxes[i][-1], class_dict[cls_boxes[i][-1]], cls_boxes[i][-2]] + list(box))
                                bbox = np.append(bbox, cls_keypoints[i].flatten())
                                video_out.append(bbox)
            results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
            if self.interpolate:
                for cord_col in COORD_COLS:
                    results[video_name][cord_col] = results[video_name][cord_col].astype(np.int32).replace(to_replace=-1, value=np.nan)
                    results[video_name][cord_col] = results[video_name][cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()

        timer.stop_timer()
        if not self.save_dir:
            if self.verbose:
                print(f'YOLO results created', timer.elapsed_time_str)
            return results
        else:
            for k, v in results.items():
                save_path = os.path.join(self.save_dir, f'{k}.csv')
                v.to_csv(save_path)
                if self.verbose:
                    print(f'YOLO results saved in {self.save_dir} directory', timer.elapsed_time_str)
                    return None
                return None
            return None


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process video using a trained YOLO model with keypoints.")
#     parser.add_argument('--weights_path', type=str, required=True, help = 'Path to trained model.')
#     parser.add_argument('--video_path', type=str, required=True, help = 'Path to video file or directory of videos.')
#     parser.add_argument('--keypoint_names', type=str, nargs='+', required=True, help = 'Space-separated list of body-part (keypoint) names.')
#     parser.add_argument('--verbose', action='store_true', help = 'Enable verbose logging.')
#     parser.add_argument('--save_dir', type=str, default=None, help = 'Optional directory to save output.')
#     parser.add_argument('--device', type=str, default='0', help = "Device to use: 'cpu' or GPU ID as string (e.g., '0').")
#     parser.add_argument('--format', type=str, default=None, help = 'Optional format specifier for model. Options: "onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite".')
#     parser.add_argument('--batch_size', type=int, default=4, help = 'Batch size for inference.')
#     parser.add_argument('--torch_threads', type=int, default=8, help = 'Number of PyTorch threads to use.')
#     parser.add_argument('--half_precision', action='store_true', help = 'Use half precision for inference.')
#     parser.add_argument('--stream', action='store_true', help = 'Enable streaming mode.')
#     parser.add_argument('--threshold', type=float, default=0.5, help = 'Retains tracking data where bounding box conf is above this value.')
#     parser.add_argument('--max_tracks', type=int, default=2, help='The maximum number of animals to detect per frame. Default: None.')
#     args = parser.parse_args()
#     keypoints_tuple = tuple(args.keypoint_names)
#     device_val = args.device if args.device == 'cpu' else int(args.device)
#     inference = YOLOPoseInference(weights_path=args.weights_path, video_path=args.video_path, keypoint_names=keypoints_tuple, verbose=args.verbose, save_dir=args.save_dir, device=device_val, format=args.format, batch_size=args.batch_size, torch_threads=args.torch_threads, half_precision=args.half_precision, stream=args.stream, threshold=args.threshold, max_tracks=args.max_tracks)
#     inference.run()

save_dir = '/mnt/c/troubleshooting/mitra/yolo_pose'
video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0521.mp4"
i = YOLOPoseInference(weights_path=r"/mnt/d/netholabs/yolo_mdls_1/train/weights/best.pt",
                        video_path=video_path,
                        save_dir=save_dir,
                        verbose=True,
                        device=0,
                        format='onnx',
                        stream=True,
                        keypoint_names=('nose', 'ear_left', 'ear_right', 'lateral_left', 'center', 'lateral_right', 'tail_base'),
                        batch_size=32,
                        imgsz=320,
                        interpolate=True)
i.run()