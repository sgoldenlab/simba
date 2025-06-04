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

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.third_party_label_appenders.converters import \
    yolo_obb_data_to_bounding_box
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple, get_fn_ext)
from simba.utils.data import df_smoother, savgol_smoother
from simba.utils.errors import SimBAGPUError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import find_core_cnt, get_video_meta_data


def fit_yolo(initial_weights: Union[str, os.PathLike],
             model_yaml: Union[str, os.PathLike],
             save_path: Union[str, os.PathLike],
             epochs: Optional[int] = 25,
             batch: Optional[Union[int, float]] = 16,
             plots: Optional[bool] = True,
             workers: int = 12):
    """
    Trains a YOLO model using specified initial weights and a configuration YAML file.

    .. note::
       `Download initial weights <https://huggingface.co/Ultralytics>`__.
       `Example model_yaml <https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml>`__.

    :param initial_weights: Path to the pre-trained YOLO model weights (usually a `.pt` file). Example weights can be found [here](https://huggingface.co/Ultralytics).
    :param model_yaml: YAML file containing paths to the training, validation, and testing datasets and the object class mappings. Example YAML file can be found [here](https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml).
    :param save_path: Directory path where the trained model, logs, and results will be saved.
    :param epochs: Number of epochs to train the model. Default is 5.
    :param batch: Batch size for training. Default is 16.
    :return: None. The trained model and associated training logs are saved in the specified `project_path`.

    :example:
    >>> fit_yolo(initial_weights=r"C:\troubleshooting\coco_data\weights\yolov8n-obb.pt", data=r"C:\troubleshooting\coco_data\model.yaml", save_path=r"C:\troubleshooting\coco_data\mdl", batch=16)

    """
    if not torch.cuda.is_available():
        raise ModuleNotFoundError('No GPU detected.')
    check_file_exist_and_readable(file_path=initial_weights)
    check_file_exist_and_readable(file_path=model_yaml)
    check_if_dir_exists(in_dir=save_path)
    check_int(name='epochs', value=epochs, min_value=1)
    check_int(name='workers', value=workers, min_value=1, max_value=find_core_cnt()[0])
    model = YOLO(initial_weights)
    model.train(data=model_yaml, epochs=epochs, project=save_path, batch=batch, plots=plots, workers=workers)

def inference_yolo(weights_path: Union[str, os.PathLike],
                   video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                   verbose: Optional[bool] = False,
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   gpu: Optional[bool] = False,
                   half_precision: Optional[bool] = True,
                   batch_size: Optional[int] = 4,
                   smoothing_method: Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] = None,
                   smoothing_time_window: Optional[int] = None,
                   interpolate: Optional[bool] = False,
                   stream: Optional[bool] = True) -> Union[None, Dict[str, pd.DataFrame]]:

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
    >>> inference_yolo(weights_path=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt", video_path=r"/mnt/c/troubleshooting/mitra/project_folder/videos/FRR_gq_Saline_0624.mp4", save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results", verbose=True, gpu=True)
    """
    COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    SMOOTHING_METHODS = ('savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential')
    NEAREST = 'nearest'

    if isinstance(video_path, list):
        check_valid_lst(data=video_path, source=f'{inference_yolo.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
    elif isinstance(video_path, str):
        check_file_exist_and_readable(file_path=video_path)
        video_path = [video_path]
    for i in video_path:
        _ = get_video_meta_data(video_path=i)
    check_file_exist_and_readable(file_path=weights_path)
    check_valid_boolean(value=[half_precision, gpu, verbose, stream, interpolate], source=inference_yolo.__name__)
    check_int(name=f'{inference_yolo.__name__} batch_size', value=batch_size, min_value=1)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=f'{inference_yolo.__name__} save_dir')
    if smoothing_method is not None:
        check_str(name=f'{inference_yolo.__name__} smoothing', value=smoothing_method, options=SMOOTHING_METHODS)
        check_float(name=f'{inference_yolo.__name__} smoothing_time_window', value=smoothing_time_window, min_value=10e-6)
    torch.set_num_threads(8)
    model = YOLO(weights_path, verbose=verbose)
    if gpu:
        model.export(format='engine')
        model.to('cuda')
    class_dict = model.names
    results = {}
    timer = SimbaTimer(start=True)
    for path in video_path:
        _, video_name, _ = get_fn_ext(filepath=path)
        video_meta_data = get_video_meta_data(video_path=path)
        video_out = []
        video_predictions = model.predict(source=path, half=half_precision, batch=batch_size, stream=stream)
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
        if interpolate:
            for cord_col in COORD_COLS:
                results[video_name][cord_col] = results[video_name][cord_col].astype(np.int32).replace(to_replace=-1, value=np.nan)
                results[video_name][cord_col] = results[video_name][cord_col].interpolate(method=NEAREST, axis=0).ffill().bfill()
        if smoothing_method is not None:
            if smoothing_method != 'savitzky-golay':
                smoothened = df_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=smoothing_time_window, source=inference_yolo.__name__, method=smoothing_method)
            else:
                smoothened = savgol_smoother(data=results[video_name][COORD_COLS], fps=video_meta_data['fps'], time_window=smoothing_time_window, source=inference_yolo.__name__)
            results[video_name].update(smoothened)
    timer.stop_timer()
    if not save_dir:
        if verbose:
            stdout_success(f'YOLO results created', elapsed_time=timer.elapsed_time_str)
        return results
    else:
        for k, v in results.items():
            save_path = os.path.join(save_dir, f'{k}.csv')
            v.to_csv(save_path)
        if verbose:
            stdout_success(f'YOLO results saved in {save_dir} directory', elapsed_time=timer.elapsed_time_str)

def inference_yolo_pose(weights_path: Union[str, os.PathLike],
                        video_path: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]],
                        keypoint_names: Tuple[str, ...],
                        verbose: Optional[bool] = False,
                        save_dir: Optional[Union[str, os.PathLike]] = None,
                        gpu: Optional[bool] = False,
                        batch_size: Optional[int] = 4,
                        torch_threads: int = 8,
                        half_precision: bool = True,
                        stream: bool = False,
                        threshold: float = 0.7,
                        max_tracks: Optional[int] = 2):

    OUT_COLS = ['FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
    if isinstance(video_path, list):
        check_valid_lst(data=video_path, source=f'{inference_yolo.__name__} video_path', valid_dtypes=(str, np.str_,), min_len=1)
    elif isinstance(video_path, str):
        check_file_exist_and_readable(file_path=video_path)
        video_path = [video_path]
    for i in video_path:
        _ = get_video_meta_data(video_path=i)
    check_file_exist_and_readable(file_path=weights_path)
    check_valid_boolean(value=[gpu, verbose], source=inference_yolo.__name__)
    check_int(name=f'{inference_yolo.__name__} batch_size', value=batch_size, min_value=1)
    check_float(name=f'{inference_yolo.__name__} threshold', value=threshold, min_value=10e-6, max_value=1.0)
    check_valid_tuple(x=keypoint_names, source=f'{inference_yolo.__name__} keypoint_names', min_integer=1, valid_dtypes=(str,))
    if max_tracks is not None:
        check_int(name=f'{inference_yolo.__name__} max_tracks', value=max_tracks, min_value=1)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=f'{inference_yolo.__name__} save_dir')
    keypoint_col_names = [f'{i}_{s}'.upper() for i in keypoint_names for s in ['x', 'y', 'p']]
    OUT_COLS.extend(keypoint_col_names)
    torch.set_num_threads(torch_threads)
    model = YOLO(weights_path, verbose=verbose)
    results = {}
    if gpu:
        model.export(format='engine')
        model.to('cuda')
    class_dict = model.names
    timer = SimbaTimer(start=True)
    for path in video_path:
        _, video_name, _ = get_fn_ext(filepath=path)
        _ = get_video_meta_data(video_path=path)
        video_out = []
        video_predictions = model.predict(source=path, half=half_precision, batch=batch_size, stream=stream)
        for frm_cnt, video_prediction in enumerate(video_predictions):
            if video_prediction.obb is not None:
                boxes = np.array(video_prediction.obb.data.cpu()).astype(np.float32)
            else:
                boxes = np.array(video_prediction.boxes.data.cpu()).astype(np.float32)
            keypoints = np.array(video_prediction.keypoints.data.cpu()).astype(np.float32)
            for c in list(class_dict.keys()):
                cls_idx = np.argwhere(boxes[:, -1] == c).flatten()
                cls_boxes, cls_keypoints = boxes[cls_idx], keypoints[cls_idx]
                if cls_boxes.shape[0] == 0:
                    bbox = np.array([frm_cnt, c, class_dict[c], -1, -1, -1, -1, -1, -1, -1, -1, -1])
                    bbox = np.append(bbox, [-1] * len(keypoint_col_names))
                    video_out.append(bbox)
                else:
                    cls_boxes = cls_boxes.reshape(-1, 6)[cls_boxes.reshape(-1, 6)[:, 4] > threshold]
                    if cls_boxes.shape[0] == 0:
                        bbox = np.array([frm_cnt, c, class_dict[c], -1, -1, -1, -1, -1, -1, -1, -1, -1])
                        bbox = np.append(bbox, [-1] * len(keypoint_col_names))
                        video_out.append(bbox)
                    else:
                        if max_tracks is not None:
                            cls_idx = np.argsort(cls_boxes[:, 4])[::-1]
                            cls_boxes = cls_boxes[cls_idx][:max_tracks, :]
                            cls_keypoints = cls_keypoints[cls_idx][:max_tracks, :]
                        for i in range(cls_boxes.shape[0]):
                            box = np.array([cls_boxes[i][0], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][3], cls_boxes[i][0], cls_boxes[i][3]]).astype(np.int32)
                            bbox = np.array([frm_cnt, cls_boxes[i][-1], class_dict[cls_boxes[i][-1]], cls_boxes[i][-2]] + list(box))
                            bbox = np.append(bbox, cls_keypoints[i].flatten())
                            video_out.append(bbox)
        results[video_name] = pd.DataFrame(video_out, columns=OUT_COLS)
    timer.stop_timer()
    if not save_dir:
        if verbose:
            stdout_success(f'YOLO results created', elapsed_time=timer.elapsed_time_str)
        return results
    else:
        for k, v in results.items():
            save_path = os.path.join(save_dir, f'{k}.csv')
            v.to_csv(save_path)
    if verbose:
        stdout_success(f'YOLO results saved in {save_dir} directory', elapsed_time=timer.elapsed_time_str)

def load_yolo_model(weights_path: Union[str, os.PathLike],
                    verbose: bool = True,
                    format: Optional[str] = None,
                    device: Union[Literal['cpu'], int] = 0):

    """
    Load a YOLO model.

    :param Union[str, os.PathLike] weights_path: Path to model weights (.pt, .engine, etc).
    :param bool verbose: Whether to print loading info.
    :param Optional[str] format: Export format, one of VALID_FORMATS or None to skip export.
    :param Union[Literal['cpu'], int]  device: Device to load model on. 'cpu', int GPU index.

    :example:
    >>> load_yolo_model(weights_path=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt", format="onnx", device=0)
    """

    VALID_FORMATS = ["onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite"]
    check_file_exist_and_readable(file_path=weights_path)
    check_valid_boolean(value=verbose, source=f'{load_yolo_model.__name__} verbose', raise_error=True)
    if format is not None: check_str(name=f'{load_yolo_model.__name__} format', value=format.lower(), options=VALID_FORMATS, raise_error=True)
    if isinstance(device, str):
        check_str(name=f'{load_yolo_model.__name__} format', value=device.lower(), options=['cpu'], raise_error=True)
    else:
        check_int(name=f'{load_yolo_model.__name__} device', value=device, min_value=0, raise_error=True)
        gpu_available, gpus = _is_cuda_available()
        if not gpu_available:
            raise SimBAGPUError(msg=f'No GPU detected but device {device} passed', source=load_yolo_model.__name__)
        if device not in list(gpus.keys()):
            raise SimBAGPUError(msg=f'Unaccepted GPU device {device} passed. Accepted: {list(gpus.keys())}', source=load_yolo_model.__name__)

    model = YOLO(weights_path, verbose=verbose, device=device)
    if format is not None: model.export(format=format)

    return model







#fit_yolo(initial_weights=r"/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml=r"/mnt/d/netholabs/yolo_data_1/map.yaml", save_path=r"/mnt/d/netholabs/yolo_mdls_1", batch=32, epochs=100)

# video_path = "/mnt/d/netholabs/videos/2025-04-17_17-09-28.h264"
# #video_path = "/mnt/d/netholabs/videos_/2025-05-27_20-59-48.mp4"
#
# inference_yolo_pose(weights_path=r"/mnt/d/netholabs/yolo_mdls_1/train/weights/best.pt",
#                     video_path=video_path,
#                     save_dir=r"/mnt/d/netholabs/yolo_test/results",
#                     verbose=True,
#                     gpu=True,
#                     keypoint_names=('nose', 'ear_left', 'ear_right', 'lateral_left', 'center', 'lateral_right', 'tail_base'),
#                     batch_size=64)






# fit_yolo(initial_weights=r"/mnt/d/yolo_weights/yolov8n.pt",
#          model_yaml=r"/mnt/d/netholabs/imgs/yolo_train_test_val/map.yaml", save_path=r"/mnt/d/netholabs/imgs/yolo_mdl", batch=32, epochs=100)

# fit_yolo(initial_weights=r"/mnt/c/troubleshooting/coco_data/weights/yolov8n-obb.pt",
#          model_yaml=r"/mnt/c/troubleshooting/coco_data/model.yaml",
#          project_path=r"/mnt/c/troubleshooting/coco_data/mdl",
#          batch=16, epochs=100)



# fit_yolo(initial_weights=r"/mnt/c/troubleshooting/coco_data/weights/yolov8n.pt",
#          model_yaml=r"/mnt/d/netholabs/yolo_test/yolo_train/map.yaml",
#          save_path=r"/mnt/d/netholabs/yolo_test/yolo_mdl", epochs=150, batch=24)



# inference_yolo(weights_path=r"/mnt/d/netholabs/yolo_test/yolo_mdl/train10/weights/best.pt",
#                video_path=r"/mnt/d/netholabs/out/2025-04-17_17-05-07.mp4",
#                save_dir=r"/mnt/d/netholabs/yolo_test/results",
#                verbose=True,
#                gpu=False)

# inference_yolo(weights=r"/mnt/c/troubleshooting/coco_data/mdl/train8/weights/best.pt",
#                video_path=r"/mnt/c/troubleshooting/mitra/project_folder/videos/FRR_gq_Saline_0624.mp4",
#                save_dir=r"/mnt/c/troubleshooting/coco_data/mdl/results",
#                verbose=True,
#                gpu=True)
#
#


# r = inference_yolo(weights=r"C:\troubleshooting\coco_data\mdl\train\weights\best.pt",
#                video_path=r"C:\troubleshooting\mitra\project_folder\videos\clipped\501_MA142_Gi_CNO_0514_clipped.mp4",
#                batch=100)


#fit_yolo(initial_weights=r"C:\troubleshooting\coco_data\weights\yolov8n-obb.pt", data=r"C:\troubleshooting\coco_data\model.yaml", project_path=r"C:\troubleshooting\coco_data\mdl", batch=16)


# """
# initial_weights=r"C:\Users\sroni\Downloads\yolov8n.pt", data=r"C:\troubleshooting\coco_data\model.yaml", epochs=30, project_path=r"C:\troubleshooting\coco_data\mdl", batch=16)
# """
#
#
#



