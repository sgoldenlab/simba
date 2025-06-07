import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import Optional, Union

import torch

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
from ultralytics import YOLO

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_array, check_valid_boolean)
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError, SimBAGPUError
from simba.utils.read_write import find_core_cnt


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
    model = load_yolo_model(weights_path=initial_weights)
    model.train(data=model_yaml, epochs=epochs, project=save_path, batch=batch, plots=plots, workers=workers)


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

    model = YOLO(weights_path, verbose=verbose)
    model.to(device=device)
    if format is not None: model.export(format=format)

    return model




def filter_yolo_keypoint_data(bbox_data: np.ndarray,
                              keypoint_data: np.ndarray,
                              class_id: Optional[int] = None,
                              confidence: Optional[float] = None,
                              class_idx: int = 6,
                              confidence_idx: int = 5):
    """
    Helper to filters YOLO bounding box and keypoint data based on class ID and/or confidence threshold.

    :param np.ndarray bbox_data: A 2D array of shape (N, M) representing YOLO bounding box data, where each row corresponds to one detection and contains class and confidence values.
    :param np.ndarray bbox_data: A 3D array of shape (N, 2, 3) representing keypoints for each detection, where K is the number of keypoints per detection.
    :param Optional[int] class_id: Target class ID to filter detections. Defaults to None.
    :param Optional[float] confidence: Minimum confidence threshold to keep detections. Must be in [0, 1]. Defaults to None.
    :param int confidence_idx: Index in `bbox_data` where confidence value is stored. Defaults to 5.
    :param int class_idx: Index in `bbox_data` where class ID is stored. Defaults to 6.
    """

    if class_id is None and confidence is None:
        raise InvalidInputError(msg='Provide at least one filter condition')

    check_valid_array(data=bbox_data, source=filter_yolo_keypoint_data.__name__, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_0=1)
    check_valid_array(data=keypoint_data, source=filter_yolo_keypoint_data.__name__, accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[bbox_data.shape[0],])
    class_id is not None and check_int(name=filter_yolo_keypoint_data.__name__, value=class_id, min_value=0, raise_error=True)
    confidence is not None and check_float(name=filter_yolo_keypoint_data.__name__, value=confidence, min_value=0, max_value=1.0, raise_error=True)
    check_int(name=f'{filter_yolo_keypoint_data.__name__} class_idx', value=class_idx, min_value=-bbox_data.shape[1], max_value=bbox_data.shape[1])
    check_int(name=f'{filter_yolo_keypoint_data.__name__} confidence_idx', value=confidence_idx, min_value=-bbox_data.shape[1], max_value=bbox_data.shape[1])

    if class_id is not None:
        cls_idx = np.argwhere(bbox_data[:, class_idx] == class_id).flatten()
        bbox_data, keypoint_data = bbox_data[cls_idx], keypoint_data[cls_idx]
    if confidence_idx is not None:
        cls_idx = np.argwhere(bbox_data[:, confidence_idx] >= confidence).flatten()
        bbox_data, keypoint_data = bbox_data[cls_idx], keypoint_data[cls_idx]

    return bbox_data, keypoint_data