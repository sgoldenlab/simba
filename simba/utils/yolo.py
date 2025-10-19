import os

import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_img,
                                check_instance, check_int, check_str,
                                check_valid_array, check_valid_boolean,
                                check_valid_device)
from simba.utils.enums import Formats, Options
from simba.utils.errors import (InvalidFileTypeError, InvalidInputError,
                                SimBAGPUError)
from simba.utils.read_write import find_core_cnt, get_video_meta_data


def fit_yolo(weights_path: Union[str, os.PathLike],
             model_yaml: Union[str, os.PathLike],
             save_path: Union[str, os.PathLike],
             epochs: int = 25,
             batch: Union[int, float] = 16,
             plots: bool = True,
             imgsz: int = 640,
             format: Optional[str] = None,
             device:  Union[Literal['cpu'], int] = 0,
             verbose: bool = True,
             workers: int = 8):
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

    if not _is_cuda_available()[0]:
        raise SimBAGPUError(msg='No GPU detected.', source=fit_yolo.__name__)
    check_file_exist_and_readable(file_path=weights_path)
    check_file_exist_and_readable(file_path=model_yaml)
    check_valid_boolean(value=verbose, source=f'{fit_yolo.__name__} verbose', raise_error=True)
    check_valid_boolean(value=plots, source=f'{fit_yolo.__name__} plots', raise_error=True)
    check_if_dir_exists(in_dir=save_path)
    if format is not None: check_str(name=f'{fit_yolo.__name__} format', value=format.lower(), options=Options.VALID_YOLO_FORMATS.value, raise_error=True)
    check_int(name=f'{fit_yolo.__name__} epochs', value=epochs, min_value=1)
    check_int(name=f'{fit_yolo.__name__} imgsz', value=imgsz, min_value=1)
    check_int(name=f'{fit_yolo.__name__} workers', value=workers, min_value=-1, unaccepted_vals=[0], max_value=find_core_cnt()[0])
    if workers == -1: workers = find_core_cnt()[0]
    check_valid_device(device=device)
    model = load_yolo_model(weights_path=weights_path, verbose=verbose, format=format, device=device)
    model.train(data=model_yaml, epochs=epochs, project=save_path, batch=batch, plots=plots, imgsz=imgsz, workers=workers)


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


    check_file_exist_and_readable(file_path=weights_path)
    check_valid_boolean(value=verbose, source=f'{load_yolo_model.__name__} verbose', raise_error=True)
    if format is not None: check_str(name=f'{load_yolo_model.__name__} format', value=format.lower(), options=Options.VALID_YOLO_FORMATS.value, raise_error=True)
    check_valid_device(device=device)
    try:
        model = YOLO(weights_path, verbose=verbose)
    except Exception as e:
        raise InvalidFileTypeError(msg=f'Could not load {weights_path} as a valid YOLO model: {e.args}', source=load_yolo_model.__name__)
    model.to(device=device)
    if format is not None:
        model.export(format=format)

    return model

def _get_undetected_obs(frm_id: int, class_id: int, class_name: str, value_cnt: int) -> np.ndarray:
    """Helper to get a missing observation of specified class and frame """
    check_int(name=f'{_get_undetected_obs.__name__} frm_id', value=frm_id, min_value=0)
    check_int(name=f'{_get_undetected_obs.__name__} class_id', value=class_id, min_value=0)
    check_str(name=f'{_get_undetected_obs.__name__} class_name', value=class_name, raise_error=True)
    check_int(name=f'{_get_undetected_obs.__name__} value_cnt', value=value_cnt, min_value=0)
    return np.array([frm_id, class_id, class_name] + [-1] * value_cnt)


def filter_yolo_keypoint_data(bbox_data: np.ndarray,
                              keypoint_data: np.ndarray,
                              class_id: Optional[int] = None,
                              confidence: Optional[float] = None,
                              class_idx: Optional[int] = None,
                              confidence_idx: Optional[int] = None):
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
    class_id is not None and check_int(name=f'{filter_yolo_keypoint_data.__name__} class_idx', value=class_idx, min_value=-bbox_data.shape[1], max_value=bbox_data.shape[1])
    confidence is not None and check_int(name=f'{filter_yolo_keypoint_data.__name__} confidence_idx', value=confidence_idx, min_value=-bbox_data.shape[1], max_value=bbox_data.shape[1])

    if class_id is not None:
        cls_idx = np.argwhere(bbox_data[:, class_idx] == class_id).flatten()
        bbox_data, keypoint_data = bbox_data[cls_idx], keypoint_data[cls_idx]
    if confidence is not None:
        cls_idx = np.argwhere(bbox_data[:, confidence_idx] >= confidence).flatten()
        bbox_data, keypoint_data = bbox_data[cls_idx], keypoint_data[cls_idx]

    return bbox_data, keypoint_data


def yolo_predict(model: YOLO,
                 source: Union[str, os.PathLike, np.ndarray],
                 half: bool = False,
                 batch_size: Optional[int] = 4,
                 stream: bool = False,
                 imgsz: int = 640,
                 iou: float = 0.75,
                 device:  Union[Literal['cpu'], int] = 0,
                 threshold: float = 0.25,
                 max_detections: int = 300,
                 verbose: bool = True,
                 retina_msk: Optional[bool] = False):

    """
    Produce YOLO predictions.

    :param Union[str, os.PathLike] model: Loaded ultralytics.YOLO model. Returned by :func:`~simba.bounding_box_tools.yolo.model.load_yolo_model`.
    :param Union[str, os.PathLike, np.ndarray] source: Path to video, video stream, directory, image, or image as loaded array.
    :param bool half: Whether to use half precision (FP16) for inference to speed up processing.
    :param bool stream: If True, return a generator that yields results one by one. Useful for stream or large videos.
    :param int imgsz: Size to resize input images to (square dimension). Must be positive integer.
    :param float iou: If max_detections > 1, then the bbox overlap allowed to detect multiple animals.
    :param Optional[int] batch_size: If stream is False, then the number of images to process in each batch.
    :param Union[Literal['cpu'], int] device: Device identifier for inference. 'cpu' to force CPU inference. E.g., integer index of the GPU device (e.g., 0 for 'cuda:0').
    :param float threshold: Confidence threshold for filtering predictions. Only detections with confidence >= threshold are returned. Must be between 0.0 and 1.0.
    :param int max_detections: Maximum number of detections per image/frame to return.
    :param bool verbose: If True, print inference progress and summary information.
    :returns: YOLO results or generator of YOLO results.
    """

    check_valid_boolean(value=half, source=f'{yolo_predict.__name__} half', raise_error=True)
    check_valid_boolean(value=stream, source=f'{yolo_predict.__name__} stream', raise_error=True)
    check_valid_boolean(value=verbose, source=f'{yolo_predict.__name__} verbose', raise_error=True)
    check_valid_boolean(value=retina_msk, source=f'{yolo_predict.__name__} retina_msk', raise_error=True)
    check_int(name=f'{yolo_predict.__name__} imgsz', value=imgsz, min_value=1, raise_error=False)
    check_int(name=f'{yolo_predict.__name__} max_detections', value=max_detections, min_value=1, raise_error=False)
    check_float(name=f'{yolo_predict.__name__} iou', value=iou, min_value=0.0, max_value=1.0, raise_error=True)
    check_float(name=f'{yolo_predict.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0, raise_error=False)
    if not stream:
        v, _ = check_int(name=f'{yolo_predict.__name__} batch_size', value=batch_size, min_value=1, raise_error=False)
        if not v: raise InvalidInputError(msg=f'If not stream, pass valid batch_size integer. Got {batch_size}', source=yolo_predict.__name__)
    check_instance(source=f'{yolo_predict.__name__} source', instance=source, accepted_types=(str, os.PathLike, np.ndarray), raise_error=True)
    if isinstance(source, (str, os.PathLike,)):
        _ = get_video_meta_data(video_path=source)
    else:
        check_if_valid_img(data=source, source=f'{yolo_predict.__name__} source', raise_error=True)
    check_instance(source=f'{yolo_predict.__name__} model', instance=model, accepted_types=(YOLO,), raise_error=True)
    check_valid_device(device=device)


    return model.predict(source=source,
                            half=half,
                            batch=batch_size,
                            stream=stream,
                            device=device,
                            conf=threshold,
                            max_det=max_detections,
                            verbose=verbose,
                            imgsz=imgsz,
                            iou=iou,
                            retina_masks=retina_msk)

def keypoint_array_to_yolo_annotation_str(x: np.ndarray,
                                          img_h: int,
                                          img_w: int,
                                          padding: Optional[float] = None) -> str:
    """
    Convert a set of keypoints into a YOLO-format annotation string that includes the normalized bounding box and keypoints.

    [x_center y_center width height x1 y1 v1 x2 y2 v2 ... xn yn vn]

    :param np.ndarray x: Array of keypoints with shape (N, 3), where each row contains (x, y, visibility).
    :param int img_h: Height of the image.
    :param int img_w: Width of the image.
    :param Optional[float] padding: Optional padding factor (between 0.0 and 1.0) to expand the bounding box around the keypoints.
    :return: YOLO string representation of the pose-estimation data including bounding box and keypoints.
    :rtype: str

    :example:
    >>> x = np.array([[100, 200, 2], [150, 250, 2], [120, 240, 1]])
    >>> keypoint_array_to_yolo_annotation_str(x=x, img_h=480, img_w=640)
    """
    check_valid_array(data=x, source=f'{keypoint_array_to_yolo_annotation_str.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2, 3], accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_0=1)
    check_int(name=f'{keypoint_array_to_yolo_annotation_str.__name__} img_h', value=img_h, min_value=1, raise_error=True)
    check_int(name=f'{keypoint_array_to_yolo_annotation_str.__name__} img_w', value=img_w, min_value=1, raise_error=True)
    if x.shape[1] == 3:
        check_valid_array(data=x[:, 2].flatten(), source=f'{keypoint_array_to_yolo_annotation_str.__name__} x visability flag', accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_0=1, accepted_values=[0, 1, 2])

    img_h, img_w, x = int(img_h), int(img_w), x.astype(np.float32)
    if padding is not None:
        check_float(name=f'{keypoint_array_to_yolo_annotation_str.__name__} padding', value=padding, max_value=1.0, min_value=0.0, raise_error=True)
        padding = float(padding)

    padding = 0.0 if padding is None else padding
    instance_str = ''
    x[np.all(x[:, 0:2] == 0.0, axis=1)] = np.nan
    x_coords, y_coords = x[:, 0], x[:, 1]
    min_x, max_x = np.nanmin(x_coords), np.nanmax(x_coords)
    min_y, max_y = np.nanmin(y_coords), np.nanmax(y_coords)
    pad_w, pad_h = padding * img_w, padding * img_h
    min_x, max_x = max(min_x - pad_w / 2, 0), min(max_x + pad_w / 2, img_w)
    min_y, max_y = max(min_y - pad_h / 2, 0), min(max_y + pad_h / 2, img_h)
    bbox_w, bbox_h = max_x - min_x, max_y - min_y
    x_center, y_center = min_x + bbox_w / 2, min_y + bbox_h / 2
    x_center /= img_w
    y_center /= img_h
    bbox_w /= img_w
    bbox_h /= img_h
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    bbox_w = np.clip(bbox_w, 0.0, 1.0)
    bbox_h = np.clip(bbox_h, 0.0, 1.0)
    x[:, 0] /= img_w
    x[:, 1] /= img_h
    x[:, 0:2] = np.clip(x[:, 0:2], 0.0, 1.0)
    instance_str += f"{x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} "
    x = np.nan_to_num(x, nan=0.0)
    for kp in x:
        instance_str += f"{kp[0]:.6f} {kp[1]:.6f} {int(kp[2])} "
    return instance_str.strip() + '\n'



#fit_yolo(weights_path="D:\yolo_weights\yolo11n-pose.pt", model_yaml=r"D:\rat_resident_intruder\yolo_3\map.yaml", save_path=r"D:\rat_resident_intruder\yolo_3\mdl", batch=12, epochs=100)




#fit_yolo(weights_path="D:\yolo_weights\yolo11n-pose.pt", model_yaml=r"D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo\map.yaml", save_path=r"D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo\mdl", batch=36, epochs=250)


#fit_yolo(weights_path="D:\yolo_weights\yolo11n-pose.pt", model_yaml=r"D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo\map.yaml", save_path=r"D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo\mdl", batch=36, epochs=250)


#fit_yolo(weights_path="D:\yolo_weights\yolov8n.pt", model_yaml=r"C:\troubleshooting\RAT_NOR\project_folder\yolo\map.yaml", save_path=r"C:\troubleshooting\RAT_NOR\project_folder\yolo\mdl", batch=36, epochs=250)


# fit_yolo(weights_path=r"C:\Users\sroni\Downloads\yolo11n (1).pt", model_yaml=r"D:\troubleshooting\two_animals_sleap\import_data\yolo\map.yaml", save_path=r"D:\troubleshooting\two_animals_sleap\import_data\yolo\mdl", batch=36, epochs=250)

#fit_yolo(weights_path=r"D:\yolo_weights\yolo11n-pose.pt", model_yaml=r"D:\rat_resident_intruder\yolo_1\map.yaml", save_path=r"D:\rat_resident_intruder\yolo_1\mdl", batch=36, epochs=250)


#fit_yolo(weights_path="/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml="/mnt/d/mouse_operant_data/yolo/map.yaml", save_path="/mnt/d/mouse_operant_data/yolo/mdl/", batch=36, epochs=250)


#fit_yolo(weights_path="/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml="/mnt/d/TS_DLC/yolo_kpt/map.yml", save_path="/mnt/d/TS_DLC/yolo_kpt/mdl", batch=32, epochs=100)


#fit_yolo(weights_path="/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml="/mnt/d/ares/data/termite_2/yolo/map.yaml", save_path="/mnt/d/ares/data/termite_2/yolo/mdl", batch=32, epochs=1)
#fit_yolo(weights_path="/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml="/mnt/d/ares/data/ant/yolo/map.yaml", save_path="/mnt/d/ares/data/ant/yolo/mdl", batch=18, epochs=100, workers=12)



#fit_yolo(weights_path="/mnt/d/yolo_weights/yolo11n-pose.pt", model_yaml="/mnt/d/rat_resident_intruder/yolo/map.yaml", save_path="/mnt/d/rat_resident_intruder/yolo/mdl", batch=18, epochs=100, workers=12)