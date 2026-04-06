import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np

try:
    import torch
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None
    torch = None

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_if_valid_img,
                                check_instance, check_int, check_str,
                                check_valid_array, check_valid_boolean,
                                check_valid_device, check_valid_lst,
                                check_valid_tuple)
from simba.utils.enums import Formats, Options
from simba.utils.errors import (InvalidFileTypeError, InvalidInputError,
                                SimBAGPUError, SimBAPAckageVersionError)
from simba.utils.printing import stdout_information
from simba.utils.read_write import (create_directory, find_core_cnt,
                                    get_fn_ext, get_video_meta_data)


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

    .. seealso::
       For the recommended wrapper class with parameter validation, see :class:`simba.model.yolo_fit.FitYolo`.

    :param initial_weights: Path to the pre-trained YOLO model weights (usually a `.pt` file). Example weights can be found [here](https://huggingface.co/Ultralytics).
    :param model_yaml: YAML file containing paths to the training, validation, and testing datasets and the object class mappings. Example YAML file can be found [here](https://github.com/sgoldenlab/simba/blob/master/misc/ex_yolo_model.yaml).
    :param save_path: Directory path where the trained model, logs, and results will be saved.
    :param epochs: Number of epochs to train the model. Default is 5.
    :param batch: Batch size for training. Default is 16.
    :return: None. The trained model and associated training logs are saved in the specified `project_path`.

    :example:
    >>> fit_yolo(initial_weights=r"C:/troubleshooting/coco_data/weights/yolov8n-obb.pt", data=r"C:/troubleshooting/coco_data/model.yaml", save_path=r"C:/troubleshooting/coco_data/mdl", batch=16)
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

    .. seealso::
       For recommended wrapper classes that use this function, see :class:`simba.model.yolo_fit.FitYolo`, :class:`simba.model.yolo_inference.YoloInference`, :class:`simba.model.yolo_pose_inference.YOLOPoseInference`, :class:`simba.model.yolo_seg_inference.YOLOSegmentationInference`, and :class:`simba.model.yolo_pose_track_inference.YOLOPoseTrackInference`.

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
    _, mdl_name, mdl_ext = get_fn_ext(filepath=weights_path)
    if device != 'cpu':
        torch.cuda.set_device(device)
    try:
        model = YOLO(weights_path, verbose=verbose)
    except Exception as e:
        raise InvalidFileTypeError(msg=f'Could not load {weights_path} as a valid YOLO model: {e.args}', source=load_yolo_model.__name__)
    if mdl_ext.lower() == '.pt':
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

    .. seealso::
       For recommended wrapper classes that use this function, see :class:`simba.model.yolo_inference.YoloInference`, :class:`simba.model.yolo_pose_inference.YOLOPoseInference`, and :class:`simba.model.yolo_seg_inference.YOLOSegmentationInference`.

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


def apply_fixed_bbox_size(data: pd.DataFrame,
                          video_name: str,
                          img_w: int,
                          img_h: int,
                          bbox_size: Tuple[int, int]) -> pd.DataFrame:
    """
    Apply a fixed axis-aligned bounding-box size to detected rows in a results table.

    The current box center is preserved, then the box is resized to ``bbox_size`` (``h, w``). If the resized box would exceed frame boundaries, the box is shifted so it remains fully inside the image while preserving the requested size.

    The function expects YOLO corner columns ``X1..Y4`` and updates them in-place on the input dataframe before returning it.

    :param pd.DataFrame data: Detection dataframe containing ``CONFIDENCE`` and corner coordinate columns ``X1, Y1, X2, Y2, X3, Y3, X4, Y4``.
    :param str video_name: Video identifier used in error messages.
    :param int img_w: Image width in pixels.
    :param int img_h: Image height in pixels.
    :param Tuple[int, int] bbox_size: Target fixed bounding-box size as ``(height, width)`` in pixels.
    :return: Input dataframe with updated fixed-size bbox coordinates for detected rows.
    :rtype: pd.DataFrame
    :raises InvalidInputError: If required columns are missing or if ``bbox_size`` is larger than image dimensions.
    """

    CONFIDENCE = "CONFIDENCE"
    COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']

    if not set(COORD_COLS).issubset(data.columns):
        raise InvalidInputError(msg=f"Missing bbox coordinate columns in data. Required: {COORD_COLS}", source=apply_fixed_bbox_size.__name__)
    if CONFIDENCE not in data.columns:
        raise InvalidInputError(msg=f"Missing {CONFIDENCE} column in data.", source=apply_fixed_bbox_size.__name__)
    check_int(name=f"{apply_fixed_bbox_size.__name__} img_w", value=img_w, min_value=1)
    check_int(name=f"{apply_fixed_bbox_size.__name__} img_h", value=img_h, min_value=1)
    box_h, box_w = int(bbox_size[0]), int(bbox_size[1])
    if box_h > int(img_h) or box_w > int(img_w):
        raise InvalidInputError(msg=f"bbox_size {bbox_size} is larger than video dimensions ({img_h}, {img_w}) for video {video_name}", source=apply_fixed_bbox_size.__name__)
    conf = pd.to_numeric(data[CONFIDENCE], errors="coerce").to_numpy(dtype=np.float32)
    detected_mask = np.isfinite(conf) & (conf >= 0.0)
    if not np.any(detected_mask):
        return data
    coords = data.loc[detected_mask, COORD_COLS].to_numpy(dtype=np.float32)
    cxs = np.mean(coords[:, [0, 2, 4, 6]], axis=1)
    cys = np.mean(coords[:, [1, 3, 5, 7]], axis=1)
    max_x, max_y = np.int32(img_w - 1), np.int32(img_h - 1)
    cx_i = np.round(cxs).astype(np.int32)
    cy_i = np.round(cys).astype(np.int32)
    x1 = cx_i - np.int32(box_w // 2)
    y1 = cy_i - np.int32(box_h // 2)
    x2 = x1 + np.int32(box_w - 1)
    y2 = y1 + np.int32(box_h - 1)
    left_shift = np.where(x1 < 0, -x1, 0).astype(np.int32)
    x1, x2 = x1 + left_shift, x2 + left_shift
    right_shift = np.where(x2 > max_x, x2 - max_x, 0).astype(np.int32)
    x1, x2 = x1 - right_shift, x2 - right_shift

    top_shift = np.where(y1 < 0, -y1, 0).astype(np.int32)
    y1, y2 = y1 + top_shift, y2 + top_shift
    bottom_shift = np.where(y2 > max_y, y2 - max_y, 0).astype(np.int32)
    y1, y2 = y1 - bottom_shift, y2 - bottom_shift

    x1 = np.clip(x1, 0, max_x).astype(np.int32)
    x2 = np.clip(x2, 0, max_x).astype(np.int32)
    y1 = np.clip(y1, 0, max_y).astype(np.int32)
    y2 = np.clip(y2, 0, max_y).astype(np.int32)
    fixed_coords = np.column_stack((x1, y1, x2, y1, x2, y2, x1, y2))
    data.loc[detected_mask, COORD_COLS] = fixed_coords

    return data


def detect_yolo_project_type(label_path: str) -> str:
    """
    Detect YOLO project type (bbox, keypoint, or segmentation) from a single label file.

    - bbox: class_id + 4 values (x_center, y_center, w, h)
    - keypoint: class_id + 4 values + N*3 keypoints (x, y, visibility)
    - segmentation: class_id + N*2 polygon vertices (N >= 3)
    """
    check_file_exist_and_readable(file_path=label_path)

    BBOX_VALUE_CNT = 4
    KPT_DIM = 3
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            n_values = len(parts) - 1
            if n_values == BBOX_VALUE_CNT:
                return 'bbox'
            elif n_values > BBOX_VALUE_CNT and (n_values - BBOX_VALUE_CNT) % KPT_DIM == 0:
                return 'keypoint'
            elif n_values >= 6 and n_values % 2 == 0:
                return 'segmentation'
            else:
                return 'bbox'
    return 'bbox'


def create_yolo_sample_visualizations(samples: List[Tuple[str, np.ndarray, str]],
                                      save_dir: Union[str, os.PathLike],
                                      names: Tuple[str, ...],
                                      palette: str = 'Set1',
                                      seg_opacity: float = 0.5,
                                      verbose: bool = True,
                                      source: str = '') -> None:
    """
    Create annotated visualizations from YOLO-format (image, label_str) samples.

    Auto-detects annotation type (bounding-box or segmentation) from the label string format and draws the appropriate overlays. Images are saved as PNG files in ``save_dir``.

    :param List[Tuple[str, np.ndarray, str]] samples: List of ``(sample_name, image, label_str)`` tuples produced by a SAM3-to-YOLO converter.
    :param Union[str, os.PathLike] save_dir: Directory where annotated images are saved. Created if it does not exist.
    :param Tuple[str, ...] names: Class names in index order.
    :param str palette: Color palette name. Default ``'Set1'``.
    :param float seg_opacity: Opacity of filled segmentation polygons (0.0–1.0). Default ``0.5``.
    :param bool verbose: Print progress messages. Default ``True``.
    :param str source: Caller class name for log messages.
    """

    import cv2

    from simba.mixins.plotting_mixin import PlottingMixin
    from simba.utils.data import create_color_palette

    check_valid_lst(data=list(samples), source=f'{source} samples', min_len=1)
    for i, s in enumerate(samples):
        if not isinstance(s, (list, tuple)) or len(s) != 3:
            raise InvalidInputError(msg=f'Each sample must be a 3-element tuple (name, image, label_str), got {type(s)} with length {len(s) if hasattr(s, "__len__") else "N/A"} at index {i}', source=source)
        if not isinstance(s[0], str):
            raise InvalidInputError(msg=f'Sample name at index {i} must be a str, got {type(s[0])}', source=source)
        check_if_valid_img(data=s[1], source=f'{source} sample image index {i}', raise_error=True)
        if not isinstance(s[2], str):
            raise InvalidInputError(msg=f'Sample label_str at index {i} must be a str, got {type(s[2])}', source=source)
    check_if_dir_exists(in_dir=os.path.dirname(save_dir), source=f'{source} save_dir')
    check_valid_tuple(x=names, source=f'{source} names', minimum_length=1, valid_dtypes=(str,))
    check_str(name=f'{source} palette', value=palette, options=Options.PALETTE_OPTIONS_CATEGORICAL.value)
    check_float(name=f'{source} seg_opacity', value=seg_opacity, min_value=0.0, max_value=1.0)
    check_valid_boolean(value=[verbose], source=f'{source} verbose', raise_error=True)

    create_directory(paths=[save_dir], overwrite=False)
    class_colors = create_color_palette(pallete_name=palette, increments=max(len(names), 10))
    if verbose:
        stdout_information(msg=f'Creating {len(samples)} annotation visualizations in {save_dir}...', source=source)

    for cnt, (sample_name, img, label_str) in enumerate(samples):
        img_h, img_w = img.shape[:2]
        thickness = max(1, PlottingMixin().get_optimal_circle_size(frame_size=(img_w, img_h), circle_frame_ratio=100))
        vis_img = img.copy()
        for line in label_str.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            color = tuple(int(c) for c in class_colors[cls_id % len(class_colors)])
            label = names[cls_id] if cls_id < len(names) else str(cls_id)
            n_values = len(parts) - 1
            if n_values == 4:
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
                cv2.putText(vis_img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, thickness // 2), cv2.LINE_AA)
            else:
                coords = [float(v) for v in parts[1:]]
                points = []
                for i in range(0, len(coords), 2):
                    points.append([int(coords[i] * img_w), int(coords[i + 1] * img_h)])
                polygon = np.array(points, dtype=np.int32)
                pts = polygon.reshape((-1, 1, 2))
                overlay = vis_img.copy()
                cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
                cv2.fillPoly(overlay, [pts], color=color)
                cv2.addWeighted(overlay, seg_opacity, vis_img, 1 - seg_opacity, 0, vis_img)
                cx, cy = int(polygon[:, 0].mean()), int(polygon[:, 1].mean())
                cv2.putText(vis_img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, max(1, thickness // 2), cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_dir, f'{sample_name}.png'), vis_img)
    if verbose:
        stdout_information(msg=f'{len(samples)} visualizations saved in {save_dir}', source=source)

def export_yolo_model(model_path: Union[str, os.PathLike],
                      export_format: Literal["onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite", "torch"],
                      imgsz: int = 256,
                      device: Union[Literal['cpu'], int] = 0,
                      int8: bool = False,
                      batch: int = 1,
                      workspace: Optional[int] = 8,
                      data: Optional[Union[str, os.PathLike]] = None,
                      task: Optional[Literal["detect", "segment", "classify", "pose", "obb"]] = None,
                      dynamic: bool = False,
                      simplify: bool = True,
                      half: bool = False) -> Union[str, os.PathLike]:
    """
    Export a YOLO model using Ultralytics ``model.export``.

    Wrapper around Ultralytics export that supports common deployment formats (including ONNX and TensorRT engine).

    .. note::
       INT8 export is only valid for ``engine`` format and cannot be combined with ``half=True``.

    :param Union[str, os.PathLike] model_path: Path to source YOLO weights (typically ``.pt``).
    :param Literal["onnx", "engine", "torchscript", "onnxsimplify", "coreml", "openvino", "pb", "tf", "tflite", "torch"] export_format: Target export format.
    :param int imgsz: Export input image size in pixels.
    :param Union[Literal['cpu'], int] device: Export device (``'cpu'`` or CUDA index).
    :param bool int8: If True, request INT8 TensorRT export. Requires ``export_format='engine'``.
    :param int batch: Export batch/profile size (must be >= 1). For INT8, ensure calibration data size is at least this value.
    :param int workspace: TensorRT workspace budget in GB (must be >= 1).
    :param Optional[Union[str, os.PathLike]] data: Optional dataset yaml path used for export/calibration.
    :param Optional[Literal["detect", "segment", "classify", "pose", "obb"]] task: Optional explicit YOLO task. Set this to avoid backend task auto-guessing warnings.
    :param bool dynamic: If True, build with dynamic input profiles.
    :param bool half: If True, request FP16 export where supported.
    :return: Path-like export artifact returned by Ultralytics.
    :rtype: Union[str, os.PathLike]
    :raises SimBAPAckageVersionError: If Ultralytics is unavailable.
    :raises InvalidInputError: On unsupported format or invalid precision combination.

    :example:
    >>> export_yolo_model(
    ...     model_path=r"F://netholabs\\primintellect_test\\mdl\\weights\\best.pt",
    ...     export_format='engine',
    ...     imgsz=256,
    ...     device=0,
    ...     int8=True,
    ...     batch=4,
    ...     workspace=8,
    ...     task='detect',
    ...     dynamic=False,
    ...     half=False
    ... )
    """

    if YOLO is None:
        raise SimBAPAckageVersionError(msg='YOLO/Ultralytics not detected in SimBA environment', source=export_yolo_model.__name__)
    check_file_exist_and_readable(file_path=model_path)
    check_int(name=f"{export_yolo_model.__name__} imgsz", value=imgsz, min_value=1)
    check_int(name=f"{export_yolo_model.__name__} batch", value=batch, min_value=1)
    if workspace is not None: check_int(name=f"{export_yolo_model.__name__} workspace", value=workspace, min_value=1)
    check_valid_device(device=device)
    check_valid_boolean(value=[half, int8, dynamic, simplify], source=export_yolo_model.__name__, raise_error=True)
    if task is not None:
        check_str(name=f"{export_yolo_model.__name__} task", value=task, options=("detect", "segment", "classify", "pose", "obb"), raise_error=True)
    export_format = str(export_format).lower()
    if data is not None: check_file_exist_and_readable(file_path=data, raise_error=True)
    if export_format not in Options.VALID_YOLO_FORMATS.value:
        raise InvalidInputError(msg=f"Unsupported format '{export_format}'. Valid: {Options.VALID_YOLO_FORMATS.value}", source=export_yolo_model.__name__)
    if int8 and export_format != "engine":
        raise InvalidInputError(msg=f"INT8 export requires 'engine' format. Got '{export_format}'.", source=export_yolo_model.__name__)
    if int8 and half:
        raise InvalidInputError(msg="Choose one precision mode: INT8 or FP16 (half).", source=export_yolo_model.__name__)
    model = YOLO(model_path) if task is None else YOLO(model_path, task=task)
    out = model.export(format=export_format, imgsz=imgsz, device=device, half=half, int8=int8, dynamic=dynamic, batch=batch, workspace=workspace, data=data, simplify=simplify)
    return out



# export_yolo_model(model_path=r"E:\open_video\open_field_2\yolo_bbox_project\mdl\train2\weights\best.pt",
#                   export_format='engine',
#                   imgsz=256,
#                   int8=False,
#                   batch=8,
#                   workspace=None,
#                   data=r"E:\open_video\open_field_2\yolo_bbox_project\map.yaml",
#                   task='detect',
#                   dynamic=True,
#                   simplify=True)

#
# export_yolo_model(
#     model_path=r"E:\litpose_yolo\bbox\mdl\train6\weights\best.pt",
#     export_format='engine',
#     imgsz=256,
#     int8=True,
#     batch=16,
#     workspace=8,
#     data=r"E:\litpose_yolo\bbox\map.yaml",
#     task='detect',
#     dynamic=False,
#     half=False
# )
# export_yolo_model(
#     model_path=r"E:\litpose_yolo\bbox\mdl\train6\weights\best.pt",
#     export_format='onnx',
#     imgsz=256,
#     batch=16,
#     task='detect'
# )



#export_yolo_model(model_path=r"E:\litpose_yolo\bbox\mdl\train3\weights\best.pt", export_format='engine', imgsz=256, int8=True, batch=10, workspace=8, data=r"E:\litpose_yolo\bbox\map.yaml", task='detect', dynamic=True)

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