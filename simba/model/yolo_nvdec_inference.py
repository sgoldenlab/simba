import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from multiprocessing import current_process
from typing import List, Optional, Tuple, Union

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cv2
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from PyNvVideoCodec import OutputColorType, SimpleDecoder
except:
    torch = None
    F = None
    SimpleDecoder = None
    OutputColorType = None
from ultralytics import YOLO

from simba.data_processors.cuda.utils import _is_cuda_available
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.data import (df_smoother, resample_geometry_vertices,
                              savgol_smoother)
from simba.utils.errors import (InvalidInputError, NoFilesFoundError,
                                SimBAGPUError)
from simba.utils.lookups import get_nvdec_count
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_core_cnt, get_fn_ext, get_pkg_version,
                                    get_video_meta_data, write_df)

DETECT = 'detect'
POSE = 'pose'
SEGMENT = 'segment'
TASKS = (DETECT, POSE, SEGMENT)
FRAME = 'FRAME'
CLASS_ID = 'CLASS_ID'
CLASS_NAME = 'CLASS_NAME'
CONFIDENCE = 'CONFIDENCE'
VERTICE = 'VERTICE'
ID = 'ID'
DETECT_COLS = [FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE, 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
COORD_COLS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
SMOOTHING_METHODS = ('savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential')
SAVITZKY_GOLAY = 'savitzky-golay'


YOLO_EXTENSIONS = ('.engine', '.pt', '.onnx', '.torchscript', '.xml', '.pb', '.tflite', '.edgetpu', '.paddle', '.ncnn', '.mnn', '.imx', '.rknn')


def read_yolo_metadata(model: Union[str, os.PathLike, YOLO]) -> dict:
    """
    Read metadata from a YOLO model file or loaded YOLO instance.

    Supports ``.engine`` (TensorRT), ``.pt`` (PyTorch), ``.onnx``, ``.torchscript``,
    and any other format that :class:`ultralytics.YOLO` can load. For ``.engine``
    files the embedded JSON header is read directly without loading the model.
    For all other formats the model is loaded via Ultralytics to extract metadata.

    :param Union[str, os.PathLike, YOLO] model: Path to a YOLO model file, or an already-loaded :class:`ultralytics.YOLO` instance.
    :return: Dictionary of model metadata. Common keys: ``batch``, ``imgsz``, ``task``, ``names``, ``stride``, ``fp16``, ``dynamic``.
    :rtype: dict
    :raises InvalidInputError: If ``model`` is not a YOLO instance, not a valid path, or has an unsupported extension.

    :example:
    >>> meta = read_yolo_metadata('/models/best.engine')
    >>> meta['batch']
    192
    >>> meta['imgsz']
    [256, 256]
    >>> meta = read_yolo_metadata('/models/best.pt')
    >>> meta['task']
    'detect'
    """

    META_KEYS = ('batch', 'imgsz', 'stride', 'task', 'names', 'fp16', 'dynamic')

    if isinstance(model, YOLO):
        meta = {}
        if hasattr(model, 'predictor') and model.predictor is not None and hasattr(model.predictor, 'model'):
            backend = model.predictor.model
            for key in META_KEYS:
                if hasattr(backend, key):
                    meta[key] = getattr(backend, key)
        if hasattr(model, 'overrides'):
            for key in META_KEYS:
                if key not in meta and key in model.overrides:
                    meta[key] = model.overrides[key]
        if hasattr(model, 'ckpt') and isinstance(model.ckpt, dict):
            train_args = model.ckpt.get('train_args', {})
            for key in META_KEYS:
                if key not in meta and key in train_args:
                    meta[key] = train_args[key]
        return meta

    if not isinstance(model, (str, os.PathLike)):
        raise InvalidInputError(msg=f'model must be a YOLO instance, str, or os.PathLike, got {type(model).__name__}.', source=read_yolo_metadata.__name__)

    path = str(model)
    check_file_exist_and_readable(file_path=path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in YOLO_EXTENSIONS:
        raise InvalidInputError(msg=f'Unsupported model extension {ext}. Supported: {", ".join(YOLO_EXTENSIONS)}.', source=read_yolo_metadata.__name__)

    if ext == '.engine':
        with open(path, 'rb') as f:
            meta_len = int.from_bytes(f.read(4), byteorder='little', signed=False)
            if 0 < meta_len < 10_000:
                return json.loads(f.read(meta_len).decode('utf-8'))
        return {}

    loaded = YOLO(path)
    meta = {}
    if hasattr(loaded, 'overrides'):
        for key in META_KEYS:
            if key in loaded.overrides:
                meta[key] = loaded.overrides[key]
    if hasattr(loaded, 'ckpt') and isinstance(loaded.ckpt, dict):
        train_args = loaded.ckpt.get('train_args', {})
        for key in META_KEYS:
            if key not in meta and key in train_args:
                meta[key] = train_args[key]
    if hasattr(loaded, 'model'):
        for key in META_KEYS:
            if key not in meta and hasattr(loaded.model, key):
                meta[key] = getattr(loaded.model, key)
    return meta



def _xyxy_to_corners(x1, y1, x2, y2):
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _process_one_video(video_path, trt_model, batch_buf, batch_size, imsz,
                       gpu_id, conf_threshold, max_detections,
                       task, class_names, vertice_cnt=30, segment_smoothing=None,
                       video_idx=None, video_cnt=None, keypoint_names=None):

    tag = current_process().name
    decoder = SimpleDecoder(video_path, gpu_id=gpu_id, use_device_memory=True, output_color_type=OutputColorType.RGBP)
    n_total = len(decoder)

    first_frames = decoder.get_batch_frames_by_index(np.array([0], dtype=int))
    first_tensor = torch.from_dlpack(first_frames[0])
    _, orig_h, orig_w = first_tensor.shape
    del first_frames, first_tensor

    decoder = SimpleDecoder(video_path, gpu_id=gpu_id, use_device_memory=True, output_color_type=OutputColorType.RGBP)

    lb_scale = min(imsz / orig_h, imsz / orig_w)
    lb_new_h, lb_new_w = int(round(orig_h * lb_scale)), int(round(orig_w * lb_scale))
    lb_pad_top, lb_pad_left = (imsz - lb_new_h) // 2, (imsz - lb_new_w) // 2

    stem = get_fn_ext(filepath=video_path)[1]
    idx_str = f' (video {video_idx}/{video_cnt})' if video_idx is not None else ''
    stdout_information(msg=f'{stem}: {n_total} frames ({orig_w}x{orig_h}){idx_str}', source=tag)

    t_decode = t_preprocess = t_infer = t_postprocess = 0.0
    frame_offset = 0
    all_rows = []

    batch_starts = list(range(0, n_total, batch_size))
    batch_sizes_list = [min(batch_size, n_total - s) for s in batch_starts]

    for batch_idx in range(len(batch_starts)):
        n_frames = batch_sizes_list[batch_idx]

        t0 = time.perf_counter()
        indices = np.arange(frame_offset, frame_offset + n_frames, dtype=int)
        decoded_frames = decoder.get_batch_frames_by_index(indices)
        torch.cuda.synchronize(gpu_id)
        t1 = time.perf_counter()
        t_decode += t1 - t0

        batch_buf[:n_frames] = 114.0 / 255.0
        for i, frame in enumerate(decoded_frames):
            t = torch.from_dlpack(frame).unsqueeze(0).to(dtype=batch_buf.dtype)
            t.mul_(1.0 / 255.0)
            if t.shape[-2:] == (imsz, imsz):
                batch_buf[i:i + 1] = t
            else:
                batch_buf[i:i + 1, :, lb_pad_top:lb_pad_top + lb_new_h, lb_pad_left:lb_pad_left + lb_new_w] = F.interpolate(t, size=(lb_new_h, lb_new_w), mode="bilinear", align_corners=False)
        del decoded_frames
        if n_frames < batch_size:
            batch_buf[n_frames:] = batch_buf[n_frames - 1:n_frames].expand(batch_size - n_frames, -1, -1, -1)
        torch.cuda.synchronize(gpu_id)
        t2 = time.perf_counter()
        t_preprocess += t2 - t1

        with torch.no_grad():
            raw_pred = trt_model(batch_buf)
        torch.cuda.synchronize(gpu_id)
        t3 = time.perf_counter()
        t_infer += t3 - t2

        if task == SEGMENT:
            first = raw_pred[0] if isinstance(raw_pred, (list, tuple)) else raw_pred
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                pred_tensor, proto = first[0], first[1]
            elif isinstance(raw_pred, (list, tuple)) and len(raw_pred) >= 2:
                pred_tensor, proto = raw_pred[0], raw_pred[1]
            else:
                pred_tensor = first
                proto = None
            if isinstance(pred_tensor, torch.Tensor) and pred_tensor.ndim == 3 and pred_tensor.shape[1] < pred_tensor.shape[2]:
                pred_tensor = pred_tensor.transpose(1, 2)
        else:
            pred_tensor = raw_pred[0] if isinstance(raw_pred, (list, tuple)) else raw_pred

        pred = pred_tensor[:n_frames].cpu()
        if task == SEGMENT and proto is not None:
            proto = proto[:n_frames].cpu()
        for i in range(n_frames):
            frm_idx = frame_offset + i
            det = pred[i]
            scores = det[:, 4]
            mask = scores > conf_threshold
            v_det = det[mask] if mask.any() else det[:0]
            if max_detections is not None and len(v_det) > max_detections:
                topk_idx = v_det[:, 4].topk(max_detections).indices
                v_det = v_det[topk_idx]

            detected_class_ids = set()
            for j in range(len(v_det)):
                vals = v_det[j]
                x1_lb, y1_lb, x2_lb, y2_lb, conf, cls_id = vals[:6].tolist()
                cls_id = int(cls_id)
                detected_class_ids.add(cls_id)
                cls_name = class_names.get(cls_id, str(cls_id))
                x1 = int(round((x1_lb - lb_pad_left) / lb_scale))
                y1 = int(round((y1_lb - lb_pad_top) / lb_scale))
                x2 = int(round((x2_lb - lb_pad_left) / lb_scale))
                y2 = int(round((y2_lb - lb_pad_top) / lb_scale))

                if task == DETECT:
                    row = [frm_idx, cls_id, cls_name, conf] + _xyxy_to_corners(x1, y1, x2, y2)
                    all_rows.append(row)

                elif task == SEGMENT and proto is not None:
                    mask_coeffs = vals[6:]
                    proto_i = proto[i]
                    n_coeffs = proto_i.shape[0]
                    mask_h, mask_w = proto_i.shape[1], proto_i.shape[2]
                    mask_pred = (mask_coeffs[:n_coeffs] @ proto_i.reshape(n_coeffs, -1)).reshape(mask_h, mask_w)
                    mask_pred = torch.sigmoid(mask_pred)
                    scale_h, scale_w = imsz / mask_h, imsz / mask_w
                    x1_m, y1_m = max(0, int(x1_lb / scale_w)), max(0, int(y1_lb / scale_h))
                    x2_m, y2_m = min(mask_w, int(x2_lb / scale_w) + 1), min(mask_h, int(y2_lb / scale_h) + 1)
                    crop_h, crop_w = y2_m - y1_m, x2_m - x1_m
                    if crop_h < 1 or crop_w < 1:
                        continue
                    crop_pred = mask_pred[y1_m:y2_m, x1_m:x2_m]
                    tgt_h = int(round(crop_h * scale_h))
                    tgt_w = int(round(crop_w * scale_w))
                    crop_up = F.interpolate(crop_pred.unsqueeze(0).unsqueeze(0), size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)[0, 0]
                    mask_np = (crop_up > 0.5).numpy().astype(np.uint8)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        pts = largest.squeeze().astype(np.float64)
                        if pts.ndim == 1:
                            pts = pts.reshape(1, 2)
                        pts[:, 0] = (pts[:, 0] + x1_m * scale_w - lb_pad_left) / lb_scale
                        pts[:, 1] = (pts[:, 1] + y1_m * scale_h - lb_pad_top) / lb_scale
                        if segment_smoothing is not None and len(pts) >= 4:
                            pts = GeometryMixin.smooth_geometry_bspline(data=pts, smooth_factor=segment_smoothing, points=len(pts))[0]
                        vertices = resample_geometry_vertices(vertices=pts.reshape(1, -1, 2), vertice_cnt=vertice_cnt).flatten()
                        row = [frm_idx, cls_id] + [int(round(v)) for v in vertices.tolist()]
                        all_rows.append(row)

            for cls_id, cls_name in class_names.items():
                if cls_id not in detected_class_ids:
                    if task == DETECT:
                        all_rows.append([frm_idx, cls_id, cls_name] + [-1] * 9)
                    elif task == POSE and keypoint_names is not None:
                        all_rows.append([frm_idx, cls_id, cls_name] + [-1] * 9 + [-1] * (len(keypoint_names) * 3))
                    elif task == SEGMENT:
                        all_rows.append([frm_idx, cls_id] + [-1] * (vertice_cnt * 2))

        t4 = time.perf_counter()
        t_postprocess += t4 - t3
        frame_offset += n_frames
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(batch_starts) - 1:
            stdout_information(msg=f'{stem}: batch {batch_idx + 1}/{len(batch_starts)} ({frame_offset}/{n_total} frames)', source=tag)
        del raw_pred

    del decoder

    timings = {"decode": t_decode, "preprocess": t_preprocess, "infer": t_infer, "postprocess": t_postprocess}
    return all_rows, timings


def _worker(task_queue, result_queue, ready_queue, engine_path,
            batch_size, imsz, gpu_id, conf_threshold,
            max_detections, task, vertice_cnt=30, segment_smoothing=None,
            save_dir=None, columns=None, interpolate=False, keypoint_names=None):

    tag = current_process().name
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    local_dev = "cuda:0"
    try:
        torch.cuda.set_device(0)
        model = YOLO(engine_path, task=task if task != 'pose' else 'pose')
        dummy = torch.zeros(batch_size, 3, imsz, imsz, dtype=torch.float16, device=local_dev)
        model.predict(source=dummy, device=0, imgsz=imsz, verbose=False)
        del dummy
        trt_model = model.predictor.model
        buf_dtype = torch.float16 if getattr(trt_model, 'fp16', False) else torch.float32
        batch_buf = torch.empty(batch_size, 3, imsz, imsz, dtype=buf_dtype, device=local_dev)
        class_names = model.names if hasattr(model, 'names') else {}
        stdout_information(msg=f'Engine ready ({task}) on physical GPU {gpu_id}.', source=tag)
    except Exception as exc:
        stdout_information(msg=f'FAILED: {exc}', source=tag)
        ready_queue.put((tag, False, {}))
        return

    ready_queue.put((tag, True, class_names))

    while True:
        item = task_queue.get()
        if item is None:
            break
        video_path, video_idx, video_cnt = item
        try:
            rows, timings = _process_one_video(
                video_path=video_path, trt_model=trt_model, batch_buf=batch_buf,
                batch_size=batch_size, imsz=imsz, gpu_id=0,
                conf_threshold=conf_threshold,
                max_detections=max_detections, task=task, class_names=class_names,
                vertice_cnt=vertice_cnt, segment_smoothing=segment_smoothing,
                video_idx=video_idx, video_cnt=video_cnt,
                keypoint_names=keypoint_names,
            )
            if save_dir is not None and columns is not None:
                t_save_start = time.perf_counter()
                df = pd.DataFrame(rows, columns=columns)
                if interpolate:
                    df = YoloNVDECInference._interpolate_df(df, task=task, keypoint_names=keypoint_names, vertice_cnt=vertice_cnt)
                video_name = get_fn_ext(filepath=video_path)[1]
                csv_path = os.path.join(save_dir, f'{video_name}.csv')
                write_df(df=df, file_type='csv', save_path=csv_path)
                timings['save'] = time.perf_counter() - t_save_start
                result_queue.put((video_path, len(df), timings, None))
            else:
                result_queue.put((video_path, rows, timings, None))
        except Exception as exc:
            stdout_information(msg=f'ERROR on {video_path}: {exc}', source=tag)
            result_queue.put((video_path, [] if save_dir is None else 0, {}, str(exc)))


class YoloNVDECInference(object):
    """
    GPU-accelerated YOLO inference on videos using NVDEC decode + TensorRT.

    Decodes video frames on GPU via NVDEC (PyNvVideoCodec), runs YOLO detection, pose-estimation, or segmentation through a TensorRT engine with GPU-side letterboxing and NMS, and stores per-frame results as DataFrames.

    .. important::
       The number of parallel NVDEC hardware decode engines varies by GPU (e.g., 1 on RTX 4070, 3 on RTX 4090, 7 on H100) and directly controls how many videos can be decoded simultaneously. More NVDEC engines means higher throughput when processing multiple videos. The count is auto-detected via
       :func:`~simba.utils.lookups.get_nvdec_count`. If your GPU is not listed or the count is incorrect, pass ``max_workers`` explicitly.

    .. important::
       When running **segmentation** (``task='segment'``), the ``imsz`` parameter is critical for mask quality.
       Segmentation requires pixel-level precision along object boundaries, so spatial detail lost to downscaling
       hurts segmentation far more than detection or pose tasks. Set ``imsz`` as large as your GPU memory allows.
       The default ``256`` may be too coarse for high-quality segmentation masks.

    .. seealso::
       * :class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_bbox.SAM3ToYoloBBox` — create a YOLO bounding-box project from SAM3 annotations.
       * :class:`~simba.third_party_label_appenders.transform.sam3_to_yolo_seg.SAM3ToYoloSeg` — create a YOLO segmentation project from SAM3 annotations.
       * :class:`~simba.model.yolo_inference.YoloInference` — CPU-based YOLO bounding-box inference.
       * :class:`~simba.model.yolo_pose_inference.YOLOPoseInference` — CPU-based YOLO pose inference.
       * :class:`~simba.model.yolo_seg_inference.YOLOSegmentationInference` — CPU-based YOLO segmentation inference.

    .. csv-table::
       :header: EXPECTED RUNTIMES BOUNDING BOX
       :file: ../../docs/tables/NVDECYoloInference_2.csv
       :widths: 10, 10, 40, 40
       :align: center
       :header-rows: 1

    :param Union[str, os.PathLike] video_path: Directory containing input video files, or path to a single video file.
    :param Union[str, os.PathLike] engine_path: Path to TensorRT engine file (.engine). If alternative model file exists, convert it to engine using :func:`simba.utils.yolo.export_yolo_model`. For multi-GPU, place the source ``.pt`` weights alongside the engine — per-GPU engines are auto-exported on first run.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory for per-video CSV output. If None, results kept in memory only. Default None.
    :param Literal['detect', 'pose', 'segment'] task: YOLO task type. Default ``'detect'``.
    :param Optional[int] imsz: Model input image size (square). If None, read from engine metadata. Default None.
    :param Optional[int] batch_size: Inference batch size. If None, read from engine metadata. Default None.
    :param Optional[int] max_workers: Number of parallel worker processes. If None, auto-detected from GPU NVDEC count. Default None.
    :param Union[int, Tuple[int, ...]] gpu_id: CUDA device index or tuple of device indices for multi-GPU inference. Workers are round-robin assigned across listed GPUs. When multiple GPUs are specified, NVDEC engine counts are summed across all GPUs. Default 0.
    :param float conf_threshold: Confidence threshold for detections. Default 0.05.
    :param float iou_threshold: IoU threshold for NMS. Default 0.45.
    :param Optional[Tuple[str, ...]] keypoint_names: Keypoint names in index order, used only when ``task='pose'`` (ignored otherwise). Required when ``task='pose'``, raises error if not provided.
    :param int vertice_cnt: Number of resampled polygon vertices, used only when ``task='segment'`` (ignored otherwise). Default 60.
    :param Optional[int] max_detections: Maximum number of detections to keep per frame after NMS (sorted by confidence). If None, keep all. Default None.
    :param Optional[int] segment_smoothing: B-spline smoothing factor for segmentation polygon vertices, used only when ``task='segment'`` (ignored otherwise). Higher values produce smoother contours. If ``None``, no smoothing is applied. Default ``None``.
    :param bool interpolate: If True, linearly interpolate missing detections across frames, used only when ``task='detect'`` (ignored otherwise). Default True.
    :param bool recursive: If True and ``video_path`` is a directory, search all subdirectories for video files. Default False.
    :param Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] smoothing_method: Smoothing method for detection coordinates, used only when ``task='detect'`` (ignored otherwise). If ``None``, no smoothing is applied. Default ``None``.
    :param Optional[int] smoothing_time_window: Time window (in ms) for coordinate smoothing. Required when ``smoothing_method`` is not None. Default ``None``.
    :param bool verbose: Print progress messages. Default True.

    :example:
    >>> detector = YoloNVDECInference(video_path=r'/videos', engine_path=r'/best.engine', task='detect')
    >>> detector.run()
    >>> detector.results['video_name']

    >>> detector = YoloNVDECInference(video_path=r'/videos', engine_path=r'/pose.engine', task='pose', keypoint_names=('NOSE', 'LEFT_EAR', 'RIGHT_EAR'))
    >>> detector.run()
    >>> detector.save()

    >>> detector = YoloNVDECInference(video_path=r'/videos/my_video.mp4', engine_path=r'/seg.engine', task='segment', save_dir=r'/output', vertice_cnt=30)
    >>> detector.run()
    >>> detector.save()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 engine_path: Union[str, os.PathLike],
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 task: Literal['detect', 'pose', 'segment'] = 'detect',
                 imsz: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 max_workers: Optional[int] = None,
                 gpu_id: Union[int, Tuple[int, ...]] = 0,
                 conf_threshold: float = 0.05,
                 iou_threshold: float = 0.45,
                 keypoint_names: Optional[Tuple[str, ...]] = None,
                 vertice_cnt: int = 60,
                 max_detections: Optional[int] = None,
                 segment_smoothing: Optional[int] = None,
                 interpolate: bool = True,
                 recursive: bool = False,
                 smoothing_method: Optional[Literal['savitzky-golay', 'bartlett', 'blackman', 'boxcar', 'cosine', 'gaussian', 'hamming', 'exponential']] = None,
                 smoothing_time_window: Optional[int] = None,
                 verbose: bool = True):

        get_pkg_version(pkg='PyNvVideoCodec', raise_error=True)
        get_pkg_version(pkg='ultralytics', raise_error=True)
        get_pkg_version(pkg='torchvision', raise_error=True)
        get_pkg_version(pkg='torch', raise_error=True)
        check_nvidea_gpu_available(raise_error=True)
        check_file_exist_and_readable(file_path=engine_path)
        if save_dir is not None: check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir')
        check_str(name=f'{self.__class__.__name__} task', value=task, options=TASKS)

        if imsz is None or batch_size is None:
            engine_meta = read_yolo_metadata(model=engine_path)
            if imsz is None:
                engine_imsz = engine_meta.get('imgsz', None)
                if engine_imsz is None:
                    raise InvalidInputError(msg='Could not read imgsz from engine metadata. Pass imsz explicitly.', source=self.__class__.__name__)
                imsz = engine_imsz[0] if isinstance(engine_imsz, (list, tuple)) else int(engine_imsz)
                if verbose: stdout_information(msg=f'Read imsz={imsz} from engine metadata.', source=self.__class__.__name__)
            if batch_size is None:
                engine_batch = engine_meta.get('batch', None)
                if engine_batch is None:
                    raise InvalidInputError(msg='Could not read batch size from engine metadata. Pass batch_size explicitly.', source=self.__class__.__name__)
                batch_size = int(engine_batch)
                if verbose: stdout_information(msg=f'Read batch_size={batch_size} from engine metadata.', source=self.__class__.__name__)

        check_int(name=f'{self.__class__.__name__} imsz', value=imsz, min_value=32)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        if isinstance(gpu_id, int):
            gpu_ids = (gpu_id,)
        else:
            check_valid_tuple(x=gpu_id, source=f'{self.__class__.__name__} gpu_id', minimum_length=1, valid_dtypes=(int,))
            gpu_ids = tuple(gpu_id)
        for gid in gpu_ids:
            check_int(name=f'{self.__class__.__name__} gpu_id', value=gid, min_value=0)
        if max_workers is not None:
            check_int(name=f'{self.__class__.__name__} max_workers', value=max_workers, min_value=1, max_value=find_core_cnt()[0])
        else:
            _, gpu_info = _is_cuda_available()
            max_workers = 0
            for gid in gpu_ids:
                gpu_name = gpu_info[gid]['model'] if gpu_info and gid in gpu_info else None
                nvdec_cnt = get_nvdec_count(gpu_name=gpu_name)
                if verbose: stdout_information(msg=f'Auto-detected {nvdec_cnt} NVDEC engine(s) for GPU {gid} ({gpu_name}).', source=self.__class__.__name__)
                max_workers += nvdec_cnt
        check_float(name=f'{self.__class__.__name__} conf_threshold', value=conf_threshold, min_value=0.0, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} iou_threshold', value=iou_threshold, min_value=0.0, max_value=1.0)
        check_valid_boolean(value=[verbose, interpolate, recursive], source=f'{self.__class__.__name__} verbose/interpolate/recursive', raise_error=True)
        if task == POSE:
            if keypoint_names is None:
                raise InvalidInputError(msg='keypoint_names is required when task is "pose".', source=self.__class__.__name__)
            check_valid_tuple(x=keypoint_names, source=f'{self.__class__.__name__} keypoint_names', minimum_length=1, valid_dtypes=(str,))
        if task == SEGMENT:
            check_int(name=f'{self.__class__.__name__} vertice_cnt', value=vertice_cnt, min_value=3)
        if max_detections is not None:
            check_int(name=f'{self.__class__.__name__} max_detections', value=max_detections, min_value=1)
        if segment_smoothing is not None:
            check_int(name=f'{self.__class__.__name__} segment_smoothing', value=segment_smoothing, min_value=1)
        if smoothing_method is not None:
            check_str(name=f'{self.__class__.__name__} smoothing_method', value=smoothing_method, options=SMOOTHING_METHODS)
            check_float(name=f'{self.__class__.__name__} smoothing_time_window', value=smoothing_time_window, min_value=10e-6)

        engine_dir = os.path.dirname(str(engine_path))
        engine_stem, engine_ext = os.path.splitext(os.path.basename(str(engine_path)))
        primary_gpu = gpu_ids[0]
        self.engine_paths = {primary_gpu: str(engine_path)}
        for gid in gpu_ids[1:]:
            if gid == primary_gpu:
                continue
            gpu_engine = os.path.join(engine_dir, f'{engine_stem}_gpu{gid}{engine_ext}')
            if not os.path.isfile(gpu_engine):
                from simba.utils.yolo import export_yolo_model
                pt_path = os.path.join(engine_dir, f'{engine_stem}.pt')
                if not os.path.isfile(pt_path):
                    pt_candidates = [f for f in os.listdir(engine_dir) if f.endswith('.pt')]
                    if pt_candidates:
                        pt_path = os.path.join(engine_dir, pt_candidates[0])
                    else:
                        raise InvalidInputError(msg=f'No .pt weights found in {engine_dir} to build engine for GPU {gid}. Export an engine per GPU or place the .pt file alongside the engine.', source=self.__class__.__name__)
                check_file_exist_and_readable(file_path=pt_path)
                if verbose: stdout_information(msg=f'Building TensorRT engine for GPU {gid} from {pt_path}...', source=self.__class__.__name__)
                engine_meta = read_yolo_metadata(model=str(engine_path))
                half = engine_meta.get('fp16', False)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_pt = os.path.join(tmp_dir, os.path.basename(pt_path))
                    shutil.copy2(pt_path, tmp_pt)
                    export_yolo_model(model_path=tmp_pt, export_format='engine', imgsz=imsz, device=gid, batch=batch_size, half=half, task=task)
                    tmp_engine = os.path.join(tmp_dir, f'{os.path.splitext(os.path.basename(pt_path))[0]}.engine')
                    shutil.move(tmp_engine, gpu_engine)
                if verbose: stdout_information(msg=f'Engine for GPU {gid} saved to {gpu_engine}.', source=self.__class__.__name__)
            self.engine_paths[gid] = gpu_engine

        self.save_dir = str(save_dir) if save_dir is not None else None
        self.task, self.imsz, self.batch_size, self.max_workers, self.gpu_ids = task, imsz, batch_size, max_workers, gpu_ids
        self.conf_threshold, self.iou_threshold, self.verbose = conf_threshold, iou_threshold, verbose
        self.keypoint_names, self.vertice_cnt, self.max_detections, self.segment_smoothing, self.interpolate = keypoint_names, vertice_cnt, max_detections, segment_smoothing, interpolate
        self.smoothing_method, self.smoothing_time_window = smoothing_method, smoothing_time_window
        if os.path.isfile(str(video_path)):
            check_file_exist_and_readable(file_path=str(video_path))
            self.video_dir = os.path.dirname(str(video_path))
            video_name = get_fn_ext(filepath=str(video_path))[1]
            self.video_paths = {video_name: str(video_path)}
        else:
            check_if_dir_exists(in_dir=video_path, source=f'{self.__class__.__name__} video_path')
            self.video_dir = str(video_path)
            if recursive:
                self.video_paths = {}
                for root, _, files in os.walk(self.video_dir):
                    for f in files:
                        if f.lower().endswith((".avi", ".mp4", ".mov", ".flv", ".m4v", ".webm")):
                            fpath = os.path.join(root, f)
                            vname = get_fn_ext(filepath=fpath)[1]
                            self.video_paths[vname] = fpath
                if not self.video_paths:
                    raise NoFilesFoundError(msg=f'No videos found recursively in {self.video_dir}.', source=self.__class__.__name__)
                if verbose: stdout_information(msg=f'Found {len(self.video_paths)} video(s) recursively in {self.video_dir}.', source=self.__class__.__name__)
            else:
                self.video_paths = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=True)
        self.results = {}
        self.timings = {}

    @staticmethod
    def _interpolate_df(df, task, keypoint_names=None, vertice_cnt=None):
        if len(df) == 0:
            return df
        if task == DETECT or task == POSE:
            interp_cols = list(COORD_COLS)
            if task == POSE and keypoint_names is not None:
                for name in keypoint_names:
                    interp_cols.extend([f'{name}_X'.upper(), f'{name}_Y'.upper()])
            group_col = CLASS_ID
        elif task == SEGMENT:
            interp_cols = []
            for i in range(vertice_cnt):
                interp_cols.extend([f'{VERTICE}_{i}_X', f'{VERTICE}_{i}_Y'])
            group_col = ID
        else:
            return df
        for grp_id in df[group_col].unique():
            if grp_id == -1:
                continue
            grp_df = df[df[group_col] == int(grp_id)].copy()
            if grp_df.empty:
                continue
            for col in interp_cols:
                if col not in grp_df.columns:
                    continue
                grp_df[col] = pd.to_numeric(grp_df[col], errors="coerce").astype(np.float32)
                grp_df[col] = grp_df[col].replace(-1, np.nan)
                grp_df[col] = (grp_df[col].interpolate(method='linear', axis=0).ffill().bfill().replace([np.inf, -np.inf], np.nan).round().fillna(-1).astype(np.int32))
            df.update(grp_df)
        if task in (DETECT, POSE):
            df[CONFIDENCE] = 0
        return df

    def _build_columns(self):
        if self.task == DETECT:
            return list(DETECT_COLS)
        elif self.task == POSE:
            kp_cols = [f'{name}_{s}'.upper() for name in self.keypoint_names for s in ('X', 'Y', 'P')]
            return list(DETECT_COLS) + kp_cols
        elif self.task == SEGMENT:
            cols = [FRAME, ID]
            for i in range(self.vertice_cnt):
                cols.extend([f'{VERTICE}_{i}_X', f'{VERTICE}_{i}_Y'])
            return cols

    def run(self):
        n_workers = min(self.max_workers, len(self.video_paths))
        n_videos = len(self.video_paths)
        if self.verbose: stdout_information(msg=f'{n_videos} video(s), {n_workers} worker(s), task={self.task}.', source=self.__class__.__name__)

        columns = self._build_columns()
        ctx = mp.get_context('spawn')
        task_queue, result_queue, ready_queue = ctx.Queue(), ctx.Queue(), ctx.Queue()
        workers = []
        for i in range(n_workers):
            worker_gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            worker_engine = self.engine_paths[worker_gpu_id]
            p = ctx.Process(
                target=_worker,
                args=(task_queue, result_queue, ready_queue, worker_engine,
                      self.batch_size, self.imsz, worker_gpu_id,
                      self.conf_threshold,
                      self.max_detections, self.task, self.vertice_cnt, self.segment_smoothing,
                      self.save_dir, columns, self.interpolate, self.keypoint_names),
                name=f"Worker-{i}-GPU{worker_gpu_id}",
            )
            p.start()
            workers.append(p)

        alive_count = 0
        class_names = {}
        for _ in range(n_workers):
            tag, ok, cn = ready_queue.get()
            if ok:
                alive_count += 1
                if not class_names: class_names = cn
            else:
                if self.verbose: stdout_information(msg=f'{tag} failed to initialize.', source=self.__class__.__name__)

        if alive_count == 0:
            raise SimBAGPUError(msg='All NVDEC workers failed to initialize. Check GPU memory and TensorRT engine compatibility.', source=self.__class__.__name__)

        timer = SimbaTimer(start=True)
        for video_idx, video_path in enumerate(self.video_paths.values(), 1):
            task_queue.put((video_path, video_idx, n_videos))
        for _ in range(alive_count): task_queue.put(None)

        total_frames, total_rows, video_cnt = 0, 0, 0
        for _ in range(n_videos):
            video_path, rows_or_cnt, timings, err = result_queue.get()
            video_name = get_fn_ext(filepath=video_path)[1]
            if err:
                if self.verbose: stdout_information(msg=f'ERROR {video_name}: {err}', source=self.__class__.__name__)
                continue
            if self.save_dir is not None:
                row_cnt = rows_or_cnt
                video_meta = get_video_meta_data(video_path=video_path)
                total_frames += video_meta['frame_count']
                total_rows += row_cnt
                self.timings[video_name] = timings
                if self.verbose: stdout_information(msg=f'Saved {video_name}.csv ({row_cnt} rows)', source=self.__class__.__name__)
            else:
                rows = rows_or_cnt
                df = pd.DataFrame(rows, columns=columns)
                if self.interpolate and len(df) > 0:
                    df = self._interpolate_df(df, task=self.task, keypoint_names=self.keypoint_names, vertice_cnt=self.vertice_cnt)
                if self.smoothing_method is not None and self.task == DETECT and len(df) > 0:
                    video_meta = get_video_meta_data(video_path=video_path)
                    if self.smoothing_method != SAVITZKY_GOLAY:
                        smoothened = df_smoother(data=df[COORD_COLS], fps=video_meta['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__, method=self.smoothing_method)
                    else:
                        smoothened = savgol_smoother(data=df[COORD_COLS], fps=video_meta['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__)
                    df.update(smoothened)
                self.timings[video_name] = timings
                video_meta = get_video_meta_data(video_path=video_path)
                total_frames += video_meta['frame_count']
                total_rows += len(df)
                self.results[video_name] = df
            video_cnt += 1

        for p in workers:
            p.join()

        timer.stop_timer()
        if self.verbose:
            elapsed_s = timer.elapsed_time
            fps_str = f', {total_frames / elapsed_s:.1f} FPS' if total_frames > 0 and elapsed_s > 0 else ''
            agg_timings = {}
            for t in self.timings.values():
                for k, v in t.items():
                    agg_timings[k] = agg_timings.get(k, 0.0) + v
            n_workers_used = max(1, min(self.max_workers, video_cnt))
            timing_parts = []
            for k in ('decode', 'preprocess', 'infer', 'postprocess', 'save'):
                if k in agg_timings:
                    wall_s = agg_timings[k] / n_workers_used
                    pct = (wall_s / elapsed_s * 100) if elapsed_s > 0 else 0
                    timing_parts.append(f'{k}={wall_s:.1f}s/{pct:.0f}%')
            timing_str = f' [{", ".join(timing_parts)}]' if timing_parts else ''
            if self.save_dir is not None:
                stdout_success(msg=f'{video_cnt} video(s), {total_frames} frames, {total_rows} rows saved in {self.save_dir}{fps_str}{timing_str}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)
            else:
                stdout_success(msg=f'{video_cnt} video(s), {total_frames} frames, {total_rows} rows{fps_str}{timing_str}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

    def save(self):
        if self.save_dir is None:
            raise InvalidInputError(msg='save_dir is None. Pass a save_dir to __init__ or set self.save_dir before calling save().', source=self.__class__.__name__)
        if len(self.results) == 0:
            raise NoFilesFoundError(msg='No results to save. Call run() first or pass save_dir to __init__ to save during run.', source=self.__class__.__name__)
        for video_name, df in self.results.items():
            csv_path = os.path.join(self.save_dir, f'{video_name}.csv')
            write_df(df=df, file_type='csv', save_path=csv_path)
            if self.verbose: stdout_information(msg=f'Saved {csv_path} ({len(df)} rows)', source=self.__class__.__name__)


# if __name__ == "__main__":
#      detector = YoloNVDECInference(video_path=r"/home/cat/simon/data/lp_videos",
#                                   engine_path=r"/home/cat/simon/yolo_project_0403/mdl/train3/weights/best.engine",
#                                   task='detect',
#                                   gpu_id=(0, 1,),
#                                   conf_threshold=0.5,
#                                   max_detections=1,
#                                   interpolate=True,
#                                   recursive=True,
#                                   segment_smoothing=10,
#                                   save_dir=r'/home/cat/simon/data/new_test_videos_to_crop/detection_multi_gpu_test',
#                                   vertice_cnt=500)
#      detector.run()
#

# if __name__ == "__main__":
#     detector = YoloNVDECInference(video_path=r"E:\open_video\open_field_2\sample\clips",
#                                  engine_path=r"E:\open_video\open_field_2\yolo_seg_project\mdl\train\weights\best.engine",
#                                  task='segment',
#                                  conf_threshold=0.5,
#                                  max_detections=1,
#                                  interpolate=True,
#                                  segment_smoothing=10,
#                                  save_dir=r'E:\open_video\open_field_2\yolo_seg_project\results',
#                                  vertice_cnt=500)
#     detector.run()


# detector = YoloNVDECInference(video_path=r'/videos',
#                              engine_path=r'/pose.engine',
#                              task='pose',
#                              keypoint_names=('NOSE', 'LEFT_EAR', 'RIGHT_EAR'))
# detector.run()
# detector.save()