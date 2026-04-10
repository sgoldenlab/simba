import os
import time
from multiprocessing import Process, Queue, current_process
from typing import Optional, Tuple, Union

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


def _xyxy_to_corners(x1, y1, x2, y2):
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _process_one_video(video_path, trt_model, batch_buf, batch_size, imsz,
                       gpu_id, conf_threshold, max_detections,
                       task, class_names, vertice_cnt=30, segment_smoothing=None):

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
    stdout_information(msg=f'{stem}: {n_total} frames ({orig_w}x{orig_h})', source=tag)

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

        pred = pred_tensor[:n_frames]
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
                x1_lb, y1_lb, x2_lb, y2_lb, conf, cls_id = vals[:6].cpu().tolist()
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
                    mask_np = (crop_up > 0.5).cpu().numpy().astype(np.uint8)
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

            if task == DETECT:
                for cls_id, cls_name in class_names.items():
                    if cls_id not in detected_class_ids:
                        all_rows.append([frm_idx, cls_id, cls_name] + [-1] * 9)

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
            max_detections, task, vertice_cnt=30, segment_smoothing=None):

    tag = current_process().name
    dev = f"cuda:{gpu_id}"
    try:
        model = YOLO(engine_path, task=task if task != 'pose' else 'pose')
        dummy = torch.zeros(batch_size, 3, imsz, imsz, dtype=torch.float16, device=dev)
        model.predict(source=dummy, device=gpu_id, imgsz=imsz, verbose=False)
        del dummy
        trt_model = model.predictor.model
        buf_dtype = torch.float16 if getattr(trt_model, 'fp16', False) else torch.float32
        batch_buf = torch.empty(batch_size, 3, imsz, imsz, dtype=buf_dtype, device=dev)
        class_names = model.names if hasattr(model, 'names') else {}
        stdout_information(msg=f'Engine ready ({task}).', source=tag)
    except Exception as exc:
        stdout_information(msg=f'FAILED: {exc}', source=tag)
        ready_queue.put((tag, False, {}))
        return

    ready_queue.put((tag, True, class_names))

    while True:
        item = task_queue.get()
        if item is None:
            break
        video_path = item
        try:
            rows, timings = _process_one_video(
                video_path=video_path, trt_model=trt_model, batch_buf=batch_buf,
                batch_size=batch_size, imsz=imsz, gpu_id=gpu_id,
                conf_threshold=conf_threshold,
                max_detections=max_detections, task=task, class_names=class_names,
                vertice_cnt=vertice_cnt, segment_smoothing=segment_smoothing,
            )
            result_queue.put((video_path, rows, timings, None))
        except Exception as exc:
            stdout_information(msg=f'ERROR on {video_path}: {exc}', source=tag)
            result_queue.put((video_path, [], {}, str(exc)))


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
    :param Union[str, os.PathLike] engine_path: Path to TensorRT engine file (.engine). If alternative model file exist, convert it to engine ysing `simba.utils.yolo.export_yolo_model`.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory for per-video CSV output. If None, results kept in memory only. Default None.
    :param Literal['detect', 'pose', 'segment'] task: YOLO task type. Default ``'detect'``.
    :param int imsz: Model input image size (square). Default 256.
    :param int batch_size: Inference batch size (must match engine batch profile). Default 192.
    :param Optional[int] max_workers: Number of parallel worker processes. If None, auto-detected from GPU NVDEC count. Default None.
    :param int gpu_id: CUDA device index. Default 0.
    :param float conf_threshold: Confidence threshold for detections. Default 0.05.
    :param float iou_threshold: IoU threshold for NMS. Default 0.45.
    :param Optional[Tuple[str, ...]] keypoint_names: Keypoint names in index order, used only when ``task='pose'`` (ignored otherwise). Required when ``task='pose'``, raises error if not provided.
    :param int vertice_cnt: Number of resampled polygon vertices, used only when ``task='segment'`` (ignored otherwise). Default 30.
    :param Optional[int] max_detections: Maximum number of detections to keep per frame after NMS (sorted by confidence). If None, keep all. Default None.
    :param Optional[int] segment_smoothing: B-spline smoothing factor for segmentation polygon vertices, used only when ``task='segment'`` (ignored otherwise). Higher values produce smoother contours. If ``None``, no smoothing is applied. Default ``None``.
    :param bool interpolate: If True, linearly interpolate missing detections across frames, used only when ``task='detect'`` (ignored otherwise). Default True.
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
                 imsz: int = 256,
                 batch_size: int = 192,
                 max_workers: Optional[int] = None,
                 gpu_id: int = 0,
                 conf_threshold: float = 0.05,
                 iou_threshold: float = 0.45,
                 keypoint_names: Optional[Tuple[str, ...]] = None,
                 vertice_cnt: int = 60,
                 max_detections: Optional[int] = None,
                 segment_smoothing: Optional[int] = None,
                 interpolate: bool = True,
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
        check_int(name=f'{self.__class__.__name__} imsz', value=imsz, min_value=32)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        if max_workers is not None:
            check_int(name=f'{self.__class__.__name__} max_workers', value=max_workers, min_value=1, max_value=find_core_cnt()[0])
        else:
            _, gpu_info = _is_cuda_available()
            gpu_name = gpu_info[gpu_id]['model'] if gpu_info and gpu_id in gpu_info else None
            max_workers = get_nvdec_count(gpu_name=gpu_name)
            if verbose: stdout_information(msg=f'Auto-detected {max_workers} NVDEC engine(s) for {gpu_name}.', source=self.__class__.__name__)
        check_int(name=f'{self.__class__.__name__} gpu_id', value=gpu_id, min_value=0)
        check_float(name=f'{self.__class__.__name__} conf_threshold', value=conf_threshold, min_value=0.0, max_value=1.0)
        check_float(name=f'{self.__class__.__name__} iou_threshold', value=iou_threshold, min_value=0.0, max_value=1.0)
        check_valid_boolean(value=[verbose, interpolate], source=f'{self.__class__.__name__} verbose/interpolate', raise_error=True)
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

        self.engine_path = str(engine_path)
        self.save_dir = str(save_dir) if save_dir is not None else None
        self.task, self.imsz, self.batch_size, self.max_workers, self.gpu_id = task, imsz, batch_size, max_workers, gpu_id
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
            self.video_paths = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=True)
        self.results = {}
        self.timings = {}

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
        if self.verbose: stdout_information(msg=f'{len(self.video_paths)} video(s), {n_workers} worker(s), task={self.task}.', source=self.__class__.__name__)

        task_queue, result_queue, ready_queue = Queue(), Queue(), Queue()
        workers = []
        for i in range(n_workers):
            p = Process(
                target=_worker,
                args=(task_queue, result_queue, ready_queue, self.engine_path,
                      self.batch_size, self.imsz, self.gpu_id,
                      self.conf_threshold,
                      self.max_detections, self.task, self.vertice_cnt, self.segment_smoothing),
                name=f"Worker-{i}",
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
        for video_path in self.video_paths.values(): task_queue.put(video_path)
        for _ in range(alive_count): task_queue.put(None)

        columns = self._build_columns()
        total_frames, video_cnt = 0, 0
        per_video_fps = []
        for _ in range(len(self.video_paths)):
            video_path, rows, timings, err = result_queue.get()
            video_name = get_fn_ext(filepath=video_path)[1]
            if err:
                if self.verbose: stdout_information(msg=f'ERROR {video_name}: {err}', source=self.__class__.__name__)
                continue
            df = pd.DataFrame(rows, columns=columns)
            if self.interpolate and self.task == DETECT and len(df) > 0:
                for cls_id in df[CLASS_ID].unique():
                    if cls_id == -1:
                        continue
                    class_df = df[df[CLASS_ID] == int(cls_id)].copy()
                    if class_df.empty:
                        continue
                    for coord_col in COORD_COLS:
                        class_df[coord_col] = pd.to_numeric(class_df[coord_col], errors="coerce").astype(np.float32)
                        class_df[coord_col] = class_df[coord_col].replace(-1, np.nan)
                        class_df[coord_col] = (class_df[coord_col].interpolate(method='linear', axis=0).ffill().bfill().replace([np.inf, -np.inf], np.nan).round().fillna(-1).astype(np.int32))
                    df.update(class_df)
                df[CONFIDENCE] = 0
            if self.smoothing_method is not None and self.task == DETECT and len(df) > 0:
                video_meta = get_video_meta_data(video_path=video_path)
                if self.smoothing_method != SAVITZKY_GOLAY:
                    smoothened = df_smoother(data=df[COORD_COLS], fps=video_meta['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__, method=self.smoothing_method)
                else:
                    smoothened = savgol_smoother(data=df[COORD_COLS], fps=video_meta['fps'], time_window=self.smoothing_time_window, source=self.__class__.__name__)
                df.update(smoothened)
            self.timings[video_name] = timings
            vid_elapsed = sum(timings.values())
            if vid_elapsed > 0:
                video_meta = get_video_meta_data(video_path=video_path)
                per_video_fps.append(video_meta['frame_count'] / vid_elapsed)
            total_frames += len(df)
            video_cnt += 1
            if self.save_dir is not None:
                csv_path = os.path.join(self.save_dir, f'{video_name}.csv')
                write_df(df=df, file_type='csv', save_path=csv_path)
                if self.verbose: stdout_information(msg=f'Saved {csv_path} ({len(df)} rows)', source=self.__class__.__name__)
            else:
                self.results[video_name] = df

        for p in workers:
            p.join()

        timer.stop_timer()
        if self.verbose:
            fps_str = ''
            if per_video_fps:
                mean_fps = float(np.mean(per_video_fps))
                std_fps = float(np.std(per_video_fps))
                fps_str = f', {mean_fps:.1f}±{std_fps:.1f} FPS'
            if self.save_dir is not None:
                stdout_success(msg=f'{video_cnt} video(s), {total_frames} rows saved in {self.save_dir}{fps_str}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)
            else:
                stdout_success(msg=f'{video_cnt} video(s), {total_frames} rows{fps_str}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)

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
#     detector = YoloNVDECInference(video_path=r"E:\open_video\open_field_2\sample\clips",
#                                  engine_path=r"E:\open_video\open_field_2\yolo_seg_project\mdl\train\weights\best.engine",
#                                  task='segment',
#                                  batch_size=1,
#                                  imsz=1240,
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