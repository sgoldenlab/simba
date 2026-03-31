from __future__ import annotations

import os
from typing import Tuple, Union, Optional
from multiprocessing import Process, Queue, current_process
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import time
try:
    from PyNvVideoCodec import SimpleDecoder, OutputColorType
    from ultralytics import YOLO
    import torch
    import torch.nn.functional as F
except:
    ultralytics = None
    SimpleDecoder = None
    OutputColorType = None
    torch = None
    F = None
from simba.utils.printing import stdout_information
from simba.utils.lookups import get_nvdec_count
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_video_meta_data, get_fn_ext, get_pkg_version
from simba.utils.enums import Formats, Options
from simba.utils.checks import (check_float, check_if_dir_exists, check_int,
                                check_nvidea_gpu_available, check_valid_boolean,
                                check_valid_device, check_valid_tuple)
from simba.utils.errors import InvalidVideoFileError
from simba.utils.yolo import apply_fixed_bbox_size

DETECTION_COLUMNS = ("frame", "x1", "y1", "x2", "y2", "confidence", "class_id")
COL_FRAME, COL_X1, COL_Y1, COL_X2, COL_Y2, COL_CONFIDENCE, COL_CLASS_ID = DETECTION_COLUMNS
FIXED_X1, FIXED_Y1, FIXED_X2, FIXED_Y2 = "X1", "Y1", "X2", "Y2"
FIXED_X3, FIXED_Y3, FIXED_X4, FIXED_Y4 = "X3", "Y3", "X4", "Y4"
FIXED_CONFIDENCE = "CONFIDENCE"

class NVDECYoloInference:
    """
    Batch object detection on videos using **NVIDIA NVDEC** (``PyNvVideoCodec.SimpleDecoder``) for GPU decode and **Ultralytics YOLO** with a TensorRT ``.engine`` for inference.


    .. csv-table::
       :header: EXPECTED RUNTIMES (SINGLE NVDEC)
       :file: ../../docs/tables/NVDECYoloInference.csv
       :widths: 10, 10, 40, 40
       :align: center
       :header-rows: 1

    .. seealso::

       :class:`~simba.model.yolo_inference.YoloInference`
          Default SimBA path for bounding-box detection on video (``.pt`` / ONNX, optional smoothing and gap interpolation, project-oriented outputs).

       :func:`~simba.utils.yolo.fit_yolo`
          Train YOLO weights on a SimBA-style dataset.

       :func:`~simba.utils.yolo.export_yolo_model`
          Export a trained checkpoint to TensorRT or other backends; an exported ``.engine`` can be used as ``engine_path`` here.

    :param video_path: Single video file or directory of videos (see ``Options.ALL_VIDEO_FORMAT_OPTIONS``).
    :param engine_path: Path to a YOLO detection TensorRT engine (``ultralytics.YOLO``).
    :param img_size: Square side length, or ``(height, width)`` for resize before inference.
    :param batch_size: Frames decoded and inferred per step (tensor batch).
    :param device: CUDA device index (integer GPU id).
    :param verbose: Log progress to stdout via :func:`simba.utils.printing.stdout_information`.
    :param save_dir: If set, write CSVs ``<stem>.csv`` here; if ``None``, decode + forward pass only (no CSV).
    :param conf_thres: YOLO confidence threshold in ``[0, 1]``.
    :param iou_thres: YOLO IoU threshold for NMS in ``[0, 1]``.
    :param max_detections: Maximum boxes per frame passed to YOLO.
    :param bbox_size: Optional ``(width, height)`` fixed box size in pixels for SimBA post-processing; ``None`` skips.

    :example:
    >>>
    >>> x = NVDECYoloInference(video_path=r"E:\litpose_yolo\pi\videos_downsampled", engine_path=r"E:\litpose_yolo\pi\yolo\mdl\train3\weights\best.engine", save_dir=r"E:\litpose_yolo\pi\csv_results")
    >>> x.run()

    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 engine_path: Union[str, os.PathLike],
                 img_size: Union[int, Tuple[int, int]] = 256,
                 batch_size: int = 10,
                 device: int = 0,
                 verbose: bool = True,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 max_detections: int = 300,
                 bbox_size: Optional[Tuple[int, int]] = None):

        check_nvidea_gpu_available(raise_error=True)
        check_valid_device(device=device, raise_error=True)
        _ = get_pkg_version(pkg='PyNvVideoCodec', raise_error=True)
        _ = get_pkg_version(pkg='ultralytics', raise_error=True)

        check_float(name="conf_thres", value=conf_thres, min_value=0.0, max_value=1.0)
        check_float(name="iou_thres", value=iou_thres, min_value=0.0, max_value=1.0)
        check_int(name=f"{self.__class__.__name__} max_detections", value=max_detections, min_value=1, raise_error=True)
        if bbox_size is not None:
            check_valid_tuple(x=bbox_size, source=f'{self.__class__.__name__} bbox_size', accepted_lengths=(2,), valid_dtypes=Formats.INTEGER_DTYPES.value, min_integer=1, raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose')
        if os.path.isdir(video_path):
            self.video_paths = find_files_of_filetypes_in_directory(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, sort_alphabetically=True, raise_error=True)
        elif os.path.isfile(video_path):
            self.video_paths = [video_path]
        else:
            raise InvalidVideoFileError(msg=f'video_path ({video_path}) is not a valid video path or video directory', source=self.__class__.__name__)
        _ = [get_video_meta_data(video_path=x) for x in self.video_paths]
        if save_dir is not None: check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', raise_error=True)
        self.n_workers = min(get_nvdec_count(), len(self.video_paths))
        if verbose: stdout_information(msg=f"Found {len(self.video_paths)} video(s). Launching {self.n_workers} worker(s).", source=self.__class__.__name__)
        self.task_queue, self.result_queue, self.ready_queue = Queue(), Queue(), Queue()
        self.engine_path, self.verbose, self.batch_size = engine_path, verbose, batch_size
        self.img_size, self.gpu_id, self.save_dir = img_size, device, save_dir
        self.conf_thres, self.iou_thres, self.max_detections, self.bbox_size = conf_thres, iou_thres, max_detections, bbox_size

    def _extend_detections_from_predict(self, rows: list, results: list, start: int, n_frames: int) -> None:
        for i in range(n_frames):
            if i >= len(results):
                break
            frame_idx = start + i
            boxes = results[i].boxes
            if boxes is None or len(boxes) == 0:
                continue
            for row in boxes.data:
                x1, y1, x2, y2, conf, cls = row[:6].tolist()
                rows.append(
                    {
                        COL_FRAME: frame_idx,
                        COL_X1: round(float(x1), 2),
                        COL_Y1: round(float(y1), 2),
                        COL_X2: round(float(x2), 2),
                        COL_Y2: round(float(y2), 2),
                        COL_CONFIDENCE: round(float(conf), 4),
                        COL_CLASS_ID: int(cls),
                    }
                )

    def _interpolate_hw(self) -> Tuple[int, int]:
        if isinstance(self.img_size, int):
            return self.img_size, self.img_size
        return int(self.img_size[0]), int(self.img_size[1])

    @staticmethod
    def _detections_df_for_fixed_bbox(df: pd.DataFrame) -> pd.DataFrame:
        y = df.copy()
        y[FIXED_X1], y[FIXED_Y1], y[FIXED_X2], y[FIXED_Y2] = y[COL_X1], y[COL_Y1], y[COL_X2], y[COL_Y1]
        y[FIXED_X3], y[FIXED_Y3], y[FIXED_X4], y[FIXED_Y4] = y[COL_X2], y[COL_Y2], y[COL_X1], y[COL_Y2]
        y[FIXED_CONFIDENCE] = y[COL_CONFIDENCE]
        return y

    @staticmethod
    def _detections_df_from_fixed_bbox(y: pd.DataFrame) -> pd.DataFrame:
        out = y.copy()
        out[COL_X1], out[COL_Y1], out[COL_X2], out[COL_Y2] = out[FIXED_X1], out[FIXED_Y1], out[FIXED_X3], out[FIXED_Y3]
        return out[list(DETECTION_COLUMNS)]

    @staticmethod
    def _no_detection_pad_dataframe(missing: list[int]) -> pd.DataFrame:
        vals = {c: (-1.0 if c == COL_CONFIDENCE else -1) for c in DETECTION_COLUMNS[1:]}
        return pd.DataFrame({COL_FRAME: missing, **vals})

    @staticmethod
    def _insert_missing_frames_no_detection(df: pd.DataFrame, total_frames: int) -> pd.DataFrame:
        if total_frames <= 0:
            return df
        seen = set(df[COL_FRAME].astype(int).unique()) if len(df) else set()
        missing = [f for f in range(total_frames) if f not in seen]
        if not missing:
            return df.sort_values(COL_FRAME, kind="mergesort").reset_index(drop=True)
        pad = NVDECYoloInference._no_detection_pad_dataframe(missing)
        out = pd.concat([df, pad], ignore_index=True)
        return out.sort_values(COL_FRAME, kind="mergesort").reset_index(drop=True)

    def _apply_bbox_size_to_dataframe(self, df: pd.DataFrame, video_label: str, img_h: int, img_w: int) -> pd.DataFrame:
        if self.bbox_size is None or len(df) == 0:
            return df
        y = self._detections_df_for_fixed_bbox(df)
        y = apply_fixed_bbox_size(data=y, video_name=video_label, img_w=img_w, img_h=img_h, bbox_size=self.bbox_size)
        return self._detections_df_from_fixed_bbox(y)

    def run_one_video(self, video_path: str, model: YOLO, batch_buf: torch.Tensor) -> Tuple[int, float]:
        engine = model.predictor.model
        tag = current_process().name
        decoder = SimpleDecoder(video_path, gpu_id=self.gpu_id, use_device_memory=True, output_color_type=OutputColorType.RGBP)

        _, video_fn, video_ext = get_fn_ext(video_path)
        video_file = f"{video_fn}{video_ext}"

        csv_path = None
        detection_rows: Optional[list] = None
        if self.save_dir is not None:
            csv_path = os.path.join(self.save_dir, f"{video_fn}.csv")
            detection_rows = []

        total_frames = len(decoder)
        if self.verbose: stdout_information(msg=f"[{tag}] {video_file}: {total_frames} frames", source=self.__class__.__name__)

        run_t0 = time.perf_counter()
        result_count = 0
        ih, iw = self._interpolate_hw()

        for start in range(0, total_frames, self.batch_size):
            end = min(start + self.batch_size, total_frames)
            n_frames = end - start
            indices = np.arange(start, end, dtype=int)

            decoded_frames = decoder.get_batch_frames_by_index(indices)

            for i, frame in enumerate(decoded_frames):
                t = torch.from_dlpack(frame).unsqueeze(0).half()
                t.mul_(1.0 / 255.0)
                batch_buf[i:i + 1] = F.interpolate(t, size=(ih, iw), mode="bilinear", align_corners=False)

            del decoded_frames
            if n_frames < self.batch_size:
                batch_buf[n_frames:] = batch_buf[n_frames - 1:n_frames].expand(self.batch_size - n_frames, -1, -1, -1)

            if detection_rows is not None:
                predict_out = model.predict(source=batch_buf, device=self.gpu_id, imgsz=self.img_size, batch=self.batch_size, verbose=False, conf=self.conf_thres, iou=self.iou_thres, max_det=self.max_detections, stream=False)
                self._extend_detections_from_predict(detection_rows, predict_out, start, n_frames)
            else:
                with torch.inference_mode():
                    _ = engine(batch_buf)

            result_count += n_frames

            if self.verbose: stdout_information(msg=f"[{tag}] {video_file} frames {start}-{end - 1}/{total_frames - 1}", source=self.__class__.__name__)

        del decoder

        if detection_rows is not None and self.save_dir is not None:
            df = pd.DataFrame(detection_rows, columns=list(DETECTION_COLUMNS))
            df = self._apply_bbox_size_to_dataframe(df=df, video_label=video_file, img_h=ih, img_w=iw)
            df = self._insert_missing_frames_no_detection(df=df, total_frames=total_frames)
            df.to_csv(csv_path, index=False)


        run_elapsed = time.perf_counter() - run_t0
        fps = total_frames / max(run_elapsed, 1e-9)
        if self.verbose:
            stdout_information(msg=f"[{tag}] {video_file} done: {result_count} frames in {run_elapsed:.4f}s ({fps:.1f} FPS)", source=self.__class__.__name__)
            if csv_path is not None:
                stdout_information(msg=f"[{tag}] Results saved to {csv_path}", source=self.__class__.__name__)
        return result_count, run_elapsed

    def _worker(self, task_queue: Queue, result_queue: Queue, ready_queue: Queue) -> None:
        tag = current_process().name
        try:
            if self.verbose: stdout_information(msg=f"[{tag}] Loading engine {self.engine_path} ...", source=self.__class__.__name__)
            model = YOLO(self.engine_path, task="detect")
            dummy = torch.zeros(self.batch_size, 3, self.img_size, self.img_size, dtype=torch.float16, device=f"cuda:{self.gpu_id}")
            model.predict(source=dummy, device=self.gpu_id, imgsz=self.img_size, verbose=False)
            del dummy
            dev = f"cuda:{self.gpu_id}"
            batch_buf = torch.empty(self.batch_size, 3, self.img_size, self.img_size, dtype=torch.float16, device=dev)
            if self.verbose: stdout_information(msg=f"[{tag}] Engine ready (raw AutoBackend).", source=self.__class__.__name__)
        except Exception as exc:
            if self.verbose: stdout_information(msg=f"[{tag}] FAILED to load engine: {exc}", source=self.__class__.__name__)
            ready_queue.put((tag, False))
            return

        ready_queue.put((tag, True))

        while True:
            video_path: str | None = task_queue.get()
            if video_path is None:
                break
            try:
                n_frames, vid_elapsed = self.run_one_video(video_path, model, batch_buf)
                result_queue.put((video_path, n_frames, vid_elapsed, None))
            except Exception as exc:
                if self.verbose: stdout_information(msg=f"[{tag}] ERROR on {video_path}: {exc}", source=self.__class__.__name__)
                result_queue.put((video_path, 0, 0.0, str(exc)))

    def _get_workers(self):
        self.workers = []
        for i in range(self.n_workers):
            p = Process(target=self._worker, args=(self.task_queue, self.result_queue, self.ready_queue), name=f"Worker-{i}")
            p.start()
            self.workers.append(p)

    def run(self):
        wall_t0 = time.perf_counter()
        self._get_workers()
        alive_count = 0
        for _ in range(self.n_workers):
            tag, ok = self.ready_queue.get()
            if ok:
                alive_count += 1
                if self.verbose: stdout_information(msg=f"[main] {tag} initialized successfully.", source=self.__class__.__name__)
            else:
                if self.verbose: stdout_information(msg=f"[main] {tag} failed to initialize.", source=self.__class__.__name__)

        if alive_count == 0:
            raise RuntimeError("All workers failed to initialize. Check GPU memory.")

        if self.verbose: stdout_information(msg=f"[main] {alive_count}/{self.n_workers} workers ready. Enqueuing {len(self.video_paths)} video(s).", source=self.__class__.__name__)

        _ = [self.task_queue.put(str(vp)) for vp in self.video_paths]
        _ = [self.task_queue.put(None) for _ in range(alive_count)]

        total_frames = 0
        errors: list[tuple[str, str]] = []
        per_video_fps: list[float] = []
        for _ in range(len(self.video_paths)):
            vpath, n_frames, vid_elapsed, err = self.result_queue.get()
            total_frames += n_frames
            if err:
                errors.append((vpath, err))
            elif n_frames > 0 and vid_elapsed > 0:
                per_video_fps.append(n_frames / vid_elapsed)

        _ = [p.join() for p in self.workers]

        wall_elapsed = round(time.perf_counter() - wall_t0, 4)
        avg_fps = float(np.mean(per_video_fps)) if per_video_fps else 0.0

        if self.verbose:
            stdout_information(msg=f"[main] All done: {len(self.video_paths)} video(s), {total_frames} frames, {wall_elapsed:.4f}s wall time, {avg_fps:.1f} FPS (mean of per-video FPS).", source=self.__class__.__name__)
        if errors:
            for vpath, err in errors:
                if self.verbose: stdout_information(msg=f"[main] FAILED: {vpath} — {err}", source=self.__class__.__name__)



# if __name__ == "__main__":
#     x = NVDECYoloInference(video_path=r"E:\litpose_yolo\pi\videos_downsampled", #
#                                 engine_path=r"E:\litpose_yolo\pi\yolo\mdl\train3\weights\best.engine",
#                                 save_dir=r"E:\litpose_yolo\pi\csv_results")
#     x.run()