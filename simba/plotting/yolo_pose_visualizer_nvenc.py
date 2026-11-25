__author__ = "Simon Nilsson"

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import math
import random
import shutil
import tempfile
import threading
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

try:
    import PyNvVideoCodec as nvc
except Exception:
    nvc = None
try:
    import torch
except Exception:
    torch = None

if nvc is not None:
    try:
        nvc.logger.setLevel(logging.WARNING)
    except Exception:
        pass

from simba.data_processors.cuda.utils import (NVC_SEEK_MARGIN,
                                              NVCFrameAlignmentError,
                                              _cuda_point_to_segment_dist,
                                              check_nvc_frame_size,
                                              concat_nvc_chunks,
                                              encode_nvc_frame,
                                              end_nvc_encoder, get_nvc_decoder,
                                              get_nvc_encoder,
                                              get_nvc_frame_timing,
                                              iter_nvc_frames,
                                              mux_nvc_raw_stream,
                                              yuv444_to_nv12_kernel)
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst, check_valid_tuple)
from simba.utils.data import create_color_palette
from simba.utils.enums import Options
from simba.utils.errors import (CountError, DataHeaderError, FrameRangeError,
                                InvalidInputError, NoDataError, SimBAGPUError,
                                SimBAPAckageVersionError)
from simba.utils.lookups import get_nvdec_count
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (bgr_to_yuv,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, get_video_meta_data,
                                    recursive_file_search)
from simba.utils.warnings import FrameRangeWarning

FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE, TRACK = 'FRAME', 'CLASS_ID', 'CLASS_NAME', 'CONFIDENCE', 'TRACK'
BOX_CORD_FIELDS = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4']
EXPECTED_COLS = [FRAME, CLASS_ID, CLASS_NAME, CONFIDENCE] + BOX_CORD_FIELDS
SKELETON_BGR = (105, 105, 105)
NVENC_CODECS = ('h264', 'hevc', 'av1')
CHROMA_OPTIONS = ('yuv444', 'yuv420')
TPB2 = (32, 8)
COORD_LIMIT = 1_000_000                                                     # px; far outside any frame, safe in int32


@cuda.jit()
def _yolo_frame_kernel(rgb: np.ndarray,
                       yuv: np.ndarray,
                       h: int,
                       f: int,
                       rects: np.ndarray,
                       boxes: np.ndarray,
                       box_on: np.ndarray,
                       kps: np.ndarray,
                       kp_on: np.ndarray,
                       det_cls: np.ndarray,
                       colors: np.ndarray,
                       skel: np.ndarray,
                       skel_on: bool,
                       skel_yuv: np.ndarray,
                       radius_sq: float,
                       box_half: float,
                       skel_half: float) -> None:
    """
    CUDA kernel: convert one decoded RGB frame to planar YUV444 (BT.601 limited range) and draw frame ``f``'s YOLO
    overlay into it. One thread per pixel (2D grid over W x H). Each pixel paints in the CPU visualizer's order (per
    detection: box outline, keypoint circles, skeleton lines; later writes win), so overlaps resolve as in
    :func:`simba.plotting.yolo_pose_visualizer._yolo_keypoint_visualizer`. ``D`` = max detections per frame, ``K`` = keypoints.

    :param np.ndarray rgb: Device array (H, W, 3) uint8: the decoded frame (NVDEC RGB output).
    :param np.ndarray yuv: Device array (3H, W) uint8, written in place: planar [Y; Cb; Cr].
    :param int h: Frame height H.
    :param int f: Index of the frame in the per-frame arrays below.
    :param np.ndarray rects: (n_frames, D, 4) int32 bounding rectangle (x0, y0, x1, y1) of everything drawn for each detection, padded by the line/circle size; pixels outside skip the detection. Empty (x0 > x1) for detections not drawn.
    :param np.ndarray boxes: (n_frames, D, 4, 2) int32 box corners (x, y).
    :param np.ndarray box_on: (n_frames, D) bool: draw this detection's box.
    :param np.ndarray kps: (n_frames, D, K, 2) int32 keypoint coordinates (x, y).
    :param np.ndarray kp_on: (n_frames, D, K) bool: keypoint visible (detection and keypoint confidence above threshold).
    :param np.ndarray det_cls: (n_frames, D) int32 class ID of each detection (row in ``colors``).
    :param np.ndarray colors: (n_classes, K + 1, 3) int32 (Y, Cb, Cr) colours per class: index 0 = box, ``k + 1`` = keypoint ``k``.
    :param np.ndarray skel: (n_edges, 2) int32 keypoint indices of the skeleton edges.
    :param bool skel_on: Draw the skeleton.
    :param np.ndarray skel_yuv: (3,) int32 (Y, Cb, Cr) skeleton colour.
    :param float radius_sq: Squared keypoint circle radius (px^2).
    :param float box_half: Half the box outline thickness (px).
    :param float skel_half: Half the skeleton line thickness (px).
    """
    x, y = cuda.grid(2)
    if y >= h or x >= yuv.shape[1] or y >= rgb.shape[0] or x >= rgb.shape[1]:   # NVDEC pads odd sizes: bound by the output
        return
    r = rgb[y, x, 0]; g = rgb[y, x, 1]; b = rgb[y, x, 2]
    c0 = 16.0 + 0.256788 * r + 0.504129 * g + 0.097906 * b
    c1 = 128.0 - 0.148223 * r - 0.290993 * g + 0.439216 * b
    c2 = 128.0 + 0.439216 * r - 0.367788 * g - 0.071427 * b
    c0 = min(255.0, max(0.0, c0)); c1 = min(255.0, max(0.0, c1)); c2 = min(255.0, max(0.0, c2))
    px, py = float(x), float(y)
    for d in range(rects.shape[1]):
        if x < rects[f, d, 0] or x > rects[f, d, 2] or y < rects[f, d, 1] or y > rects[f, d, 3]:
            continue                                                         # outside this detection (also: not drawn)
        cls = det_cls[f, d]
        if box_on[f, d]:
            for e in range(4):
                e2 = (e + 1) % 4
                if _cuda_point_to_segment_dist(px, py, float(boxes[f, d, e, 0]), float(boxes[f, d, e, 1]), float(boxes[f, d, e2, 0]), float(boxes[f, d, e2, 1])) <= box_half:
                    c0 = colors[cls, 0, 0]; c1 = colors[cls, 0, 1]; c2 = colors[cls, 0, 2]
                    break
        for k in range(kps.shape[2]):
            if kp_on[f, d, k]:
                dx = x - kps[f, d, k, 0]; dy = y - kps[f, d, k, 1]
                if dx * dx + dy * dy <= radius_sq:
                    c0 = colors[cls, k + 1, 0]; c1 = colors[cls, k + 1, 1]; c2 = colors[cls, k + 1, 2]
        for s in range(skel.shape[0]):
            k1 = skel[s, 0]; k2 = skel[s, 1]
            if skel_on and kp_on[f, d, k1] and kp_on[f, d, k2]:
                if _cuda_point_to_segment_dist(px, py, float(kps[f, d, k1, 0]), float(kps[f, d, k1, 1]), float(kps[f, d, k2, 0]), float(kps[f, d, k2, 1])) <= skel_half:
                    c0 = skel_yuv[0]; c1 = skel_yuv[1]; c2 = skel_yuv[2]
    yuv[y, x] = np.uint8(c0)
    yuv[h + y, x] = np.uint8(c1)
    yuv[2 * h + y, x] = np.uint8(c2)


class YOLOPoseVisualizerNVENC(object):
    """
    Visualize YOLO pose-estimation results (bounding boxes, keypoints, optional skeleton) on video, fully on the GPU
    (NVDEC decode -> CUDA drawing -> NVENC encode). GPU counterpart of :class:`~simba.plotting.yolo_pose_visualizer.YOLOPoseVisualizer`,
    drawing the same overlay: per detection with ``CONFIDENCE > threshold``, the box (palette colour 0), a filled circle
    per keypoint with ``_P > threshold`` (palette colour ``k + 1``), and optional grey skeleton lines.

    .. seealso::
       For the CPU version, see :class:`~simba.plotting.yolo_pose_visualizer.YOLOPoseVisualizer`.
       To create YOLO pose data, see :class:`~simba.model.yolo_pose_inference.YOLOPoseInference`.
       For the general pose-on-video GPU plotter, see :class:`~simba.plotting.pose_plotter_nvenc.PosePlotterNVENC`.

    .. note::
       Requires PyNvVideoCodec (NVDEC/NVENC), torch (DLPack decode interop) and a CUDA GPU.
       ``chroma='yuv444'`` (default) keeps full-resolution colour so small coloured keypoints stay sharp, but some players
       (browsers, QuickTime, PowerPoint) cannot play 4:4:4 H.264/HEVC; VLC, ffmpeg, OpenCV and mpv can. ``chroma='yuv420'``
       plays everywhere at the cost of softer colour edges on small circles. Drawing is not anti-aliased, and the output is
       lossy, so pixels are close to, but not bit-identical with, the CPU path.

    .. csv-table:: Benchmark: GPU (NVDEC -> CUDA -> NVENC) vs CPU, one 1280x960 25 fps video, 10,132 frames, 2 classes x 7 keypoints, boxes on. RTX 4070 (1 NVDEC engine); CPU = :class:`~simba.plotting.yolo_pose_visualizer.YOLOPoseVisualizer` on 8 cores of an i9-14900KF. Wall time incl. setup, mean of 3 (GPU) / 2 (CPU) runs.
       :header: Method, Time (s), FPS, Speedup
       :widths: 40, 20, 20, 20
       :align: center

       "CPU, 8 cores", 82.1, 123, 1.0x
       "GPU, ``chroma='yuv444'``", 13.8, 732, 5.9x
       "GPU, ``chroma='yuv420'``", 12.7, 800, 6.5x

    .. note::
       If the number of frames in the data and the video differ (many videos report one more frame in the header than
       can be decoded), the first ``min(video frames, data frames)`` frames are rendered with a
       :func:`~simba.utils.warnings.FrameRangeWarning`, as in the CPU visualizer. If the decoder runs out early, rendering
       stops at the last decodable frame and warns.

    :param Union[str, os.PathLike] data_path: CSV of YOLO pose results (output of YOLO pose inference), or a directory of such CSVs.
    :param Union[str, os.PathLike] video_path: The video, or a directory of videos, matched to the data by file name.
    :param Union[str, os.PathLike] save_dir: Existing directory where ``<video_name>.mp4`` is written.
    :param Optional[Union[str, Tuple[str, ...]]] palettes: Categorical palette name(s), one per class. If the count does not match the number of classes, the first categorical palettes are used (as in the CPU visualizer). Default 'Set1'.
    :param float threshold: Detections with ``CONFIDENCE > threshold`` are drawn, and keypoints/skeleton ends with ``_P > threshold``. Default 0.0.
    :param Optional[int] thickness: Box outline thickness (px). None: equal to ``circle_size``. Default None.
    :param Optional[int] circle_size: Keypoint circle radius (px). None: ``min(width, height) / 80``. Default None.
    :param bool bbox: Draw the bounding boxes. Default True.
    :param Optional[List[Tuple[str, str]]] skeleton: Keypoint-name pairs to connect with grey lines (thickness ``max(1, circle_size // 2)``) when both ends are visible. Default None (no lines).
    :param bool recursive: Search ``data_path`` / ``video_path`` directories recursively. Default False.
    :param Optional[int] sample_n: Render a random sample of ``sample_n`` data files. Default None (all).
    :param str codec: NVENC codec ('h264', 'hevc' or 'av1'). Default 'h264'.
    :param int qp: Constant quantization parameter (0-51); lower = higher quality and larger files. Default 15.
    :param str chroma: 'yuv444' (sharp colour, not all players) or 'yuv420' (plays everywhere). Default 'yuv444'.
    :param int buffer_size: Frames the threaded NVDEC decoder decodes ahead. Default 16.
    :param Optional[int] n_workers: Parallel decode/encode pipelines (one per NVDEC engine); the video is split into chunks and concatenated. None = the GPU's NVDEC count. Default None.
    :param bool verbose: Print progress. Default True.

    :example:

    >>> YOLOPoseVisualizerNVENC(data_path=r'F:/yolo_results/Trial_1.csv', video_path=r'F:/videos/Trial_1.mp4', save_dir=r'F:/yolo_videos').run()

    :example II - directories, with a skeleton, browser-playable output:

    >>> skeleton = [('NOSE', 'LEFT_EAR'), ('NOSE', 'RIGHT_EAR'), ('LEFT_EAR', 'CENTER'), ('RIGHT_EAR', 'CENTER'), ('CENTER', 'TAIL_BASE')]
    >>> YOLOPoseVisualizerNVENC(data_path=r'F:/yolo_results', video_path=r'F:/videos', save_dir=r'F:/yolo_videos', skeleton=skeleton, threshold=0.5, chroma='yuv420').run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 video_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 palettes: Optional[Union[str, Tuple[str, ...]]] = 'Set1',
                 threshold: float = 0.0,
                 thickness: Optional[int] = None,
                 circle_size: Optional[int] = None,
                 bbox: bool = True,
                 skeleton: Optional[List[Tuple[str, str]]] = None,
                 recursive: bool = False,
                 sample_n: Optional[int] = None,
                 codec: str = 'h264',
                 qp: int = 15,
                 chroma: str = 'yuv444',
                 buffer_size: int = 16,
                 n_workers: Optional[int] = None,
                 verbose: bool = True):

        missing = [n for n, m in (('PyNvVideoCodec', nvc), ('torch', torch)) if m is None]
        if missing:
            raise SimBAPAckageVersionError(msg=f'{self.__class__.__name__} requires {" and ".join(missing)}, which could not be found in this environment.', source=self.__class__.__name__)
        if not torch.cuda.is_available():
            raise SimBAGPUError(msg='No CUDA GPU detected.', source=self.__class__.__name__)
        check_valid_boolean(value=[recursive, bbox, verbose], source=self.__class__.__name__, raise_error=True)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        if circle_size is not None:
            check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        if thickness is not None:
            check_int(name=f'{self.__class__.__name__} thickness', value=thickness, min_value=1)
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_str(name=f'{self.__class__.__name__} chroma', value=chroma, options=CHROMA_OPTIONS)
        check_int(name=f'{self.__class__.__name__} qp', value=qp, min_value=0, max_value=51)
        check_int(name=f'{self.__class__.__name__} buffer_size', value=buffer_size, min_value=1)
        check_if_dir_exists(in_dir=save_dir)
        if isinstance(palettes, str):
            palettes = (palettes,)
        else:
            check_valid_tuple(x=palettes, source=f'{self.__class__.__name__} palettes', minimum_length=1, valid_dtypes=(str,), accepted_values=Options.PALETTE_OPTIONS_CATEGORICAL.value)
        if skeleton is not None:
            check_valid_lst(data=skeleton, source=f'{self.__class__.__name__} skeleton', valid_dtypes=(list, tuple,), min_len=1, raise_error=True)
            for pair in skeleton:
                check_valid_tuple(x=tuple(pair), source=f'{self.__class__.__name__} skeleton {pair}', accepted_lengths=(2,), valid_dtypes=(str,))
        available = max(1, get_nvdec_count(gpu_name=torch.cuda.get_device_name(0)))
        if n_workers is None:
            n_workers = available
        else:
            check_int(name=f'{self.__class__.__name__} n_workers', value=n_workers, min_value=1)
            if n_workers > available:
                raise SimBAGPUError(msg=f'n_workers={n_workers} was requested but GPU "{torch.cuda.get_device_name(0)}" has only {available} NVDEC engine(s). Pass n_workers <= {available}.', source=self.__class__.__name__)

        if os.path.isdir(data_path):
            if not recursive:
                data_paths = find_files_of_filetypes_in_directory(directory=data_path, extensions=['.csv'], as_dict=True, raise_error=True)
            else:
                data_paths = recursive_file_search(directory=data_path, extensions='csv', as_dict=True, raise_error=True)
            if sample_n is not None:
                check_int(name=f'{self.__class__.__name__} sample_n', min_value=1, raise_error=True, value=sample_n)
                data_paths = dict(random.sample(list(data_paths.items()), min(sample_n, len(data_paths))))
        elif os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            data_paths = {get_fn_ext(filepath=data_path)[1]: data_path}
        else:
            raise InvalidInputError(msg=f'{data_path} is not a valid directory path or file path', source=self.__class__.__name__)
        if os.path.isdir(video_path):
            if not recursive:
                video_paths = find_files_of_filetypes_in_directory(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, as_dict=True, raise_error=True)
            else:
                video_paths = recursive_file_search(directory=video_path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS_2.value, as_dict=True, raise_error=True)
        elif os.path.isfile(video_path):
            video_paths = {get_fn_ext(filepath=video_path)[1]: video_path}
        else:
            raise InvalidInputError(msg=f'{video_path} is not a valid directory path or file path', source=self.__class__.__name__)
        missing_videos = [x for x in data_paths.keys() if x not in video_paths.keys()]
        if missing_videos:
            raise NoDataError(msg=f'The data file(s) {missing_videos} do not have a representative video.', source=self.__class__.__name__)

        self.data_paths, self.video_paths, self.save_dir = data_paths, video_paths, save_dir
        self.palettes, self.threshold, self.thickness, self.circle_size = palettes, threshold, thickness, circle_size
        self.bbox, self.skeleton, self.verbose = bbox, skeleton, verbose
        self.codec, self.qp, self.chroma, self.buffer_size, self._n_workers_req = codec, qp, chroma, buffer_size, n_workers

    def _prepare(self, data_path: str, video_path: str, video_name: str) -> None:
        """Read one CSV and vectorise it ONCE into dense per-frame GPU arrays (no per-frame pandas filtering)."""
        df = pd.read_csv(data_path, index_col=0)
        check_valid_dataframe(df=df, source=f'{self.__class__.__name__} {data_path}', required_fields=EXPECTED_COLS)
        n_value_cols = len(df.columns) - len(EXPECTED_COLS)                 # palette increments, as in the CPU visualizer
        if TRACK in df.columns:
            df = df.drop([TRACK], axis=1)
        kp_cols = [c for c in df.columns if c not in EXPECTED_COLS]
        if len(kp_cols) % 3 != 0:
            raise CountError(msg=f'{data_path}: {len(kp_cols)} keypoint columns is not a multiple of 3 (X, Y, P).', source=self.__class__.__name__)
        kp_names = [c[:-2] for c in kp_cols[0::3]]
        n_kp = len(kp_names)
        if n_kp == 0:
            raise CountError(msg=f'{data_path} has no keypoint columns (<KEYPOINT>_X, <KEYPOINT>_Y, <KEYPOINT>_P) after the box columns.', source=self.__class__.__name__)
        if self.skeleton is not None:
            missing = [f'{x}_{s}' for pair in self.skeleton for x in pair for s in ('X', 'Y', 'P') if f'{x}_{s}' not in df.columns]
            if missing:
                raise DataHeaderError(msg=f'Columns {missing} missing in file {data_path} as passed by skeleton.', source=self.__class__.__name__)

        self.meta = get_video_meta_data(video_path=video_path, fps_as_int=False)
        self.w, self.h = self.meta['width'], self.meta['height']
        if self.chroma == 'yuv420' and (self.w % 2 or self.h % 2):
            raise InvalidInputError(msg=f'chroma="yuv420" needs an even width and height; {video_name} is {self.w}x{self.h}. Use chroma="yuv444".', source=self.__class__.__name__)
        df_frm_cnt = int(np.unique(df[FRAME].values).shape[0])
        if df_frm_cnt == 0:
            raise NoDataError(msg=f'{data_path} contains no rows (no frames) to visualize.', source=self.__class__.__name__)
        self.n = int(min(self.meta['frame_count'], df_frm_cnt))
        if self.meta['frame_count'] != df_frm_cnt:
            FrameRangeWarning(msg=f'The bounding boxes contain data for {df_frm_cnt} frames, while the video is {self.meta["frame_count"]} frames ({video_name}). Visualizing the first {self.n} frames.', source=self.__class__.__name__)

        classes = np.unique(df[CLASS_NAME].values)
        palettes = self.palettes
        if len(classes) != len(palettes):
            if len(classes) > len(Options.PALETTE_OPTIONS_CATEGORICAL.value):
                raise CountError(msg=f'There are more classes ({len(classes)}) than available color palettes ({len(Options.PALETTE_OPTIONS_CATEGORICAL.value)})', source=self.__class__.__name__)
            palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value[:len(classes) + 1]
        class_ids = df[CLASS_ID].values.astype(np.int64)
        n_categorical = len(Options.PALETTE_OPTIONS_CATEGORICAL.value)
        if class_ids.min() < 0 or class_ids.max() >= n_categorical:
            raise InvalidInputError(msg=f'{data_path} has CLASS_ID values {sorted(np.unique(class_ids).tolist())}; CLASS_ID must be 0 to {n_categorical - 1} (one colour palette per class).', source=self.__class__.__name__)
        if class_ids.max() >= len(palettes):                                 # e.g. only classes 0 and 5 in this file: the CPU
            palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value[:int(class_ids.max()) + 1]   # visualizer raises KeyError; same palette order, so same colours
        colors_bgr = np.zeros((len(palettes), n_kp + 1, 3), dtype=np.int32)
        for cnt, pal in enumerate(palettes):
            clrs = np.array(create_color_palette(pallete_name=pal, increments=n_value_cols)).astype(np.int32)
            colors_bgr[cnt] = clrs[:n_kp + 1]

        circle_size = self.circle_size if self.circle_size is not None else PlottingMixin().get_optimal_circle_size(frame_size=(self.w, self.h), circle_frame_ratio=80)
        thickness = self.thickness if self.thickness is not None else circle_size
        skel_thickness = int(max(1, int(circle_size / 2)))

        frm = df[FRAME].values.astype(np.int64)
        keep = (frm >= 0) & (frm < self.n) & (df[CONFIDENCE].values > self.threshold)
        sub = df.loc[keep]
        frm = frm[keep]
        slot = sub.groupby(FRAME).cumcount().values.astype(np.int64)        # row order within the frame = CPU drawing order
        max_dets = max(1, int(slot.max()) + 1) if len(slot) else 1
        boxes = np.zeros((self.n, max_dets, 4, 2), dtype=np.int32)
        kps = np.zeros((self.n, max_dets, n_kp, 2), dtype=np.int32)
        kp_on = np.zeros((self.n, max_dets, n_kp), dtype=np.bool_)
        det_on = np.zeros((self.n, max_dets), dtype=np.bool_)
        det_cls = np.zeros((self.n, max_dets), dtype=np.int32)
        if len(sub):
            bv = sub[BOX_CORD_FIELDS].values.astype(np.float64).reshape(-1, 4, 2)
            kv = sub[kp_cols].values.astype(np.float64).reshape(-1, n_kp, 3)
            box_ok = np.isfinite(bv).all(axis=(1, 2))                       # missing (NaN) coordinates: not drawn
            kp_ok = np.isfinite(kv).all(axis=2)
            boxes[frm, slot] = np.clip(np.nan_to_num(bv), -COORD_LIMIT, COORD_LIMIT).astype(np.int32)   # clip: no int32 overflow
            kps[frm, slot] = np.clip(np.nan_to_num(kv[:, :, :2]), -COORD_LIMIT, COORD_LIMIT).astype(np.int32)
            kp_on[frm, slot] = kp_ok & (np.nan_to_num(kv[:, :, 2]) > self.threshold)
            det_on[frm, slot] = box_ok
            det_cls[frm, slot] = sub[CLASS_ID].values.astype(np.int32)
        box_on = det_on & bool(self.bbox)                                   # det_on: detected and box coordinates present

        margin = int(max(circle_size, math.ceil(thickness / 2), math.ceil(skel_thickness / 2))) + 1
        big = np.iinfo(np.int32).max // 2
        kx = np.where(kp_on, kps[..., 0], big); ky = np.where(kp_on, kps[..., 1], big)
        kx2 = np.where(kp_on, kps[..., 0], -big); ky2 = np.where(kp_on, kps[..., 1], -big)
        bx = np.where(box_on[..., None], boxes[..., 0], big); by = np.where(box_on[..., None], boxes[..., 1], big)
        bx2 = np.where(box_on[..., None], boxes[..., 0], -big); by2 = np.where(box_on[..., None], boxes[..., 1], -big)
        rects = np.stack([np.minimum(kx.min(-1), bx.min(-1)) - margin, np.minimum(ky.min(-1), by.min(-1)) - margin,
                          np.maximum(kx2.max(-1), bx2.max(-1)) + margin, np.maximum(ky2.max(-1), by2.max(-1)) + margin], axis=-1).astype(np.int32)
        empty = ~(box_on | kp_on.any(-1))
        rects[empty] = np.array([1, 1, 0, 0], dtype=np.int32)                # x0 > x1: never inside

        if self.skeleton is not None:
            skel = np.array([[kp_names.index(a), kp_names.index(b)] for a, b in self.skeleton], dtype=np.int32)
        else:
            skel = np.zeros((1, 2), dtype=np.int32)
        self.skel_on = self.skeleton is not None
        self.radius_sq = float(circle_size * circle_size)
        self.box_half, self.skel_half = float(thickness) / 2.0, float(skel_thickness) / 2.0
        self.rects_d, self.boxes_d, self.box_on_d = cuda.to_device(rects), cuda.to_device(boxes), cuda.to_device(box_on)
        self.kps_d, self.kp_on_d, self.det_cls_d = cuda.to_device(kps), cuda.to_device(kp_on), cuda.to_device(det_cls)
        self.colors_d = cuda.to_device(np.ascontiguousarray(bgr_to_yuv(colors_bgr)))
        self.skel_d = cuda.to_device(skel)
        self.skel_yuv_d = cuda.to_device(bgr_to_yuv(np.array(SKELETON_BGR)).astype(np.int32))
        self.video_path, self.video_name = video_path, video_name
        self.n_workers = max(1, min(self._n_workers_req, self.n))

    def _encode_chunk(self, start: int, end: int, out_path: str, timing: Optional[Tuple[int, float]] = None) -> int:
        """
        Decode -> draw -> encode frames [start, end) and mux to ``out_path``. Returns the number of frames encoded.

        With ``timing`` = (timestamp of frame 0, timestamp step) the frame index is read from each frame's timestamp
        instead of counted: seeking the decoder to ``start`` can land a few frames late (e.g. MPEG-4 with B-frames), so
        the chunk seeks ``NVC_SEEK_MARGIN`` frames early and skips to exactly ``start`` (see
        :func:`simba.data_processors.cuda.utils.iter_nvc_frames`). Raises ``NVCFrameAlignmentError`` if the timestamps are
        irregular or the seek overshoots ``start``; ``_render`` then falls back to one worker.
        """
        yuv_t = torch.empty((3 * self.h, self.w), dtype=torch.uint8, device='cuda:0')
        yuv_dev = cuda.as_cuda_array(yuv_t)
        if self.chroma == 'yuv420':
            enc_t = torch.empty((3 * self.h // 2, self.w), dtype=torch.uint8, device='cuda:0')
            enc_dev = cuda.as_cuda_array(enc_t)
            encoder = get_nvc_encoder(width=self.w, height=self.h, fps=self.meta['fps'], codec=self.codec, fmt='NV12', qp=self.qp)
        else:
            enc_t, enc_dev = yuv_t, None
            encoder = get_nvc_encoder(width=self.w, height=self.h, fps=self.meta['fps'], codec=self.codec, fmt='YUV444', qp=self.qp)
        seek = start if timing is None else max(0, start - NVC_SEEK_MARGIN)
        grid = (math.ceil(self.w / TPB2[0]), math.ceil(self.h / TPB2[1]))
        raw_path = f'{out_path}.raw.{self.codec}'
        written, decoder = 0, None
        try:
            decoder = get_nvc_decoder(video_path=self.video_path, output_color_type=nvc.OutputColorType.RGB, use_device_memory=True, threaded=True, buffer_size=self.buffer_size, start_frame=seek)
            written = self._encode_frames(decoder=decoder, encoder=encoder, enc_t=enc_t, yuv_dev=yuv_dev, enc_dev=enc_dev, grid=grid, raw_path=raw_path, start=start, end=end, timing=timing)
            mux_nvc_raw_stream(raw_path=raw_path, fps=self.meta['fps'], save_path=out_path, source=self.__class__.__name__)
        finally:
            if decoder is not None:
                try:
                    decoder.end()
                except Exception:
                    pass
            if os.path.isfile(raw_path):
                os.remove(raw_path)
        return written

    def _encode_frames(self, decoder, encoder, enc_t: 'torch.Tensor', yuv_dev: np.ndarray, enc_dev: Optional[np.ndarray], grid: Tuple[int, int],
                       raw_path: str, start: int, end: int, timing: Optional[Tuple[int, float]]) -> int:
        """Frame loop of :meth:`_encode_chunk`: decode, draw, encode frames [start, end) into the raw stream at
        ``raw_path``. Returns the number of frames encoded."""
        written = 0
        with open(raw_path, 'wb') as raw:
            for i, fr in iter_nvc_frames(decoder=decoder, start=start, end=end, buffer_size=self.buffer_size, timing=timing):
                rgb = cuda.as_cuda_array(torch.from_dlpack(fr))
                if written == 0:
                    check_nvc_frame_size(frame=rgb, width=self.w, height=self.h, video_path=self.video_path, source=self.__class__.__name__)
                _yolo_frame_kernel[grid, TPB2](rgb, yuv_dev, self.h, i, self.rects_d, self.boxes_d, self.box_on_d, self.kps_d, self.kp_on_d,
                                               self.det_cls_d, self.colors_d, self.skel_d, self.skel_on, self.skel_yuv_d, self.radius_sq, self.box_half, self.skel_half)
                if self.chroma == 'yuv420':
                    yuv444_to_nv12_kernel[grid, TPB2](yuv_dev, enc_dev, self.h)
                raw.write(encode_nvc_frame(encoder=encoder, frame=enc_t))
                written += 1
                if self.verbose and (i + 1) % 1000 == 0:
                    stdout_information(msg=f'Encoded {i + 1}/{self.n} frames ({self.video_name})...', source=self.__class__.__name__)
            raw.write(end_nvc_encoder(encoder=encoder))
        return written

    def _render(self, save_path: str) -> int:
        """Render the prepared video, split over ``n_workers`` NVDEC engines when > 1. Returns frames written.
        Chunks find their exact start frame by timestamp; if that is not possible for this video, it is rendered by
        one worker instead (with a warning)."""
        timing = get_nvc_frame_timing(video_path=self.video_path) if self.n_workers > 1 else None
        if self.n_workers > 1 and timing is None:
            FrameRangeWarning(msg=f'Could not read frame timestamps of {self.video_name}; rendering with 1 worker.', source=self.__class__.__name__)
        if self.n_workers <= 1 or timing is None:
            return self._encode_chunk(0, self.n, save_path)
        bounds = np.linspace(0, self.n, self.n_workers + 1).astype(int)
        tmp = tempfile.mkdtemp(prefix='simba_yolo_nvenc_')                      # unique per render: no name clashes
        chunk_paths = [os.path.join(tmp, f'chunk_{k}.mp4') for k in range(self.n_workers)]
        errors, counts, threads = [], [0] * self.n_workers, []

        def _work(k: int) -> None:
            try:
                counts[k] = self._encode_chunk(int(bounds[k]), int(bounds[k + 1]), chunk_paths[k], timing)
            except Exception as e:
                errors.append((k, e))

        for k in range(self.n_workers):
            t = threading.Thread(target=_work, args=(k,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        try:
            if errors:
                real_errors = [(k, e) for k, e in errors if not isinstance(e, NVCFrameAlignmentError)]
                if real_errors:
                    raise FrameRangeError(msg=f'{self.__class__.__name__} chunk {real_errors[0][0]} failed: {real_errors[0][1]}', source=self.__class__.__name__)
                FrameRangeWarning(msg=f'Multi-worker rendering of {self.video_name} failed ({errors[0][1]}); rendering with 1 worker.', source=self.__class__.__name__)
                return self._encode_chunk(0, self.n, save_path)
            concat_nvc_chunks(chunk_paths=chunk_paths, save_path=save_path, source=self.__class__.__name__)
            return int(sum(counts))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def run(self) -> None:
        timer = SimbaTimer(start=True)
        n_videos = len(self.data_paths)
        for cnt, (video_name, data_path) in enumerate(self.data_paths.items()):
            video_timer = SimbaTimer(start=True)
            self._prepare(data_path=data_path, video_path=self.video_paths[video_name], video_name=video_name)
            save_path = os.path.join(self.save_dir, f'{video_name}.mp4')
            if self.verbose:
                stdout_information(msg=f'Visualizing video {video_name} ({self.n} frames, video {cnt + 1}/{n_videos}, workers: {self.n_workers})...', source=self.__class__.__name__)
            written = self._render(save_path=save_path)
            if written < self.n:
                FrameRangeWarning(msg=f'Only {written} of {self.n} frames could be decoded from {video_name}; the video was saved with {written} frames.', source=self.__class__.__name__)
            video_timer.stop_timer()
            if self.verbose:
                stdout_success(msg=f'YOLO pose video saved at {save_path} (video {cnt + 1}/{n_videos}, {written / max(video_timer.elapsed_time, 1e-9):.0f} fps)', source=self.__class__.__name__, elapsed_time=video_timer.elapsed_time_str)
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'{n_videos} YOLO pose video(s) saved in directory {self.save_dir}', source=self.__class__.__name__, elapsed_time=timer.elapsed_time_str)


# if __name__ == "__main__":
#     YOLOPoseVisualizerNVENC(data_path=r"F:\maplight_1126\tracking_test\yolo26_imgsz640\Trial_1_Day3_1.csv",
#                             video_path=r"F:\maplight_1126\RI-4 Camera",
#                             save_dir=r"F:\maplight_1126\tracking_test\nvenc_test").run()
