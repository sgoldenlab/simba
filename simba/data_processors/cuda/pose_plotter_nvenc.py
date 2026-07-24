__author__ = "Simon Nilsson"

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import math
import subprocess
import tempfile
import threading
import warnings
from typing import Optional, Union

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
        nvc.logger.setLevel(logging.WARNING)   # silence PyNvVideoCodec INFO logs (e.g. "Cache size ... Setting cache size to: 1")
    except Exception:
        pass

from simba.data_processors.cuda.utils import get_nvc_decoder
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_array, check_valid_boolean)
from simba.utils.data import create_color_palette
from simba.utils.enums import Formats
from simba.utils.errors import (CountError, FrameRangeError,
                                InvalidFilepathError, InvalidInputError,
                                MissingColumnsError, SimBAGPUError,
                                SimBAPAckageVersionError)
from simba.utils.lookups import get_nvdec_count
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    find_video_of_file, get_fn_ext,
                                    get_video_meta_data, read_df)

TPB2 = (32, 32, 1)
TPB1 = 64
NVENC_CODECS = ('h264', 'hevc', 'av1')


@cuda.jit()
def _rgb_to_yuv444_kernel(rgb: np.ndarray, yuv: np.ndarray, h: int) -> None:
    """Decoded RGB(H,W,3) -> planar YUV444 (3H, W) buffer [Y; Cb; Cr] for NVENC 'YUV444' input
    (BT.601 limited range). Full-resolution chroma keeps small/colored overlay circles crisp -
    the 4:2:0 path subsamples + quantizes chroma and mottles or erases them."""
    x, y = cuda.grid(2)
    if y >= rgb.shape[0] or x >= rgb.shape[1]:
        return
    r = rgb[y, x, 0]; g = rgb[y, x, 1]; b = rgb[y, x, 2]
    yy = 16.0 + 0.256788 * r + 0.504129 * g + 0.097906 * b
    cb = 128.0 - 0.148223 * r - 0.290993 * g + 0.439216 * b
    cr = 128.0 + 0.439216 * r - 0.367788 * g - 0.071427 * b
    yuv[y, x] = np.uint8(min(255.0, max(0.0, yy)))
    yuv[h + y, x] = np.uint8(min(255.0, max(0.0, cb)))
    yuv[2 * h + y, x] = np.uint8(min(255.0, max(0.0, cr)))


@cuda.jit()
def _draw_circles_yuv444_kernel(yuv: np.ndarray, bps: np.ndarray, circle_size: np.ndarray, resolution: np.ndarray, colors_yuv: np.ndarray, h: int) -> None:
    """Draw one frame's body-part circles into the planar YUV444 buffer. One thread per body-part;
    ``colors_yuv`` holds each palette colour pre-converted to (Y, Cb, Cr)."""
    bp_n = cuda.grid(1)
    if bp_n < 0 or bp_n > (bps.shape[0] - 1):
        return
    bp, col, cs = bps[bp_n], colors_yuv[bp_n], circle_size[0]
    w, ht = resolution[0], resolution[1]
    for x1 in range(bp[0] - cs, bp[0] + cs):
        for y1 in range(bp[1] - cs, bp[1] + cs):
            if (0 < x1 < w) and (0 < y1 < ht):
                if ((x1 - bp[0]) ** 2 + (y1 - bp[1]) ** 2) < (cs ** 2):
                    yuv[y1, x1] = col[0]
                    yuv[h + y1, x1] = col[1]
                    yuv[2 * h + y1, x1] = col[2]


class PosePlotterNVENC(object):
    """
    Overlay pose-estimation data on a video, fully on the GPU (NVDEC -> CUDA -> NVENC).

    .. seealso::
       For the CPU-encode GPU version, see :func:`simba.data_processors.cuda.image.pose_plotter`.
       For CPU methods, see :func:`~simba.plotting.path_plotter.PathPlotterSingleCore` and :func:`~simba.plotting.path_plotter_mp.PathPlotterMulticore`.

    .. note::
       Requires PyNvVideoCodec (NVDEC/NVENC) and torch (for the DLPack decode interop); the
       constructor raises :class:`~simba.utils.errors.SimBAPAckageVersionError` if either is
       missing. Encoding is **4:4:4** (full-resolution chroma) so small/colored body-part
       circles stay crisp - 4:2:0 subsamples chroma and mottles or erases them. Trade-off: a
       4:4:4 H.264/HEVC stream needs a 4:4:4-capable player (VLC, ffmpeg/OpenCV, mpv all work;
       some browsers and QuickTime do not). Output is lossy, so pixels are perceptually
       equivalent but not bit-identical to the CPU path.

    .. csv-table:: Expected runtimes by image size
       :file: ../../../docs/tables/pose_plotter_nvenc.csv
       :widths: 40, 30, 30
       :align: center
       :header-rows: 1

    .. csv-table:: Multi-worker scaling (100K frames, 1280x1024, 7 body-parts, RTX PRO 6000 Blackwell)
       :header: Workers, Time (s), FPS, Speedup
       :widths: 20, 20, 20, 20
       :align: center

       1, 100.7, 993, 1.00x
       4, 55.8, 1794, 1.81x

    Three input styles are supported:

    * **Single file** - ``data`` is a CSV path (or a ``(n_frames, n_bodyparts, 2)`` array) and ``video_path`` is a
      single video. ``save_path`` is the output file (or a directory, in which case ``<video_name>.mp4`` is written).
    * **Directory (batch)** - ``data`` and ``video_path`` are both directories; each data file is matched to the
      video of the same name and rendered. ``save_path`` must be an existing directory. If one of ``data`` /
      ``video_path`` is a directory the other must be too.
    * **SimBA project** - ``config_path`` points at a ``project_config.ini``; data defaults to the project's
      outlier-corrected directory and videos to ``project_folder/videos`` (either can be overridden by passing
      ``data`` / ``video_path`` directories). ``save_path`` defaults to ``project_folder/frames/output/pose_nvenc``.

    :param Union[str, os.PathLike, np.ndarray] data: A CSV path or ``(n_frames, n_bodyparts, 2)`` array (single mode), or a directory of pose files (batch mode). In project mode, an optional directory overriding the project default.
    :param Optional[Union[str, os.PathLike]] video_path: A single video (single mode) or a directory of videos (batch mode). If either ``data`` or ``video_path`` is a directory, both must be. Optional in project mode.
    :param Optional[Union[str, os.PathLike]] save_path: Output video file (single mode) or output directory (batch/project mode). Optional in project mode (defaults into the project).
    :param Optional[Union[str, os.PathLike]] config_path: Path to a SimBA ``project_config.ini`` to drive project-mode batch rendering. Default None.
    :param Optional[float] sample_time: If not None, render only the first ``sample_time`` seconds of each video (``int(fps * sample_time)`` frames). Useful for quick previews. Default None (full video). When set, the data/video frame counts need not match exactly.
    :param Optional[int] circle_size: Radius of the body-part circles. If None, inferred from resolution.
    :param Optional[str] colors: Palette name for body-part colors. Default 'Set1'.
    :param str codec: NVENC codec ('h264', 'hevc', or 'av1'). Default 'h264'.
    :param int qp: Constant quantization parameter (0-51) for the NVENC encoder (rc='constqp', high-quality tuning, 4:4:4 chroma). Lower = higher quality and larger files. Default 15.
    :param int buffer_size: Number of frames the threaded NVDEC decoder decodes ahead on its background thread (overlaps decoding with GPU compute/encode). Default 16.
    :param Optional[int] n_workers: Number of parallel decode/encode pipelines (one per NVDEC/NVENC engine); the video is split into this many chunks and concatenated. None auto-detects the GPU's NVDEC count. Must be 1..(NVDEC count) - a larger value raises. Default None.
    :param bool verbose: If True, print progress. Default True.

    :example:

    >>> plotter = PosePlotterNVENC(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH, circle_size=10)
    >>> plotter.run()

    :example II - directory batch, first 10s of each:

    >>> plotter = PosePlotterNVENC(data=DATA_DIR, video_path=VIDEO_DIR, save_path=OUT_DIR, sample_time=10)
    >>> plotter.run()

    :example III - SimBA project:

    >>> plotter = PosePlotterNVENC(config_path='project_folder/project_config.ini')
    >>> plotter.run()
    """

    def __init__(self,
                 data: Union[str, os.PathLike, np.ndarray] = None,
                 video_path: Optional[Union[str, os.PathLike]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 config_path: Optional[Union[str, os.PathLike]] = None,
                 sample_time: Optional[float] = None,
                 circle_size: Optional[int] = None,
                 colors: Optional[str] = 'Set1',
                 codec: str = 'h264',
                 qp: int = 15,
                 buffer_size: int = 16,
                 n_workers: Optional[int] = None,
                 verbose: bool = True):

        missing = [n for n, m in (('PyNvVideoCodec', nvc), ('torch', torch)) if m is None]
        if missing:
            raise SimBAPAckageVersionError(msg=f'{self.__class__.__name__} requires {" and ".join(missing)}, which could not be found in this environment.', source=self.__class__.__name__)

        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_int(name=f'{self.__class__.__name__} qp', value=qp, min_value=0, max_value=51)
        check_int(name=f'{self.__class__.__name__} buffer_size', value=buffer_size, min_value=1)
        if sample_time is not None:
            check_float(name=f'{self.__class__.__name__} sample_time', value=sample_time, min_value=10e-6)
        if circle_size is not None:
            check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
        available = max(1, get_nvdec_count(gpu_name=torch.cuda.get_device_name(0)))
        if n_workers is None:
            n_workers = available
        else:
            check_int(name=f'{self.__class__.__name__} n_workers', value=n_workers, min_value=1)
            if n_workers > available:
                raise SimBAGPUError(msg=f'n_workers={n_workers} was requested but GPU "{torch.cuda.get_device_name(0)}" has only {available} NVDEC engine(s). Pass n_workers <= {available}.', source=self.__class__.__name__)

        self.sample_time = sample_time
        self._circle_size_param = circle_size
        self._colors_param = colors
        self.codec, self.qp, self.verbose, self.buffer_size, self._n_workers_req = codec, qp, verbose, buffer_size, n_workers
        self.res_dev, self._config = None, None
        self.jobs = self._resolve_jobs(data=data, video_path=video_path, save_path=save_path, config_path=config_path)

    def _resolve_single_save(self, save_path: Optional[Union[str, os.PathLike]], video_path: Union[str, os.PathLike]) -> str:
        """Resolve the output path for single-file mode (a directory yields ``<video_name>.mp4``)."""
        if save_path is None:
            raise InvalidInputError(msg='save_path is required when rendering a single video.', source=self.__class__.__name__)
        if isinstance(save_path, str) and os.path.isdir(save_path):
            return os.path.join(save_path, f'{get_fn_ext(video_path)[1]}.mp4')
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        return str(save_path)

    def _build_dir_jobs(self, data_dir: Union[str, os.PathLike], video_dir: Union[str, os.PathLike], save_dir: Union[str, os.PathLike], file_type: str) -> list:
        """Pair every pose file in ``data_dir`` with the video of the same name in ``video_dir``."""
        exts = list({f'.{file_type}', '.csv'})
        files = find_files_of_filetypes_in_directory(directory=data_dir, extensions=exts, raise_error=True)
        jobs = []
        for f in sorted(files):
            stem = get_fn_ext(f)[1]
            vid = find_video_of_file(video_dir=video_dir, filename=stem, raise_error=True)
            jobs.append({'data': f, 'video_path': vid, 'save_path': os.path.join(save_dir, f'{stem}.mp4')})
        return jobs

    def _resolve_jobs(self, data, video_path, save_path, config_path) -> list:
        """Resolve the constructor inputs into a list of per-file render jobs."""
        if isinstance(data, np.ndarray):
            check_valid_array(data=data, source=self.__class__.__name__, accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            if not (isinstance(video_path, str) and os.path.isfile(video_path)):
                raise InvalidInputError(msg='When data is an array, video_path must be a single video file.', source=self.__class__.__name__)
            return [{'data': data, 'video_path': video_path, 'save_path': self._resolve_single_save(save_path, video_path)}]

        if config_path is not None:
            check_file_exist_and_readable(file_path=config_path)
            from simba.mixins.config_reader import ConfigReader
            cfg = ConfigReader(config_path=config_path, read_video_info=False, create_logger=False)
            self._config = cfg
            if isinstance(data, str) and os.path.isfile(data):
                stem = get_fn_ext(data)[1]
                if isinstance(video_path, str) and os.path.isfile(video_path):
                    vid = video_path
                else:
                    vdir = video_path if (isinstance(video_path, str) and os.path.isdir(video_path)) else cfg.video_dir
                    vid = find_video_of_file(video_dir=vdir, filename=stem, raise_error=True)
                if save_path is None:
                    save_path = os.path.join(cfg.project_path, 'frames', 'output', 'pose_nvenc')
                    os.makedirs(save_path, exist_ok=True)
                return [{'data': data, 'video_path': vid, 'save_path': self._resolve_single_save(save_path, vid)}]
            data_dir = data if (isinstance(data, str) and os.path.isdir(data)) else cfg.outlier_corrected_dir
            video_dir = video_path if (isinstance(video_path, str) and os.path.isdir(video_path)) else cfg.video_dir
            check_if_dir_exists(in_dir=data_dir); check_if_dir_exists(in_dir=video_dir)
            if save_path is None:
                save_dir = os.path.join(cfg.project_path, 'frames', 'output', 'pose_nvenc')
                os.makedirs(save_dir, exist_ok=True)
            elif isinstance(save_path, str) and os.path.isdir(save_path):
                save_dir = save_path
            else:
                raise InvalidInputError(msg='In project mode, save_path must be an existing directory (or None to use the project default).', source=self.__class__.__name__)
            return self._build_dir_jobs(data_dir=data_dir, video_dir=video_dir, save_dir=save_dir, file_type=cfg.file_type)

        data_is_dir = isinstance(data, str) and os.path.isdir(data)
        video_is_dir = isinstance(video_path, str) and os.path.isdir(video_path)
        if data_is_dir != video_is_dir:
            raise InvalidInputError(msg='If either data or video_path is a directory, both must be directories.', source=self.__class__.__name__)
        if data_is_dir:
            if not (isinstance(save_path, str) and os.path.isdir(save_path)):
                raise InvalidInputError(msg='In directory (batch) mode, save_path must be an existing directory.', source=self.__class__.__name__)
            return self._build_dir_jobs(data_dir=data, video_dir=video_path, save_dir=save_path, file_type='csv')

        if not (isinstance(data, str) and os.path.isfile(data)):
            raise InvalidFilepathError(msg=f'data must be a pose file, a directory of pose files, or a 3D array; got {data}.', source=self.__class__.__name__)
        check_file_exist_and_readable(file_path=data)
        if not (isinstance(video_path, str) and os.path.isfile(video_path)):
            raise InvalidInputError(msg='video_path must be a single video file when data is a single file.', source=self.__class__.__name__)
        return [{'data': data, 'video_path': video_path, 'save_path': self._resolve_single_save(save_path, video_path)}]

    def _prepare(self, job: dict) -> None:
        """Load one job's pose data + video onto the GPU (respecting ``sample_time``) ahead of encoding."""
        data, video_path = job['data'], job['video_path']
        if isinstance(data, str):
            file_type = self._config.file_type if self._config is not None else get_fn_ext(data)[2].lower().lstrip('.')
            df = read_df(file_path=data, file_type=file_type, check_multiindex=True)
            if self._config is not None and 'input_csv' in os.path.dirname(str(data)):
                df = self._config.insert_column_headers_for_outlier_correction(data_df=df, new_headers=self._config.bp_headers, filepath=data)
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0).reset_index(drop=True)
            if self._config is not None:
                cols = [c for animal in self._config.animal_bp_dict.values() for pair in zip(animal['X_bps'], animal['Y_bps']) for c in pair]
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    raise MissingColumnsError(msg=f'The pose file {os.path.basename(str(data))} is missing {len(missing)} body-part column(s) expected by this SimBA project (e.g. {missing[:5]}). The project pose-configuration has {len(cols) // 2} body-part(s); ensure the data matches it.', source=self.__class__.__name__)
            else:
                cols = [x for x in df.columns if not str(x).lower().endswith('_p')]
            arr = df[cols].values
            if arr.shape[1] % 2 != 0:
                raise CountError(msg=f'The pose data in {os.path.basename(str(data))} has {arr.shape[1]} coordinate columns, which is not an even number of (x, y) pairs - it does not resolve to a whole number of body-parts. Ensure the data matches the expected body-part count (each body-part needs an x and a y column).', source=self.__class__.__name__)
            arr = arr.reshape(arr.shape[0], arr.shape[1] // 2, 2)
        else:
            arr = data
        arr = np.ascontiguousarray(arr.astype(np.int32))
        self.video_meta = get_video_meta_data(video_path=video_path)
        self.w, self.h = self.video_meta['width'], self.video_meta['height']
        n_video, n_data = self.video_meta['frame_count'], arr.shape[0]
        if self.sample_time is None:
            if n_data != n_video:
                raise FrameRangeError(msg=f'The data for {os.path.basename(str(video_path))} contains {n_data} frames while the video contains {n_video} frames. Pass sample_time to render a shorter clip.', source=self.__class__.__name__)
            self.n = n_video
        else:
            self.n = max(1, min(int(self.video_meta['fps'] * self.sample_time), n_video, n_data))
        arr = np.ascontiguousarray(arr[:self.n])
        if self._circle_size_param is None:
            circle_size = np.array([PlottingMixin().get_optimal_circle_size(frame_size=(self.w, self.h))]).astype(np.int32)
        else:
            circle_size = np.array([self._circle_size_param]).astype(np.int32)
        self.video_path = video_path
        self.n_workers = min(self._n_workers_req, self.n)
        self.n_bp = arr.shape[1]
        colors_bgr = np.array(create_color_palette(pallete_name=self._colors_param, increments=self.n_bp))
        b, g, r = colors_bgr[:, 0], colors_bgr[:, 1], colors_bgr[:, 2]
        yy = 16.0 + 0.256788 * r + 0.504129 * g + 0.097906 * b
        cb = 128.0 - 0.148223 * r - 0.290993 * g + 0.439216 * b
        cr = 128.0 + 0.439216 * r - 0.367788 * g - 0.071427 * b
        colors_yuv = np.clip(np.round(np.stack([yy, cb, cr], axis=1)), 0, 255).astype(np.int32)
        self.cs_dev = cuda.to_device(circle_size)
        self.colors_dev = cuda.to_device(np.ascontiguousarray(colors_yuv))
        self.res_dev = cuda.to_device(np.array([self.w, self.h]))
        self.data_dev = cuda.to_device(arr)

    def _encode_chunk(self, start: int, end: int, out_path: str) -> None:
        """Decode -> draw -> NVENC-encode frames [start, end) and mux to ``out_path``.
        Each chunk runs its own ThreadedDecoder (seeked to ``start``) + Encoder, so several
        chunks execute concurrently across the GPU's NVDEC/NVENC engines."""
        yuv_t = torch.empty((3 * self.h, self.w), dtype=torch.uint8, device='cuda:0')
        yuv_dev = cuda.as_cuda_array(yuv_t)
        decoder = get_nvc_decoder(video_path=self.video_path, output_color_type=nvc.OutputColorType.RGB, use_device_memory=True, threaded=True, buffer_size=self.buffer_size, start_frame=start)
        encoder = nvc.CreateEncoder(self.w, self.h, 'YUV444', False, codec=self.codec, tuning_info='high_quality', rc='constqp', qp=self.qp)
        grid_fill = (math.ceil(self.w / TPB2[0]), math.ceil(self.h / TPB2[1]))
        grid_circle = math.ceil(self.n_bp / TPB1)
        raw_path = f'{out_path}.raw.{self.codec}'
        i = start
        with open(raw_path, 'wb') as raw:
            while i < end:
                frames = decoder.get_batch_frames(min(self.buffer_size, end - i))
                if not frames:
                    break
                for f in frames:
                    if i >= end:
                        break
                    rgb = cuda.as_cuda_array(torch.from_dlpack(f))
                    _rgb_to_yuv444_kernel[grid_fill, TPB2](rgb, yuv_dev, self.h)
                    _draw_circles_yuv444_kernel[grid_circle, TPB1](yuv_dev, self.data_dev[i], self.cs_dev, self.res_dev, self.colors_dev, self.h)
                    bitstream = encoder.Encode(yuv_t)
                    if bitstream:
                        raw.write(bytes(bitstream))
                    i += 1
                    if self.verbose and i % 500 == 0:
                        stdout_information(msg=f'Encoded {i}/{self.n} frames...', source=self.__class__.__name__)
            tail = encoder.EndEncode()
            if tail:
                raw.write(bytes(tail))
        decoder.end()
        subprocess.call(f'ffmpeg -y -loglevel error -framerate {self.video_meta["fps"]} -i "{raw_path}" -c copy "{out_path}"', shell=True)
        try:
            os.remove(raw_path)
        except OSError:
            pass

    def _render(self, save_path: Union[str, os.PathLike]) -> None:
        """Render the currently-prepared job to ``save_path``. With multiple NVDEC engines
        (``n_workers`` > 1) the video is split into contiguous chunks processed concurrently,
        then concatenated."""
        timer = SimbaTimer(start=True)
        if self.n_workers <= 1:
            self._encode_chunk(0, self.n, str(save_path))
        else:
            bounds = np.linspace(0, self.n, self.n_workers + 1).astype(int)
            tmp = tempfile.gettempdir()
            chunk_paths = [os.path.join(tmp, f'_posechunk_{os.getpid()}_{k}.mp4') for k in range(self.n_workers)]
            errors, threads = [], []

            def _work(k: int) -> None:
                try:
                    self._encode_chunk(int(bounds[k]), int(bounds[k + 1]), chunk_paths[k])
                except Exception as e:
                    errors.append((k, e))

            for k in range(self.n_workers):
                t = threading.Thread(target=_work, args=(k,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            if errors:
                k, exc = errors[0]
                raise FrameRangeError(msg=f'{self.__class__.__name__} chunk {k} failed: {exc}', source=self.__class__.__name__)
            list_path = os.path.join(tmp, f'_poselist_{os.getpid()}.txt')
            with open(list_path, 'w') as fh:
                for p in chunk_paths:
                    fh.write(f"file '{p}'\n")
            subprocess.call(f'ffmpeg -y -loglevel error -f concat -safe 0 -i "{list_path}" -c copy "{save_path}"', shell=True)
            for p in chunk_paths + [list_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'Pose-estimation video saved at {save_path} (workers: {self.n_workers}).', elapsed_time=timer.elapsed_time_str)

    def run(self) -> None:
        """Render every resolved job (one for a single video, several in directory/project mode) to its output path."""
        total_timer = SimbaTimer(start=True)
        n_jobs = len(self.jobs)
        for idx, job in enumerate(self.jobs):
            if self.verbose:
                stdout_information(msg=f'Processing pose video {idx + 1}/{n_jobs} ({os.path.basename(str(job["video_path"]))})...', source=self.__class__.__name__)
            self._prepare(job=job)
            self._render(save_path=job['save_path'])
        total_timer.stop_timer()
        if self.verbose and n_jobs > 1:
            stdout_success(msg=f'{n_jobs} pose-estimation videos saved.', elapsed_time=total_timer.elapsed_time_str)


# if __name__ == "__main__":
#     DATA_PATH = r"E:\troubleshooting\mitra_emergence_hour\project_folder\csv\outlier_corrected_movement_location\Box1_180mISOcontrol_Females.csv"
#     VIDEO_PATH = r"E:\troubleshooting\mitra_emergence_hour\project_folder\videos\Box1_180mISOcontrol_Females.mp4"
#     SAVE_PATH = r"E:\troubleshooting\mitra_emergence_hour\project_folder\gpu_visualization\Box1_180mISOcontrol_Females.mp4"
#     plotter = PosePlotterNVENC(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH)
#     plotter.run()
#


#if __name__ == "__main__":
    # DATA_PATH =r'I:\mitra\nick_ressler\project_folder\csv\input_csv'  # r"I:\mitra\nick_ressler\project_folder\csv\input_csv\2025-10-15 12-27-07_box1.csv" #r"I:\mitra\nick_ressler\project_folder\csv\input_csv"
    # VIDEO_PATH = r"I:\mitra\nick_ressler\project_folder\videos" #r"I:\mitra\nick_ressler\project_folder\videos\2025-10-15 12-27-07_box1.mp4" #r"I:\mitra\nick_ressler\project_folder\videos"
    # SAVE_PATH = r"I:\mitra\nick_ressler\raw_pose"
    # SAMPLE_TIME = 30
    # CONFIG_PATH = r"I:\mitra\nick_ressler\project_folder\project_config.ini"
    # QUALITY = 12
    # CIRCLE_SIZE = 6
    # plotter = PosePlotterNVENC(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH, config_path=CONFIG_PATH, sample_time=SAMPLE_TIME, qp=QUALITY, circle_size=CIRCLE_SIZE)
    # plotter.run()

