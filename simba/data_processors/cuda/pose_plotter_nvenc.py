__author__ = "Simon Nilsson"

import math
import os
import subprocess
import tempfile
import threading
from typing import Optional, Union

import numpy as np
from numba import cuda

try:
    import PyNvVideoCodec as nvc
except Exception:
    nvc = None
try:
    import torch
except Exception:
    torch = None

from simba.data_processors.cuda.utils import get_nvc_decoder
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int, check_str,
                                check_valid_array, check_valid_boolean)
from simba.utils.data import create_color_palette
from simba.utils.enums import Formats
from simba.utils.errors import (FrameRangeError, SimBAGPUError,
                                SimBAPAckageVersionError)
from simba.utils.lookups import get_nvdec_count
from simba.utils.printing import (SimbaTimer, stdout_information,
                                  stdout_success)
from simba.utils.read_write import get_video_meta_data, read_df

TPB2 = (32, 32, 1)
TPB1 = 64
NVENC_CODECS = ('h264', 'hevc', 'av1')


@cuda.jit()
def _fill_argb_kernel(rgb: np.ndarray, argb: np.ndarray) -> None:
    """RGB(H,W,3) decoded frame -> ARGB(H,W,4) for NVENC. NVENC 'ARGB' wants byte order
    [B, G, R, A] (verified by per-slot calibration)."""
    x, y = cuda.grid(2)
    if y < rgb.shape[0] and x < rgb.shape[1]:
        argb[y][x][0] = rgb[y][x][2]
        argb[y][x][1] = rgb[y][x][1]
        argb[y][x][2] = rgb[y][x][0]
        argb[y][x][3] = 255


@cuda.jit()
def _draw_circles_argb_kernel(argb: np.ndarray, bps: np.ndarray, circle_size: np.ndarray, resolution: np.ndarray, colors: np.ndarray) -> None:
    """Draw one frame's body-part circles into the ARGB buffer. One thread per body-part.
    Palette channel c is written to ARGB slot c (matches the original kernel's channel map)."""
    bp_n = cuda.grid(1)
    if bp_n < 0 or bp_n > (bps.shape[0] - 1):
        return
    bp, color, cs = bps[bp_n], colors[bp_n], circle_size[0]
    for x1 in range(bp[0] - cs, bp[0] + cs):
        for y1 in range(bp[1] - cs, bp[1] + cs):
            if (0 < x1 < resolution[0]) and (0 < y1 < resolution[1]):
                if ((x1 - bp[0]) ** 2 + (y1 - bp[1]) ** 2) < (cs ** 2):
                    argb[y1][x1][0] = color[0]
                    argb[y1][x1][1] = color[1]
                    argb[y1][x1][2] = color[2]


class PosePlotterNVENC(object):
    """
    Overlay pose-estimation data on a video, fully on the GPU (NVDEC -> CUDA -> NVENC).

    .. seealso::
       For the CPU-encode GPU version, see :func:`simba.data_processors.cuda.image.pose_plotter`.
       For CPU methods, see :func:`~simba.plotting.path_plotter.PathPlotterSingleCore` and :func:`~simba.plotting.path_plotter_mp.PathPlotterMulticore`.

    .. note::
       Requires PyNvVideoCodec (NVDEC/NVENC) and torch (for the DLPack decode interop); the
       constructor raises :class:`~simba.utils.errors.SimBAPAckageVersionError` if either is
       missing. NVENC encodes H.264 (YUV420), so output pixels are perceptually equivalent but
       not bit-identical to the cv2 (mpeg4) path - both are lossy; the overlay geometry and
       colors match.

    .. csv-table:: Expected runtimes by image size
       :file: ../../../docs/tables/pose_plotter_nvenc.csv
       :widths: 40, 30, 30
       :align: center
       :header-rows: 1

    :param Union[str, os.PathLike, np.ndarray] data: Path to a CSV with pose-estimation data, or a 3D array (n_frames, n_bodyparts, 2) of locations.
    :param Union[str, os.PathLike] video_path: Path to the video the ``data`` was pose-estimated on.
    :param Union[str, os.PathLike] save_path: Output video path.
    :param Optional[int] circle_size: Radius of the body-part circles. If None, inferred from resolution.
    :param Optional[str] colors: Palette name for body-part colors. Default 'Set1'.
    :param str codec: NVENC codec ('h264', 'hevc', or 'av1'). Default 'h264'.
    :param int buffer_size: Number of frames the threaded NVDEC decoder decodes ahead on its background thread (overlaps decoding with GPU compute/encode). Default 16.
    :param Optional[int] n_workers: Number of parallel decode/encode pipelines (one per NVDEC/NVENC engine); the video is split into this many chunks and concatenated. None auto-detects the GPU's NVDEC count. Must be 1..(NVDEC count) - a larger value raises. Default None.
    :param bool verbose: If True, print progress. Default True.

    :example:

    >>> plotter = PosePlotterNVENC(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH, circle_size=10)
    >>> plotter.run()
    """

    def __init__(self,
                 data: Union[str, os.PathLike, np.ndarray],
                 video_path: Union[str, os.PathLike],
                 save_path: Union[str, os.PathLike],
                 circle_size: Optional[int] = None,
                 colors: Optional[str] = 'Set1',
                 codec: str = 'h264',
                 buffer_size: int = 16,
                 n_workers: Optional[int] = None,
                 verbose: bool = True):

        missing = [n for n, m in (('PyNvVideoCodec', nvc), ('torch', torch)) if m is None]
        if missing:
            raise SimBAPAckageVersionError(msg=f'{self.__class__.__name__} requires {" and ".join(missing)}, which could not be found in this environment.', source=self.__class__.__name__)

        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_int(name=f'{self.__class__.__name__} buffer_size', value=buffer_size, min_value=1)
        available = max(1, get_nvdec_count(gpu_name=torch.cuda.get_device_name(0)))
        if n_workers is None:
            n_workers = available
        else:
            check_int(name=f'{self.__class__.__name__} n_workers', value=n_workers, min_value=1)
            if n_workers > available:
                raise SimBAGPUError(msg=f'n_workers={n_workers} was requested but GPU "{torch.cuda.get_device_name(0)}" has only {available} NVDEC engine(s). Pass n_workers <= {available}.', source=self.__class__.__name__)
        if isinstance(data, str):
            check_file_exist_and_readable(file_path=data)
            df = read_df(file_path=data, file_type='csv')
            cols = [x for x in df.columns if not x.lower().endswith('_p')]
            data = df[cols].values
            data = data.reshape(data.shape[0], int(data.shape[1] / 2), 2)
        else:
            check_valid_array(data=data, source=self.__class__.__name__, accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        data = np.ascontiguousarray(data.astype(np.int32))
        self.video_meta = get_video_meta_data(video_path=video_path)
        self.w, self.h, self.n = self.video_meta['width'], self.video_meta['height'], self.video_meta['frame_count']
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        if data.shape[0] != self.n:
            raise FrameRangeError(msg=f'The data contains {data.shape[0]} frames while the video contains {self.n} frames')
        if circle_size is None:
            circle_size = np.array([PlottingMixin().get_optimal_circle_size(frame_size=(self.w, self.h))]).astype(np.int32)
        else:
            check_int(name=f'{self.__class__.__name__} circle_size', value=circle_size, min_value=1)
            circle_size = np.array([circle_size]).astype(np.int32)

        self.video_path, self.save_path, self.codec, self.verbose, self.buffer_size = video_path, save_path, codec, verbose, buffer_size
        self.n_workers = min(n_workers, self.n)
        self.n_bp = data.shape[1]
        colors = np.array(create_color_palette(pallete_name=colors, increments=self.n_bp)).astype(np.int32)
        self.cs_dev = cuda.to_device(circle_size)
        self.colors_dev = cuda.to_device(colors)
        self.res_dev = cuda.to_device(np.array([self.w, self.h]))
        self.data_dev = cuda.to_device(data)

    def _encode_chunk(self, start: int, end: int, out_path: str) -> None:
        """Decode -> draw -> NVENC-encode frames [start, end) and mux to ``out_path``.
        Each chunk runs its own ThreadedDecoder (seeked to ``start``) + Encoder, so several
        chunks execute concurrently across the GPU's NVDEC/NVENC engines."""
        argb_t = torch.empty((1, self.h, self.w, 4), dtype=torch.uint8, device='cuda:0')
        argb_dev = cuda.as_cuda_array(argb_t)[0]
        decoder = get_nvc_decoder(video_path=self.video_path, output_color_type=nvc.OutputColorType.RGB, use_device_memory=True, threaded=True, buffer_size=self.buffer_size, start_frame=start)
        encoder = nvc.CreateEncoder(self.w, self.h, 'ARGB', False, codec=self.codec)
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
                    _fill_argb_kernel[grid_fill, TPB2](rgb, argb_dev)
                    _draw_circles_argb_kernel[grid_circle, TPB1](argb_dev, self.data_dev[i], self.cs_dev, self.res_dev, self.colors_dev)
                    bitstream = encoder.Encode(argb_t[0])
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

    def run(self) -> None:
        """Render the overlay video and write it to ``save_path``. With multiple NVDEC engines
        (``n_workers`` > 1) the video is split into contiguous chunks processed concurrently,
        then concatenated."""
        timer = SimbaTimer(start=True)
        if self.n_workers <= 1:
            self._encode_chunk(0, self.n, str(self.save_path))
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
            subprocess.call(f'ffmpeg -y -loglevel error -f concat -safe 0 -i "{list_path}" -c copy "{self.save_path}"', shell=True)
            for p in chunk_paths + [list_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'Pose-estimation video saved at {self.save_path} (workers: {self.n_workers}).', elapsed_time=timer.elapsed_time_str)


# if __name__ == "__main__":
#     DATA_PATH = r"E:\troubleshooting\mitra_emergence_hour\project_folder\csv\outlier_corrected_movement_location\Box1_180mISOcontrol_Females.csv"
#     VIDEO_PATH = r"E:\troubleshooting\mitra_emergence_hour\project_folder\videos\Box1_180mISOcontrol_Females.mp4"
#     SAVE_PATH = r"E:\troubleshooting\mitra_emergence_hour\project_folder\gpu_visualization\Box1_180mISOcontrol_Females.mp4"
#     plotter = PosePlotterNVENC(data=DATA_PATH, video_path=VIDEO_PATH, save_path=SAVE_PATH)
#     plotter.run()


