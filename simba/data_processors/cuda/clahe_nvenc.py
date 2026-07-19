__author__ = "Simon Nilsson"

"""
Colour-preserving CLAHE, fully on the GPU (NVDEC -> CUDA -> NVENC).

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the luminance (Y) channel
in YCrCb colour space; the chroma (Cr, Cb) channels are carried through unchanged, so the output
stays in colour (unlike SimBA's greyscale ``clahe_enhance_video``). Y/Cr/Cb use cv2's BT.601
coefficients, so the colour round-trip is validatable against ``cv2.cvtColor``.

Per frame the pipeline runs four CUDA kernels:
  1. ``_ycc_hist_kernel``   - RGB -> YCrCb (store Y,Cr,Cb) + accumulate a per-tile Y histogram.
  2. ``_clahe_lut_kernel``  - per tile: clip the histogram, redistribute the excess, CDF -> 0..255 LUT.
  3. ``_clahe_apply_kernel``- per pixel: bilinearly interpolate the neighbouring tile LUTs for Y,
                              recombine (Y', Cr, Cb) -> RGB, write into the ARGB buffer.
  (``_zero_i32_kernel`` clears the histogram between frames.)

Follows the GreyscaleNVENC / PosePlotterNVENC idiom: self-contained ``object`` subclass, module-level
kernels, sharing the ``cuda/utils.py`` NVDEC/NVENC factory functions. GPU-only.
"""

import math
import os
import subprocess
import tempfile
import threading
from typing import Optional, Tuple, Union

import numpy as np

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from numba import cuda

try:
    import torch
except Exception:
    torch = None

from simba.data_processors.cuda.utils import (_cuda_digital_pixel_to_grey,
                                              get_nvc_decoder, get_nvc_encoder)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.errors import (FrameRangeError, InvalidInputError,
                                SimBAGPUError)
from simba.utils.lookups import get_nvdec_count
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    get_pkg_version, get_video_meta_data)

NVENC_CODECS = ('h264', 'hevc', 'av1')
TPB = (16, 16)     # threads-per-block for the per-pixel kernels
TPB_TILE = 64      # threads-per-block for the per-tile LUT kernel
TPB_ZERO = 256     # threads-per-block for the histogram-zeroing kernel


@cuda.jit()
def _zero_i32_kernel(hist):
    """Zero a (n_tiles_y, n_tiles_x, 256) int32 histogram between frames. One thread per bin."""
    idx = cuda.grid(1)
    ntx, nb = hist.shape[1], hist.shape[2]
    total = hist.shape[0] * ntx * nb
    if idx < total:
        ty = idx // (ntx * nb)
        rem = idx % (ntx * nb)
        hist[ty, rem // nb, rem % nb] = 0


@cuda.jit()
def _ycc_hist_kernel(rgb, ycc, hist, tile_h, tile_w):
    """RGB(H,W,3) -> YCrCb stored in ``ycc`` (H,W,3) + per-tile Y histogram (atomic add).
    Y uses the shared BT.601 primitive _cuda_digital_pixel_to_grey; Cr/Cb use cv2's BT.601 formula."""
    x, y = cuda.grid(2)
    if y < rgb.shape[0] and x < rgb.shape[1]:
        r, g, b = rgb[y][x][0], rgb[y][x][1], rgb[y][x][2]
        yf = _cuda_digital_pixel_to_grey(r, g, b)
        yi = int(yf + 0.5)
        if yi > 255:
            yi = 255
        cr = (r - yf) * 0.713 + 128.0
        cb = (b - yf) * 0.564 + 128.0
        if cr < 0.0: cr = 0.0
        elif cr > 255.0: cr = 255.0
        if cb < 0.0: cb = 0.0
        elif cb > 255.0: cb = 255.0
        ycc[y][x][0] = yi
        ycc[y][x][1] = int(cr + 0.5)
        ycc[y][x][2] = int(cb + 0.5)
        # tile_h/tile_w are ceil(dim / n_tiles), so the grid covers the whole frame and every
        # pixel maps to a valid tile (the last tile row/col is simply partial when the dimension
        # is not divisible). Per-tile normalization in _clahe_lut_kernel handles the partial tile.
        cuda.atomic.add(hist, (y // tile_h, x // tile_w, yi), 1)


@cuda.jit()
def _clahe_lut_kernel(hist, lut, clip_limit):
    """Per tile (one thread each): clip the histogram, redistribute the clipped excess, then build
    a cumulative 0..255 LUT. The clip level and LUT scale are derived from the tile's ACTUAL pixel
    count (sum of the histogram), so partial edge tiles (from ceil tiling) normalise correctly."""
    idx = cuda.grid(1)
    nty, ntx = lut.shape[0], lut.shape[1]
    if idx >= nty * ntx:
        return
    ty, tx = idx // ntx, idx % ntx
    total = 0
    for i in range(256):
        total += hist[ty, tx, i]
    if total <= 0:
        for i in range(256):
            lut[ty, tx, i] = 0
        return
    clip_scaled = int(clip_limit * total / 256)
    if clip_scaled < 1:
        clip_scaled = 1
    excess = 0
    for i in range(256):
        c = hist[ty, tx, i]
        if c > clip_scaled:
            excess += c - clip_scaled
            hist[ty, tx, i] = clip_scaled
    redist = excess // 256
    residual = excess - redist * 256
    for i in range(256):
        hist[ty, tx, i] += redist
    if residual > 0:                     # strided single pass, matching cv2 (no mop-up)
        step = 256 // residual
        if step < 1:
            step = 1
        i = 0
        while i < 256 and residual > 0:
            hist[ty, tx, i] += 1
            residual -= 1
            i += step
    lut_scale = 255.0 / total
    cdf = 0
    for i in range(256):
        cdf += hist[ty, tx, i]
        v = int(cdf * lut_scale + 0.5)
        if v > 255:
            v = 255
        lut[ty, tx, i] = v


@cuda.jit()
def _clahe_apply_kernel(ycc, lut, argb, tile_h, tile_w):
    """Per pixel: bilinearly interpolate the 4 neighbouring tile LUTs for this pixel's Y, recombine
    (Y', Cr, Cb) -> RGB (cv2 BT.601), and write [B, G, R, 255] into the ARGB buffer."""
    x, y = cuda.grid(2)
    if y < ycc.shape[0] and x < ycc.shape[1]:
        nty, ntx = lut.shape[0], lut.shape[1]
        yi = ycc[y][x][0]
        gy = (y - tile_h * 0.5) / tile_h
        ty1 = int(math.floor(gy))
        fy = gy - ty1
        ty2 = ty1 + 1
        if ty1 < 0: ty1 = 0
        elif ty1 > nty - 1: ty1 = nty - 1
        if ty2 < 0: ty2 = 0
        elif ty2 > nty - 1: ty2 = nty - 1
        gx = (x - tile_w * 0.5) / tile_w
        tx1 = int(math.floor(gx))
        fx = gx - tx1
        tx2 = tx1 + 1
        if tx1 < 0: tx1 = 0
        elif tx1 > ntx - 1: tx1 = ntx - 1
        if tx2 < 0: tx2 = 0
        elif tx2 > ntx - 1: tx2 = ntx - 1
        top = lut[ty1, tx1, yi] * (1.0 - fx) + lut[ty1, tx2, yi] * fx
        bot = lut[ty2, tx1, yi] * (1.0 - fx) + lut[ty2, tx2, yi] * fx
        y_new = top * (1.0 - fy) + bot * fy
        crd = ycc[y][x][1] - 128.0
        cbd = ycc[y][x][2] - 128.0
        rr = y_new + 1.403 * crd
        gg = y_new - 0.714 * crd - 0.344 * cbd
        bb = y_new + 1.773 * cbd
        if rr < 0.0: rr = 0.0
        elif rr > 255.0: rr = 255.0
        if gg < 0.0: gg = 0.0
        elif gg > 255.0: gg = 255.0
        if bb < 0.0: bb = 0.0
        elif bb > 255.0: bb = 255.0
        argb[y][x][0] = int(bb + 0.5)
        argb[y][x][1] = int(gg + 0.5)
        argb[y][x][2] = int(rr + 0.5)
        argb[y][x][3] = 255


class ClaheNVENC(object):
    """Colour-preserving CLAHE, fully on the GPU (NVDEC -> CUDA -> NVENC). GPU-only.

    CLAHE is applied to the Y (luma) channel in YCrCb space; Cr/Cb are preserved, so the output
    stays in colour. Matches the parameter defaults of SimBA's CPU ``clahe_enhance_video``.

    .. seealso::
       CPU reference (greyscale output): :func:`~simba.video_processors.video_processing.clahe_enhance_video`
       and its multicore variant :func:`~simba.video_processors.video_processing.clahe_enhance_video_mp`.
       For GPU greyscale (no CLAHE), see :class:`~simba.data_processors.cuda.greyscale_nvenc.GreyscaleNVENC`.

    .. csv-table:: Measured runtime (RTX 4070, h264, threaded decode, n_workers=1)
       :header: Resolution, Frames, Source FPS, Time (s), Encode FPS
       :widths: 22, 16, 16, 16, 16
       :align: center

       800x600, 108000, 30, 68.2, 1584

    .. note::
       Validated against ``cv2.createCLAHE`` over the same YCrCb pipeline. For tile grids that
       divide the frame evenly, the GPU output matches cv2 to ~1 grey level (max ~3, from uint8
       YCrCb rounding). When the frame is not divisible by the tile count, this class uses
       ceil-sized tiles with per-tile normalisation rather than cv2's reflect-padding, so edge
       tiles differ by a few levels (e.g. 800x600 @ (16,16): mean ~3, 84% of pixels within 5).

    :param Union[str, os.PathLike] data_path: A single video file, OR a directory of videos.
    :param Union[str, os.PathLike] save_dir: Output directory. Each result is written to ``save_dir/<name>.mp4`` (created if needed).
    :param int clip_limit: CLAHE contrast-amplification limit. Default 2.
    :param Tuple[int, int] tile_grid_size: Number of tiles as (n_tiles_x, n_tiles_y), matching cv2's tileGridSize. Default (16, 16).
    :param str codec: NVENC codec ('h264', 'hevc', or 'av1'). Default 'h264'.
    :param int buffer_size: Number of frames the threaded NVDEC decoder reads ahead per worker (overlaps decode with GPU compute/encode). Default 16.
    :param Optional[int] n_workers: Number of parallel decode/encode pipelines (one per NVDEC/NVENC engine); each video is split into this many contiguous chunks, encoded concurrently, and concatenated. None auto-detects the GPU's NVDEC count. Must be 1..(NVDEC count); larger raises. On a single-NVDEC GPU this is 1 (single stream). Default None.
    :param bool verbose: If True, print progress. Default True.

    :example:

    >>> ClaheNVENC(data_path='in.mp4', save_dir='/out').run()
    >>> ClaheNVENC(data_path='project_folder/videos', save_dir='/out', clip_limit=3).run()
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 clip_limit: int = 2,
                 tile_grid_size: Tuple[int, int] = (16, 16),
                 codec: Literal['h264', 'hevc', 'av1'] = 'h264',
                 buffer_size: int = 16,
                 n_workers: Optional[int] = None,
                 verbose: bool = True):

        get_pkg_version(pkg='PyNvVideoCodec', raise_error=True)
        get_pkg_version(pkg='torch', raise_error=True)
        check_nvidea_gpu_available(raise_error=True)
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_int(name=f'{self.__class__.__name__} clip_limit', value=clip_limit, min_value=1)
        check_int(name=f'{self.__class__.__name__} buffer_size', value=buffer_size, min_value=1)
        check_valid_tuple(x=tile_grid_size, source=f'{self.__class__.__name__} tile_grid_size', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        available = max(1, get_nvdec_count(gpu_name=torch.cuda.get_device_name(0)))
        if n_workers is None:
            n_workers = available
        else:
            check_int(name=f'{self.__class__.__name__} n_workers', value=n_workers, min_value=1)
            if n_workers > available:
                raise SimBAGPUError(msg=f'n_workers={n_workers} was requested but GPU "{torch.cuda.get_device_name(0)}" has only {available} NVDEC engine(s). Pass n_workers <= {available}.', source=self.__class__.__name__)
        if os.path.isdir(data_path):
            self.video_paths = list(find_all_videos_in_directory(directory=data_path, raise_error=True, as_dict=True).values())
        elif os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.video_paths = [data_path]
        else:
            raise InvalidInputError(msg=f'{data_path} is not a valid file path or directory path.', source=self.__class__.__name__)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', create_if_not_exist=True)
        self.save_dir, self.clip_limit, self.tile_grid_size = save_dir, clip_limit, tile_grid_size
        self.codec, self.verbose, self.buffer_size, self.n_workers = codec, verbose, buffer_size, n_workers

    def _encode_chunk(self, video_path: Union[str, os.PathLike], start: int, end: int, out_path: str, video_meta: dict) -> None:
        """Decode -> CLAHE (4 kernels) -> NVENC-encode frames [start, end) of ``video_path`` and mux to
        ``out_path``. Each chunk owns its own ThreadedDecoder (seeked to ``start``), Encoder and device
        buffers, so several chunks run concurrently across the GPU's NVDEC/NVENC engines."""
        w, h, n = video_meta['width'], video_meta['height'], video_meta['frame_count']
        ntx, nty = self.tile_grid_size
        tile_w, tile_h = math.ceil(w / ntx), math.ceil(h / nty)   # ceil so tiles cover the whole frame
        argb_t = torch.empty((1, h, w, 4), dtype=torch.uint8, device='cuda:0')
        argb_dev = cuda.as_cuda_array(argb_t)[0]
        ycc_dev = cuda.device_array((h, w, 3), dtype='uint8')
        hist_dev = cuda.device_array((nty, ntx, 256), dtype='int32')
        lut_dev = cuda.device_array((nty, ntx, 256), dtype='uint8')
        decoder = get_nvc_decoder(video_path=video_path, use_device_memory=True, threaded=True, buffer_size=self.buffer_size, start_frame=start)
        encoder = get_nvc_encoder(width=w, height=h, codec=self.codec, fmt='ARGB')
        grid_px = (math.ceil(w / TPB[0]), math.ceil(h / TPB[1]))
        grid_tile = math.ceil((nty * ntx) / TPB_TILE)
        grid_zero = math.ceil((nty * ntx * 256) / TPB_ZERO)
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
                    _zero_i32_kernel[grid_zero, TPB_ZERO](hist_dev)
                    _ycc_hist_kernel[grid_px, TPB](rgb, ycc_dev, hist_dev, tile_h, tile_w)
                    _clahe_lut_kernel[grid_tile, TPB_TILE](hist_dev, lut_dev, self.clip_limit)
                    _clahe_apply_kernel[grid_px, TPB](ycc_dev, lut_dev, argb_dev, tile_h, tile_w)
                    bitstream = encoder.Encode(argb_t[0])
                    if bitstream:
                        raw.write(bytes(bitstream))
                    i += 1
                    if self.verbose and i % 500 == 0:
                        stdout_information(msg=f'Encoded {i}/{n} frames...', source=self.__class__.__name__)
            tail = encoder.EndEncode()
            if tail:
                raw.write(bytes(tail))
        decoder.end()
        subprocess.call(f'ffmpeg -y -loglevel error -framerate {video_meta["fps"]} -i "{raw_path}" -c copy "{out_path}"', shell=True)
        try:
            os.remove(raw_path)
        except OSError:
            pass

    def _run_nvenc(self, video_path: Union[str, os.PathLike], save_path: str, video_meta: dict) -> None:
        """CLAHE-encode ``video_path`` to ``save_path``. With ``n_workers`` > 1 the video is split into
        contiguous frame chunks encoded concurrently (one per NVDEC/NVENC engine), then concatenated."""
        w, h, n = video_meta['width'], video_meta['height'], video_meta['frame_count']
        ntx, nty = self.tile_grid_size
        if (ntx > w) or (nty > h):
            raise InvalidInputError(msg=f'tile_grid_size {self.tile_grid_size} is larger than the video ({w}x{h}).', source=self.__class__.__name__)
        if self.n_workers <= 1:
            self._encode_chunk(video_path, 0, n, str(save_path), video_meta)
            return
        bounds = np.linspace(0, n, self.n_workers + 1).astype(int)
        tmp = tempfile.gettempdir()
        chunk_paths = [os.path.join(tmp, f'_clahechunk_{os.getpid()}_{k}.mp4') for k in range(self.n_workers)]
        errors, threads = [], []

        def _work(k: int) -> None:
            try:
                self._encode_chunk(video_path, int(bounds[k]), int(bounds[k + 1]), chunk_paths[k], video_meta)
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
        list_path = os.path.join(tmp, f'_clahelist_{os.getpid()}.txt')
        with open(list_path, 'w') as fh:
            for p in chunk_paths:
                fh.write(f"file '{p}'\n")
        subprocess.call(f'ffmpeg -y -loglevel error -f concat -safe 0 -i "{list_path}" -c copy "{save_path}"', shell=True)
        for p in chunk_paths + [list_path]:
            try:
                os.remove(p)
            except OSError:
                pass

    def run(self) -> None:
        """CLAHE-enhance every input video on the GPU and write each to ``save_dir``."""
        timer = SimbaTimer(start=True)
        for cnt, video_path in enumerate(self.video_paths):
            _, name, _ = get_fn_ext(filepath=video_path)
            save_path = os.path.join(self.save_dir, f'{name}.mp4')
            if os.path.abspath(save_path) == os.path.abspath(str(video_path)):
                raise InvalidInputError(msg=f'Output {save_path} would overwrite the source video - choose a save_dir that is not the source directory.', source=self.__class__.__name__)
            if self.verbose:
                stdout_information(msg=f'CLAHE-enhancing {name} ({cnt + 1}/{len(self.video_paths)})...', source=self.__class__.__name__)
            self._run_nvenc(video_path=video_path, save_path=save_path, video_meta=get_video_meta_data(video_path=video_path))
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'{len(self.video_paths)} CLAHE video(s) saved in {self.save_dir}.', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


if __name__ == '__main__':
    # ClaheNVENC(data_path=r'C:\path\to\in.mp4', save_dir=r'C:\path\to\out').run()
    pass
