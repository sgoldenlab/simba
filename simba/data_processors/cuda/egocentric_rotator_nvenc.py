__author__ = "Simon Nilsson"

"""
Egocentric video rotation, fully on the GPU (NVDEC -> CUDA -> NVENC).

Rotates and translates each frame so a chosen anchor body-part is centered at a fixed location and
the animal is aligned to a fixed direction - the video equivalent of egocentric pose alignment. This
is the GPU counterpart of the CPU-multiprocessing ``EgocentricVideoRotator``; it takes the same
per-frame ``centers`` and ``rotation_vectors`` (from ``egocentrically_align_pose``), combines the
center-rotation and target-translation into a single affine per frame, and applies the inverse warp
with bilinear interpolation in one CUDA kernel.

Follows the GreyscaleNVENC / PosePlotterNVENC idiom: self-contained ``object`` subclass, module-level
kernel, sharing the ``cuda/utils.py`` NVDEC/NVENC factory functions. GPU-only.
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

from simba.data_processors.cuda.utils import get_nvc_decoder, get_nvc_encoder
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_if_valid_rgb_tuple,
                                check_int, check_nvidea_gpu_available,
                                check_str, check_valid_array,
                                check_valid_boolean, check_valid_tuple)
from simba.utils.data import (align_target_warpaffine_vectors,
                              center_rotation_warpaffine_vectors)
from simba.utils.enums import Formats
from simba.utils.errors import (FrameRangeError, InvalidInputError,
                                SimBAGPUError)
from simba.utils.lookups import get_nvdec_count
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (get_fn_ext, get_pkg_version,
                                    get_video_meta_data)

NVENC_CODECS = ('h264', 'hevc', 'av1')
TPB = (16, 16)


@cuda.jit()
def _egocentric_warp_argb_kernel(rgb, inv_m, argb, fill_b, fill_g, fill_r):
    """Inverse-affine warp of one frame into the ARGB buffer, with bilinear interpolation.

    ``inv_m`` is this frame's 2x3 INVERSE affine [[ia00, ia01, it0], [ia10, ia11, it1]] mapping an
    output pixel (x=col, y=row) back to a source location. Out-of-bounds samples get ``fill`` colour.
    RGB in (index 0=R); ARGB out written as [B, G, R, A] (NVENC 'ARGB')."""
    x, y = cuda.grid(2)
    h, w = rgb.shape[0], rgb.shape[1]
    if y < h and x < w:
        sx = inv_m[0, 0] * x + inv_m[0, 1] * y + inv_m[0, 2]
        sy = inv_m[1, 0] * x + inv_m[1, 1] * y + inv_m[1, 2]
        x0 = int(math.floor(sx))
        y0 = int(math.floor(sy))
        if x0 < 0 or x0 + 1 >= w or y0 < 0 or y0 + 1 >= h:
            argb[y][x][0] = fill_b
            argb[y][x][1] = fill_g
            argb[y][x][2] = fill_r
        else:
            dx = sx - x0
            dy = sy - y0
            w00 = (1.0 - dx) * (1.0 - dy)
            w01 = dx * (1.0 - dy)
            w10 = (1.0 - dx) * dy
            w11 = dx * dy
            r = rgb[y0][x0][0] * w00 + rgb[y0][x0 + 1][0] * w01 + rgb[y0 + 1][x0][0] * w10 + rgb[y0 + 1][x0 + 1][0] * w11
            g = rgb[y0][x0][1] * w00 + rgb[y0][x0 + 1][1] * w01 + rgb[y0 + 1][x0][1] * w10 + rgb[y0 + 1][x0 + 1][1] * w11
            b = rgb[y0][x0][2] * w00 + rgb[y0][x0 + 1][2] * w01 + rgb[y0 + 1][x0][2] * w10 + rgb[y0 + 1][x0 + 1][2] * w11
            argb[y][x][0] = b
            argb[y][x][1] = g
            argb[y][x][2] = r
        argb[y][x][3] = 255


class EgocentricRotatorNVENC(object):
    """Egocentric video rotation, fully on the GPU (NVDEC -> CUDA -> NVENC). GPU-only.

    GPU counterpart of :class:`~simba.video_processors.egocentric_video_rotator.EgocentricVideoRotator`,
    taking the same ``centers`` and ``rotation_vectors``. Each frame is rotated about its ``center`` and
    translated so the center lands on ``anchor_location``; combined into one affine and applied as an
    inverse warp with bilinear interpolation.

    .. seealso::
       CPU version: :class:`~simba.video_processors.egocentric_video_rotator.EgocentricVideoRotator`.
       To produce ``centers`` / ``rotation_vectors``: :func:`~simba.utils.data.egocentrically_align_pose_numba`
       or :func:`~simba.utils.data.egocentrically_align_pose`.

    .. csv-table:: Measured runtime (RTX 4070, h264, threaded decode, n_workers=1)
       :header: Resolution, Frames, Source FPS, Time (s), Encode FPS
       :widths: 22, 16, 16, 16, 16
       :align: center

       800x600, 108000, 30, 65.7, 1643

    :param Union[str, os.PathLike] video_path: Path to the video to rotate.
    :param np.ndarray centers: (n_frames, 2) anchor location per frame (pre-alignment).
    :param np.ndarray rotation_vectors: (n_frames, 2, 2) per-frame rotation matrices.
    :param Tuple[int, int] anchor_location: (x, y) target location the anchor is moved to.
    :param Union[str, os.PathLike] save_path: Output path. If None, ``<video>_rotated.mp4`` next to the source.
    :param Tuple[int, int, int] fill_clr: RGB fill for out-of-frame pixels. Default (0, 0, 0).
    :param str codec: NVENC codec ('h264', 'hevc', or 'av1'). Default 'h264'.
    :param int buffer_size: Threaded NVDEC read-ahead per worker. Default 16.
    :param Optional[int] n_workers: Parallel decode/encode pipelines (one per NVDEC engine); None auto-detects. Default None.
    :param bool verbose: If True, print progress. Default True.

    :example:

    >>> _, centers, rot = egocentrically_align_pose(data=data, anchor_1_idx=6, anchor_2_idx=2, anchor_location=np.array([250, 250]), direction=0)
    >>> EgocentricRotatorNVENC(video_path='in.mp4', centers=centers, rotation_vectors=rot, anchor_location=(250, 250)).run()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 centers: np.ndarray,
                 rotation_vectors: np.ndarray,
                 anchor_location: Tuple[int, int],
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 fill_clr: Tuple[int, int, int] = (0, 0, 0),
                 codec: Literal['h264', 'hevc', 'av1'] = 'h264',
                 buffer_size: int = 16,
                 n_workers: Optional[int] = None,
                 verbose: bool = True):

        get_pkg_version(pkg='PyNvVideoCodec', raise_error=True)
        get_pkg_version(pkg='torch', raise_error=True)
        check_nvidea_gpu_available(raise_error=True)
        check_file_exist_and_readable(file_path=video_path)
        self.video_meta = get_video_meta_data(video_path=video_path)
        n = self.video_meta['frame_count']
        check_valid_array(data=centers, source=f'{self.__class__.__name__} centers', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_axis_0_shape=[n], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=rotation_vectors, source=f'{self.__class__.__name__} rotation_vectors', accepted_ndims=(3,), accepted_axis_0_shape=[n], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_tuple(x=anchor_location, source=f'{self.__class__.__name__} anchor_location', accepted_lengths=(2,), valid_dtypes=(int,), min_integer=1)
        check_if_valid_rgb_tuple(data=fill_clr)
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_int(name=f'{self.__class__.__name__} buffer_size', value=buffer_size, min_value=1)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        available = max(1, get_nvdec_count(gpu_name=torch.cuda.get_device_name(0)))
        if n_workers is None:
            n_workers = available
        else:
            check_int(name=f'{self.__class__.__name__} n_workers', value=n_workers, min_value=1)
            if n_workers > available:
                raise SimBAGPUError(msg=f'n_workers={n_workers} requested but GPU has only {available} NVDEC engine(s).', source=self.__class__.__name__)
        video_dir, video_name, _ = get_fn_ext(filepath=video_path)
        if save_path is None:
            save_path = os.path.join(video_dir, f'{video_name}_rotated.mp4')
        else:
            check_if_dir_exists(in_dir=os.path.dirname(save_path), source=f'{self.__class__.__name__} save_path')
        if os.path.abspath(str(save_path)) == os.path.abspath(str(video_path)):
            raise InvalidInputError(msg=f'save_path {save_path} would overwrite the source video.', source=self.__class__.__name__)
        self.video_path, self.save_path, self.anchor_location = video_path, save_path, anchor_location
        self.codec, self.verbose, self.buffer_size, self.n_workers = codec, verbose, buffer_size, n_workers
        self.fill_clr = fill_clr
        self.inv_m = self._inverse_affines(centers=np.asarray(centers, dtype=np.float64), rotation_vectors=np.asarray(rotation_vectors, dtype=np.float64), anchor=np.array(anchor_location, dtype=np.float64))

    @staticmethod
    def _inverse_affines(centers: np.ndarray, rotation_vectors: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Combine center-rotation + target-translation into one affine per frame and return the
        (N, 2, 3) INVERSE maps used by the warp kernel (output pixel -> source location)."""
        rc = center_rotation_warpaffine_vectors(rotation_vectors=rotation_vectors, centers=centers)  # (N,2,3)
        tr = align_target_warpaffine_vectors(centers=centers, target=anchor)                          # (N,2,3)
        n = rc.shape[0]
        inv = np.empty((n, 2, 3), dtype=np.float32)
        for i in range(n):
            rc3 = np.eye(3); rc3[:2] = rc[i]
            tr3 = np.eye(3); tr3[:2] = tr[i]
            m = tr3 @ rc3                       # forward: rotate about center, then translate to target
            a = m[:2, :2]
            ia = np.linalg.inv(a)
            it = -ia @ m[:2, 2]
            inv[i, :, :2] = ia
            inv[i, :, 2] = it
        return inv

    def _encode_chunk(self, start: int, end: int, out_path: str) -> None:
        """Decode -> warp kernel -> NVENC-encode frames [start, end) and mux to ``out_path``."""
        w, h, n = self.video_meta['width'], self.video_meta['height'], self.video_meta['frame_count']
        fb, fg, fr = self.fill_clr[2], self.fill_clr[1], self.fill_clr[0]   # RGB tuple -> B,G,R
        argb_t = torch.empty((1, h, w, 4), dtype=torch.uint8, device='cuda:0')
        argb_dev = cuda.as_cuda_array(argb_t)[0]
        inv_dev = cuda.to_device(self.inv_m)
        decoder = get_nvc_decoder(video_path=self.video_path, use_device_memory=True, threaded=True, buffer_size=self.buffer_size, start_frame=start)
        encoder = get_nvc_encoder(width=w, height=h, codec=self.codec, fmt='ARGB')
        grid = (math.ceil(w / TPB[0]), math.ceil(h / TPB[1]))
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
                    _egocentric_warp_argb_kernel[grid, TPB](rgb, inv_dev[i], argb_dev, fb, fg, fr)
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
        subprocess.call(f'ffmpeg -y -loglevel error -framerate {self.video_meta["fps"]} -i "{raw_path}" -c copy "{out_path}"', shell=True)
        try:
            os.remove(raw_path)
        except OSError:
            pass

    def run(self) -> None:
        """Rotate the whole video on the GPU and write it to ``save_path``. With ``n_workers`` > 1 the
        video is split into contiguous chunks encoded concurrently, then concatenated."""
        timer = SimbaTimer(start=True)
        n = self.video_meta['frame_count']
        if self.n_workers <= 1:
            self._encode_chunk(0, n, str(self.save_path))
        else:
            bounds = np.linspace(0, n, self.n_workers + 1).astype(int)
            tmp = tempfile.gettempdir()
            chunk_paths = [os.path.join(tmp, f'_egochunk_{os.getpid()}_{k}.mp4') for k in range(self.n_workers)]
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
            list_path = os.path.join(tmp, f'_egolist_{os.getpid()}.txt')
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
            stdout_success(msg=f'Egocentrically rotated video saved at {self.save_path}.', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


if __name__ == '__main__':
    # See EgocentricVideoRotator for how to produce centers/rotation_vectors from pose data.
    pass
