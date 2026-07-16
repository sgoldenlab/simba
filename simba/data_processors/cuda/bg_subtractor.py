__author__ = "Simon Nilsson"

import math
import os
import queue
import subprocess
import tempfile
import threading
from typing import Optional, Tuple, Union

import cv2
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

from simba.data_processors.cuda.utils import _cuda_luminance_pixel_to_grey
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_if_valid_img,
                                check_if_valid_rgb_tuple, check_int, check_str,
                                check_valid_boolean)
from simba.utils.enums import Formats
from simba.utils.errors import (FFMPEGCodecGPUError,
                                SimBAPAckageVersionError)
from simba.utils.printing import (SimbaTimer, stdout_information,
                                  stdout_success)
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    is_video_color,
                                    read_img_batch_from_video_gpu)

THREADS_PER_BLOCK = (32, 32, 1)
NVENC_CODECS = ('h264', 'hevc', 'av1')


@cuda.jit()
def _bg_subtraction_kernel(imgs, avg_img, results, is_clr, fg_clr, threshold, bg_clr):
    """Device: uint8 background subtraction, 3-channel in/out (BGR). Writes bg colour
    explicitly so the results buffer needs no host pre-fill and can be reused."""
    x, y, n = cuda.grid(3)
    if n < 0 or n > (imgs.shape[0] - 1):
        return
    if y < 0 or y > (imgs.shape[1] - 1):
        return
    if x < 0 or x > (imgs.shape[2] - 1):
        return
    if is_clr[0] == 1:
        b1, g1, r1 = imgs[n][y][x][0], imgs[n][y][x][1], imgs[n][y][x][2]
        b2, g2, r2 = avg_img[y][x][0], avg_img[y][x][1], avg_img[y][x][2]
        b_diff = b1 - b2 if b1 > b2 else b2 - b1
        g_diff = g1 - g2 if g1 > g2 else g2 - g1
        r_diff = r1 - r2 if r1 > r2 else r2 - r1
        grey_diff = _cuda_luminance_pixel_to_grey(b_diff, g_diff, r_diff)
        if grey_diff > threshold[0]:
            if fg_clr[0] != -1:
                results[n][y][x][0] = fg_clr[0]
                results[n][y][x][1] = fg_clr[1]
                results[n][y][x][2] = fg_clr[2]
            else:
                results[n][y][x][0] = b1
                results[n][y][x][1] = g1
                results[n][y][x][2] = r1
        else:
            results[n][y][x][0] = bg_clr[0]
            results[n][y][x][1] = bg_clr[1]
            results[n][y][x][2] = bg_clr[2]
    else:
        v1, v2 = imgs[n][y][x][0], avg_img[y][x][0]
        grey_diff = v1 - v2 if v1 > v2 else v2 - v1
        out = (v1 if fg_clr[0] != -1 else 255) if grey_diff > threshold[0] else 0
        results[n][y][x][0] = out
        results[n][y][x][1] = out
        results[n][y][x][2] = out


@cuda.jit()
def _bg_subtraction_argb_kernel(imgs, avg_img, results, is_clr, fg_clr, threshold, bg_clr):
    """Device: RGB(NVDEC) in -> ARGB out for NVENC. R/B are swapped in the luminance
    call so the mask matches the CPU (bgr24) path. NVENC 'ARGB' wants byte order
    [B, G, R, A] (little-endian ARGB), verified by per-slot calibration."""
    x, y, n = cuda.grid(3)
    if n < 0 or n > (imgs.shape[0] - 1):
        return
    if y < 0 or y > (imgs.shape[1] - 1):
        return
    if x < 0 or x > (imgs.shape[2] - 1):
        return
    if is_clr[0] == 1:
        r1, g1, b1 = imgs[n][y][x][0], imgs[n][y][x][1], imgs[n][y][x][2]
        r2, g2, b2 = avg_img[y][x][0], avg_img[y][x][1], avg_img[y][x][2]
        r_diff = r1 - r2 if r1 > r2 else r2 - r1
        g_diff = g1 - g2 if g1 > g2 else g2 - g1
        b_diff = b1 - b2 if b1 > b2 else b2 - b1
        grey_diff = _cuda_luminance_pixel_to_grey(b_diff, g_diff, r_diff)
        if grey_diff > threshold[0]:
            if fg_clr[0] != -1:
                out_r, out_g, out_b = fg_clr[0], fg_clr[1], fg_clr[2]
            else:
                out_r, out_g, out_b = r1, g1, b1
        else:
            out_r, out_g, out_b = bg_clr[0], bg_clr[1], bg_clr[2]
    else:
        v1, v2 = imgs[n][y][x][0], avg_img[y][x][0]
        out_r = out_g = out_b = (v1 if fg_clr[0] != -1 else 255) if (v1 - v2 if v1 > v2 else v2 - v1) > threshold[0] else 0
    results[n][y][x][0] = out_b
    results[n][y][x][1] = out_g
    results[n][y][x][2] = out_r
    results[n][y][x][3] = 255


class BackgroundSubtractorCUDA(object):
    """
    Remove background from a video using GPU acceleration.

    .. video:: _static/img/video_bg_subtraction.webm
       :width: 800
       :autoplay:
       :loop:
       :muted:
       :align: center

    Uses a fully on-GPU pipeline when PyNvVideoCodec is available: frames are decoded on
    the NVDEC engine straight to device memory, background-subtracted by a CUDA kernel,
    and re-encoded on the NVENC engine - no host round-trips and no CPU video encoding.
    When PyNvVideoCodec is not installed, transparently falls back to an ffmpeg-decode +
    CUDA-kernel + cv2-encode pipeline whose three stages are overlapped across threads.

    .. seealso::
       Reference CPU implementation: :func:`simba.video_processors.video_processing.video_bg_subtraction`.
       To create an ``avg_frm``, see :func:`simba.video_processors.video_processing.create_average_frm`.

    .. note::
       The NVENC back-end encodes H.264 (YUV420), so its output pixels are perceptually
       equivalent but not bit-identical to the CPU (mpeg4) back-end - both are lossy. The
       computed foreground/background classification is the same for both.

    .. csv-table:: Expected runtimes by back-end (RTX 4070, 1280x1024, 1802 frames)
       :header: BACK-END, TIME (S), FPS, SPEEDUP
       :widths: 45, 18, 18, 18
       :align: center

       "nvenc (NVDEC -> CUDA -> NVENC)", 2.47, 731, 20x
       "cv2 fallback (ffmpeg + CUDA + cv2)", 10.87, 166, 4.6x
       "original bg_subtraction_cuda", 50.08, 36, 1x

    .. csv-table:: NVENC back-end: FPS by image size
       :file: ../../../docs/tables/bg_subtraction_cuda_nvenc.csv
       :widths: 40, 30, 30
       :align: center
       :header-rows: 1

    .. note::
       ``batch_size`` does not affect the NVENC back-end, which decodes and encodes
       frame-by-frame - measured FPS at 1280x1024 is flat across batch sizes
       (1 -> 729, 100 -> 735, 500 -> 736). ``batch_size`` only tunes the cv2 fallback.

    :param Union[str, os.PathLike] video_path: Path to the video to remove the background from.
    :param np.ndarray avg_frm: Average frame of the video (BGR), e.g. from :func:`~simba.video_processors.video_processing.create_average_frm`.
    :param Optional[Union[str, os.PathLike]] save_path: Output path. If None, saved next to the input with a ``_bg_removed`` suffix.
    :param Optional[Tuple[int, int, int]] bg_clr: Background fill colour. Default (0, 0, 0).
    :param Optional[Tuple[int, int, int]] fg_clr: Foreground colour. If None, original pixel colours are kept.
    :param Optional[int] batch_size: Frames processed per batch on the fallback (cv2) back-end. Default 500.
    :param Optional[int] threshold: Difference threshold (0-255) above which a pixel is foreground. Default 50.
    :param Optional[bool] use_nvenc: Force the back-end. None (default) = auto (NVENC if available, else cv2). True raises if PyNvVideoCodec is missing.
    :param str codec: NVENC codec when using the NVENC back-end ('h264', 'hevc', or 'av1'). Default 'h264'.
    :param int queue_size: Pipeline queue depth for the cv2 fallback back-end. Default 3.
    :param bool verbose: If True, print progress. Default True.

    :example:

    >>> from simba.video_processors.video_processing import create_average_frm
    >>> video_path = "/mnt/c/troubleshooting/mitra/project_folder/videos/clipped/592_MA147_Gq_CNO_0515.mp4"
    >>> avg_frm = create_average_frm(video_path=video_path)
    >>> subtractor = BackgroundSubtractorCUDA(video_path=video_path, avg_frm=avg_frm, fg_clr=(255, 255, 255))
    >>> subtractor.run()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 avg_frm: np.ndarray,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 bg_clr: Optional[Tuple[int, int, int]] = (0, 0, 0),
                 fg_clr: Optional[Tuple[int, int, int]] = None,
                 batch_size: Optional[int] = 500,
                 threshold: Optional[int] = 50,
                 use_nvenc: Optional[bool] = None,
                 codec: str = 'h264',
                 queue_size: int = 3,
                 verbose: bool = True):

        check_file_exist_and_readable(file_path=video_path)
        check_if_valid_img(data=avg_frm, source=self.__class__.__name__)
        check_if_valid_rgb_tuple(data=bg_clr)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0, max_value=255)
        check_int(name=f'{self.__class__.__name__} queue_size', value=queue_size, min_value=1)
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        if use_nvenc is not None:
            check_valid_boolean(value=use_nvenc, source=f'{self.__class__.__name__} use_nvenc', raise_error=True)
        if fg_clr is not None:
            check_if_valid_rgb_tuple(data=fg_clr)
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(str(save_path)), source=f'{self.__class__.__name__} save_path')

        if use_nvenc is None or use_nvenc:
            missing = [name for name, mod in (('PyNvVideoCodec', nvc), ('torch', torch)) if mod is None]
            nvenc_ok = not missing
            if use_nvenc and not nvenc_ok:
                raise SimBAPAckageVersionError(msg=f'use_nvenc=True but the NVENC back-end requires {" and ".join(missing)}, which could not be found in this environment.', source=self.__class__.__name__)
            use_nvenc = nvenc_ok

        self.video_path = video_path
        self.batch_size, self.threshold = batch_size, threshold
        self.use_nvenc, self.codec, self.queue_size, self.verbose = use_nvenc, codec, queue_size, verbose
        self.video_meta = get_video_meta_data(video_path=video_path)
        self.w, self.h = self.video_meta['width'], self.video_meta['height']
        if save_path is None:
            in_dir, video_name, _ = get_fn_ext(filepath=video_path)
            save_path = os.path.join(in_dir, f'{video_name}_bg_removed.mp4')
        self.save_path = save_path

        avg = cv2.resize(avg_frm, (self.w, self.h)).astype(np.uint8)
        if use_nvenc:
            avg = avg[..., ::-1].copy()
        is_color = np.array([1]) if is_video_color(video_path) else np.array([0])
        fg = np.array(fg_clr).astype(np.int32) if fg_clr is not None else np.array([-1]).astype(np.int32)
        self.y_dev = cuda.to_device(avg)
        self.fg_dev = cuda.to_device(fg)
        self.is_clr_dev = cuda.to_device(is_color)
        self.thr_dev = cuda.to_device(np.array([threshold]).astype(np.int32))
        self.bg_dev = cuda.to_device(np.array(bg_clr).astype(np.int32))

    def run(self) -> Union[str, os.PathLike]:
        """Run background subtraction and write the output video. Returns the save path."""
        timer = SimbaTimer(start=True)
        self._run_nvenc() if self.use_nvenc else self._run_cv2()
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'Video saved at {self.save_path} (back-end: {"nvenc" if self.use_nvenc else "cv2"})', elapsed_time=timer.elapsed_time_str)
        return self.save_path

    def _run_nvenc(self) -> None:
        """Fully on-GPU back-end: NVDEC decode -> CUDA kernel -> NVENC encode -> mux."""
        argb_t = torch.empty((1, self.h, self.w, 4), dtype=torch.uint8, device='cuda:0')
        argb_dev = cuda.as_cuda_array(argb_t)
        decoder = nvc.SimpleDecoder(str(self.video_path), use_device_memory=True, output_color_type=nvc.OutputColorType.RGB)
        encoder = nvc.CreateEncoder(self.w, self.h, 'ARGB', False, codec=self.codec)
        grid = (math.ceil(self.w / 32), math.ceil(self.h / 32), 1)
        raw_path = os.path.join(tempfile.gettempdir(), f'_bgsub_{os.getpid()}.{self.codec}')
        n_dec = len(decoder)
        with open(raw_path, 'wb') as raw:
            for i in range(n_dec):
                t = torch.from_dlpack(decoder[i]).unsqueeze(0)
                x_dev = cuda.as_cuda_array(t)
                _bg_subtraction_argb_kernel[grid, THREADS_PER_BLOCK](x_dev, self.y_dev, argb_dev, self.is_clr_dev, self.fg_dev, self.thr_dev, self.bg_dev)
                bitstream = encoder.Encode(argb_t[0])
                if bitstream:
                    raw.write(bytes(bitstream))
                if self.verbose and (i + 1) % 500 == 0:
                    stdout_information(msg=f'Encoded {i + 1}/{n_dec} frames...', source=self.__class__.__name__)
            tail = encoder.EndEncode()
            if tail:
                raw.write(bytes(tail))
        command = f'ffmpeg -y -loglevel error -framerate {self.video_meta["fps"]} -i "{raw_path}" -c copy "{self.save_path}"'
        subprocess.call(command, shell=True)
        try:
            os.remove(raw_path)
        except OSError:
            pass

    def _run_cv2(self) -> None:
        """Fallback back-end: ffmpeg GPU decode + CUDA kernel + cv2 encode, with the
        decode/compute/encode stages overlapped across a thread pipeline."""
        n = self.video_meta['frame_count']
        batch_cnt = int(max(1, np.ceil(n / self.batch_size)))
        frm_batches = np.array_split(np.arange(0, n), batch_cnt)
        writer = cv2.VideoWriter(str(self.save_path), cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value), self.video_meta['fps'], (self.w, self.h))
        results_dev = cuda.device_array((self.batch_size, self.h, self.w, 3), dtype=np.uint8)
        grid_x, grid_y = math.ceil(self.w / 32), math.ceil(self.h / 32)
        decode_q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        encode_q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        errors = []

        def _decoder():
            try:
                for fb in frm_batches:
                    imgs = read_img_batch_from_video_gpu(video_path=self.video_path, start_frm=fb[0], end_frm=fb[-1], out_format='array')
                    decode_q.put(np.ascontiguousarray(imgs))
            except Exception as e:
                errors.append(('decoder', e))
            finally:
                decode_q.put(None)

        def _encoder():
            try:
                while True:
                    res = encode_q.get()
                    if res is None:
                        break
                    for img in res:
                        writer.write(img)
            except Exception as e:
                errors.append(('encoder', e))

        dec_t = threading.Thread(target=_decoder, daemon=True)
        enc_t = threading.Thread(target=_encoder, daemon=True)
        dec_t.start()
        enc_t.start()
        processed = 0
        while True:
            imgs = decode_q.get()
            if imgs is None:
                break
            batch_n = imgs.shape[0]
            x_dev = cuda.to_device(imgs)
            bpg = (grid_x, grid_y, math.ceil(batch_n / THREADS_PER_BLOCK[2]))
            _bg_subtraction_kernel[bpg, THREADS_PER_BLOCK](x_dev, self.y_dev, results_dev, self.is_clr_dev, self.fg_dev, self.thr_dev, self.bg_dev)
            encode_q.put(results_dev[:batch_n].copy_to_host())
            processed += 1
            if self.verbose:
                stdout_information(msg=f'Processed batch {processed}/{len(frm_batches)}...', source=self.__class__.__name__)
        encode_q.put(None)
        dec_t.join()
        enc_t.join()
        writer.release()
        if errors:
            stage, exc = errors[0]
            raise FFMPEGCodecGPUError(msg=f'{stage} thread failed: {exc}', source=self.__class__.__name__)
