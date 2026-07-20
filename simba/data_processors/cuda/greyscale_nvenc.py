__author__ = "Simon Nilsson"

"""
Convert a video to greyscale fully on the GPU (NVDEC -> CUDA -> NVENC).

Follows the idiom of the two production reference classes ``BackgroundSubtractorCUDA`` and
``PosePlotterNVENC``: a self-contained plain ``object`` subclass that owns its own
``run()`` / ``_run_nvenc()`` and a module-level kernel, sharing only the ``cuda/utils.py``
factory functions (``get_nvc_decoder`` / ``get_nvc_encoder``). GPU-only.
"""

import math
import os
import subprocess
import tempfile
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from numba import cuda

try:
    import torch
except Exception:
    torch = None

from simba.data_processors.cuda.utils import (_cuda_luminance_pixel_to_grey,
                                              get_nvc_decoder, get_nvc_encoder)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_nvidea_gpu_available, check_str,
                                check_valid_boolean, is_video_color)
from simba.utils.errors import InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    get_pkg_version, get_video_meta_data)

NVENC_CODECS = ('h264', 'hevc', 'av1')
TPB = (16, 16)


@cuda.jit()
def _grey_argb_kernel(rgb, argb):
    """RGB (H,W,3) -> greyscale written into ARGB (H,W,4) as [B, G, R, A] = [L, L, L, 255].
    One thread per pixel. Per-pixel luminance uses the shared device primitive
    _cuda_luminance_pixel_to_grey (cuda/utils.py)."""
    x, y = cuda.grid(2)
    if y < rgb.shape[0] and x < rgb.shape[1]:
        lum = _cuda_luminance_pixel_to_grey(rgb[y][x][0], rgb[y][x][1], rgb[y][x][2])
        argb[y][x][0] = lum
        argb[y][x][1] = lum
        argb[y][x][2] = lum
        argb[y][x][3] = 255


class GreyscaleNVENC(object):
    """Convert a video to greyscale, fully on the GPU (NVDEC -> CUDA -> NVENC). GPU-only.

    .. csv-table:: Measured runtime (RTX 4070, h264, single NVDEC/NVENC stream)
       :header: Resolution, Frames, Source FPS, Time (s), Encode FPS
       :widths: 22, 16, 16, 16, 16
       :align: center

       800x600, 108000, 30, 64.1, 1684

    .. csv-table:: Measured batch runtime (RTX 4070, h264, sequential over a directory)
       :header: N videos, Resolution, Total frames, Time (s), Overall FPS, Per-video avg (s)
       :widths: 12, 14, 18, 12, 14, 18
       :align: center

       20, 800x600, 2160000, 1488.9, 1451, 74.4

    :param Union[str, os.PathLike] data_path: A single video file, OR a directory of videos to convert.
    :param Union[str, os.PathLike] save_dir: Output directory. Each converted video is written to ``save_dir/<name>.mp4`` (created if it does not exist).
    :param str codec: NVENC codec ('h264', 'hevc', or 'av1'). Default 'h264'.
    :param bool verbose: If True, print progress. Default True.

    :example:

    >>> GreyscaleNVENC(data_path='in.mp4', save_dir='/out').run()                 # single video
    >>> GreyscaleNVENC(data_path='project_folder/videos', save_dir='/out').run()  # whole directory
    """

    def __init__(self,
                 data_path: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike],
                 codec: Literal['h264', 'hevc', 'av1'] = 'h264',
                 verbose: bool = True):

        get_pkg_version(pkg='PyNvVideoCodec', raise_error=True)
        get_pkg_version(pkg='torch', raise_error=True)
        check_nvidea_gpu_available(raise_error=True)
        check_str(name=f'{self.__class__.__name__} codec', value=codec, options=NVENC_CODECS)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose')
        if os.path.isdir(data_path):
            self.video_paths = find_all_videos_in_directory(directory=data_path, raise_error=True, as_dict=True)
            self.video_paths = list(self.video_paths.values())
        elif os.path.isfile(data_path):
            check_file_exist_and_readable(file_path=data_path)
            self.video_paths = [data_path]
        else:
            raise InvalidInputError(msg=f'{data_path} is not a valid file path or directory path.', source=self.__class__.__name__)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} save_dir', create_if_not_exist=True)
        self.save_dir, self.codec, self.verbose = save_dir, codec, verbose

    def _run_nvenc(self, video_path: Union[str, os.PathLike], save_path: str, video_meta: dict) -> None:
        """Decode -> greyscale kernel -> NVENC-encode every frame of ``video_path`` and mux to ``save_path``."""
        w, h, n = video_meta['width'], video_meta['height'], video_meta['frame_count']
        argb_t = torch.empty((1, h, w, 4), dtype=torch.uint8, device='cuda:0')
        argb_dev = cuda.as_cuda_array(argb_t)[0]
        decoder = get_nvc_decoder(video_path=video_path, use_device_memory=True, threaded=False)  # RGB (H,W,3)
        encoder = get_nvc_encoder(width=w, height=h, codec=self.codec, fmt='ARGB')
        grid = (math.ceil(w / TPB[0]), math.ceil(h / TPB[1]))
        raw_path = os.path.join(tempfile.gettempdir(), f'_grey_{os.getpid()}.{self.codec}')
        with open(raw_path, 'wb') as raw:
            for i in range(len(decoder)):
                rgb = cuda.as_cuda_array(torch.from_dlpack(decoder[i]))
                _grey_argb_kernel[grid, TPB](rgb, argb_dev)
                bitstream = encoder.Encode(argb_t[0])
                if bitstream:
                    raw.write(bytes(bitstream))
                if self.verbose and (i + 1) % 500 == 0:
                    stdout_information(msg=f'Encoded {i + 1}/{n} frames...', source=self.__class__.__name__)
            tail = encoder.EndEncode()
            if tail:
                raw.write(bytes(tail))
        subprocess.call(f'ffmpeg -y -loglevel error -framerate {video_meta["fps"]} -i "{raw_path}" -c copy "{save_path}"', shell=True)
        try:
            os.remove(raw_path)
        except OSError:
            pass

    def run(self) -> None:
        timer = SimbaTimer(start=True)
        for cnt, video_path in enumerate(self.video_paths):
            _, name, _ = get_fn_ext(filepath=video_path)
            if not is_video_color(video=str(video_path)):
                stdout_information(msg=f'Skipping {name} ({cnt + 1}/{len(self.video_paths)}) - already greyscale.', source=self.__class__.__name__)
                continue
            save_path = os.path.join(self.save_dir, f'{name}.mp4')
            if os.path.abspath(save_path) == os.path.abspath(str(video_path)):
                raise InvalidInputError(msg=f'Output {save_path} would overwrite the source video - choose a save_dir that is not the source directory.', source=self.__class__.__name__)
            if self.verbose:
                stdout_information(msg=f'Greyscaling {name} ({cnt + 1}/{len(self.video_paths)})...', source=self.__class__.__name__)
            self._run_nvenc(video_path=video_path, save_path=save_path, video_meta=get_video_meta_data(video_path=video_path))
        timer.stop_timer()
        if self.verbose:
            stdout_success(msg=f'{len(self.video_paths)} greyscale video(s) saved in {self.save_dir}.', elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


if __name__ == '__main__':
    # GreyscaleNVENC(data_path=r'C:\path\to\in.mp4', save_dir=r'C:\path\to\out').run()
    pass
