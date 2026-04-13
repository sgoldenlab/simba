import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import queue
import threading
from typing import Generator, Optional, Union

import cv2
import numpy as np

try:
    import PyNvVideoCodec as nvc
    import torch
except ImportError:
    nvc = None
    torch = None

from simba.data_processors.cuda.utils import get_nvc_decoder
from simba.utils.checks import (check_int, check_nvidea_gpu_available,
                                check_valid_boolean)
from simba.utils.errors import SimBAGPUError, SimBAModuleNotFoundError


class NvDecReader:
    """
    Async GPU-accelerated video frame reader using NVIDIA NVDEC.

    Decodes frames on a background thread via ``PyNvVideoCodec.SimpleDecoder``
    and yields them as numpy arrays (or GPU tensors when ``use_device_memory=True``)
    from a bounded queue, providing backpressure so that at most
    ``max_batches_pending`` decoded batches are held in memory at any time.

    .. csv-table:: NVDEC DECODE RUNTIMES (2 VIDEOS, 216000 FRAMES, 800x600)
       :file: ../../docs/tables/NvDecReader.csv
       :widths: 33, 33, 33
       :align: center
       :header-rows: 1

    :param Union[str, os.PathLike] video_path: Path to the video file.
    :param int gpu_id: NVIDIA GPU device index. Default 0.
    :param int batch_size: Frames decoded per batch. Default 32.
    :param int max_batches_pending: Max decoded batches buffered in memory. Controls peak memory usage. Default 3.
    :param int n_decoders: Number of concurrent NVDEC decoder sessions. Each occupies one hardware NVDEC engine. Set to the number of NVDEC engines on your GPU for max throughput (e.g. 2 for RTX 4090, 5 for RTX 3090). Pass -1 to auto-detect and use all available NVDEC engines. Default 1.
    :param bool use_device_memory: If True, keep decoded frames as GPU tensors. If False, return CPU numpy arrays. Default False.
    :param Optional[nvc.OutputColorType] output_color_type: Output color format. Default ``nvc.OutputColorType.RGB``.
    :param bool bgr: If True, convert decoded RGB frames to BGR for OpenCV. Only applies when ``use_device_memory=False``. Default True.

    :example:
    >>> reader = NvDecReader(video_path='test.mp4', batch_size=64, max_batches_pending=2, n_decoders=2)
    >>> reader.start()
    >>> for frame_idx, frame in reader:
    ...     cv2.imshow('frame', frame)
    >>> reader.kill()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 gpu_id: int = 0,
                 batch_size: int = 32,
                 max_batches_pending: int = 3,
                 n_decoders: int = 1,
                 use_device_memory: bool = False,
                 output_color_type: Optional[nvc.OutputColorType] = None,
                 bgr: bool = True):

        if nvc is None:
            raise SimBAModuleNotFoundError(msg='PyNvVideoCodec is required but not installed. Install via: pip install pynvvideocodec', source=self.__class__.__name__)
        if torch is None:
            raise SimBAModuleNotFoundError(msg='PyTorch is required but not installed.', source=self.__class__.__name__)
        if not check_nvidea_gpu_available():
            raise SimBAGPUError(msg='No NVIDIA GPU detected.', source=self.__class__.__name__)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1)
        check_int(name=f'{self.__class__.__name__} max_batches_pending', value=max_batches_pending, min_value=1)
        check_int(name=f'{self.__class__.__name__} n_decoders', value=n_decoders, min_value=-1)
        check_valid_boolean(value=bgr, source=f'{self.__class__.__name__} bgr', raise_error=True)
        if output_color_type is None:
            output_color_type = nvc.OutputColorType.RGB
        from simba.data_processors.cuda.utils import _is_cuda_available
        from simba.utils.lookups import get_nvdec_count
        gpu_available, gpu_devices = _is_cuda_available()
        gpu_name = gpu_devices[gpu_id]['model'] if gpu_devices and gpu_id in gpu_devices else None
        max_nvdec = get_nvdec_count(gpu_name=gpu_name)
        if n_decoders == -1:
            n_decoders = max_nvdec
        elif n_decoders > max_nvdec:
            raise SimBAGPUError(msg=f'{self.__class__.__name__} n_decoders={n_decoders} exceeds the max NVDEC engines ({max_nvdec}) on {gpu_name}.', source=self.__class__.__name__)
        self.video_path, self.gpu_id, self.batch_size = video_path, gpu_id, batch_size
        self.max_batches_pending, self.n_decoders = max_batches_pending, n_decoders
        self.use_device_memory, self.output_color_type, self.bgr = use_device_memory, output_color_type, bgr
        self._decoder = get_nvc_decoder(video_path=video_path, gpu_id=gpu_id, use_device_memory=use_device_memory, output_color_type=output_color_type)
        self._stop, self._threads, self.frame_queue, self.frame_count = False, [], None, len(self._decoder)

    @property
    def _batch_indices(self):
        batches = []
        for batch_idx, start in enumerate(range(0, self.frame_count, self.batch_size)):
            end = min(start + self.batch_size, self.frame_count)
            batches.append((batch_idx, list(range(start, end))))
        return batches

    def _decode_worker(self, assigned_batches: list, result_dict: dict, ready_events: dict):
        try:
            decoder = get_nvc_decoder(video_path=self.video_path, gpu_id=self.gpu_id, use_device_memory=self.use_device_memory, output_color_type=self.output_color_type)
            for batch_idx, frame_indices in assigned_batches:
                if self._stop:
                    break
                frames = decoder.get_batch_frames_by_index(frame_indices)
                batch_frames = []
                for i, frame in enumerate(frames):
                    if self._stop:
                        break
                    if self.use_device_memory:
                        img = torch.from_dlpack(frame)
                    else:
                        img = torch.from_dlpack(frame).cpu().numpy().astype(np.uint8)
                        if self.bgr and img.ndim == 3 and img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    batch_frames.append((frame_indices[i], img))
                result_dict[batch_idx] = batch_frames
                ready_events[batch_idx].set()
        except Exception as e:
            for batch_idx, _ in assigned_batches:
                if batch_idx not in result_dict:
                    result_dict[batch_idx] = e
                    ready_events[batch_idx].set()

    def _reorder_worker(self, result_dict: dict, ready_events: dict, all_batches: list):
        try:
            for batch_idx, _ in all_batches:
                if self._stop:
                    break
                ready_events[batch_idx].wait()
                batch_data = result_dict.pop(batch_idx)
                if isinstance(batch_data, Exception):
                    self.frame_queue.put(batch_data)
                    return
                for item in batch_data:
                    if self._stop:
                        break
                    self.frame_queue.put(item)
        finally:
            self.frame_queue.put(None)

    def start(self) -> None:
        if len(self._threads) > 0 and any(t.is_alive() for t in self._threads):
            return
        self._stop = False
        self._threads = []
        self.frame_queue = queue.Queue(maxsize=self.max_batches_pending * self.batch_size)
        all_batches = self._batch_indices
        result_dict = {}
        ready_events = {batch_idx: threading.Event() for batch_idx, _ in all_batches}
        per_decoder = [[] for _ in range(self.n_decoders)]
        for i, batch in enumerate(all_batches):
            per_decoder[i % self.n_decoders].append(batch)
        for dec_id in range(self.n_decoders):
            if len(per_decoder[dec_id]) == 0:
                continue
            t = threading.Thread(target=self._decode_worker, args=(per_decoder[dec_id], result_dict, ready_events), daemon=True)
            t.start()
            self._threads.append(t)
        reorder_t = threading.Thread(target=self._reorder_worker, args=(result_dict, ready_events, all_batches), daemon=True)
        reorder_t.start()
        self._threads.append(reorder_t)

    def stop(self) -> None:
        """Signal decode threads to stop after the current batch."""
        self._stop = True

    def kill(self) -> None:
        """Stop all decode threads, drain the queue, and clear state."""
        self.stop()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []
        if self.frame_queue is not None:
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            self.frame_queue = None

    def is_running(self) -> bool:
        """Return True if any decode thread is alive and not stopped."""
        return len(self._threads) > 0 and any(t.is_alive() for t in self._threads) and not self._stop

    def __iter__(self) -> Generator:
        if len(self._threads) == 0 or not any(t.is_alive() for t in self._threads):
            raise RuntimeError(f'{self.__class__.__name__}: call start() before iterating.')
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def __len__(self):
        return self.frame_count

if __name__ == "__main__":
    import glob
    import time

    from simba.utils.printing import stdout_information
    from simba.utils.read_write import get_fn_ext

    VIDEO_DIR = r"E:\troubleshooting\mitra_emergence_hour\project_folder\frames\output\pose_gpu"
    BATCH_SIZE = 250
    MAX_BATCHES_PENDING = 2
    N_DECODERS = -1

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_paths = glob.glob(os.path.join(VIDEO_DIR, '*'))
    video_paths = [v for v in video_paths if os.path.isfile(v) and os.path.splitext(v)[1].lower() in VIDEO_EXTS]
    total_start = time.perf_counter()
    for video_cnt, video_path in enumerate(video_paths):
        _, video_name, _ = get_fn_ext(filepath=video_path)
        video_start = time.perf_counter()
        reader = NvDecReader(video_path=video_path, batch_size=BATCH_SIZE, max_batches_pending=MAX_BATCHES_PENDING, n_decoders=N_DECODERS)
        reader.start()
        frm_cnt = 0
        for frame_idx, frame in reader:
            frm_cnt += 1
        reader.kill()
        video_elapsed = time.perf_counter() - video_start
        stdout_information(msg=f'Video {video_cnt + 1}/{len(video_paths)} ({video_name}): {frm_cnt} frames in {video_elapsed:.2f}s ({frm_cnt / video_elapsed:.1f} fps) (decoders: {reader.n_decoders})', source='NvDecReader benchmark')
    total_elapsed = time.perf_counter() - total_start
    stdout_information(msg=f'All {len(video_paths)} videos completed in {total_elapsed:.2f}s', source='NvDecReader benchmark')