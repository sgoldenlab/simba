import os
import threading
from queue import Queue
from typing import Optional, Tuple, Union

import numpy as np

from simba.utils.checks import (check_instance, check_int,
                                check_nvidea_gpu_available,
                                check_valid_boolean)
from simba.utils.errors import SimBAGPUError
from simba.utils.printing import SimbaTimer
from simba.utils.read_write import (get_video_meta_data,
                                    read_img_batch_from_video,
                                    read_img_batch_from_video_gpu)


class AsyncVideoFrameReader:

    """
    Asynchronous video frame reader that loads and queues video frames in batches using a background thread.

    .. note::
       Wrapper of :func:`~simba.utils.read_write.read_img_batch_from_video_gpu` or :func:`~simba.utils.read_write.read_img_batch_from_video` which allows parallel decoding of video frames.


    :param Union[str, os.PathLike] video_path: Path to the input video file.
    :param int batch_size: Number of frames to read and enqueue per batch (default=100).
    :param int max_que_size: Maximum number of batches to store in the queue. Controls memory use and producer-consumer backpressure (default=2).
    :param Optional[int] start_idx: Frame index to start reading from. Defaults to the beginning of the video.
    :param Optional[int] end_idx: Frame index to stop reading at. Defaults to the last frame of the video.
    :param bool gpu: Whether to use GPU-accelerated video decoding (default=True).
    :param bool verbose: Whether to print progress messages (default=True).
    :param bool greyscale: Whether to convert frames to grayscale (default=False).
    :param bool black_and_white: Whether to convert frames to black and white using thresholding (default=False).

    :example:
    >>> video_path = "/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/03152021_NOB_IOT_8.mp4"
    >>> runner = AsyncVideoFrameReader(video_path=video_path, batch_size=500)
    >>> reader_thread = threading.Thread(target=runner.run, daemon=True)
    >>> reader_thread.start()
    """

    def __init__(self,
                 video_path: Union[str, os.PathLike],
                 batch_size: int = 100,
                 max_que_size: int = 2,
                 start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None,
                 gpu: bool = True,
                 verbose: bool = True,
                 greyscale: bool = False,
                 black_and_white: bool = False,
                 clahe: bool = False):

        self.video_meta_data = get_video_meta_data(video_path=video_path)
        self.start_idx = 0 if start_idx is None else start_idx
        self.end_idx = self.video_meta_data['frame_count'] if end_idx is None else end_idx

        check_int(name=f'{self.__class__.__name__} max_que_size', value=max_que_size, min_value=1, raise_error=True)
        check_int(name=f'{self.__class__.__name__} batch_size', value=batch_size, min_value=1, raise_error=True)
        check_valid_boolean(value=gpu, source=f'{self.__class__.__name__} gpu', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=greyscale, source=f'{self.__class__.__name__} greyscale', raise_error=True)
        check_valid_boolean(value=black_and_white, source=f'{self.__class__.__name__} black_and_white', raise_error=True)
        check_valid_boolean(value=clahe, source=f'{self.__class__.__name__} clahe', raise_error=True)
        self.frame_queue = Queue(maxsize=max_que_size)
        self.batch_size, self.video_path, self.gpu, self.clahe = batch_size, video_path, gpu, clahe
        self.verbose, self.greyscale, self.black_and_white = verbose, greyscale, black_and_white
        if self.gpu and not check_nvidea_gpu_available():
            raise SimBAGPUError(msg=f'GPU passed but no GPU device detected on machine', source=self.__class__.__name__)
        self.batches = [(i, min(i + batch_size, self.end_idx)) for i in range(self.start_idx, self.end_idx, batch_size)]
        self.batch_cnt = len(self.batches)
        self._stop, self._thread = False, None

    def run(self):
        try:
            for batch_cnt, (batch_start_idx, batch_end_idx) in enumerate(self.batches):
                batch_timer = SimbaTimer(start=True)
                if self._stop:
                    break
                if self.gpu:
                    imgs = read_img_batch_from_video_gpu(video_path=self.video_path, start_frm=batch_start_idx, end_frm=batch_end_idx-1, greyscale=self.greyscale, black_and_white=self.black_and_white)
                else:
                    imgs = read_img_batch_from_video(video_path=self.video_path, start_frm=batch_start_idx, end_frm=batch_end_idx-1, greyscale=self.greyscale, black_and_white=self.black_and_white, clahe=self.clahe)
                imgs = np.stack(list(imgs.values()), axis=0)
                self.frame_queue.put((batch_start_idx, batch_end_idx-1, imgs))
                batch_timer.stop_timer()
                if self.verbose:
                    print(f'[{self.__class__.__name__}] ({self.video_meta_data["video_name"]}) frames queued {batch_start_idx}-{batch_end_idx-1} (elapsed time: {batch_timer.elapsed_time_str}s).')
        except Exception as e:
            if self.verbose:
                print(f"[{self.__class__.__name__}] ERROR: {e.args}")
            self.frame_queue.put(e)
        finally:
            self.frame_queue.put(None)

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def kill(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        self.frame_queue, self.batch_end_idxs, self.video_meta_data  = None, None, None
        self.video_path, self._stop = None, None
        if self.verbose:
            print(f"[{self.__class__.__name__}] Reader thread killed and state cleared.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop

def get_async_frame_batch(batch_reader: AsyncVideoFrameReader,
                          timeout: int = 10) -> Tuple[int, int, np.ndarray]:
    """
    Retrieve the next batch of video frames from an `AsyncVideoFrameReader` instance.

    :param AsyncVideoFrameReader batch_reader: An instance of `AsyncVideoFrameReader` that is currently running.
    :param AsyncVideoFrameReader timeout: Maximum time (in seconds) to wait for a frame batch before raising `queue.Empty`.
    :returns: A tuple containing: start frame index of the batch of video (int), end frame index of the batch of video (int) and batch of frames (np.ndarray of shape [batch_size, H, W, C] or similar).
    :rtype: Tuple[int, int, np.ndarray]
    """

    check_int(name=f'{get_async_frame_batch.__name__} timeout', min_value=0, raise_error=True, value=timeout)
    check_instance(source=f'{get_async_frame_batch.__name__} batch_reader', instance=batch_reader, accepted_types=(AsyncVideoFrameReader,), raise_error=True)
    #if not batch_reader.is_running():
    #    raise InvalidInputError(msg=f'batch_reader is not running. Run start() is self before running get_async_frame_batch() ', source=get_async_frame_batch.__name__)
    x = batch_reader.frame_queue.get(timeout=timeout)
    if isinstance(x, Exception):
        raise x
    else:
        return x
