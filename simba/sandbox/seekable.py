from typing import Union, Optional
import os
import cv2
import numpy as np

from simba.utils.checks import check_file_exist_and_readable, check_instance

from simba.utils.errors import InvalidInputError, CorruptedFileError
from simba.utils.warnings import CorruptedFileWarning

def check_if_video_corrupted(video: Union[str, os.PathLike, cv2.VideoCapture],
                             frame_interval: Optional[int] = None,
                             frame_n: Optional[int] = 20,
                             raise_error: Optional[bool] = True) -> None:

    """
    Check if a video file is corrupted by inspecting a set of its frames.

    :param Union[str, os.PathLike] video_path: Path to the video file or cv2.VideoCapture OpenCV object.
    :param Optional[int] frame_interval: Interval between frames to be checked. If None, ``frame_n`` will be used.
    :param Optional[int] frame_n: Number of frames to be checked. If None, ``frame_interval`` will be used.
    :param Optional[bool] raise_error: Whether to raise an error if corruption is found. If False, prints warning.
    :return None:

    :example:
    >>> check_if_video_corrupted(video_path='/Users/simon/Downloads/NOR ENCODING FExMP8.mp4')
    """
    check_instance(source=f'{check_if_video_corrupted.__name__} video', instance=video, accepted_types=(str, cv2.VideoCapture))
    if isinstance(video, str):
        check_file_exist_and_readable(file_path=video)
        cap = cv2.VideoCapture(video)
    else:
        cap = video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if (frame_interval is not None and frame_n is not None) or (frame_interval is None and frame_n is None):
        raise InvalidInputError(msg='Pass frame_interval OR frame_n', source=check_if_video_corrupted.__name__)
    if frame_interval is not None:
        frms_to_check = list(range(0, frame_count, frame_interval))
    else:
        frms_to_check = np.array_split(np.arange(0, frame_count), frame_n)
        frms_to_check = [x[-1] for x in frms_to_check]
    errors = []
    for frm_id in frms_to_check:
        cap.set(1, frm_id)
        ret, _ = cap.read()
        if not ret: errors.append(frm_id)
    if len(errors) > 0:
        if raise_error:
            raise CorruptedFileError(msg=f'Found {len(errors)} corrupted frame(s) at indexes {errors} in video {video}', source=check_if_video_corrupted.__name__)
        else:
            CorruptedFileWarning(msg=f'Found {len(errors)} corrupted frame(s) at indexes {errors} in video {video}', source=check_if_video_corrupted.__name__)
    else:
        pass


#detect_corrupted_frames(video_file_path='/Users/simon/Downloads/NOR ENCODING FExMP8.mp4')


