import cv2
from simba.misc_tools import get_fn_ext
from simba.utils.errors import InvalidVideoFileError
import os
from simba.enums import Paths

def get_video_meta_data(video_path: str):
    """
    Helper to read video metadata (fps, resolution, frame cnt etc.) from video file.

    Parameters
    ----------
    video_path: str
        Path to video file.

    Returns
    -------
    vdata: dict
        Python dictionary holding video meta data

    Notes
    ----------

    Examples
    >>> get_video_meta_data('/Users/simon/Desktop/envs/simba_dev/tests/test_data/video_tests/Together_1.avi')
    {'video_name': 'Together_1', 'fps': 30, 'width': 400, 'height': 600, 'frame_count': 300, 'resolution_str': '400 x 600', 'video_length_s': 10}

    """
    d = os.path.join(os.path.dirname(os.path.abspath("__file__")), 'tests/test_data/video_tests/Together_1.avi')
    video_data = {}
    cap = cv2.VideoCapture(video_path)
    _, video_data['video_name'], _ = get_fn_ext(video_path)
    video_data['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    video_data['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_data['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_data['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k, v in video_data.items():
        if v == 0:
            raise InvalidVideoFileError(msg=f'SIMBA WARNING: Video {video_data["video_name"]} has {k} of {str(v)}.')
    video_data['resolution_str'] = str(f'{video_data["width"]} x {video_data["height"]}')
    video_data['video_length_s'] = int(video_data['frame_count'] / video_data['fps'])
    return video_data

# = get_video_meta_data('tests/test_data/video_tests/Together_1.avi')