from typing import List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import functools
import multiprocessing
import os

import cv2

from simba.utils.checks import (check_if_dir_exists, check_int, check_str,
                                check_valid_boolean)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt, get_video_meta_data,
                                    read_frm_of_video)

JPEG, PNG, WEBP = 'jpeg', 'png', 'webp'

def _video_to_frms_helper(img_batch: Tuple[int, List[int]],
                          verbose: bool,
                          img_format: str,
                          quality: int,
                          include_video_name_in_filename: bool,
                          greyscale: bool,
                          black_and_white: bool,
                          clahe: bool,
                          video_path: str,
                          save_dir: str):

    batch_cnt, frm_idxs = img_batch
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    for frm_idx in frm_idxs:
        if include_video_name_in_filename:
            save_path = os.path.join(save_dir, f'{video_meta_data["video_name"]}_{frm_idx}.{img_format}')
        else:
            save_path = os.path.join(save_dir, f'{frm_idx}.{img_format}')
        if verbose:
            print(f"Saving image {save_path} ...")
        img = read_frm_of_video(video_path=cap, frame_index=frm_idx, greyscale=greyscale, clahe=clahe, black_and_white=black_and_white)
        if img_format == WEBP:
            cv2.imwrite(save_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
        elif img_format == JPEG:
            cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return batch_cnt

def video_to_frames(video_path: Union[str, os.PathLike],
                    save_dir: Union[str, os.PathLike],
                    quality: Optional[int] = 95,
                    img_format: Literal['png', 'webp'] = 'png',
                    verbose: bool = True,
                    clahe: bool = False,
                    greyscale: bool = False,
                    core_cnt: Optional[int] = -1,
                    black_and_white: bool = False,
                    include_video_name_in_filename: bool = True):


    """
    Extract all frames from a video file and save them as individual image files.

    .. note::
       Uses multiprocessing for faster frame extraction. Frames are saved with sequential numbering (0, 1, 2, ...).

    :param Union[str, os.PathLike] video_path: Path to the video file to extract frames from.
    :param Union[str, os.PathLike] save_dir: Directory where extracted frames will be saved.
    :param Optional[int] quality: Image quality for JPEG format (1-100). Higher values = better quality + larger file size. Default: 95.
    :param Literal['png', 'webp'] img_format: Format of the output images. One of: 'png', 'webp'. Default: 'png'.
    :param bool verbose: If True, prints progress messages during extraction. Default: True.
    :param bool clahe: If True, applies Contrast Limited Adaptive Histogram Equalization to each frame. Default: False.
    :param bool greyscale: If True, converts frames to greyscale. Default: False.
    :param Optional[int] core_cnt: Number of CPU cores to use for multiprocessing. -1 uses all available cores. Default: -1.
    :param bool black_and_white: If True, converts frames to black and white. Default: False.
    :param bool include_video_name_in_filename: If True, includes video name in frame filenames. Default: True.
    :return: None. Frames are saved to disk in the specified directory.

    :example:
    >>> video_to_frames(video_path=r"C:\troubleshooting\SDS_pre_post\project_folder\videos\SDI100 x ALR2 post_d7.mp4", 
    ...                save_dir=r'C:\troubleshooting\SDS_pre_post\project_folder\videos\test', 
    ...                black_and_white=False, 
    ...                verbose=True, 
    ...                img_format='webp', 
    ...                clahe=True)
    """

    timer = SimbaTimer(start=True)
    video_meta_data = get_video_meta_data(video_path=video_path)
    check_if_dir_exists(in_dir=save_dir, source=video_to_frames.__name__, raise_error=True)
    check_valid_boolean(value=verbose, source=f'{video_to_frames.__name__} verbose')
    check_valid_boolean(value=clahe, source=f'{video_to_frames.__name__} clahe')
    check_valid_boolean(value=greyscale, source=f'{video_to_frames.__name__} greyscale')
    check_valid_boolean(value=black_and_white, source=f'{video_to_frames.__name__} black_and_white')
    check_valid_boolean(value=include_video_name_in_filename, source=f'{video_to_frames.__name__} include_video_name_in_filename')
    check_int(name=f'{video_to_frames.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0], raise_error=True)
    check_int(name=f'{video_to_frames.__name__} quality', value=quality, min_value=1, max_value=100, raise_error=True)
    core_cnt = find_core_cnt()[0] if core_cnt -1 or core_cnt > find_core_cnt()[0] else core_cnt
    check_str(name=f'{video_to_frames.__name__} img_format', value=img_format, options=('jpeg', 'png', 'webp'))
    frm_ids = list(range(0, video_meta_data['frame_count']))
    frm_ids = [frm_ids[i * len(frm_ids) // core_cnt: (i + 1) * len(frm_ids) // core_cnt] for i in range(core_cnt)]
    frm_ids = [(i, j) for i, j in enumerate(frm_ids)]
    with multiprocessing.Pool(core_cnt, maxtasksperchild=100) as pool:
        constants = functools.partial(_video_to_frms_helper,
                                          verbose=verbose,
                                          img_format=img_format,
                                          quality=quality,
                                          greyscale=greyscale,
                                          black_and_white=black_and_white,
                                          include_video_name_in_filename=include_video_name_in_filename,
                                          video_path=video_path,
                                          clahe=clahe,
                                          save_dir=save_dir)
        for cnt, batch_id in enumerate(pool.imap(constants, frm_ids, chunksize=1)):
            if verbose:
                print(f'Video frame batch {batch_id} (of {core_cnt}) complete...')
    pool.join()
    pool.terminate()
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'All frames for video {video_path} saved in {save_dir}', elapsed_time=timer.elapsed_time_str)

# if __name__ == "__main__":
#     video_to_frames(video_path=r"C:\troubleshooting\SDS_pre_post\project_folder\videos\SDI100 x ALR2 post_d7.mp4",
#                     save_dir=r'C:\troubleshooting\SDS_pre_post\project_folder\videos\test',
#                     black_and_white=False,
#                     verbose=True,
#                     img_format='webp',
#                     clahe=True)