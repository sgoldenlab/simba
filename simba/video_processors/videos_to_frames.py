from typing import Union, Optional, Dict
try:
    from typing import Literal
except:
    from typing_extensions import Literal
import os
import cv2

import numpy as np
from PIL import Image
from simba.utils.read_write import get_video_meta_data, read_img_batch_from_video_gpu, read_img_batch_from_video, find_core_cnt
from simba.utils.checks import check_if_dir_exists, check_valid_boolean, check_int, check_str, check_nvidea_gpu_available
from simba.utils.errors import SimBAGPUError
from simba.utils.printing import SimbaTimer, stdout_success


import multiprocessing
import functools
JPEG, PNG = 'jpeg', 'png'

def _video_to_frms_helper(img_batch: Dict[int, np.ndarray],
                          save_dir: str,
                          verbose: bool,
                          img_format: str,
                          quality: int):

    for img_id, img in img_batch.items():
        save_path = os.path.join(save_dir, f'{img_id}.{img_format}')
        if verbose:
            print(f"Saving image {img_id} ...")
        if img_format in [JPEG, PNG]:
            if img.ndim == 2:
                img = Image.fromarray(img, mode='L')
            elif img.ndim == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]
                img = Image.fromarray(img, mode='RGB')
            else:
                img = Image.fromarray(img, mode='RGBA')
            if img.mode in ('RGBA', 'LA'): img = img.convert('RGB')

            if img_format == JPEG:
                img.save(save_path, format='JPEG', quality=quality)
            else:
                img.save(save_path, 'PNG')
        else:
            cv2.imwrite(save_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])




def video_to_frames(video_path: Union[str, os.PathLike],
                    save_dir: Union[str, os.PathLike],
                    gpu: bool = False,
                    batch_size: Optional[int] = None,
                    core_cnt: int = -1,
                    quality: Optional[int] = 95,
                    img_format: Literal['jpeg', 'png', 'webp'] = 'png',
                    verbose: bool = True):

    timer = SimbaTimer(start=True)
    video_meta_data = get_video_meta_data(video_path=video_path)
    check_if_dir_exists(in_dir=save_dir, source=video_to_frames.__name__, raise_error=True)
    check_valid_boolean(value=gpu, source=f'{video_to_frames.__name__} gpu')
    check_valid_boolean(value=verbose, source=f'{video_to_frames.__name__} verbose')
    check_int(name=f'{video_to_frames.__name__} core_cnt', min_value=-1, max_value=find_core_cnt()[0], unaccepted_vals=[0], value=core_cnt)
    core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
    if gpu and not check_nvidea_gpu_available():
        raise SimBAGPUError(msg='No GPU detected but gpu set to True.', source=video_to_frames.__name__)
    if batch_size is not None:
        check_int(name=f'{video_to_frames.__name__} batch_size', min_value=1, max_value=video_meta_data['frame_count'], value=batch_size)
        batch_size = min(batch_size, video_meta_data['frame_count'])
    else:
        batch_size = video_meta_data['frame_count']
    check_str(name=f'{video_to_frames.__name__} img_format', value=img_format, options=('jpeg', 'png', 'webp'))
    if img_format != PNG:
        check_int(name=f'{video_to_frames.__name__} quality', min_value=1, max_value=100, value=quality)
    for start_idx in range(0, video_meta_data['frame_count'], batch_size):
        end_idx = min((start_idx + batch_size-1), video_meta_data['frame_count'])
        if gpu:
            img_batch = read_img_batch_from_video_gpu(video_path=video_path, start_frm=start_idx, end_frm=end_idx, verbose=verbose, out_format='dict')
        else:
            img_batch = read_img_batch_from_video(video_path=video_path, start_frm=start_idx, end_frm=end_idx, verbose=verbose, core_cnt=core_cnt)
        img_batch = [{k: img_batch[k] for k in list(img_batch.keys())[i::core_cnt]} for i in range(core_cnt)]
        if verbose:
            print(f'Saving frames {start_idx}-{end_idx+1}...')
        with multiprocessing.Pool(core_cnt, maxtasksperchild=100) as pool:
            constants = functools.partial(_video_to_frms_helper,
                                          save_dir=save_dir,
                                          verbose=verbose,
                                          quality=quality,
                                          img_format=img_format)
            for cnt, _ in enumerate(pool.imap(constants, img_batch, chunksize=1)):
                if verbose:
                    print(f'Video frames {start_idx}-{end_idx+1} (of {video_meta_data["frame_count"]}) complete...')
            pool.join()
            pool.terminate()
    timer.stop_timer()
    if verbose:
        stdout_success(msg=f'All frames for video {video_path} saved in {save_dir}', elapsed_time=timer.elapsed_time_str)
