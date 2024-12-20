import os
from typing import Union, Optional
from cupyx.scipy.ndimage import rotate
import cupy as cp
import numpy as np
from simba.utils.read_write import read_img_batch_from_video_gpu, get_video_meta_data, get_fn_ext
from simba.utils.checks import check_valid_array, check_int
from simba.mixins.image_mixin import ImageMixin
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
import cv2

def rotate_img_stack_cupy(imgs: np.ndarray,
                          rotation_degrees: Optional[float] = 180,
                          batch_size: Optional[int] = 500) -> np.ndarray:
    """
    Rotates a stack of images by a specified number of degrees using GPU acceleration with CuPy.

    Accepts a 3D (single-channel images) or 4D (multichannel images) NumPy array, rotates each image in the stack by the specified degree around the center, and returns the result as a NumPy array.

    :param np.ndarray imgs: The input stack of images to be rotated. Expected to be a NumPy array with 3 or 4 dimensions.  3D shape: (num_images, height, width) - 4D shape: (num_images, height, width, channels)
    :param Optional[float] rotation_degrees: The angle by which the images should be rotated, in degrees. Must be between 1 and 359 degrees. Defaults to 180 degrees.
    :param Optional[int] batch_size: Number of images to process on GPU in each batch. Decrease if data can't fit on GPU RAM.
    :returns: A NumPy array containing the rotated images with the same shape as the input.
    :rtype: np.ndarray

    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/F0_gq_Saline_0626_clipped.mp4"
    >>> imgs = read_img_batch_from_video_gpu(video_path=video_path)
    >>> imgs = np.stack(np.array(list(imgs.values())), axis=0)
    >>> imgs = rotate_img_stack_cupy(imgs=imgs, rotation=50)
    """

    check_valid_array(data=imgs, source=f'{rotate_img_stack_cupy.__name__} imgs', accepted_ndims=(3, 4))
    check_int(name=f'{rotate_img_stack_cupy.__name__} rotation', value=rotation_degrees, min_value=1, max_value=359)
    results = cp.full_like(imgs, fill_value=np.nan, dtype=np.uint8)
    for l in range(0, imgs.shape[0], batch_size):
        r = l + batch_size
        batch_imgs = cp.array(imgs[l:r])
        results[l:r] = rotate(input=batch_imgs, angle=rotation_degrees, axes=(2, 1), reshape=True)
    return results.get()


def rotate_video_cupy(video_path: Union[str, os.PathLike],
                      save_path: Optional[Union[str, os.PathLike]] = None,
                      rotation_degrees: Optional[float] = 180,
                      batch_cnt: Optional[int] = 1) -> None:
    """
    Rotates a video by a specified angle using GPU acceleration and CuPy for image processing.

    :param Union[str, os.PathLike] video_path: Path to the input video file.
    :param Optional[Union[str, os.PathLike]] save_path: Path to save the rotated video. If None, saves the video in the same directory as the input with '_rotated_<rotation_degrees>' appended to the filename.
    :param nptional[float] rotation_degrees:  Degrees to rotate the video. Must be between 1 and 359 degrees. Default is 180.
    :param Optional[int] batch_cnt: Number of batches to split the video frames into for processing. Higher values reduce memory usage. Default is 1.
    :returns: None.

    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/F0_gq_Saline_0626_clipped.mp4"
    >>> rotate_video_cupy(video_path=video_path, rotation_degrees=45)
    """

    timer = SimbaTimer(start=True)
    check_int(name=f'{rotate_img_stack_cupy.__name__} rotation', value=rotation_degrees, min_value=1, max_value=359)
    check_int(name=f'{rotate_img_stack_cupy.__name__} batch_cnt', value=batch_cnt, min_value=1)
    if save_path is None:
        video_dir, video_name, _ = get_fn_ext(filepath=video_path)
        save_path = os.path.join(video_dir, f'{video_name}_rotated_{rotation_degrees}.mp4')
    video_meta_data = get_video_meta_data(video_path=video_path)
    fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
    is_clr = ImageMixin.is_video_color(video=video_path)
    frm_ranges = np.arange(0, video_meta_data['frame_count'])
    frm_ranges = np.array_split(frm_ranges, batch_cnt)
    for frm_batch, frm_range in enumerate(frm_ranges):
        imgs = read_img_batch_from_video_gpu(video_path=video_path, start_frm=frm_range[0], end_frm=frm_range[-1])
        imgs = np.stack(np.array(list(imgs.values())), axis=0)
        imgs = rotate_img_stack_cupy(imgs=imgs, rotation_degrees=rotation_degrees)
        if frm_batch == 0:
            writer = cv2.VideoWriter(save_path, fourcc, video_meta_data['fps'], (imgs.shape[2], imgs.shape[1]), isColor=is_clr)
        for img in imgs: writer.write(img)
    writer.release()
    timer.stop_timer()
    stdout_success(f'Rotated video saved at {save_path}', source=rotate_video_cupy.__name__)

# video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/F0_gq_Saline_0626_clipped.mp4"
# rotate_video_cupy(video_path=video_path, rotation_degrees=45)

import time
its = 3
for i in [500, 1000, 2000, 4000, 8000, 16000, 32000]: #1000, 2000, 4000, 8000, 16000, 32000]
    imgs = np.random.randint(0, 255, (i, 320, 240))
    times = []
    for j in range(its):
        print(j)
        start_time = time.perf_counter()
        _ = rotate_img_stack_cupy(imgs=imgs)
        end_time = time.perf_counter() - start_time
        times.append(end_time)
    print(i, '\t'* 3, np.mean(times), '\t' * 3, np.std(times))





# imgs = read_img_batch_from_video_gpu(video_path=video_path)
# imgs = np.stack(np.array(list(imgs.values())), axis=0)
#
# imgs = rotate_img_stack_cupy(imgs=imgs, rotation=50)
#
# cv2. imshow('sasdasdsad', imgs[0].astype(np.uint8))
# cv2.waitKey(5000)
#









#
# def rotate_video_cupy(video_path: Union[str, os.PathLike], rotation: Optional[int] = 180):
