__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional

import cupy as cp
import numpy as np

from simba.utils.checks import check_if_valid_img, check_instance
from simba.utils.read_write import read_img_batch_from_video_gpu


def img_stack_to_grayscale_cupy(imgs: np.ndarray,
                                batch_size: Optional[int] = 250) -> np.ndarray:
    """
    Converts a stack of color images to grayscale using GPU acceleration with CuPy.

    :param np.ndarray imgs: A 4D NumPy array representing a stack of images with shape (num_images, height, width, channels). The images are expected to have 3 channels (RGB).
    :param Optional[int] batch_size: The number of images to process in each batch. Defaults to 250. Adjust this parameter to fit your GPU's memory capacity.
    :return np.ndarray: m A 3D NumPy array of shape (num_images, height, width) containing the grayscale images. If the input array is not 4D, the function returns the input as is.

    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_IOT_1_cropped.mp4", verbose=False, start_frm=0, end_frm=i)
    >>> imgs = np.stack(list(imgs.values()), axis=0).astype(np.uint8)
    >>> gray_imgs = img_stack_to_grayscale_cupy(imgs=imgs)
    """


    check_instance(source=img_stack_to_grayscale_cupy.__name__, instance=imgs, accepted_types=(np.ndarray,))
    check_if_valid_img(data=imgs[0], source=img_stack_to_grayscale_cupy.__name__)
    if imgs.ndim != 4:
        return imgs
    results = cp.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
    n = int(np.ceil((imgs.shape[0] / batch_size)))
    imgs = np.array_split(imgs, n)
    start = 0
    for i in range(len(imgs)):
        img_batch = cp.array(imgs[i])
        batch_cnt = img_batch.shape[0]
        end = start + batch_cnt
        vals = (0.07 * img_batch[:, :, :, 2] + 0.72 * img_batch[:, :, :, 1] + 0.21 * img_batch[:, :, :, 0])
        results[start:end] = vals.astype(cp.uint8)
        start = end
    return results.get()