__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import numpy as np
from numba import cuda

from simba.utils.checks import check_if_valid_img, check_instance
from simba.utils.read_write import read_img_batch_from_video_gpu


@cuda.jit()
def _img_stack_to_grayscale(data, results):
    y, x, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > results.shape[0] - 1 or x > results.shape[1] - 1 or y > results.shape[2] - 1:
        return
    else:
        b = 0.07 * data[i][x][y][2]
        g = 0.72 * data[i][x][y][1]
        r = 0.21 * data[i][x][y][0]
        val = b + g + r
        results[i][x][y] = val

def img_stack_to_grayscale_cuda(x: np.ndarray) -> np.ndarray:
    """
    Convert image stack to grayscale using CUDA.

    :param np.ndarray x: 4d array of color images in numpy format.
    :return np.ndarray: 3D array of greyscaled images.

    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"/mnt/c/troubleshooting/mitra/project_folder/videos/temp_2/592_MA147_Gq_Saline_0516_downsampled.mp4", verbose=False, start_frm=0, end_frm=i)
    >>> imgs = np.stack(list(imgs.values()), axis=0).astype(np.uint8)
    >>> grey_images = img_stack_to_grayscale_cuda(x=imgs)
    """
    check_instance(source=img_stack_to_grayscale_cuda.__name__, instance=imgs, accepted_types=(np.ndarray,))
    check_if_valid_img(data=x[0], source=img_stack_to_grayscale_cuda.__name__)
    if x.ndim != 4:
        return x
    x = np.ascontiguousarray(x).astype(np.uint8)
    x_dev = cuda.to_device(x)
    results = cuda.device_array((x.shape[0], x.shape[1], x.shape[2]), dtype=np.uint8)
    grid_x = (x.shape[1] + 16 - 1) // 16
    grid_y = (x.shape[2] + 16 - 1) // 16
    grid_z = x.shape[0]
    threads_per_block = (16, 16, 1)
    blocks_per_grid = (grid_y, grid_x, grid_z)
    _img_stack_to_grayscale[blocks_per_grid, threads_per_block](x_dev, results)
    results = results.copy_to_host()
    return results


