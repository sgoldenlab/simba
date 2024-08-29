import time
from copy import deepcopy
from typing import Optional, Union

import numpy as np

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from numba import cuda

from simba.utils.checks import check_if_valid_img, check_instance
from simba.utils.read_write import read_img_batch_from_video_gpu

PHOTOMETRIC = 'photometric'
DIGITAL = 'digital'

@cuda.jit()
def _photometric(data, results):
    y, x, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > results.shape[0] - 1 or x > results.shape[1]  - 1 or y > results.shape[2] - 1:
        return
    else:
        r, g, b = data[i][x][y][0], data[i][x][y][1], data[i][x][y][2]
        results[i][x][y] = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)

@cuda.jit()
def _digital(data, results):
    y, x, i = cuda.grid(3)
    if i < 0 or x < 0 or y < 0:
        return
    if i > results.shape[0] - 1 or x > results.shape[1]  - 1 or y > results.shape[2] - 1:
        return
    else:
        r, g, b = data[i][x][y][0], data[i][x][y][1], data[i][x][y][2]
        results[i][x][y] = (0.299 * r) + (0.587 * g) + (0.114 * b)

def img_stack_brightness(x: np.ndarray,
                         method: Optional[Literal['photometric', 'digital']] = 'digital',
                         ignore_black: Optional[bool] = True) -> np.ndarray:
    """
    Calculate the average brightness of a stack of images using a specified method.


    - **Photometric Method**: The brightness is calculated using the formula:

    .. math::
       \text{brightness} = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B

    - **Digital Method**: The brightness is calculated using the formula:

    .. math::
       \text{brightness} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B


    :param np.ndarray x: A 4D array of images with dimensions (N, H, W, C), where N is the number of images, H and W are the height and width, and C is the number of channels (RGB).
    :param Optional[Literal['photometric', 'digital']] method:  The method to use for calculating brightness. It can be 'photometric' for  the standard luminance calculation or 'digital' for an alternative set of coefficients. Default is 'digital'.
    :param Optional[bool] ignore_black: If True, black pixels (i.e., pixels with brightness value 0) will be ignored in the calculation of the average brightness. Default is True.
    :return np.ndarray: A 1D array of average brightness values for each image in the stack. If `ignore_black` is True, black pixels are ignored in the averaging process.


    :example:
    >>> imgs = read_img_batch_from_video_gpu(video_path=r"/mnt/c/troubleshooting/RAT_NOR/project_folder/videos/2022-06-20_NOB_DOT_4_downsampled.mp4", start_frm=0, end_frm=5000)
    >>> imgs = np.stack(list(imgs.values()), axis=0)
    >>> x = img_stack_brightness(x=imgs)
    """

    check_instance(source=img_stack_brightness.__name__, instance=x, accepted_types=(np.ndarray,))
    check_if_valid_img(data=x[0], source=img_stack_brightness.__name__)
    x = np.ascontiguousarray(x).astype(np.uint8)
    if x.ndim == 4:
        grid_x = (x.shape[1] + 16 - 1) // 16
        grid_y = (x.shape[2] + 16 - 1) // 16
        grid_z = x.shape[0]
        threads_per_block = (16, 16, 1)
        blocks_per_grid = (grid_y, grid_x, grid_z)
        x_dev = cuda.to_device(x)
        results = cuda.device_array((x.shape[0], x.shape[1], x.shape[2]), dtype=np.uint8)
        if method == PHOTOMETRIC:
            _photometric[blocks_per_grid, threads_per_block](x_dev, results)
        else:
            _digital[blocks_per_grid, threads_per_block](x_dev, results)
        results = results.copy_to_host()
        if ignore_black:
            masked_array = np.ma.masked_equal(results, 0)
            results = np.mean(masked_array, axis=(1, 2)).filled(0)
    else:
        results = deepcopy(x)
        results = np.mean(results, axis=(1, 2))

    return results