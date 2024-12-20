from typing import Optional

import numpy as np
from numba import cuda

from simba.utils.checks import (check_if_valid_img, check_instance,
                                check_valid_array)


@cuda.jit()
def _grey_mse(data, ref_img, stride, batch_cnt, mse_arr):
    y, x, i = cuda.grid(3)
    stride = stride[0]
    batch_cnt = batch_cnt[0]
    if batch_cnt == 0:
        if (i - stride) < 0 or x < 0 or y < 0:
            return
        else:
            if i < 0 or x < 0 or y < 0:
                return
    if i > mse_arr.shape[0] - 1 or x > mse_arr.shape[1]  - 1 or y > mse_arr.shape[2] - 1:
        return
    else:
        img_val = data[i][x][y]
        if i == 0:
            prev_val = ref_img[x][y]
        else:
            img_val = data[i][x][y]
            prev_val = data[i - stride][x][y]
        mse_arr[i][x][y] = (img_val - prev_val) ** 2


@cuda.jit()
def _rgb_mse(data, ref_img, stride, batch_cnt, mse_arr):
    y, x, i = cuda.grid(3)
    stride = stride[0]
    batch_cnt = batch_cnt[0]
    if batch_cnt == 0:
        if (i - stride) < 0 or x < 0 or y < 0:
            return
        else:
            if i < 0 or x < 0 or y < 0:
                return
    if i > mse_arr.shape[0] - 1 or x > mse_arr.shape[1]  - 1 or y > mse_arr.shape[2] - 1:
        return
    else:
        img_val = data[i][x][y]
        if i != 0:
            prev_val = data[i - stride][x][y]
        else:
            prev_val = ref_img[x][y]
        r_diff = (img_val[0] - prev_val[0]) ** 2
        g_diff = (img_val[1] - prev_val[1]) ** 2
        b_diff = (img_val[2] - prev_val[2]) ** 2
        mse_arr[i][x][y] = r_diff + g_diff + b_diff

def stack_sliding_mse(x: np.ndarray,
                      stride: Optional[int] = 1,
                      batch_size: Optional[int] = 1000) -> np.ndarray:
    """
    Computes the Mean Squared Error (MSE) between each image in a stack and a reference image,
    where the reference image is determined by a sliding window approach with a specified stride.
    The function is optimized for large image stacks by processing them in batches.

    :param np.ndarray x: Input array of images, where the first dimension corresponds to the stack of images. The array should be either 3D (height, width, channels) or 4D (batch, height, width, channels).
    :param Optional[int] stride: The stride or step size for the sliding window that determines the reference image. Defaults to 1, meaning the previous image in the stack is used as the reference.
    :param Optional[int] batch_size: The number of images to process in a single batch. Larger batch sizes may improve performance but require more GPU memory.  Defaults to 1000.
    :return: A 1D NumPy array containing the MSE for each image in the stack compared to its corresponding reference image. The length of the array is equal to the number of images in the input stack.
    :rtype: np.ndarray

    """

    check_instance(source=stack_sliding_mse.__name__, instance=x, accepted_types=(np.ndarray,))
    check_if_valid_img(data=x[0], source=stack_sliding_mse.__name__)
    check_valid_array(data=x, source=stack_sliding_mse.__name__, accepted_ndims=[3, 4])
    stride = np.array([stride], dtype=np.int32)
    stride_dev = cuda.to_device(stride)
    out = np.full((x.shape[0]), fill_value=0.0, dtype=np.float32)
    for batch_cnt, l in enumerate(range(0, x.shape[0], batch_size)):
        r = l + batch_size
        batch_x = x[l:r]
        if batch_cnt != 0:
            if x.ndim == 3:
                ref_img = x[l-stride].astype(np.uint8).reshape(x.shape[1], x.shape[2])
            else:
                ref_img = x[l-stride].astype(np.uint8).reshape(x.shape[1], x.shape[2], 3)
        else:
            ref_img = np.full_like(x[l], dtype=np.uint8, fill_value=0)
        ref_img = ref_img.astype(np.uint8)
        grid_x = (batch_x.shape[1] + 16 - 1) // 16
        grid_y = (batch_x.shape[2] + 16 - 1) // 16
        grid_z = batch_x.shape[0]
        threads_per_block = (16, 16, 1)
        blocks_per_grid = (grid_y, grid_x, grid_z)
        ref_img_dev = cuda.to_device(ref_img)
        x_dev = cuda.to_device(batch_x)
        results = cuda.device_array((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]), dtype=np.uint8)
        batch_cnt_dev = np.array([batch_cnt], dtype=np.int32)
        if x.ndim == 3:
            _grey_mse[blocks_per_grid, threads_per_block](x_dev, ref_img_dev, stride_dev, batch_cnt_dev,  results)
        else:
            _rgb_mse[blocks_per_grid, threads_per_block](x_dev, ref_img_dev, stride_dev, batch_cnt_dev,  results)
        results = results.copy_to_host()
        results = np.mean(results, axis=(1, 2))
        out[l:r] = results
    return out