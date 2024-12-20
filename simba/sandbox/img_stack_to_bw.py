import time

import numpy as np
import cv2
from simba.mixins.image_mixin import ImageMixin
from numba import jit


@jit(nopython=True)
def img_stack_to_bw(imgs: np.ndarray,
                    lower_thresh: int,
                    upper_thresh: int,
                    invert: bool):
    """
    Convert a stack of color images into black and white format.

    .. note::
       If converting a single image, consider ``simba.mixins.imgage_mixin.ImageMixin.img_to_bw()``

    :param np.ndarray img: 4-dimensional array of color images.
    :param Optional[int] lower_thresh: Lower threshold value for binary conversion. Pixels below this value become black. Default is 20.
    :param Optional[int] upper_thresh: Upper threshold value for binary conversion. Pixels above this value become white. Default is 250.
    :param Optional[bool] invert: Flag indicating whether to invert the binary image (black becomes white and vice versa). Default is True.
    :return np.ndarray: 4-dimensional array with black and white image.

    :example:
    >>> imgs = ImageMixin.read_img_batch_from_video(video_path='/Users/simon/Downloads/3A_Mouse_5-choice_MouseTouchBasic_a1.mp4', start_frm=0, end_frm=100)
    >>> imgs = np.stack(imgs.values(), axis=0)
    >>> bw_imgs = img_stack_to_bw(imgs=imgs, upper_thresh=255, lower_thresh=20, invert=False)
    """

    results = np.full((imgs.shape[:3]), np.nan)
    for cnt in range(imgs.shape[0]):
        arr =  imgs[cnt]
        m, n, _ = arr.shape
        img_mean = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                total = 0.0
                for k in range(arr.shape[2]):
                    total += arr[i, j, k]
                img_mean[i, j] = total / arr.shape[2]
        img = np.where(img_mean < lower_thresh, 0, img_mean)
        img = np.where(img > upper_thresh, 1, img)
        if invert:
            img = 1 - img
        results[cnt] = img
    return results

# imgs = ImageMixin.read_img_batch_from_video(video_path='/Users/simon/Downloads/3A_Mouse_5-choice_MouseTouchBasic_a1.mp4', start_frm=0, end_frm=100)
# imgs = np.stack(imgs.values(), axis=0)
# bw_imgs = img_stack_to_bw(imgs=imgs, upper_thresh=255, lower_thresh=20, invert=False)



#mgs[0]

# img = cv2.imread('/Users/simon/Desktop/test.png')


cv2.imshow('sdsdf', bw_imgs[0])
cv2.waitKey(5000)