import time

import numpy as np
import cv2
from simba.mixins.image_mixin import ImageMixin
from numba import jit, prange

@jit(nopython=True, parallel=True)
def segment_img_stack_horizontal(imgs: np.ndarray, pct: int, lower: bool, both: bool) -> np.ndarray:
    """
    Segment a horizontal part of all images in stack.

    :example:
    >>> imgs = ImageMixin.read_img_batch_from_video(video_path='/Users/simon/Downloads/3A_Mouse_5-choice_MouseTouchBasic_a1.mp4', start_frm=0, end_frm=400)
    >>> imgs = np.stack(imgs.values(), axis=0)
    >>> sliced_imgs = segment_img_stack_horizontal(imgs=imgs, pct=50, lower=True, both=False)
    """
    results = []
    for cnt in range(imgs.shape[0]):
        img = imgs[cnt]
        sliced_height = int(img.shape[0] * pct / 100)
        if both:
            sliced_img = img[sliced_height: img.shape[0] - sliced_height, :]
        elif lower:
            sliced_img = img[img.shape[0] - sliced_height:, :]
        else:
            sliced_img = img[:sliced_height, :]
        results.append(sliced_img)
    stacked_results = np.full((len(results), results[0].shape[0], results[0].shape[1], 3), np.nan)
    for i in prange(len(results)): stacked_results[i] = results[i]
    return results

imgs = ImageMixin.read_img_batch_from_video(video_path='/Users/simon/Downloads/3A_Mouse_5-choice_MouseTouchBasic_a1.mp4', start_frm=0, end_frm=400)
imgs = np.stack(imgs.values(), axis=0)
imgs_gray = ImageMixin.img_stack_to_greyscale(imgs=imgs)
cv2.imshow('ssss', imgs_gray[0])
cv2.waitKey(5000)


