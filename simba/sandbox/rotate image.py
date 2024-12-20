import time

import numpy as np
from numba import njit, uint8, types, bool_
import cv2


@njit([(uint8[:,:,:], bool_)])
def rotate_img(img: np.ndarray, right: bool) -> np.ndarray:
    """
    Flip a color image 90 degrees to the left or right

    .. image:: _static/img/rotate_img.png
       :width: 600
       :align: center

    :param np.ndarray img: Input image as numpy array in uint8 format.
    :param bool right: If True, flips to the right. If False, flips to the left.
    :returns: The rotated image as a numpy array of uint8 format.

    :example:
    >>> img = cv2.imread('/Users/simon/Desktop/test.png')
    >>> rotated_img = rotate_img(img=img, right=False)
    """

    if right:
        img = np.transpose(img[:, ::-1, :], axes=(1, 0, 2))
    else:
        img = np.transpose(img[::-1, :, :], axes=(1, 0, 2))
    return np.ascontiguousarray(img).astype(np.uint8)


img = cv2.imread('/Users/simon/Desktop/test.png')
start = time.time()
for i in range(10000):
    rotated_img = rotate_img(img=img, right=True)
print(time.time() - start)
# cv2.imshow('sdsf', rotated_img)
# cv2.waitKey(5000)