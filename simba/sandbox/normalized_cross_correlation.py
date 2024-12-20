from skimage.metrics import structural_similarity
import numpy as np
import cv2
from simba.utils.checks import check_if_valid_img, check_valid_lst, check_int
from typing import List, Optional
from simba.mixins.image_mixin import ImageMixin
from numba import jit, njit, prange


@njit(["(uint8[:, :], uint8[:, :])",
        "(uint8[:, :, :], uint8[:, :, :])"])
def cross_correlation_similarity(img_1: np.ndarray, img_2: np.ndarray) -> float:
    """
    Computes the Normalized Cross-Correlation (NCC) similarity between two images.

    The NCC measures the similarity between two images by calculating the correlation
    coefficient of their pixel values. The output value ranges from -1 to 1, where 1 indicates perfect positive correlation, 0 indicates no correlation, and -1 indicates perfect negative correlation.

    :param np.ndarray img_1: The first input image. It can be a 2D grayscale image or a 3D color image.
    :param np.ndarray img_2:  The second input image. It must have the same dimensions as img_1.
    :return float: The NCC value representing the similarity between the two images. Returns 0.0 if the denominator is zero, indicating no similarity.

    :example:
    >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/a.png').astype(np.uint8)
    >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/f.png').astype(np.uint8)
    >>> cross_correlation_similarity(img_1=img_1, img_2=img_2)
    """

    img1_flat = img_1.flatten()
    img2_flat = img_2.flatten()
    mean_1, mean_2 = np.mean(img1_flat), np.mean(img2_flat)
    N = np.sum((img1_flat - mean_1) * (img2_flat - mean_2))
    D = np.sqrt(np.sum((img1_flat - mean_1) ** 2) * np.sum((img2_flat - mean_2) ** 2))
    if D == 0:
        return 0.0
    else:
        return N / D

@njit(["(uint8[:, :, :], int64)",
        "(uint8[:, :, :, :], int64)"])
def sliding_cross_correlation_similarity(imgs: np.ndarray,
                                         stride: int) -> np.ndarray:
    """
    Computes the Normalized Cross-Correlation (NCC) similarity for a sequence of images using a sliding window approach.

    This function calculates the NCC between each image and the image that is `stride` positions before it in the sequence. The result is an array of NCC values representing
    the similarity between successive images.

    .. seealso::
       ``simba.mixins.image_mixin.ImageMixin.cross_correlation_similarity``
       ``simba.mixins.image_mixin.ImageMixin.cross_correlation_matrix``

    :param np.ndarray imgs: A 3D array (for grayscale images) or a 4D array (for color images) containing the sequence of images.  Each image should have the same size.
    :param int stride: The stride length for comparing images. Determines how many steps back in the sequence each image is compared to.
    :return np.ndarray: A 1D array of NCC values representing the similarity between each image and the image `stride` positions before it. The length of the array is the same as the number of images.

    :example:
    >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat11_12_frames')
    >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
    >>> imgs = np.stack(list(imgs.values()))
    >>> results = sliding_cross_correlation_similarity(imgs=imgs, stride=1)
    """
    results = np.ones((imgs.shape[0]), dtype=np.float32)
    for i in prange(stride, imgs.shape[0]):
        img1_flat, img2_flat = imgs[i-stride].flatten(), imgs[i].flatten()
        mean_1, mean_2 = np.mean(img1_flat), np.mean(img2_flat)
        N = np.sum((img1_flat - mean_1) * (img2_flat - mean_2))
        D = np.sqrt(np.sum((img1_flat - mean_1) ** 2) * np.sum((img2_flat - mean_2) ** 2))
        if D == 0:
            results[i] = 0.0
        else:
            results[i] = N / D
    return results

@njit(["(uint8[:, :, :],)",
        "(uint8[:, :, :, :],)"])
def cross_correlation_matrix(imgs: np.array) -> np.array:
    """
    Computes the cross-correlation matrix for a given array of images.

    This function calculates the cross-correlation coefficient between each pair of images in the input array.
    The cross-correlation coefficient is a measure of similarity between two images, with values ranging from
    -1 (completely dissimilar) to 1 (identical).

    The function uses the `numba` library for Just-In-Time (JIT) compilation to optimize performance, and
    `prange` for parallel execution over the image pairs.

    .. seealso::
       ``simba.mixins.image_mixin.ImageMixin.cross_correlation_similarity``
       ``simba.mixins.image_mixin.ImageMixin.sliding_cross_correlation_similarity``

    .. note::
       Use greyscale images for faster runtime. Ideally should be move dto GPU.


    :param np.array imgs: A 3D (or 4D) numpy array of images where the first dimension indexes the images,
                          and the remaining dimensions are the image dimensions (height, width, [channels]).
                          - For grayscale images: shape should be (n_images, height, width)
                          - For color images: shape should be (n_images, height, width, channels)

    :return np.array: A 2D numpy array representing the cross-correlation matrix, where the element at [i, j]
                      contains the cross-correlation coefficient between the i-th and j-th images.

    :example:
    >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
    >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat11_12_frames')
    >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
    >>> imgs = np.stack(list(imgs.values()))
    >>> imgs = ImageMixin.img_stack_to_greyscale(imgs=imgs)
    >>> results = cross_correlation_matrix(imgs=imgs)
    """

    results = np.ones((imgs.shape[0], imgs.shape[0]), dtype=np.float32)
    for i in prange(imgs.shape[0]):
        img1_flat = imgs[i].flatten()
        mean_1 = np.mean(img1_flat)
        for j in range(i + 1, imgs.shape[0]):
            img2_flat = imgs[j].flatten()
            mean_2 = np.mean(img2_flat)
            N = np.sum((img1_flat - mean_1) * (img2_flat - mean_2))
            D = np.sqrt(np.sum((img1_flat - mean_1) ** 2) * np.sum((img2_flat - mean_2) ** 2))
            if D == 0: val = 0.0
            else: val = N / D
            results[i, j] = val
            results[j, i] = val
    return results








#
imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
# imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat11_12_frames')
imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
#imgs = list(imgs.values())[0:10]
imgs = np.stack(list(imgs.values()))
imgs = ImageMixin.img_stack_to_greyscale(imgs=imgs)
results = cross_correlation_matrix(imgs=imgs)

import time
start = time.time()
results = cross_correlation_matrix(imgs=imgs)
print(time.time() - start)


# import time
# start = time.time()
# results = sliding_cross_correlation_similarity(imgs=imgs, stride=1)
# print(time.time() - start)


# imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/08102021_DOT_Rat11_12_frames')
# imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
# imgs = np.stack(list(imgs.values()))
# results = sliding_cross_correlation_similarity(imgs=imgs, stride=1)
#



# img_1 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/a.png').astype(np.uint8)
# img_2 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/f.png').astype(np.uint8)
# normalized_cross_correlation(img_1=img_1, img_2=img_2)
# img_1 = ImageMixin.img_to_greyscale(img=img_1)
# img_2 = ImageMixin.img_to_greyscale(img=img_2)


# ImageMixin.img_emd(img_1=img_1, img_2=img_2, lower_bound=0.5, verbose=True)







#results = sliding_structural_similarity_matrix(imgs=imgs)






