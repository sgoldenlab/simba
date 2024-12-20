from skimage.metrics import structural_similarity
import numpy as np
import cv2
from simba.utils.checks import check_if_valid_img, check_valid_lst, check_int
from typing import List, Optional
from simba.mixins.image_mixin import ImageMixin

def structural_similarity_index(img_1: np.ndarray, img_2: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSI) between two images.

    The function evaluates the SSI between two input images `img_1` and `img_2`. If the images have different numbers
    of channels, they are converted to greyscale before computing the SSI. If the images are multi-channel (e.g., RGB),
    the SSI is computed for each channel.

    :param np.ndarray img_1: The first input image represented as a NumPy array.
    :param np.ndarray img_2: The second input image represented as a NumPy array.
    :return float: The SSI value representing the similarity between the two images.
    """
    check_if_valid_img(data=img_1, source=f'{structural_similarity_index.__name__} img_1')
    check_if_valid_img(data=img_2, source=f'{structural_similarity_index.__name__} img_2')
    multichannel = False
    if img_1.ndim != img_2.ndim:
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    if img_1.ndim > 2: multichannel = True
    return abs(structural_similarity(im1=img_1.astype(np.uint8), im2=img_2.astype(np.uint8), multichannel=multichannel))

def img_to_greyscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a single color image to greyscale.

    The function takes an RGB image and converts it to a greyscale image using a weighted sum approach.
    If the input image is already in greyscale (2D array), it is returned as is.

    :param np.ndarray img: Input image represented as a NumPy array. For a color image, the array should have three channels (RGB).
    :return np.ndarray: The greyscale image as a 2D NumPy array.
    """
    check_if_valid_img(data=img, source=img_to_greyscale.__name__)
    if len(img.shape) != 2:
        return (0.07 * img[:, :, 2] + 0.72 * img[:, :, 1] + 0.21 * img[:, :, 0])
    else:
        return img

def sliding_structural_similarity_index(imgs: List[np.ndarray],
                                        stride: Optional[int] = 1,
                                        verbose: Optional[bool] = False) -> np.ndarray:

    """
    Computes the Structural Similarity Index (SSI) between consecutive images in an array with a specified stride.

    The function evaluates the SSI between pairs of images in the input array `imgs` using a sliding window approach
    with the specified `stride`. The SSI is computed for each pair of images and the results are stored in an output
    array. If the images are multi-channel (e.g., RGB), the SSI is computed for each channel.

    High SSI values (close to 1) indicate high similarity between images, while low SSI values (close to 0 or negative)
    indicate low similarity.

    :param np.ndarray imgs: A list of images. Each element in the list is expected to be a numpy array representing an image.
    :param Optional[int] stride: The number of images to skip between comparisons. Default is 1.
    :param Optional[bool] verbose: If True, prints progress messages. Default is False.
    :return np.ndarray: A numpy array containing the SSI values for each pair of images.

    :example:
    >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
    >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
    >>> imgs = list(imgs.values())
    >>> results = sliding_structural_similarity_index(imgs=imgs, stride=1, verbose=True)
    """

    check_valid_lst(data=imgs, valid_dtypes=(np.ndarray,), min_len=2)
    check_int(name=f'{sliding_structural_similarity_index.__name__} stride', min_value=1, max_value=len(imgs), value=stride)
    ndims, multichannel = set(), False
    for i in imgs:
        check_if_valid_img(data=i, source=sliding_structural_similarity_index.__name__)
        ndims.add(i.ndim)
    if len(list(ndims)) > 1:
        imgs = ImageMixin.img_stack_to_greyscale(imgs=imgs)
    if imgs[0].ndim > 2: multichannel = True
    results = np.zeros((len(imgs)), np.float32)
    for cnt, i in enumerate(range(stride, len(imgs))):
        img_1, img_2 = imgs[i-stride], imgs[i]
        results[i] = structural_similarity(im1=img_1, im2=img_2, multichannel=multichannel)
        if verbose:
            print(f'SSI computed ({cnt+1}/{len(imgs)-stride})')
    return results


def structural_similarity_matrix(imgs: List[np.array], verbose: Optional[bool] = False) -> np.array:
    """
    Computes a matrix of Structural Similarity Index (SSI) values for a list of images.

    This function takes a list of images and computes the SSI between each pair of images and produce a symmetric matrix.

    :param List[np.array] imgs: A list of images represented as numpy arrays. If not all images are greyscale or color, they are converted and processed as greyscale.
    :param Optional[bool] verbose: If True, prints progress messages showing which SSI values have been computed.  Default is False.
    :return np.array: A square numpy array where the element at [i, j] represents the SSI between imgs[i] and imgs[j].

    :example:
    >>> imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
    >>> imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
    >>> imgs = list(imgs.values())[0:10]
    >>> results = sliding_structural_similarity_matrix(imgs=imgs)
    """

    check_valid_lst(data=imgs, valid_dtypes=(np.ndarray,), min_len=2)
    ndims, multichannel = set(), False
    for i in imgs:
        check_if_valid_img(data=i, source=sliding_structural_similarity_index.__name__)
        ndims.add(i.ndim)
    if len(list(ndims)) > 1:
        imgs = ImageMixin.img_stack_to_greyscale(imgs=imgs)
    if imgs[0].ndim > 2: multichannel = True
    results = np.ones((len(imgs), len(imgs)), np.float32)
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            if verbose:
                print(f'SSI matrix position ({i}, {j}) complete...')
            val = structural_similarity(im1=imgs[i], im2=imgs[j], multichannel=multichannel)
            results[i, j] = val
            results[j, i] = val
    return results




#
# img_1 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/a.png').astype(np.float32)
# img_2 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/b.png').astype(np.float32)
# # ImageMixin.img_emd(img_1=img_1, img_2=img_2, lower_bound=0.5, verbose=True)

# imgs = ImageMixin.read_all_img_in_dir(dir='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/test')
# imgs = {k: imgs[k] for k in sorted(imgs, key=lambda x: int(x.split('.')[0]))}
# imgs = list(imgs.values())[0:10]
# results = sliding_structural_similarity_matrix(imgs=imgs)


# results = sliding_structural_similarity_index(imgs=imgs, stride=1, verbose=True)
#



#
# img_1 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/a.png', 0).astype(np.float32)
# img_2 = cv2.imread('/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/examples/e.png', 0).astype(np.float32)
# structural_similarity_index(img_1=img_1, img_2=img_2)
