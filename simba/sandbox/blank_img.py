import os
import cv2
from typing import Tuple, Union, Optional
from simba.utils.checks import check_if_valid_rgb_tuple, check_valid_tuple, check_if_dir_exists, check_int
import numpy as np

def create_uniform_img(size: Tuple[int, int],
                       color: Tuple[int, int, int],
                       save_path: Optional[Union[str, os.PathLike]] = None) -> Union[None, np.ndarray]:

    """
    Creates an image of specified size and color, and optionally saves it to a file.

    :param Tuple[int, int] size: A tuple of two integers representing the width and height of the image.
    :param Tuple[int, int, int] color: A tuple of three integers representing the RGB color (e.g., (255, 0, 0) for red).
    :param Optional[Union[str, os.PathLike]] save_path: a string representing the file path to save the image.  If not provided, the function returns the image as a numpy array.
    :return Union[None, np.ndarray]: If save_path is provided, the function saves the image to the specified path and returns None. f save_path is not provided, the function returns the image as a numpy ndarray.
    """

    check_valid_tuple(x=size, accepted_lengths=(2,), valid_dtypes=(int,))
    check_if_valid_rgb_tuple(data=color)
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color[::-1]
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
        cv2.imwrite(save_path, img)
    else:
        return img


def interpolate_color_palette(start_color: Tuple[int, int, int],
                              end_color: Tuple[int, int, int],
                              n: Optional[int] = 10):
    """
    Generate a list of colors interpolated between two passed RGB colors.

    :param start_color: Tuple of RGB values for the start color.
    :param end_color: Tuple of RGB values for the end color.
    :param n: Number of colors to generate.
    :return: List of interpolated RGB colors.

    :example:
    >>> red, black = (255, 0, 0), (0, 0, 0)
    >>> colors = interpolate_color_palette(start_color=red, end_color=black, n = 10)
    """

    check_if_valid_rgb_tuple(data=start_color)
    check_if_valid_rgb_tuple(data=end_color)
    check_int(name=f'{interpolate_color_palette.__name__} n', value=n, min_value=3)
    return [(
        int(start_color[0] + (end_color[0] - start_color[0]) * i / (n - 1)),
        int(start_color[1] + (end_color[1] - start_color[1]) * i / (n - 1)),
        int(start_color[2] + (end_color[2] - start_color[2]) * i / (n - 1))
    ) for i in range(n)]
