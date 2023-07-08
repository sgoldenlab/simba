import math

import numpy as np


def rectangle_size_calc(rectangle_dict: dict, px_mm: float) -> dict:
    """
    Compute metric height, width and area of rectangle.

    :param dict rectangle_dict: The rectangle width and height in pixels.
    :param float px_mm: Pixels per millimeter in the video.

    :example:
    >>> rectangle_size_calc(rectangle_dict={'height': 500, 'width': 500}, px_mm=10)
    >>> {'height': 500, 'width': 500, 'height_cm': 5.0, 'width_cm': 5.0, 'area_cm': 25.0}

    """

    rectangle_dict["height_cm"] = round((rectangle_dict["height"] / px_mm) / 10, 2)
    rectangle_dict["width_cm"] = round((rectangle_dict["width"] / px_mm) / 10, 2)
    rectangle_dict["area_cm"] = round(
        rectangle_dict["width_cm"] * rectangle_dict["height_cm"], 2
    )
    return rectangle_dict


def circle_size_calc(circle_dict, px_mm) -> dict:
    """
    Compute metric radius and area of circle.

    :param dict circle_dict: The circle radius in pixels
    :param float px_mm: Pixels per millimeter in the video.

    :example:
    >>> circle_size_calc(circle_dict={'radius': 100}, px_mm=5)
    >>> {'radius': 100, 'radius_cm': 2.0, 'area_cm': 12.57}
    """

    radius_cm = round((circle_dict["radius"] / px_mm) / 10, 2)
    circle_dict["radius_cm"] = radius_cm
    circle_dict["area_cm"] = round(math.pi * (radius_cm**2), 2)
    return circle_dict


def polygon_size_calc(polygon_dict, px_mm) -> dict:
    """
    Compute metric area of polygon.

    :param dict polygon_dict: The polygon vertices as np.ndarray
    :param float px_mm: Pixels per millimeter in the video.

    :example:
    >>> polygon_size_calc(polygon_dict={'vertices': np.array([[0, 2], [200, 98], [100, 876], [10, 702]])}, px_mm=5)
    >>> {'vertices': [[  0,   2], [200,  98], [100, 876], [ 10, 702]], 'area_cm': 45.29}
    """

    y_vals = polygon_dict["vertices"][:, 0]
    x_vals = polygon_dict["vertices"][:, 1]
    poly_area_px = 0.5 * np.abs(
        np.dot(x_vals, np.roll(y_vals, 1)) - np.dot(y_vals, np.roll(x_vals, 1))
    )
    polygon_dict["area_cm"] = round((poly_area_px / px_mm) / 500, 2)

    return polygon_dict


polygon_size_calc(
    polygon_dict={"vertices": np.array([[0, 2], [200, 98], [100, 876], [10, 702]])},
    px_mm=5,
)
