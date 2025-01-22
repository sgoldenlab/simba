__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from simba.utils.checks import check_int, check_str, check_valid_tuple
from simba.utils.enums import Formats


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
    rectangle_dict["area_cm"] = round(rectangle_dict["width_cm"] * rectangle_dict["height_cm"], 2)
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
    polygon = polygon_dict["vertices"]
    area = round((ConvexHull(polygon).area / px_mm) / 10, 2)
    polygon_dict["area_cm"] = area

    return polygon_dict

def get_half_circle_vertices(center: Tuple[int, int],
                             radius: int,
                             direction: str,
                             n_points: Optional[int] = 50) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:

    """
    Generates vertices for a half-circle with a given radius and direction, centered at a specific point.

    :param Tuple[int, int] center: A tuple (x, y) representing the center coordinates of the half-circle. Must contain exactly two numeric values.
    :param int radius: The radius of the half-circle. Must be a positive integer.
    :param str direction: The direction in which the half-circle is oriented.
    :param Optional[int] n_points: The number of vertices used to approximate the half-circle. Defaults to 50.
    :return: A tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='radius', value=radius, min_value=1)
    check_str(name='direction', options=['NORTH', 'SOUTH', 'WEST', 'EAST' 'NORTH-EAST', 'NORTH-WEST', 'SOUTH-EAST', 'SOUTH-WEST'], value=direction)
    x_c, y_c = center
    if direction == "WEST":
        a = np.linspace(np.pi / 2, 3 * np.pi / 2, n_points)
    elif direction == "EAST":
        a = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    elif direction == "SOUTH":
        a = np.linspace(0, np.pi, n_points)
    elif direction == 'SOUTH-WEST':
        a = np.linspace(np.pi / 4, 5 * np.pi / 4, n_points)
    elif direction == 'SOUTH-EAST':
        a = np.linspace(-np.pi / 4, 3 * np.pi / 4, n_points)
    elif direction == 'NORTH-WEST':
        a = np.linspace(3 * np.pi / 4, 7 * np.pi / 4, n_points)
    elif direction == 'NORTH-EAST':
        a = np.linspace(5 * np.pi / 4, 9 * np.pi / 4, n_points)
    else:
        a = np.linspace(np.pi, 2 * np.pi, n_points)
    x, y = x_c + radius * np.cos(a), y_c + radius * np.sin(a)
    vertices = np.column_stack((x, y)).astype(np.int32)
    shape_center = np.array(Polygon(vertices).centroid.coords)[0].astype(np.int32)
    vertices_dict = {"Center_tag": (shape_center[0], shape_center[1])}

    for tag_id in range(vertices.shape[0]):
        vertices_dict[f"Tag_{tag_id}"] = (vertices[tag_id][0], vertices[tag_id][1])
    return (np.array(vertices).astype("int32"), vertices_dict)



def get_vertices_hexagon(center: Tuple[int, int],
                         radius: int) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Generates the vertices of a regular hexagon centered at a given point with a specified radius.

    :param Tuple[int, int] center:  A tuple (x, y) representing the center coordinates of the hexagon. Must contain exactly two numeric values.
    :param int radius: The radius of the hexagon, which represents the distance from the center to any of the vertices. Must be a positive integer.
    :return: A tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """


    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='radius', value=radius, min_value=1)
    vertices = []
    x_c, y_c = center
    for i in range(6):
        angle_rad = np.deg2rad(60 * i)
        x_i = x_c + radius * np.cos(angle_rad)
        y_i = y_c + radius * np.sin(angle_rad)
        vertices.append((x_i, y_i))

    vertices_dict = {"Center_tag": (center[0], center[1])}
    for tag_id, tag in enumerate(vertices):
        vertices_dict[f"Tag_{tag_id}"] = (int(tag[0]), int(tag[1]))
    return (np.array(vertices).astype("int32"), vertices_dict)


def get_ear_tags_for_rectangle(center: Tuple[int, int], width: int, height: int) -> Dict[str, Union[int, Tuple[int, int]]]:
    """
    Knowing the center, width, and height of rectangle, return its vertices.

    :param Tuple[int, int] center: The center x and y coordinates of the rectangle
    :param int width: The width of the rectangle in pixels.
    :param Tuple[int, int] width: The width of the rectangle in pixels.
    :return: Dictionary holding the name of the tag as key and coordinates as value.
    :rtype: Dict[str, Union[int, Tuple[int, int]]]
    """

    check_valid_tuple(x=center, source=get_ear_tags_for_rectangle.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='width', value=width, min_value=1)
    check_int(name='height', value=height, min_value=1)
    tags = {}
    tags['top_left_x'] = int((center[1] - (width/2)))
    tags['top_left_y'] = int(center[0] - (height/2))
    tags['bottom_right_x'] = int(center[1] + (width/2))
    tags['bottom_right_y'] = int(center[0] + (height/2))
    tags['top_right_tag'] = (int(center[1] + (width/2)), int(center[0] - (height/2)))
    tags['bottom_left_tag'] = (int(center[1] - (width / 2)), int(center[0] + (height / 2)))
    tags['top_tag'] = (int(center[1]), int(center[0] - (height / 2)))
    tags['right_tag'] = (int(center[1] + (width / 2)), int(center[0]))
    tags['left_tag'] = (int(center[1] - (width / 2)), int(center[0]))
    tags['bottom_tag'] = (int(center[1]), int(center[0] + (height / 2)))
    return tags


def get_triangle_vertices(center: Tuple[int, int], side_length: int, direction: int) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Find equilateral triangle vertices knowing the center, direction and length side.

    :param Tuple[int, int] center: A tuple (x, y) representing the center coordinates of the hexagon. Must contain exactly two numeric values.
    :param int side_length: The length of each of the three sides in pixels.
    :param int direction: The direction to point the vertices.
    :return: 2-part tuple containing vertices as array and dict.
    :rtype: Tuple[np.ndarray, Dict[str, Tuple[int, int]]]
    """
    check_valid_tuple(x=center, source=get_vertices_hexagon.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name='side_length', value=side_length, min_value=1)
    check_int(name='direction', value=direction, min_value=0, max_value=360)

    direction_radians = np.radians(direction)
    radius = side_length / np.sqrt(3)

    top_vertex = (center[0] + radius * np.cos(direction_radians), center[1] + radius * np.sin(direction_radians))
    vertex2 = (center[0] + radius * np.cos(direction_radians + np.radians(120)), center[1] + radius * np.sin(direction_radians + np.radians(120)))
    vertex3 = (center[0] + radius * np.cos(direction_radians + np.radians(-120)), center[1] + radius * np.sin(direction_radians + np.radians(-120)))
    vertices = np.array([top_vertex, vertex2, vertex3, top_vertex]).astype(np.int32)
    vertices_dict = {"Center_tag": (int(center[0]), int(center[1])), 'Tag_0':  (int(top_vertex[0]), int(top_vertex[1])), 'Tag_1': (int(vertex2[0]), int(vertex2[1])), 'Tag_2': (int(vertex3[0]), int(vertex3[1]))}
    return (vertices, vertices_dict)
