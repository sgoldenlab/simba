from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon
from simba.utils.checks import check_valid_lst, check_valid_tuple
from simba.mixins.geometry_mixin import GeometryMixin
import cv2



def adjust_geometry_locations(geometries: List[Polygon],
                              shift: Tuple[int, int],
                              minimum: Optional[Tuple[int, int]] = (0, 0),
                              maximum: Optional[Tuple[int, int]] = (np.inf, np.inf)) -> List[Polygon]:
    """
    Shift a set of geometries specified distance in the x and/or y-axis.

    .. image:: _static/img/adjust_geometry_locations.png
       :width: 600
       :align: center

    :param  List[Polygon] geometries: List of input polygons to be adjusted.
    :param Tuple[int, int] shift: Tuple specifying the shift distances in the x and y-axis.
    :param Optional[Tuple[int, int]] minimum: Minimim allowed coordinates of Polygon points on x and y axes. Default: (0,0).
    :param Optional[Tuple[int, int]] maximum: Maximum allowed coordinates of Polygon points on x and y axes. Default: (np.inf, np.inf).
    :return List[Polygon]: List of adjusted polygons.

    :example:
    >>> shapes = GeometryMixin().adjust_geometry_locations(geometries=shapes, shift=(0, 333))
    """


    check_valid_tuple(x=shift, source=f"{adjust_geometry_locations.__name__} shift", accepted_lengths=(2,), valid_dtypes=(int,))
    check_valid_tuple(x=shift, source=f"{adjust_geometry_locations.__name__} minimum", accepted_lengths=(2,), valid_dtypes=(int,))
    check_valid_tuple(x=shift, source=f"{adjust_geometry_locations.__name__} maximum", accepted_lengths=(2,), valid_dtypes=(int,))
    check_valid_lst(data=geometries, source=f"{adjust_geometry_locations.__name__} geometries", valid_dtypes=(Polygon,), min_len=1)
    results = []
    for shape_cnt, shape in enumerate(geometries):
        shape_results = []
        for x, y in list(shape.exterior.coords):
            x_shift, y_shift = int(np.ceil(y + shift[1])), int(np.ceil(x + shift[0]))
            x_shift, y_shift = max(minimum[0], x_shift), max(minimum[1], y_shift)
            x_shift, y_shift = min(maximum[0], x_shift), min(maximum[1], y_shift)
            shape_results.append([y_shift, x_shift])
        results.append(Polygon(shape_results))
    return results






    #     results.append(Polygon([(int(abs(x + shift[0])), int(abs(y + shift[1]))) for x, y in list(shape.exterior.coords)]))
    # return results


geometries = GeometryMixin().bodyparts_to_polygon(np.array([[[50, 50],
                                                             [100, 100],
                                                             [50, 100],
                                                             [100, 50]]]))
geometries_shifted = adjust_geometry_locations(geometries=geometries, shift=(-25, 100), maximum=(500, 500))

geometries = [geometries[0], geometries_shifted[0]]

img = GeometryMixin.view_shapes(shapes=geometries)
cv2.imshow('sdasdas', img)
cv2.waitKey(3000)