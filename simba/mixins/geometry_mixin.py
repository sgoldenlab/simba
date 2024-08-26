import functools
import itertools
import math
import multiprocessing
import os
import platform
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import imutils
import numpy as np
import pandas as pd
from numba import njit, prange
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              MultiPoint, MultiPolygon, Point, Polygon)
from shapely.ops import linemerge, split, triangulate, unary_union

try:
    from typing_extensions import Literal
except:
    from typing import Literal

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import (check_float,
                                check_if_2d_array_has_min_unique_values,
                                check_if_dir_exists, check_if_valid_img,
                                check_if_valid_input, check_if_valid_rgb_tuple,
                                check_instance, check_int,
                                check_iterable_length, check_str,
                                check_that_column_exist, check_valid_array,
                                check_valid_boolean, check_valid_lst,
                                check_valid_tuple)
from simba.utils.data import create_color_palette, create_color_palettes
from simba.utils.enums import (Defaults, Formats, GeometryEnum, Options,
                               TextOptions)
from simba.utils.errors import CountError, InvalidInputError
from simba.utils.read_write import (SimbaTimer, find_core_cnt,
                                    find_max_vertices_coordinates, read_df,
                                    read_frm_of_video, stdout_success)


class GeometryMixin(object):
    """
    Methods to perform geometry transformation of pose-estimation data. This includes creating bounding boxes,
    line objects, circles etc. from pose-estimated body-parts and computing metric representations
    of the relationships between created shapes or their attributes (sizes, distances etc.).

    As of 01/24, very much wip and relies heavily on `shapley <https://shapely.readthedocs.io/en/stable/manual.html>`_.

    .. note::
       These methods generally do not create visualizations - they mainly generate geometry data-objects or metrics.
       To create visualizations with geometries overlay on videos, pass returned shapes to `simba.plotting.geometry_plotter.GeometryPlotter`.

    """

    def __init__(self):
        if platform.system() == "Darwin":
            if not multiprocessing.get_start_method(allow_none=True):
                multiprocessing.set_start_method("fork", force=True)
        pass

    @staticmethod
    def bodyparts_to_polygon(data: np.ndarray,
                             cap_style: Optional[Literal["round", "square", "flat"]] = "round",
                             parallel_offset: Optional[int] = 1,
                             pixels_per_mm: Optional[int] = 1,
                             simplify_tolerance: Optional[float] = 2,
                             preserve_topology: Optional[bool] = True) -> List[Polygon]:


        """
        .. note::
           To convert multiple frame body-part coordinates to polygon, use ``simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodyparts_to_polygon``

        .. image:: _static/img/bodyparts_to_polygon.png
           :width: 400
           :align: center

        :param np.ndarray data: 3D array with body-part coordinates where rows are frames and columns are x and y coordinates.
        :param Literal["round", "square", "flat"] cap_style: How intersections between lines are handled in the polygon. Default: round.
        :param int parallel_offset: How much to "buffer" the polygon from the original size in millimeters. Default: 1.
        :param int pixels_per_mm: The pixels per millimeter conversion factor used for buffering. Default: 1.
        :param float simplify_tolerance: The higher this value, the smaller the number of vertices in the resulting polygon. Default 2.
        :param bool preserve_topology: If True, operation will avoid creating invalid geometries (checking for collapses, ring-intersections, etc). Deafult True.

        :example:
        >>> data = [[[364, 308],[383, 323],[403, 335], [423, 351]]]
        >>> GeometryMixin().bodyparts_to_polygon(data=data)
        """

        check_valid_array(
            source=f"{GeometryMixin().bodyparts_to_polygon.__name__} data",
            data=data,
            accepted_ndims=(3,),
            accepted_dtypes=(np.float64, np.float32, np.int32, np.int64, int, float),
        )

        check_str(
            name=f"{GeometryMixin().bodyparts_to_polygon.__name__} cap style",
            value=cap_style,
            options=list(GeometryEnum.CAP_STYLE_MAP.value.keys()),
        )
        check_float(
            name=f"{GeometryMixin().bodyparts_to_polygon.__name__} parallel_offset",
            value=parallel_offset,
            min_value=0,
        )
        check_float(
            name=f"{GeometryMixin().bodyparts_to_polygon.__name__} pixels_per_mm",
            value=pixels_per_mm,
            min_value=0.01,
        )
        check_float(
            name=f"{GeometryMixin().bodyparts_to_polygon.__name__} simplify_tolerance",
            value=simplify_tolerance,
            min_value=1,
        )
        results = []

        if parallel_offset > 0:
            buffer = int(parallel_offset / pixels_per_mm)
        else:
            buffer = 0

        for i in range(data.shape[0]):
            if not check_if_2d_array_has_min_unique_values(data=data[i], min=3):
                results.append(Polygon([(0, 0), (0, 0), (0, 0)]))
            else:
                results.append(
                    Polygon(
                        LineString(data[i].tolist())
                        .buffer(
                            distance=buffer,
                            cap_style=GeometryEnum.CAP_STYLE_MAP.value[cap_style],
                        )
                        .simplify(
                            tolerance=simplify_tolerance,
                            preserve_topology=preserve_topology,
                        )
                        .convex_hull
                    )
                )

        return results

    @staticmethod
    def bodyparts_to_points(data: np.ndarray, buffer: Optional[int] = None, px_per_mm: Optional[int] = None) -> List[Union[Point, Polygon]]:
        """
        Convert body-parts coordinate to Point geometries.

        :param np.ndarray data: 2D array with body-part coordinates where rows are frames and columns are x and y coordinates.
        :param Optional[int] buffer: If not None, then the area of the Point. Thus, if not None, then returns Polygons representing the Points.
        :param Optional[int] px_per_mm: Pixels to millimeter convertion factor. Required if buffer is not None.

        .. note:
           If buffer and px_per_mm is not None, then the points will be *buffered* and a 2D share polygon created with the specified buffered area.
           If buffer is provided, then also provide px_per_mm for accurate convesion factor between pixels and millimeters.

           If having a large number of body-parts, consider using ``simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodypart_to_point``
           which uses CPU multiprocessing.

        :example:
        >>> data = np.random.randint(0, 100, (1, 2))
        >>> GeometryMixin().bodyparts_to_points(data=data)
        """

        check_valid_array(
            source=f"{GeometryMixin.bodyparts_to_points} data",
            data=data,
            accepted_dtypes=(np.int64, np.int32, np.int8),
            accepted_ndims=(2,),
        )
        area = None
        if buffer is not None:
            check_float(
                name=f"{GeometryMixin.bodyparts_to_points} buffer",
                value=buffer,
                min_value=1,
            )
            check_float(
                name=f"{GeometryMixin.bodyparts_to_points} px_per_mm",
                value=px_per_mm,
                min_value=1,
            )
            area = buffer / px_per_mm
        results = []
        for i in range(data.shape[0]):
            if area is not None:
                results.append(Point(data[i]).buffer(area / 2, cap_style=3))
            else:
                results.append(Point(data[i]))
        return results

    @staticmethod
    def to_linestring(data: np.ndarray) -> LineString:
        """
        Convert a 2D array of x and y coordinates to a shapely linestring.

        Linestrings are useful for representing an animal path, and to answer questions like (i)
        "How far along the animals paths was the animal most proximal to geometry X"?
        "How far had the animal travelled at time T?"
        "When does the animal path intersect geometry X?"

        :param np.ndarray data: 2D array with floats or ints of size Nx2 representing body-part coordinates.

        :example:
        >>> data = np.load('/Users/simon/Desktop/envs/simba/simba/simba/sandbox/data.npy')
        >>> linestring = GeometryMixin.to_linestring(data=data)
        """
        check_valid_array(
            data=data,
            source=GeometryMixin.to_linestring.__name__,
            accepted_ndims=(2,),
            accepted_dtypes=(float, np.float32, np.float64, np.int32, np.int64),
        )
        return LineString(data)

    @staticmethod
    def bodyparts_to_circle(data: np.ndarray,
                            parallel_offset: Optional[float] = 1,
                            pixels_per_mm: Optional[int] = 1) -> Union[Polygon, List[Polygon]]:
        """
        Create a circle geometry from a single body-part (x,y) coordinate.

        .. note::
           For multiple frames, call this method using :func:`~simba.mixins.geometry_mixin.GeometryMixin.multiframe_bodyparts_to_circle`

        .. image:: _static/img/bodyparts_to_circle.png
           :width: 400
           :align: center

        .. image:: _static/img/bodyparts_to_circle.gif
           :width: 450
           :align: center

        :param np.ndarray data: The body-part coordinate xy as a 1d array. E.g., np.array([364, 308])
        :param float parallel_offset: The radius of the resultant circle in millimeters.
        :param int pixels_per_mm: The pixels per millimeter of the video. If not passed, 1 will be used meaning revert to radius in pixels rather than millimeters.
        :returns Polygon: Shapely Polygon of curcular shape.

        :example:
        >>> data = np.array([364, 308])
        >>> polygon = GeometryMixin().bodyparts_to_circle(data=data, parallel_offset=10, pixels_per_mm=4)
        """

        check_valid_array(data=data, accepted_ndims=([1, 2, ]), accepted_dtypes=Formats.NUMERIC_DTYPES.value, max_axis_1=2)
        check_float(name=f"{GeometryMixin.bodyparts_to_circle.__name__} parallel_offset", value=pixels_per_mm, min_value=1)
        check_float(name=f"{GeometryMixin.bodyparts_to_circle.__name__} pixels_per_mm", value=pixels_per_mm, min_value=1)

        if data.shape == (2,):
            return Point(data).buffer(parallel_offset / pixels_per_mm)
        else:
            results = []
            for i in range(data.shape[0]):
                results.append(Point(data[i]).buffer(parallel_offset / pixels_per_mm))
            return results

    @staticmethod
    def bodyparts_to_multistring_skeleton(data: np.ndarray) -> MultiLineString:
        """
        Create a multistring skeleton from a 3d array where each 2d array represents start and end coordinates of a line
        within the skeleton.

        :param np.ndarray data: A 3D numpy array where each 2D array represents the start position and end position of each LineString.
        :returns MultiLineString: Shapely MultiLineString representing animal skeleton.

        .. image:: _static/img/bodyparts_to_multistring_skeleton.png
           :width: 400
           :align: center

        .. image:: _static/img/bodyparts_to_multistring_skeleton.gif
           :width: 450
           :align: center

        :example:
        >>> skeleton = np.array([[[5, 5], [1, 10]], [[5, 5], [9, 10]], [[9, 10], [1, 10]], [[9, 10], [9, 25]], [[1, 10], [1, 25]], [[9, 25], [5, 50]], [[1, 25], [5, 50]]])
        >>> shape_multistring = GeometryMixin().bodyparts_to_multistring_skeleton(data=skeleton)
        """

        if data.ndim != 3:
            raise InvalidInputError(msg=f"Body-parts to skeleton expects a 3D array, got {data.ndim}",
                                    source=GeometryMixin.bodyparts_to_line.__name__,
            )
        shape_skeleton = []
        for i in data:
            shape_skeleton.append(GeometryMixin().bodyparts_to_line(data=i))
        shape_skeleton = linemerge(MultiLineString(shape_skeleton))

        return shape_skeleton

    @staticmethod
    def buffer_shape(shape: Union[Polygon, LineString],
                     size_mm: int,
                     pixels_per_mm: float,
                     cap_style: Literal["round", "square", "flat"] = "round") -> Polygon:
        """
        Create a buffered shape by applying a buffer operation to the input polygon or linestring.

        .. image:: _static/img/buffer_shape.png
           :width: 400
           :align: center

        :param Union[Polygon, LineString] shape: The input Polygon or LineString to be buffered.
        :param int size_mm: The size of the buffer in millimeters. Use a negative value for an inward buffer.
        :param float pixels_per_mm: The conversion factor from millimeters to pixels.
        :param Literal['round', 'square', 'flat'] cap_style: The cap style for the buffer. Valid values are 'round', 'square', or 'flat'. Defaults to 'round'.
        :return Polygon: The buffered shape.

        :example:
        >>> polygon = GeometryMixin().bodyparts_to_polygon(np.array([[100, 110],[100, 100],[110, 100],[110, 110]]))
        >>> buffered_polygon = GeometryMixin().buffer_shape(shape=polygon, size_mm=-1, pixels_per_mm=1)
        """

        check_instance(
            source=GeometryMixin.buffer_shape.__name__,
            instance=shape,
            accepted_types=(LineString, Polygon),
        )
        check_int(name="BUFFER SHAPE size_mm", value=size_mm)
        check_float(name="BUFFER SHAPE pixels_per_mm", value=pixels_per_mm, min_value=1)
        return shape.buffer(distance=int(size_mm / pixels_per_mm), cap_style=GeometryEnum.CAP_STYLE_MAP.value[cap_style])

    @staticmethod
    def compute_pct_shape_overlap(shapes: np.ndarray, denominator: Optional[Literal["difference", "shape_1", "shape_2"]] = "difference") -> int:

        """
        Compute the percentage of overlap between two shapes.

        .. image:: _static/img/compute_pct_shape_overlap.png
           :width: 400
           :align: center

        :param List[Union[LineString, Polygon]] shapes: A 2D array, where each sub-array has two Polygon or LineString shapes.
        :param Optional[Literal['union', 'shape_1', 'shape_2']] denominator: If ``difference``, then percent overlap is calculated using non-intersection area as denominator. If ``shape_1``, percent overlap is calculated using the area of the first shape as denominator. If ``shape_2``, percent overlap is calculated using the area of the second shape as denominator. Default: ``difference``.
        :return float: The percentage of overlap between the two shapes as integer.

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
        >>> polygon_1 = [polygon_1 for x in range(100)]
        >>> polygon_2 = [polygon_2 for x in range(100)]
        >>> data = np.column_stack((polygon_1, polygon_2))
        >>> results = GeometryMixin.compute_pct_shape_overlap(shapes=data)
        """

        check_valid_array(
            data=shapes,
            source=GeometryMixin.compute_pct_shape_overlap.__name__,
            accepted_ndims=(2,),
            max_axis_1=2,
            min_axis_0=1,
            accepted_dtypes=(Polygon, LineString),
        )
        results = np.full((shapes.shape[0],), np.nan)

        for i in range(shapes.shape[0]):
            if shapes[i][0].intersects(shapes[i][1]):
                intersection = shapes[i][0].intersection(shapes[i][1])
                if denominator == "difference":
                    results[i] = np.round(
                        (
                            intersection.area
                            / (
                                (shapes[i][0].area + shapes[i][1].area)
                                - intersection.area
                            )
                            * 100
                        ),
                        2,
                    )
                elif denominator == "shape_1":
                    results[i] = np.round(
                        (intersection.area / shapes[i][0].area) * 100, 2
                    )
                else:
                    results[i] = np.round(
                        (intersection.area / shapes[i][1].area) * 100, 2
                    )
            else:
                results[i] = 0

        return results

    @staticmethod
    def compute_shape_overlap(shapes: List[Union[Polygon, LineString]]) -> int:
        """
        Computes if two geometrical shapes (Polygon or LineString) overlaps or are disjoint.

        .. note::
           Only returns if two shapes are overlapping or not overlapping. If the amount of overlap is required, use
           ``GeometryMixin().compute_shape_overlap()``.

        .. image:: _static/img/compute_overlap.png
           :width: 400
           :align: center

        :param List[Union[LineString, Polygon]] shapes: A list of two input Polygon or LineString shapes.
        :return float: Returns 1 if the two shapes overlap, otherwise returns 0.
        """

        for shape in shapes:
            check_instance(
                source=GeometryMixin.compute_shape_overlap.__name__,
                instance=shape,
                accepted_types=(LineString, Polygon),
            )
        check_iterable_length(
            source=GeometryMixin.compute_shape_overlap.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )
        if shapes[0].intersects(shapes[1]):
            return 1
        else:
            return 0

    @staticmethod
    def crosses(shapes: List[LineString]) -> bool:
        """
        Check if two LineString objects cross each other.

        .. image:: _static/img/are_lines_crossing.png
           :width: 400
           :align: center

        :param List[LineString] shapes: A list containing two LineString objects.
        :return bool: True if the LineStrings cross each other, False otherwise.

        :example:
        >>> line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 10],[20, 10],[30, 10],[40, 10]]))
        >>> line_2 = GeometryMixin().bodyparts_to_line(np.array([[25, 5],[25, 20],[25, 30],[25, 40]]))
        >>> GeometryMixin().crosses(shapes=[line_1, line_2])
        >>> True
        """

        check_iterable_length(
            source=GeometryMixin.crosses.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )
        for shape in shapes:
            check_instance(
                source=GeometryMixin.crosses.__name__,
                instance=shape,
                accepted_types=LineString,
            )
        return shapes[0].crosses(shapes[1])

    @staticmethod
    def is_shape_covered(shapes: List[Union[LineString, Polygon, MultiPolygon, MultiPoint]]) -> bool:
        """
        Check if one geometry fully covers another.

        .. image:: _static/img/is_line_covered.png
           :width: 400
           :align: center

        :param Union[LineString, Polygon, MultiPolygon, MultiPoint] shapes: List of 2 geometries, checks if the second geometry fully covers the first geometry.
        :return bool: True if the second geometry fully covers the first geometry, otherwise False.

        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 25], [25, 75], [90, 25], [90, 75]]))
        >>> GeometryMixin().is_shape_covered(shapes=[polygon_2, polygon_1])
        >>> True

        """
        check_valid_lst(
            data=shapes,
            source=GeometryMixin.is_shape_covered.__name__,
            valid_dtypes=(LineString, Polygon, MultiPolygon, MultiPoint),
            exact_len=2,
        )
        return shapes[1].covers(shapes[0])

    @staticmethod
    def area(shape: Union[MultiPolygon, Polygon], pixels_per_mm: Optional[float]) -> float:
        """
        Calculate the area of a geometry in square millimeters.

        .. note::
           If certain that the input data is a valid Polygon, consider using :func:`simba.feature_extractors.perimeter_jit.jitted_hull`

        :param Union[MultiPolygon, Polygon] shape: The geometry (MultiPolygon or Polygon) for which to calculate the area.
        :param float pixels_per_mm: The pixel-to-millimeter conversion factor.
        :return float: The area of the geometry in square millimeters.

        :example:
        >>> polygon = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
        >>> GeometryMixin().area(shape=polygon, pixels_per_mm=4.9)
        >>> 1701.556313816644
        """

        check_instance(
            source=f"{GeometryMixin().area.__name__} shape",
            instance=shape,
            accepted_types=(MultiPolygon, Polygon),
        )
        check_float(name=f"{GeometryMixin().area.__name__} shape",value=pixels_per_mm,min_value=0.01,)

        return shape.area / pixels_per_mm

    @staticmethod
    def shape_distance(shapes: List[Union[LineString, Polygon, Point]], pixels_per_mm: float, unit: Literal["mm", "cm", "dm", "m"] = "mm") -> float:
        """
        Calculate the distance between two geometries in specified units.

        :param List[Union[LineString, Polygon]] shapes: A list containing two LineString or Polygon geometries.
        :param float pixels_per_mm: The conversion factor from pixels to millimeters.
        :param Literal['mm', 'cm', 'dm', 'm'] unit: The desired unit for the distance calculation. Options: 'mm', 'cm', 'dm', 'm'. Defaults to 'mm'.
        :return float: The distance between the two geometries in the specified unit.

        .. image:: _static/img/shape_distance.png
           :width: 400
           :align: center

        >>> shape_1 = Polygon([(0, 0), 10, 10), 0, 10), 10, 0)])
        >>> shape_2 = Polygon([(0, 0), 10, 10), 0, 10), 10, 0)])
        >>> GeometryMixin.shape_distance(shapes=[shape_1, shape_2], pixels_per_mm=1)
        >>> 0
        """

        check_if_valid_input(name="UNIT", input=unit, options=["mm", "cm", "dm", "m"])
        for shape in shapes:
            check_instance(
                source=GeometryMixin.shape_distance.__name__,
                instance=shape,
                accepted_types=(LineString, Polygon, Point),
            )
        check_iterable_length(
            source=GeometryMixin.shape_distance.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )

        D = shapes[0].distance(shapes[1]) / pixels_per_mm
        if unit == "cm":
            D = D / 10
        elif unit == "dm":
            D = D / 100
        elif unit == "m":
            D = D / 1000
        return D

    @staticmethod
    def bodyparts_to_line(
        data: np.ndarray,
        buffer: Optional[int] = None,
        px_per_mm: Optional[float] = None,
    ) -> Union[Polygon, LineString]:
        """
        Convert body-part coordinates to a Linestring.

        .. note::
          If buffer and px_per_mm is provided, then the returned object will be linestring buffered to a 2D object rectangle
          with specificed area.

        .. image:: _static/img/bodyparts_to_line.png
           :width: 400
           :align: center

        :example:
        >>> data = np.array([[364, 308],[383, 323], [403, 335],[423, 351]])
        >>> line = GeometryMixin().bodyparts_to_line(data=data)
        >>> line = GeometryMixin().bodyparts_to_line(data=data, buffer=10, px_per_mm=4)
        """

        if buffer is not None:
            check_int(
                name=f"{GeometryMixin.bodyparts_to_line} buffer",
                value=buffer,
                min_value=1,
            )
            check_float(
                name=f"{GeometryMixin.bodyparts_to_line} px_per_mm",
                value=px_per_mm,
                min_value=1,
            )
            area = buffer * px_per_mm
        else:
            area = None

        if data.ndim != 2:
            raise InvalidInputError(
                msg=f"Body-parts to linestring expects a 2D array, got {data.ndim}",
                source=GeometryMixin.bodyparts_to_line.__name__,
            )

        if area is None:
            return LineString(data.tolist())
        else:
            return LineString(data.tolist()).buffer(distance=area, cap_style=3)

    @staticmethod
    def get_center(shape: Union[LineString, Polygon, MultiPolygon, List[Union[LineString, Polygon, MultiPolygon]]]) -> np.ndarray:
        """
        Get the center coordinate of a shape or a list of shapes.


        .. image:: _static/img/get_center.png
           :width: 500
           :align: center

        :example:
        >>> multipolygon = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
        >>> GeometryMixin().get_center(shape=multipolygon)
        >>> [33.96969697, 62.32323232]

        """

        check_instance(source=GeometryMixin.get_center.__name__, instance=shape, accepted_types=(MultiPolygon, LineString, Polygon, list))
        if not isinstance(shape, list):
            return np.array(shape.centroid)
        else:
            results = np.full((len(shape), 2), np.nan)
            check_valid_lst(data=shape, source=GeometryMixin.get_center.__name__, valid_dtypes=(MultiPolygon, LineString, Polygon,), min_len=1)
            for i in range(len(shape)):
                results[i] = np.array(shape[i].centroid)
            return results



        return

    @staticmethod
    def is_touching(shapes=List[Union[LineString, Polygon]]) -> bool:
        """
        Check if two geometries touch each other.

        .. image:: _static/img/touches.png
           :width: 500
           :align: center

        .. note::
           Different from GeometryMixin().crosses: Touches requires a common boundary, and does not require the sharing of interior space.

        :param List[Union[LineString, Polygon]] shapes: A list containing two LineString or Polygon geometries.
        :return bool: True if the geometries touch each other, False otherwise.

        :example:
        >>> rectangle_1 = Polygon(np.array([[0, 0], [10, 10], [0, 10], [10, 0]]))
        >>> rectangle_2 = Polygon(np.array([[20, 20], [30, 30], [20, 30], [30, 20]]))
        >>> GeometryMixin().is_touching(shapes=[rectangle_1, rectangle_2])
        >>> False
        """

        for i in shapes:
            check_instance(
                source=GeometryMixin.is_touching.__name__,
                instance=i,
                accepted_types=(LineString, Polygon),
            )
        check_iterable_length(
            source=GeometryMixin.is_touching.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )
        return shapes[0].touches(shapes[1])

    @staticmethod
    def is_containing(shapes=List[Union[LineString, Polygon]]) -> bool:
        """
        Check if the first shape in a list contains a second shape in a list.

        .. image:: _static/img/is_containing.png
           :width: 500
           :align: center

        """
        for i in shapes:
            check_instance(
                source=GeometryMixin.is_containing.__name__,
                instance=i,
                accepted_types=(LineString, Polygon),
            )
        check_iterable_length(
            source=GeometryMixin.is_containing.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )

        return shapes[0].contains(shapes[1])

    @staticmethod
    def difference(shapes=List[Union[LineString, Polygon, MultiPolygon]]) -> Polygon:
        """
        Calculate the difference between a shape and one or more potentially overlapping shapes.

        .. image:: _static/img/difference.png
           :width: 400
           :align: center

        .. image:: _static/img/difference_1.png
           :width: 400
           :align: center

        :param List[Union[LineString, Polygon, MultiPolygon]] shapes: A list of geometries.
        :return: The first geometry in ``shapes`` is returned where all parts that overlap with the other geometries in ``shapes have been removed.

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 25],[25, 75],[90, 25],[90, 75]]))
        >>> polygon_3 = GeometryMixin().bodyparts_to_polygon(np.array([[1, 25],[1, 75],[110, 25],[110, 75]]))
        >>> difference = GeometryMixin().difference(shapes = [polygon_1, polygon_2, polygon_3])
        """

        check_iterable_length(
            source=GeometryMixin.difference.__name__, val=len(shapes), min=2
        )
        for shape in shapes:
            check_instance(
                source=GeometryMixin.difference.__name__,
                instance=shape,
                accepted_types=(LineString, Polygon, MultiPolygon),
            )

        results = deepcopy(shapes[0])
        for overlap_shap in shapes[1:]:
            if isinstance(overlap_shap, MultiPolygon):
                for geo in overlap_shap.geoms:
                    results = results.difference(geo)
            else:
                results = results.difference(overlap_shap)
        return results

    @staticmethod
    def union(
        shapes: List[Union[LineString, Polygon, MultiPolygon]]
    ) -> Union[MultiPolygon, Polygon, MultiLineString]:
        """
        Compute the union of multiple geometries.

        .. image:: _static/img/union.png
           :width: 400
           :align: center

        :param List[Union[LineString, Polygon, MultiPolygon]] shapes: A list of LineString, Polygon, or MultiPolygon geometries to be unioned.
        :return Union[MultiPolygon, Polygon]: The resulting geometry after performing the union operation.

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[1, 25],[1, 75],[110, 25],[110, 75]]))
        >>> union = GeometryMixin().union(shape = polygon_1, overlap_shapes=[polygon_2, polygon_2])
        """

        check_iterable_length(
            source=GeometryMixin.union.__name__, val=len(shapes), min=2
        )
        for shape in shapes:
            check_instance(
                source=GeometryMixin.union.__name__,
                instance=shape,
                accepted_types=(LineString, Polygon, MultiPolygon),
            )
        return unary_union(shapes)

    @staticmethod
    def symmetric_difference(shapes: List[Union[LineString, Polygon, MultiPolygon]]) -> List[Union[Polygon, MultiPolygon]]:
        """
        Computes a new geometry consisting of the parts that are exclusive to each input geometry.

        In other words, it includes the parts that are unique to each geometry while excluding the parts that are common to both.

        .. image:: _static/img/symmetric_difference.png
           :width: 400
           :align: center

        :param List[Union[LineString, Polygon, MultiPolygon]] shapes: A list of LineString, Polygon, or MultiPolygon geometries to find the symmetric difference.
        :return List[Union[Polygon, MultiPolygon]]: A list containing the resulting geometries after performing symmetric difference operations.

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[1, 25], [1, 75], [110, 25], [110, 75]]))
        >>> symmetric_difference = symmetric_difference(shapes=[polygon_1, polygon_2])
        """
        check_iterable_length(source=GeometryMixin.union.__name__, val=len(shapes), min=2)
        for shape in shapes:
            check_instance(source=GeometryMixin.symmetric_difference.__name__, instance=shape, accepted_types=(LineString, Polygon, MultiPolygon))
        results = deepcopy(shapes)
        for c in itertools.combinations(list(range(0, len(shapes))), 2):
            results[c[0]] = results[c[0]].convex_hull.difference(results[c[1]].convex_hull)
            results[c[1]] = results[c[1]].convex_hull.difference(results[c[0]].convex_hull)

        results = [geometry for geometry in results if not geometry.is_empty]
        return results

    @staticmethod
    def view_shapes(shapes: List[Union[LineString, Polygon, MultiPolygon, MultiLineString]],
                    bg_img: Optional[np.ndarray] = None,
                    bg_clr: Optional[Tuple[int]] = None,
                    size: Optional[int] = None,
                    color_palette: Optional[str] = None,
                    thickness: Optional[int] = 2,
                    pixel_buffer: Optional[int] = 200) -> np.ndarray:

        """
        Helper function to draw shapes on white canvas or specified background image. Useful for quick troubleshooting.

        :example:
        >>> multipolygon_1 = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[100, 110],[100, 100],[110, 100],[110, 110]]))
        >>> line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
        >>> img = GeometryMixin.view_shapes(shapes=[line_1, polygon_1, multipolygon_1])
        """

        check_valid_lst(data=shapes, source=GeometryMixin.view_shapes.__name__, valid_dtypes=(LineString, Polygon, MultiPolygon, MultiLineString, Point), min_len=1)
        check_int(name='pixel_buffer', value=pixel_buffer, min_value=0)
        max_vertices = find_max_vertices_coordinates(shapes=shapes, buffer=pixel_buffer)
        if bg_img is None:
            if bg_clr is None:
                img = (np.ones((max_vertices[0], max_vertices[1], 3), dtype=np.uint8) * 255)
            else:
                check_if_valid_rgb_tuple(data=bg_clr)
                img = np.full((max_vertices[0], max_vertices[1], 3), bg_clr, dtype=np.uint8)
        else:
            img = bg_img

        check_str(name='color_palette', value=color_palette, options=Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value)
        colors = create_color_palette(pallete_name=color_palette, increments=len(shapes)+1)
        for shape_cnt, shape in enumerate(shapes):
            if isinstance(shape, Polygon):
                cv2.polylines(img, [np.array(shape.exterior.coords).astype(np.int32)], True, (colors[shape_cnt][::-1]), thickness=thickness)
                interior_coords = [np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2)) for interior in shape.interiors]
                for interior in interior_coords:
                    cv2.polylines(img, [interior], isClosed=True, color=(colors[shape_cnt][::-1]), thickness=thickness,)
            if isinstance(shape, LineString):
                if color_palette is None:
                    cv2.polylines(img, [np.array(shape.coords, dtype=np.int32)], False, (colors[shape_cnt][::-1]), thickness=thickness)
                else:
                    lines = np.array(shape.coords, dtype=np.int32)
                    palette = create_color_palette(pallete_name=color_palette, increments=lines.shape[0])
                    for i in range(1, lines.shape[0]):
                        p1, p2 = lines[i - 1], lines[i]
                        cv2.line(img, tuple(p1), tuple(p2), palette[i], thickness)

            if isinstance(shape, MultiPolygon):
                multi_polygon_clrs = create_color_palette(pallete_name=color_palette, increments=len(shape.geoms)+1)
                for polygon_cnt, polygon in enumerate(shape.geoms):
                    polygon_np = np.array((polygon.convex_hull.exterior.coords), dtype=np.int32)
                    cv2.polylines(img, [polygon_np], True, (multi_polygon_clrs[polygon_cnt][::-1]), thickness=thickness)
            if isinstance(shape, MultiLineString):
                for line_cnt, line in enumerate(shape.geoms):
                    cv2.polylines(img, [np.array(shape[line_cnt].coords, dtype=np.int32)], False, (colors[shape_cnt][::-1]), thickness=thickness)
            if isinstance(shape, Point):
                cv2.circle(
                    img,
                    (
                        int(np.array(shape.centroid)[0]),
                        int(np.array(shape.centroid)[1]),
                    ),
                    0,
                    colors[shape_cnt][::-1],
                    thickness,
                )
        if size:
            return imutils.resize(img, width=size)
        else:
            return img

    @staticmethod
    def geometry_video(shapes: List[List[Union[LineString, Polygon, MultiPolygon, MultiLineString, MultiPoint]]],
                       size: Optional[Tuple[int]],
                       save_path: Optional[Union[str, os.PathLike]] = None,
                       fps: Optional[int] = 10,
                       verbose: Optional[bool] = False,
                       bg_img: Optional[np.ndarray] = None,
                       bg_clr: Optional[Tuple[int]] = None,
                       thickness: Optional[int] = 2) -> None:
        """
        Helper to create a geometry video from a list of shapes.

        .. note::
           If more aesthetic videos are needed, overlaid on video, then use ``simba.plotting.geometry_plotter.GeometryPlotter``
           If single images of geometries are needed, then use ``simba.mixins.geometry_mixin.view_shapes``

        .. image:: _static/img/geometry_video.gif
           :width: 500
           :align: center

        :param List[List[Union[LineString, Polygon, MultiPolygon, MultiPoint, MultiLineString]]] shapes: List of lists containing geometric shapes to be included in the video. Each sublist represents a frame, and each element within the sublist represents a shape for that frame.
        :param Union[str, os.PathLike] save_path: Path where the resulting video will be saved.
        :param Optional[Tuple[int]] size: Tuple specifying the size of the output video in pixels (width, height).
        :param Optional[int] fps: Frames per second of the output video. Defaults to 10.
        :param Optional[bool] verbose: If True, then prints progress frmae-by-frame. Default: False.
        :param Optional[np.ndarray] bg_img: Background image to be used as the canvas for drawing shapes. Defaults to None. Could be e.g., a low opacity image of the arena.
        :param Optional[Tuple[int]] bg_clr: Background color specified as a tuple of RGB values. Defaults to white.
        """

        timer = SimbaTimer(start=True)
        for i in shapes:
            for j in i:
                check_instance(source=GeometryMixin.geometry_video.__name__, instance=j, accepted_types=( LineString, Polygon, MultiPolygon, MultiPoint, MultiLineString, Point))

        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
        check_int(name="fps", value=fps, min_value=1)
        if bg_img is not None:
            check_if_valid_img(data=bg_img, source=GeometryMixin.geometry_video.__name__)
        if bg_clr is not None:
            check_if_valid_rgb_tuple(data=bg_clr)
        check_instance(source=GeometryMixin.geometry_video.__name__, instance=size, accepted_types=(tuple))
        if len(size) != 2:
            raise InvalidInputError(msg=f"Size has to be 2 values, got {len(size)}", source=GeometryMixin.geometry_video.__name__)
        for i in size:
            check_instance(source=GeometryMixin.geometry_video.__name__, instance=i, accepted_types=(int,))
        if bg_img is None:
            if bg_clr is None:
                img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
            else:
                img = np.full((size[0], size[1], 3), bg_clr, dtype=np.uint8)
        else:
            img = bg_img
        clrs = create_color_palettes(no_animals=len(shapes), map_size=1)
        fourcc = cv2.VideoWriter_fourcc(*Formats.MP4_CODEC.value)
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (size[1], size[0]))
        for frm_cnt, frm_shapes in enumerate(zip(*shapes)):
            frm_img = deepcopy(img)
            for shape_cnt, shape in enumerate(frm_shapes):
                if isinstance(shape, Polygon):
                    cv2.polylines(frm_img, [np.array(shape.exterior.coords).astype(np.int)], True, (clrs[shape_cnt][0][::-1]), thickness=thickness)
                    interior_coords = [np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2)) for interior in shape.interiors]
                    for interior in interior_coords:
                        cv2.polylines(frm_img, [interior], isClosed=True, color=(clrs[shape_cnt][0][::-1]), thickness=thickness,)
                elif isinstance(shape, LineString):
                    cv2.polylines(frm_img, [np.array(shape.coords, dtype=np.int32)], False, (clrs[shape_cnt][0][::-1]), thickness=thickness)
                elif isinstance(shape, MultiPolygon):
                    for polygon_cnt, polygon in enumerate(shape.geoms):
                        polygon_np = np.array((polygon.convex_hull.exterior.coords), dtype=np.int32)
                        cv2.polylines(frm_img, [polygon_np], True, (clrs[shape_cnt + polygon_cnt + 1][::-1]), thickness=thickness)
                elif isinstance(shape, MultiLineString):
                    for line_cnt, line in enumerate(shape.geoms):
                        cv2.polylines(frm_img, [np.array(shape[line_cnt].coords, dtype=np.int32)], False, (clrs[shape_cnt][0][::-1]), thickness=thickness)
                elif isinstance(shape, MultiPoint):
                    for point in shape:
                        cv2.circle(frm_img, (int(np.array(point.centroid)[0]), int(np.array(point.centroid)[1])), 0, clrs[shape_cnt][0][::-1], thickness)
                elif isinstance(shape, Point):
                    cv2.circle(frm_img, (int(np.array(shape.centroid)[0]), int(np.array(shape.centroid)[1])), 0, clrs[shape_cnt][0][::-1], thickness)
            video_writer.write(frm_img.astype(np.uint8))
            if verbose:
                print(f"Geometry frame complete ({frm_cnt+1} / {len(shapes)})")
        video_writer.release()
        timer.stop_timer()
        if save_path is not None:
            msg = f"Video complete and saved at {save_path}!"
        else:
            msg = f"Video complete!"
        stdout_success(msg=msg, elapsed_time=timer.elapsed_time_str, source=GeometryMixin.geometry_video.__name__)

    @staticmethod
    def minimum_rotated_rectangle(shape=Polygon) -> Polygon:
        """
        Calculate the minimum rotated rectangle that bounds a given polygon.

        The minimum rotated rectangle, also known as the minimum bounding rectangle (MBR) or oriented bounding box (OBB), is the smallest rectangle that can fully contain a given polygon or set of points while allowing rotation. It is defined by its center, dimensions (length and width), and rotation angle.

        .. image:: _static/img/minimum_rotated_rectangle.png
           :width: 500
           :align: center

        :param Polygon shape: The Polygon for which the minimum rotated rectangle is to be calculated.
        :return Polygon: The minimum rotated rectangle geometry that bounds the input polygon.

        :example:
        >>> polygon = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
        >>> rectangle = GeometryMixin().minimum_rotated_rectangle(shape=polygon)
        """

        check_instance(
            source=GeometryMixin.minimum_rotated_rectangle.__name__,
            instance=shape,
            accepted_types=Polygon,
        )
        rotated_rectangle = shape.minimum_rotated_rectangle
        if isinstance(rotated_rectangle, Point):
            return Polygon([(0, 0), (0, 0), (0, 0)])
        else:
            return rotated_rectangle

    @staticmethod
    def length(shape: Union[LineString, MultiLineString],
               pixels_per_mm: float,
               unit: Literal["mm", "cm", "dm", "m"] = "mm") -> float:
        """
        Calculate the length of a LineString geometry.

        .. image:: _static/img/length.png
           :width: 400
           :align: center

        :param LineString shape: The LineString geometry for which the length is to be calculated.
        :param Literal['mm', 'cm', 'dm', 'm'] unit: The desired unit for the length measurement ('mm', 'cm', 'dm', 'm').
        :return float: The length of the LineString geometry in the specified unit.

        :example:
        >>> line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
        >>> GeometryMixin().length(shape=line_1, pixels_per_mm=1.0)
        >>> 50.6449510224598
        """

        check_float(name="line_length pixels_per_mm", value=pixels_per_mm, min_value=0)
        check_instance(source=GeometryMixin.length.__name__, instance=shape, accepted_types=LineString)
        L = shape.length / pixels_per_mm
        if unit == "cm":
            L = L / 10
        elif unit == "dm":
            L = L / 100
        elif unit == "m":
            L = L / 1000

        return L

    def multiframe_bodyparts_to_polygon(self,
                                        data: np.ndarray,
                                        video_name: Optional[str] = None,
                                        animal_name: Optional[str] = None,
                                        verbose: Optional[bool] = False,
                                        cap_style: Optional[Literal["round", "square", "flat"]] = "round",
                                        parallel_offset: Optional[int] = 1,
                                        pixels_per_mm: Optional[float] = None,
                                        simplify_tolerance: Optional[float] = 2,
                                        preserve_topology: bool = True,
                                        core_cnt: int = -1) -> List[Polygon]:
        """
        Convert multidimensional NumPy array representing body part coordinates to a list of Polygons.

        .. note::
           To convert single frame animal body-part coordinates to polygon, use ``simba.mixins.geometry_mixin.GeometryMixin.bodyparts_to_polygon``

        :param np.ndarray data: NumPy array of body part coordinates. Each subarray represents the coordinates of a body part.
        :param Literal['round', 'square', 'flat'] cap_style: Style of line cap for parallel offset. Options: 'round', 'square', 'flat'.
        :param int parallel_offset: Offset distance for parallel lines. Default is 1.
        :param float simplify_tolerance: Tolerance parameter for simplifying geometries. Default is 2.

        :example:
        >>> data = np.array([[[364, 308], [383, 323], [403, 335], [423, 351]],[[356, 307], [376, 319], [396, 331], [419, 347]]])
        >>> GeometryMixin().multiframe_bodyparts_to_polygon(data=data)
        """

        check_valid_array(
            data=data,
            source=GeometryMixin().multiframe_bodyparts_to_polygon.__name__,
            accepted_ndims=(3,),
            accepted_dtypes=(float, int, np.float32, np.float64, np.int32, np.int64),
        )
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if pixels_per_mm is not None:
            check_float(
                name="PIXELS PER MM",
                value=pixels_per_mm,
                min_value=0.1,
                raise_error=True,
            )
            parallel_offset = np.ceil(parallel_offset * pixels_per_mm)
            if parallel_offset < 1:
                parallel_offset = 1
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        data = np.array_split(data, core_cnt)

        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.bodyparts_to_polygon,
                parallel_offset=parallel_offset,
                cap_style=cap_style,
                simplify_tolerance=simplify_tolerance,
                preserve_topology=preserve_topology,
            )
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                if verbose:
                    if not video_name and not animal_name:
                        print(f"Computing polygon batch {cnt+1}/{len(data)}...")
                    elif not video_name and animal_name:
                        print(
                            f"Computing polygon batch {cnt + 1}/{len(data)} (Animal: {animal_name})..."
                        )
                    elif video_name and not animal_name:
                        print(
                            f"Computing polygon batch {cnt + 1}/{len(data)} (Video: {video_name})..."
                        )
                    else:
                        print(
                            f"Computing polygon batch {cnt + 1}/{len(data)} (Video: {video_name}, Animal: {animal_name})..."
                        )
                results.append(mp_return)
        #
        timer.stop_timer()
        stdout_success(msg="Polygons complete.", elapsed_time=timer.elapsed_time_str)
        pool.join()
        pool.terminate()
        return [l for ll in results for l in ll]

    @staticmethod
    def multiframe_bodypart_to_point(data: np.ndarray,
                                     core_cnt: Optional[int] = -1,
                                     buffer: Optional[int] = None,
                                     px_per_mm: Optional[int] = None) -> Union[List[Point], List[List[Point]]]:
        """
        Process multiple frames of body part data in parallel and convert them to shapely Points.

        This function takes a multi-frame body part data represented as an array and
        converts it into points. It utilizes multiprocessing for parallel processing.

        :param np.ndarray data: 2D or 3D array with body-part coordinates where rows are frames and columns are x and y coordinates.
        :param Optional[int] core_cnt: The number of cores to use. If -1, then all available cores.
        :param Optional[int] px_per_mm: Pixels ro millimeter convertion factor. Required if buffer is not None.
        :param Optional[int] buffer: If not None, then the area of the Point. Thus, if not None, then returns Polygons representing the Points.
        :param Optional[int] px_per_mm: Pixels to millimeter convertion factor. Required if buffer is not None.
        :returns Union[List[Point], List[List[Point]]]: If input is a 2D array, then list of Points. If 3D array, then list of list of Points.

        .. note::
           If buffer and px_per_mm is not None, then the points will be *buffered* and a 2D share polygon created with the specified buffered area.
           If buffer is provided, then also provide px_per_mm for accurate conversion factor between pixels and millimeters.

        :example:
        >>> data = np.random.randint(0, 100, (100, 2))
        >>> points_lst = GeometryMixin().multiframe_bodypart_to_point(data=data, buffer=10, px_per_mm=4)
        >>> data = np.random.randint(0, 100, (10, 10, 2))
        >>> point_lst_of_lst = GeometryMixin().multiframe_bodypart_to_point(data=data)
        """

        check_valid_array(
            data=data,
            accepted_dtypes=(np.int64, np.int32, np.int8),
            accepted_ndims=(2, 3),
        )
        check_int(
            name=GeometryMixin().multiframe_bodypart_to_point.__name__,
            value=core_cnt,
            min_value=-1,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results = []
        data_ndim = data.ndim
        if data_ndim == 2:
            data = np.array_split(data, core_cnt)
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.bodyparts_to_points, buffer=buffer, px_per_mm=px_per_mm
            )
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(result)
        pool.join()
        pool.terminate()
        if data_ndim == 2:
            return [i for s in results for i in s]
        else:
            return results

    def multiframe_bodyparts_to_circle(
        self,
        data: np.ndarray,
        parallel_offset: int = 1,
        core_cnt: int = -1,
        pixels_per_mm: Optional[int] = 1,
    ) -> List[Polygon]:
        """
        Convert a set of pose-estimated key-points to circles with specified radius using multiprocessing.

        :param np.ndarray data: The body-part coordinates xy as a 2d array where rows are frames and columns represent x and y coordinates . E.g., np.array([[364, 308], [369, 309]])
        :param int data: The radius of the resultant circle in millimeters.
        :param int core_cnt: Number of CPU cores to use. Defaults to -1 meaning all available cores will be used.
        :param int pixels_per_mm: The pixels per millimeter of the video. If not passed, 1 will be used meaning revert to radius in pixels rather than millimeters.
        :returns Polygon: List of shapely Polygons of circular shape of size data.shape[0].

        :example:
        >>> data = np.random.randint(0, 100, (100, 2))
        >>> circles = GeometryMixin().multiframe_bodyparts_to_circle(data=data)
        """

        timer = SimbaTimer(start=True)
        check_int( name="CORE COUNT", value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True,)
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(
                GeometryMixin.bodyparts_to_circle,
                parallel_offset=parallel_offset,
                pixels_per_mm=pixels_per_mm,
            )
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

        pool.join()
        pool.terminate()
        timer.stop_timer()
        stdout_success(
            msg="Multiframe body-parts to circle complete",
            source=GeometryMixin.multiframe_bodyparts_to_circle.__name__,
            elapsed_time=timer.elapsed_time_str,
        )
        return results

    @staticmethod
    def delaunay_triangulate_keypoints(data: np.ndarray) -> List[Polygon]:
        """
        Triangulates a set of 2D keypoints. E.g., use to polygonize animal hull, or triangulate a gridpoint areana.

        This method takes a 2D numpy array representing a set of keypoints and
        triangulates them using the Delaunay triangulation algorithm. The input
        array should have two columns corresponding to the x and y coordinates of
        the keypoints.

        .. image:: _static/img/delaunay_triangulate_keypoints.png
           :width: 400
           :align: center

        .. image:: _static/img/delaunay_triangulate_keypoints.gif
           :width: 450
           :align: center

        .. image:: _static/img/delaunay_triangulate_keypoints_2.png
           :width: 450
           :align: center

        :param np.ndarray data: NumPy array of body part coordinates. Each subarray represents the coordinates of a body part.
        :returns  List[Polygon]: A list of `Polygon` objects representing the triangles formed by the Delaunay triangulation.

        :example:
        >>> data = np.array([[126, 122],[152, 116],[136,  85],[167, 172],[161, 206],[197, 193],[191, 237]])
        >>> triangulated_hull = GeometryMixin().delaunay_triangulate_keypoints(data=data)
        """

        check_instance(
            source=GeometryMixin().delaunay_triangulate_keypoints.__name__,
            instance=data,
            accepted_types=np.ndarray,
        )
        if data.ndim != 2:
            raise InvalidInputError(
                msg=f"Triangulate requires 2D array, got {data.ndim}",
                source=GeometryMixin.delaunay_triangulate_keypoints.__name__,
            )
        return triangulate(MultiPoint(data.astype(np.int64)))

    def multiframe_bodyparts_to_line(self,
                                     data: np.ndarray,
                                     buffer: Optional[int] = None,
                                     px_per_mm: Optional[float] = None,
                                     core_cnt: Optional[int] = -1) -> List[LineString]:
        """
        Convert multiframe body-parts data to a list of LineString objects using multiprocessing.

        :param np.ndarray data: Input array representing multiframe body-parts data. It should be a 3D array with dimensions (frames, points, coordinates).
        :param Optional[int] buffer: If not None, then the linestring will be expanded into a 2D geometry polygon with area ``buffer``.
        :param Optional[int] px_per_mm: If ``buffer`` if not None, then provide the pixels to millimeter
        :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. If set to -1, the function will automatically determine the available core count.
        :return List[LineString]: A list of LineString objects representing the body-parts trajectories.

        :example:
        >>> data = np.random.randint(0, 100, (100, 2))
        >>> data = data.reshape(50,-1, data.shape[1])
        >>> lines = GeometryMixin().multiframe_bodyparts_to_line(data=data)
        """

        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        if data.ndim != 3:
            raise InvalidInputError(
                msg=f"Multiframe body-parts to linestring expects a 3D array, got {data.ndim}",
                source=GeometryMixin.bodyparts_to_line.__name__,
            )
        if buffer is not None:
            check_float(
                name=f"{GeometryMixin.bodyparts_to_points} buffer",
                value=buffer,
                min_value=1,
            )
            check_float(
                name=f"{GeometryMixin.bodyparts_to_points} px_per_mm",
                value=px_per_mm,
                min_value=1,
            )
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.bodyparts_to_line, buffer=buffer, px_per_mm=px_per_mm
            )
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(result)
        pool.join()
        pool.terminate()
        return results

    def multiframe_compute_pct_shape_overlap(self,
                                             shape_1: List[Polygon],
                                             shape_2: List[Polygon],
                                             core_cnt: Optional[int] = -1,
                                             video_name: Optional[str] = None,
                                             verbose: Optional[bool] = False,
                                             animal_names: Optional[Tuple[str]] = None,
                                             denominator: Optional[Literal["difference", "shape_1", "shape_2"]] = "difference") -> List[float]:
        """
        Compute the percentage overlap between corresponding Polygons in two lists.

        .. image:: _static/img/multiframe_compute_pct_shape_overlap.png
           :width: 600
           :align: center

        :param List[Polygon] shape_1: List of Polygons.
        :param List[Polygon] shape_2: List of Polygons with the same length as shape_1.
        :param int core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        :param Optional[str] video_name: If not None, then the name of the video being processed for interpretable progress msgs.
        :param Optional[bool] video_name: If True, then prints interpretable progress msgs.
        :param Optional[Tuple[str]] animal_names: If not None, then a two-tuple of animal names (or alternative shape names) interpretable progress msgs.

        :return List[float]: List of percentage overlap between corresponding Polygons.

        :example:
        """
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2):
            raise InvalidInputError(
                msg=f"shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}",
                source=GeometryMixin.multiframe_compute_pct_shape_overlap.__name__,
            )
        input_dtypes = list(
            set([type(x) for x in shape_1] + [type(x) for x in shape_2])
        )
        if len(input_dtypes) > 1:
            raise InvalidInputError(
                msg=f"shape_1 and shape_2 contains more than 1 dtype {input_dtypes}",
                source=GeometryMixin.multiframe_compute_pct_shape_overlap.__name__,
            )
        check_instance(
            source=GeometryMixin.multiframe_compute_pct_shape_overlap.__name__,
            instance=shape_1[0],
            accepted_types=(LineString, Polygon),
        )
        data, results, timer = (
            np.column_stack((shape_1, shape_2)),
            [],
            SimbaTimer(start=True),
        )
        data = np.array_split(data, core_cnt)
        pool = multiprocessing.Pool(
            processes=core_cnt,
            maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value,
        )
        constants = functools.partial(
            GeometryMixin.compute_pct_shape_overlap, denominator=denominator
        )
        for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
            if verbose:
                if not video_name and not animal_names:
                    print(f"Computing % overlap batch {cnt+1}/{len(data)}...")
                elif not video_name and animal_names:
                    print(
                        f"Computing % overlap batch {cnt + 1}/{len(data)} (Animals/Shapes: {animal_names})..."
                    )
                elif video_name and not animal_names:
                    print(
                        f"Computing % overlap batch {cnt + 1}/{len(data)} (Video: {video_name})..."
                    )
                else:
                    print(
                        f"Computing % overlap batch {cnt + 1}/{len(data)} (Video: {video_name}, Animals: {animal_names})..."
                    )
            results.append(result)
        timer.stop_timer()
        stdout_success(
            msg="Compute overlap complete.", elapsed_time=timer.elapsed_time_str
        )
        pool.close()
        pool.join()
        pool.terminate()
        return np.hstack(results)

    def multiframe_compute_shape_overlap(
        self,
        shape_1: List[Polygon],
        shape_2: List[Polygon],
        core_cnt: Optional[int] = -1,
        verbose: Optional[bool] = False,
        names: Optional[Tuple[str]] = None,
    ) -> List[int]:
        """
        Multiprocess compute overlap between corresponding Polygons in two lists.

        .. note::
           Only returns if two shapes are overlapping or not overlapping. If the amount of overlap is required, use
           ``GeometryMixin().multifrm_compute_pct_shape_overlap()``.

        :param List[Polygon] shape_1: List of Polygons.
        :param List[Polygon] shape_2: List of Polygons with the same length as shape_1.
        :param int core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        :return List[float]: List of overlap between corresponding Polygons. If overlap 1, else 0.
        """

        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2):
            raise InvalidInputError(
                msg=f"shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}",
                source=GeometryMixin.multiframe_compute_shape_overlap.__name__,
            )
        input_dtypes = list(
            set([type(x) for x in shape_1] + [type(x) for x in shape_2])
        )
        if len(input_dtypes) > 1:
            raise InvalidInputError(
                msg=f"shape_1 and shape_2 contains more than 1 dtype {input_dtypes}",
                source=GeometryMixin.multiframe_compute_shape_overlap.__name__,
            )
        check_instance(
            source=GeometryMixin.multiframe_compute_shape_overlap.__name__,
            instance=shape_1[0],
            accepted_types=(LineString, Polygon),
        )
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin.compute_shape_overlap, data, chunksize=1)
            ):
                if verbose:
                    if not names:
                        print(f"Computing overlap {cnt + 1}/{data.shape[0]}...")
                    else:
                        print(
                            f"Computing overlap {cnt + 1}/{data.shape[0]} (Shape 1: {names[0]}, Shape 2: {names[1]}, Video: {names[2]}...)"
                        )
                results.append(result)

        pool.join()
        pool.terminate()
        return results

    def multiframe_shape_distance(self,
                                  shape_1: List[Union[LineString, Polygon]],
                                  shape_2: List[Union[LineString, Polygon]],
                                  pixels_per_mm: float,
                                  unit: Literal["mm", "cm", "dm", "m"] = "mm",
                                  core_cnt=-1) -> List[float]:
        """
        Compute shape distances between corresponding shapes in two lists of LineString or Polygon geometries for multiple frames.

        :param List[Union[LineString, Polygon]] shape_1: List of LineString or Polygon geometries.
        :param List[Union[LineString, Polygon]] shape_2: List of LineString or Polygon geometries with the same length as shape_1.
        :param float pixels_per_mm: Conversion factor from pixels to millimeters.
        :param Literal['mm', 'cm', 'dm', 'm'] unit: Unit of measurement for the result. Options: 'mm', 'cm', 'dm', 'm'. Default: 'mm'.
        :param core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        :return List[float]: List of shape distances between corresponding shapes in passed unit.
        """

        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        check_float(name="PIXELS PER MM", value=pixels_per_mm, min_value=0.0)
        check_if_valid_input(name="UNIT", input=unit, options=["mm", "cm", "dm", "m"])
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2):
            raise InvalidInputError(msg=f"shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}", source=GeometryMixin.multiframe_shape_distance.__name__)
        check_float(name="pixels_per_mm", value=pixels_per_mm, min_value=0.0)
        data = [list(x) for x in zip(shape_1, shape_2)]
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.shape_distance, pixels_per_mm=pixels_per_mm, unit=unit
            )
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(result)

        pool.join()
        pool.terminate()
        return results

    def multiframe_minimum_rotated_rectangle(
        self,
        shapes: List[Polygon],
        video_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        animal_name: Optional[bool] = None,
        core_cnt: int = -1,
    ) -> List[Polygon]:
        """
        Compute the minimum rotated rectangle for each Polygon in a list using mutiprocessing.

        :param List[Polygon] shapes: List of Polygons.
        :param core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        """

        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin.minimum_rotated_rectangle, shapes, chunksize=1)
            ):
                if verbose:
                    if not video_name and not animal_name:
                        print(f"Rotating polygon {cnt+1}/{len(shapes)}...")
                    elif not video_name and animal_name:
                        print(
                            f"Rotating polygon {cnt + 1}/{len(shapes)} (Animal: {animal_name})..."
                        )
                    elif video_name and not animal_name:
                        print(
                            f"Rotating polygon {cnt + 1}/{len(shapes)} (Video: {video_name})..."
                        )
                    else:
                        print(
                            f"Rotating polygon {cnt + 1}/{len(shapes)} (Video: {video_name}, Animal: {animal_name})..."
                        )
                results.append(result)

        timer.stop_timer()
        stdout_success(
            msg="Rotated rectangles complete.", elapsed_time=timer.elapsed_time_str
        )
        pool.join()
        pool.terminate()
        return results

    @staticmethod
    @njit("(float32[:,:,:], float64[:])")
    def static_point_lineside(lines: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Determine the relative position (left vs right) of a static point with respect to multiple lines.

        .. image:: _static/img/static_point_lineside.png
           :width: 400
           :align: center

        .. note::
           Modified from `rayryeng <https://stackoverflow.com/a/62886424>`__.

        :param numpy.ndarray lines: An array of shape (N, 2, 2) representing N lines, where each line is defined by two points. The first point that denotes the beginning of the line, the second point denotes the end of the line.
        :param numpy.ndarray point: A 2-element array representing the coordinates of the static point.
        :return np.ndarray: An array of length N containing the results for each line. 2 if the point is on the right side of the line. 1 if the point is on the left side of the line. 0 if the point is on the line.

        :example:
        >>> line = np.array([[[25, 25], [25, 20]], [[15, 25], [15, 20]], [[15, 25], [50, 20]]]).astype(np.float32)
        >>> point = np.array([20, 0]).astype(np.float64)
        >>> GeometryMixin().static_point_lineside(lines=line, point=point)
        >>> [1. 2. 1.]
        """

        results = np.full((lines.shape[0]), np.nan)
        threshhold = 1e-9
        for i in prange(lines.shape[0]):
            v = (lines[i][1][0] - lines[i][0][0]) * (point[1] - lines[i][0][1]) - (lines[i][1][1] - lines[i][0][1]) * (point[0] - lines[i][0][0])
            if v >= threshhold:
                results[i] = 2
            elif v <= -threshhold:
                results[i] = 1
            else:
                results[i] = 0
        return results

    @staticmethod
    @njit("(float32[:,:,:], float32[:, :])")
    def point_lineside(lines: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Determine the relative position of a point (left vs right) with respect to a lines in each frame.

        .. image:: _static/img/point_lineside.png
           :width: 400
           :align: center

        :param numpy.ndarray lines: An array of shape (N, 2, 2) representing N lines, where each line is defined by two points. The first point that denotes the beginning of the line, the second point denotes the end of the line.
        :param numpy.ndarray point: An array of shape (N, 2) representing N points.
        :return np.ndarray: An array of length N containing the results for each line. 2 if the point is on the right side of the line. 1 if the point is on the left side of the line. 0 if the point is on the line.

        :example:
        >>> lines = np.array([[[25, 25], [25, 20]], [[15, 25], [15, 20]], [[15, 25], [50, 20]]]).astype(np.float32)
        >>> points = np.array([[20, 0], [15, 20], [90, 0]]).astype(np.float32)
        >>> GeometryMixin().point_lineside(lines=lines, points=points)
        >>> [1., 0., 1.]
        """
        results = np.full((lines.shape[0]), np.nan)
        threshhold = 1e-9
        for i in prange(lines.shape[0]):
            line, point = lines[i], points[i]
            v = (line[1][0] - line[0][0]) * (point[1] - line[0][1]) - (
                line[1][1] - line[0][1]
            ) * (point[0] - line[0][0])
            if v >= threshhold:
                results[i] = 2
            elif v <= -threshhold:
                results[i] = 1
            else:
                results[i] = 0
        return results

    @staticmethod
    @njit("(int64[:,:], int64[:])")
    def extend_line_to_bounding_box_edges(line_points: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
        """
        Jitted extend a line segment defined by two points to fit within a bounding box.

        .. image:: _static/img/extend_line_to_bounding_box_edges.png
           :width: 400
           :align: center

        :param np.ndarray line_points: Coordinates of the line segment's two points. Two rows and each row represents a point (x, y).
        :param np.ndarray bounding_box: Bounding box coordinates in the format (min_x, min_y, max_x, max_y).
        :returns np.ndarray: Intersection points where the extended line crosses the bounding box edges. The shape of the array is (2, 2), where each row represents a point (x, y).

        :example:
        >>> line_points = np.array([[25, 25], [45, 25]]).astype(np.float32)
        >>> bounding_box = np.array([0, 0, 50, 50]).astype(np.float32)
        >>> GeometryMixin().extend_line_to_bounding_box_edges(line_points, bounding_box)
        >>> [[ 0. 25.] [50. 25.]]
        """

        x1, y1 = line_points[0]
        x2, y2 = line_points[1]
        min_x, min_y, max_x, max_y = bounding_box

        if x1 == x2:
            intersection_points = np.array(
                [[x1, max(min_y, 0)], [x1, min(max_y, min_y)]]
            ).astype(np.float32)
        elif y1 == y2:
            intersection_points = np.array([[min_x, y1], [max_x, y1]]).astype(
                np.float32
            )
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            x_min_intersection = (min_y - intercept) / slope
            x_max_intersection = (max_y - intercept) / slope

            # x_min_intersection = np.clip(x_min_intersection, min_x, max_x)
            # x_max_intersection = np.clip(x_max_intersection, min_x, max_x)

            intersection_points = np.array(
                [[x_min_intersection, min_y], [x_max_intersection, max_y]]
            ).astype(np.float32)

        return intersection_points

    @staticmethod
    def line_split_bounding_box(intersections: np.ndarray, bounding_box: np.ndarray) -> GeometryCollection:
        """
        Split a bounding box into two parts using an extended line.

        .. note::
          Extended line can be found by body-parts using ``GeometryMixin().extend_line_to_bounding_box_edges``.

        .. image:: _static/img/line_split_bounding_box.png
           :width: 400
           :align: center

        .. image:: _static/img/extend_line_to_bounding_box_edge.gif
           :width: 450
           :align: center

        :param np.ndarray line_points: Intersection points where the extended line crosses the bounding box edges. The shape of the array is (2, 2), where each row represents a point (x, y).
        :param np.ndarray bounding_box: Bounding box coordinates in the format (min_x, min_y, max_x, max_y).
        :returns GeometryCollection: A collection of polygons resulting from splitting the bounding box with the extended line.

        :example:
        >>> line_points = np.array([[25, 25], [45, 25]]).astype(np.float32)
        >>> bounding_box = np.array([0, 0, 50, 50]).astype(np.float32)
        >>> intersection_points = GeometryMixin().extend_line_to_bounding_box_edges(line_points, bounding_box)
        >>> GeometryMixin().line_split_bounding_box(intersections=intersection_points, bounding_box=bounding_box)
        """

        extended_line = LineString(intersections)
        original_polygon = Polygon(
            [
                (bounding_box[0], bounding_box[1]),
                (bounding_box[2], bounding_box[1]),
                (bounding_box[2], bounding_box[3]),
                (bounding_box[0], bounding_box[3]),
            ]
        )

        return split(original_polygon, extended_line)

    def multiframe_length(
        self,
        shapes: List[Union[LineString, MultiLineString]],
        pixels_per_mm: float,
        core_cnt: int = -1,
        unit: Literal["mm", "cm", "dm", "m"] = "mm",
    ) -> List[float]:
        """
        :example:
        >>> data = np.random.randint(0, 100, (5000, 2))
        >>> data = data.reshape(2500,-1, data.shape[1])
        >>> lines = GeometryMixin().multiframe_bodyparts_to_line(data=data)
        >>> lengths = GeometryMixin().multiframe_length(shapes=lines, pixels_per_mm=1.0)
        """

        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        check_float(name="PIXELS PER MM", value=pixels_per_mm, min_value=0.0)
        check_if_valid_input(name="UNIT", input=unit, options=["mm", "cm", "dm", "m"])
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.length, pixels_per_mm=pixels_per_mm, unit=unit
            )
            for cnt, result in enumerate(pool.imap(constants, shapes, chunksize=1)):
                results.append(result)
        pool.join()
        pool.terminate()
        return results

    def multiframe_union(
        self, shapes: Iterable[Union[LineString, MultiLineString]], core_cnt: int = -1
    ) -> Iterable[Union[LineString, MultiLineString]]:
        """
        :example:
        >>> data_1 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
        >>> data_2 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
        >>> polygon_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_1)
        >>> polygon_2 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_2)
        >>> data = np.array([polygon_1, polygon_2]).T
        >>> unions = GeometryMixin().multiframe_union(shapes=data)
        """

        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin().union, shapes, chunksize=1)
            ):
                results.append(result)
        pool.join()
        pool.terminate()
        return results

    def multiframe_symmetric_difference(
        self, shapes: Iterable[Union[LineString, MultiLineString]], core_cnt: int = -1
    ):
        """
        Compute the symmetric differences between corresponding LineString or MultiLineString geometries usng multiprocessing.

        :example:
        >>> data_1 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
        >>> data_2 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
        >>> polygon_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_1)
        >>> polygon_2 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_2)
        >>> data = np.array([polygon_1, polygon_2]).T
        >>> symmetric_differences = GeometryMixin().multiframe_symmetric_difference(shapes=data)
        """
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin().symmetric_difference, shapes, chunksize=1)
            ):
                results.append(result)
        pool.join()
        pool.terminate()
        return results

    def multiframe_delaunay_triangulate_keypoints(
        self, data: np.ndarray, core_cnt: int = -1
    ) -> List[List[Polygon]]:
        """
        >>> data_path = '/Users/simon/Desktop/envs/troubleshooting/Rat_NOR/project_folder/csv/machine_results/08102021_DOT_Rat7_8(2).csv'
        >>> data = pd.read_csv(data_path, index_col=0).head(1000).iloc[:, 0:21]
        >>> data = data[data.columns.drop(list(data.filter(regex='_p')))]
        >>> animal_data = data.values.reshape(len(data), -1, 2).astype(int)
        >>> tri = GeometryMixin().multiframe_delaunay_triangulate_keypoints(data=animal_data)
        """

        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        check_instance(
            source=GeometryMixin().multiframe_delaunay_triangulate_keypoints.__name__,
            instance=data,
            accepted_types=np.ndarray,
        )
        if data.ndim != 3:
            raise InvalidInputError(
                msg=f"Multiframe delaunay triangulate keypointstriangulate keypoints expects a 3D array, got {data.ndim}",
                source=GeometryMixin.multiframe_delaunay_triangulate_keypoints.__name__,
            )
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(
                    GeometryMixin().delaunay_triangulate_keypoints, data, chunksize=1
                )
            ):
                results.append(result)

        pool.join()
        pool.terminate()
        return results

    def multiframe_difference(
        self,
        shapes: Iterable[Union[LineString, Polygon, MultiPolygon]],
        core_cnt: Optional[int] = -1,
        verbose: Optional[bool] = False,
        animal_names: Optional[str] = None,
        video_name: Optional[str] = None,
    ) -> List[Union[Polygon, MultiPolygon]]:
        """
        Compute the multi-frame difference for a collection of shapes using parallel processing.

        :param Iterable[Union[LineString, Polygon, MultiPolygon]] shapes: A collection of shapes, where each shape is a list containing two geometries.
        :param int core_cnt: The number of CPU cores to use for parallel processing. Default is -1, which automatically detects the available cores.
        :param Optional[bool] verbose: If True, print progress messages during computation. Default is False.
        :param Optional[str] animal_names: Optional string representing the names of animals for informative messages.
        :param Optional[str]video_name: Optional string representing the name of the video for informative messages.
        :return List[Union[Polygon, MultiPolygon]]: A list of geometries representing the multi-frame difference.
        """

        check_instance(
            source=f"{GeometryMixin().multiframe_difference.__name__} shapes",
            instance=shapes,
            accepted_types=list,
        )
        for i in shapes:
            check_instance(
                source=f"{GeometryMixin().multiframe_difference.__name__} shapes {i}",
                instance=i,
                accepted_types=list,
            )
            check_iterable_length(
                f"{GeometryMixin().multiframe_difference.__name__} shapes {i}",
                val=len(i),
                exact_accepted_length=2,
            )
            for j in i:
                check_instance(
                    source=f"{GeometryMixin().multiframe_difference.__name__} shapes",
                    instance=j,
                    accepted_types=(LineString, Polygon, MultiPolygon),
                )
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin().difference, shapes, chunksize=1)
            ):
                if verbose:
                    if not video_name and not animal_names:
                        print(
                            f"Computing geometry difference {cnt + 1}/{len(shapes)}..."
                        )
                    elif not video_name and animal_names:
                        print(
                            f"Computing geometry difference {cnt + 1}/{len(shapes)} (Animals: {animal_names})..."
                        )
                    elif video_name and not animal_names:
                        print(
                            f"Computing geometry difference {cnt + 1}/{len(shapes)} (Video: {video_name})..."
                        )
                    else:
                        print(
                            f"Computing geometry difference {cnt + 1}/{len(shapes)} (Video: {video_name}, Animals: {animal_names})..."
                        )
                results.append(result)

        timer.stop_timer()
        stdout_success(
            msg="Multi-frame difference compute complete",
            elapsed_time=timer.elapsed_time_str,
        )
        pool.join()
        pool.terminate()
        return results

    def multiframe_area(
        self,
        shapes: List[Union[MultiPolygon, Polygon]],
        pixels_per_mm: float,
        core_cnt: Optional[int] = -1,
        verbose: Optional[bool] = False,
        video_name: Optional[bool] = False,
        animal_names: Optional[bool] = False,
    ) -> np.ndarray:

        check_instance(
            source=f"{GeometryMixin().multiframe_area.__name__} shapes",
            instance=shapes,
            accepted_types=list,
        )
        for i in shapes:
            check_instance(
                source=f"{GeometryMixin().multiframe_difference.__name__} shapes {i}",
                instance=i,
                accepted_types=(MultiPolygon, Polygon),
            )
        check_float(
            name=f"{self.__class__.__name__} pixels_per_mm",
            value=pixels_per_mm,
            min_value=0.01,
        )
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.area, pixels_per_mm=pixels_per_mm
            )
            for cnt, result in enumerate(pool.imap(constants, shapes, chunksize=1)):
                if verbose:
                    if not video_name and not animal_names:
                        print(f"Computing area {cnt + 1}/{len(shapes)}...")
                    elif not video_name and animal_names:
                        print(
                            f"Computing % area {cnt + 1}/{len(shapes)} (Animals: {animal_names})..."
                        )
                    elif video_name and not animal_names:
                        print(
                            f"Computing % area {cnt + 1}/{len(shapes)} (Video: {video_name})..."
                        )
                    else:
                        print(
                            f"Computing % area {cnt + 1}/{len(shapes)} (Video: {video_name}, Animals: {animal_names})..."
                        )
                results.append(result)

        timer.stop_timer()
        stdout_success(
            msg="Multi-frame area compute complete", elapsed_time=timer.elapsed_time_str
        )
        pool.join()
        pool.terminate()
        return results

    def multiframe_bodyparts_to_multistring_skeleton(
        self,
        data_df: pd.DataFrame,
        skeleton: Iterable[str],
        core_cnt: Optional[int] = -1,
        verbose: Optional[bool] = False,
        video_name: Optional[bool] = False,
        animal_names: Optional[bool] = False,
    ) -> List[Union[LineString, MultiLineString]]:
        """
        Convert body parts to LineString skeleton representations in a videos using multiprocessing.

        :param pd.DataFrame data_df: Pose-estimation data.
        :param Iterable[str] skeleton: Iterable of body part pairs defining the skeleton structure. Eg., [['Center', 'Lat_left'], ['Center', 'Lat_right'], ['Center', 'Nose'], ['Center', 'Tail_base']]
        :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        :param Optional[bool] verbose: If True, print progress information during computation. Default is False.
        :param Optional[bool] video_name: If True, include video name in progress information. Default is False.
        :param Optional[bool] animal_names: If True, include animal names in progress information. Default is False.
        :return List[Union[LineString, MultiLineString]]: List of LineString or MultiLineString objects representing the computed skeletons.

        :example:
        >>> df = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/Rat_NOR/project_folder/csv/machine_results/08102021_DOT_Rat7_8(2).csv', nrows=500).fillna(0).astype(int)
        >>> skeleton = [['Center', 'Lat_left'], ['Center', 'Lat_right'], ['Center', 'Nose'], ['Center', 'Tail_base'], ['Lat_left', 'Tail_base'], ['Lat_right', 'Tail_base'], ['Nose', 'Ear_left'], ['Nose', 'Ear_right'], ['Ear_left', 'Lat_left'], ['Ear_right', 'Lat_right']]
        >>> geometries = GeometryMixin().multiframe_bodyparts_to_multistring_skeleton(data_df=df, skeleton=skeleton, core_cnt=2, verbose=True)
        """

        timer = SimbaTimer(start=True)
        check_instance(
            source=f"{GeometryMixin().multiframe_bodyparts_to_multistring_skeleton.__name__} data",
            instance=data_df,
            accepted_types=pd.DataFrame,
        )
        for i in skeleton:
            check_instance(
                source=f"{GeometryMixin().multiframe_bodyparts_to_multistring_skeleton.__name__} skeleton {i}",
                instance=i,
                accepted_types=(
                    list,
                    tuple,
                ),
            )
            check_iterable_length(
                source=f"{GeometryMixin().multiframe_bodyparts_to_multistring_skeleton.__name__} skeleton",
                val=len(i),
                exact_accepted_length=2,
            )
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        skeleton_data, results = None, []
        for node_cnt, nodes in enumerate(skeleton):
            bp_1, bp_2 = (
                data_df[[f"{nodes[0]}_x", f"{nodes[0]}_y"]].values,
                data_df[[f"{nodes[1]}_x", f"{nodes[1]}_y"]].values,
            )
            line = np.hstack((bp_1, bp_2)).reshape(-1, 2, 2)
            if node_cnt == 0:
                skeleton_data = deepcopy(line)
            else:
                skeleton_data = np.concatenate((skeleton_data, line), axis=1)
        skeleton_data = skeleton_data.reshape(len(data_df), len(skeleton), 2, -1)
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(
                    GeometryMixin.bodyparts_to_multistring_skeleton,
                    skeleton_data,
                    chunksize=1,
                )
            ):
                if verbose:
                    if not video_name and not animal_names:
                        print(f"Computing skeleton {cnt + 1}/{len(data_df)}...")
                    elif not video_name and animal_names:
                        print(
                            f"Computing skeleton {cnt + 1}/{len(data_df)} (Animals: {animal_names})..."
                        )
                    elif video_name and not animal_names:
                        print(
                            f"Computing skeleton {cnt + 1}/{len(data_df)} (Video: {video_name})..."
                        )
                    else:
                        print(
                            f"Computing skeleton {cnt + 1}/{len(data_df)} (Video: {video_name}, Animals: {animal_names})..."
                        )
                results.append(result)

        timer.stop_timer()
        stdout_success(
            msg="Multistring skeleton complete.",
            elapsed_time=timer.elapsed_time_str,
            source=self.__class__.__name__,
        )
        return results

    @staticmethod
    def get_geometry_brightness_intensity(img: Union[np.ndarray, Tuple[cv2.VideoCapture, int]],
                                          geometries: List[Union[np.ndarray, Polygon]],
                                          ignore_black: Optional[bool] = True) -> np.ndarray:
        """
        Calculate the average brightness intensity within a geometry region-of-interest of an image.

        E.g., can be used with hardcoded thresholds or model kmeans in `simba.mixins.statistics_mixin.Statistics.kmeans_1d` to detect if a light source is ON or OFF state.

        .. image:: _static/img/get_geometry_brightness_intensity.png
           :width: 500
           :align: center

        :param np.ndarray img: Either an image in numpy array format OR a tuple with cv2.VideoCapture object and the frame index.
        :param List[Union[Polygon, np.ndarray]] geometries: A list of shapes either as vertices in a numpy array, or as shapely Polygons.
        :param Optional[bool] ignore_black: If non-rectangular geometries, then pixels that don't belong to the geometry are masked in black. If True, then these pixels will be ignored when computing averages.

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/1.png').astype(np.uint8)
        >>> data_path = '/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv'
        >>> data = pd.read_csv(data_path, usecols=['Nose_x', 'Nose_y']).sample(n=3).fillna(1).values.astype(np.int64)
        >>> geometries = []
        >>> for frm_data in data: geometries.append(GeometryMixin().bodyparts_to_circle(frm_data, 100))
        >>> GeometryMixin().get_geometry_brightness_intensity(img=img, geometries=geometries, ignore_black=False)
        >>> [125.0, 113.0, 118.0]
        """

        check_instance(
            source=f"{GeometryMixin().get_geometry_brightness_intensity.__name__} img",
            instance=img,
            accepted_types=(tuple, np.ndarray),
        )
        check_instance(
            source=f"{GeometryMixin().get_geometry_brightness_intensity.__name__} geometries",
            instance=geometries,
            accepted_types=list,
        )
        for geom_cnt, geometry in enumerate(geometries):
            check_instance(
                source=f"{GeometryMixin().get_geometry_brightness_intensity.__name__} geometry {geom_cnt}",
                instance=geometry,
                accepted_types=(Polygon, np.ndarray),
            )
        sliced_imgs = ImageMixin().slice_shapes_in_img(img=img, geometries=geometries)
        return ImageMixin().brightness_intensity(
            imgs=sliced_imgs, ignore_black=ignore_black
        )

    @staticmethod
    def geometry_histocomparison(
        imgs: List[Union[np.ndarray, Tuple[cv2.VideoCapture, int]]],
        geometry: Polygon = None,
        method: Optional[
            Literal[
                "chi_square",
                "correlation",
                "intersection",
                "bhattacharyya",
                "hellinger",
                "chi_square_alternative",
                "kl_divergence",
            ]
        ] = "correlation",
        absolute: Optional[bool] = True,
    ) -> float:
        """
        Retrieve histogram similarities within a geometry inside two images.

        For example, the polygon may represent an area around a rodents head. While the front paws are not pose-estimated, computing the histograms of the geometry in two sequential images gives indication of non-freezing.

        .. note::
           If shapes is None, the entire two images passed as ``imgs`` will be compared.

           `Documentation <https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#gga994f53817d621e2e4228fc646342d386ad75f6e8385d2e29479cf61ba87b57450>`__.

        .. important::
           If there is non-pose related noise in the environment (e.g., there are non-experiment related light sources that goes on and off, or waving window curtains causing changes in histgram values w/o affecting pose) this will negatively affect the realiability of histogram comparisons.

        .. image:: _static/img/geometry_histocomparison.png
           :width: 700
           :align: center

        :parameter List[Union[np.ndarray, Tuple[cv2.VideoCapture, int]]] imgs: List of two input images. Can be either an two image in numpy array format OR a two tuples with cv2.VideoCapture object and the frame index.
        :parameter Optional[Polygon] geometry: If Polygon, then the geometry in the two images that should be compared. If None, then entire images will be histocompared.
        :parameter Literal['correlation', 'chi_square'] method: The method used for comparison. E.g., if `correlation`, then small output values suggest large differences between the current versus prior image. If `chi_square`, then large output values  suggest large differences between the geometries.
        :parameter Optional[bool] absolute: If True, the absolute difference between the two histograms. If False, then (image2 histogram) - (image1 histogram)
        :return float: Value representing the histogram similarities between the geometry in the two images.

        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/1.png')
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/2.png')
        >>> data_path = '/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv'
        >>> data = pd.read_csv(data_path, nrows=1, usecols=['Nose_x', 'Nose_y']).fillna(-1).values.astype(np.int64)
        >>> polygon = GeometryMixin().bodyparts_to_circle(data[0], 100)
        >>> GeometryMixin().geometry_histocomparison(imgs=[img_1, img_2], geometry=polygon, method='correlation')
        >>> 0.9999769684923543
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/41411.png')
        >>> GeometryMixin().geometry_histocomparison(imgs=[img_1, img_2], geometry=polygon, method='correlation')
        >>> 0.6732792208872572
        >>> img_1 = (cv2.VideoCapture('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4'), 1)
        >>> img_2 = (cv2.VideoCapture('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4'), 2)
        >>> GeometryMixin().geometry_histocomparison(imgs=[img_1, img_2], geometry=polygon, method='correlation')
        >>> 0.9999769684923543
        """
        check_instance(
            source=f"{GeometryMixin().geometry_histocomparison.__name__} imgs",
            instance=imgs,
            accepted_types=list,
        )
        check_iterable_length(
            f"{GeometryMixin().geometry_histocomparison.__name__} imgs",
            val=len(imgs),
            min=2,
            max=2,
        )
        check_str(
            name=f"{GeometryMixin().geometry_histocomparison.__name__} method",
            value=method,
            options=list(GeometryEnum.HISTOGRAM_COMPARISON_MAP.value.keys()),
        )
        corrected_imgs = []
        for i in range(len(imgs)):
            check_instance(
                source=f"{GeometryMixin().geometry_histocomparison.__name__} imgs {i}",
                instance=imgs[i],
                accepted_types=(np.ndarray, tuple),
            )
            if isinstance(imgs[i], tuple):
                check_iterable_length(
                    f"{GeometryMixin().geometry_histocomparison.__name__} imgs {i}",
                    val=len(imgs),
                    min=2,
                    max=2,
                )
                check_instance(
                    source=f"{GeometryMixin().geometry_histocomparison.__name__} imgs {i} 0",
                    instance=imgs[i][0],
                    accepted_types=cv2.VideoCapture,
                )
                corrected_imgs.append(
                    read_frm_of_video(video_path=imgs[i][0], frame_index=imgs[i][1])
                )
            else:
                corrected_imgs.append(imgs[i])
        imgs = corrected_imgs
        del corrected_imgs
        if geometry is not None:
            sliced_imgs = []
            check_instance(
                source=f"{GeometryMixin().geometry_histocomparison.__name__} geometry",
                instance=geometry,
                accepted_types=Polygon,
            )
            for img in imgs:
                sliced_imgs.append(
                    ImageMixin().slice_shapes_in_img(img=img, geometries=[geometry])[0]
                )
            imgs = sliced_imgs
            del sliced_imgs
        return ImageMixin().get_histocomparison(
            img_1=imgs[0], img_2=imgs[1], method=method, absolute=absolute
        )

    def multiframe_is_shape_covered(
        self,
        shape_1: List[Polygon],
        shape_2: List[Polygon],
        core_cnt: Optional[int] = -1,
    ) -> List[bool]:
        """
        For each shape in time-series of shapes, check if another shape in the same time-series fully covers the
        first shape.

        .. image:: _static/img/multiframe_is_shape_covered.png
           :width: 600
           :align: center


        :example:
        >>> shape_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=np.random.randint(0, 200, (100, 6, 2)))
        >>> shape_2 = [Polygon([[0, 0], [20, 20], [20, 10], [10, 20]]) for x in range(len(shape_1))]
        >>> GeometryMixin.multiframe_is_shape_covered(shape_1=shape_1, shape_2=shape_2, core_cnt=3)
        """
        check_valid_lst(
            data=shape_1,
            source=GeometryMixin.multiframe_is_shape_covered.__name__,
            valid_dtypes=(
                LineString,
                Polygon,
                MultiPolygon,
            ),
        )
        check_valid_lst(
            data=shape_2,
            source=GeometryMixin.multiframe_is_shape_covered.__name__,
            valid_dtypes=(
                LineString,
                Polygon,
                MultiPolygon,
            ),
        )
        if len(shape_1) != len(shape_2):
            raise InvalidInputError(
                msg=f"shape_1 ({len(shape_1)}) and shape_2 ({len(shape_2)}) are unequal length",
                source=GeometryMixin.multiframe_is_shape_covered.__name__,
            )
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        shapes = [list(x) for x in zip(shape_1, shape_2)]
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, mp_return in enumerate(
                pool.imap(GeometryMixin.is_shape_covered, shapes, chunksize=1)
            ):
                results.append(mp_return)
        pool.join()
        pool.terminate()
        return results

    @staticmethod
    def geometry_contourcomparison(
        imgs: List[Union[np.ndarray, Tuple[cv2.VideoCapture, int]]],
        geometry: Optional[Polygon] = None,
        method: Optional[Literal["all", "exterior"]] = "all",
        canny: Optional[bool] = True,
    ) -> float:
        """
        Compare contours between a geometry in two images using shape matching.

        .. image:: _static/img/geometry_contourcomparison.png
           :width: 700
           :align: center

        .. important::
           If there is non-pose related noise in the environment (e.g., there are non-experiment related intermittant light or shade sources that goes on and off, this will negatively affect the reliability of contour comparisons.

           Used to pick up very subtle changes around pose-estimated body-part locations.

        :parameter List[Union[np.ndarray, Tuple[cv2.VideoCapture, int]]] imgs: List of two input images. Can be either be two images in numpy array format OR a two tuples with cv2.VideoCapture object and the frame index.
        :parameter Optional[Polygon] geometry: If Polygon, then the geometry in the two images that should be compared. If None, then entire images will be contourcompared.
        :parameter Literal['all', 'exterior'] method: The method used for contour comparison.
        :parameter Optional[bool] canny: If True, applies Canny edge detection before contour comparison. Helps reduce noise and enhance contours.  Default is True.
        :returns float: Contour matching score between the two images. Lower scores indicate higher similarity.


        :example:
        >>> img_1 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/1978.png').astype(np.uint8)
        >>> img_2 = cv2.imread('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1_frames/1977.png').astype(np.uint8)
        >>> data = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv', nrows=1, usecols=['Nose_x', 'Nose_y']).fillna(-1).values.astype(np.int64)
        >>> geometry = GeometryMixin().bodyparts_to_circle(data[0, :], 100)
        >>> GeometryMixin().geometry_contourcomparison(imgs=[img_1, img_2], geometry=geometry, canny=True, method='exterior')
        >>> 22.54
        """

        check_instance(
            source=f"{GeometryMixin().geometry_contourcomparison.__name__} imgs",
            instance=imgs,
            accepted_types=list,
        )
        check_iterable_length(
            f"{GeometryMixin().geometry_contourcomparison.__name__} imgs",
            val=len(imgs),
            min=2,
            max=2,
        )
        check_str(
            name=f"{GeometryMixin().geometry_contourcomparison.__name__} method",
            value=method,
            options=list(GeometryEnum.CONTOURS_MAP.value.keys()),
        )
        corrected_imgs = []
        for i in range(len(imgs)):
            check_instance(
                source=f"{GeometryMixin().geometry_contourcomparison.__name__} imgs {i}",
                instance=imgs[i],
                accepted_types=(np.ndarray, tuple),
            )
            if isinstance(imgs[i], tuple):
                check_iterable_length(
                    f"{GeometryMixin().geometry_contourcomparison.__name__} imgs {i}",
                    val=len(imgs),
                    min=2,
                    max=2,
                )
                check_instance(
                    source=f"{GeometryMixin().geometry_contourcomparison.__name__} imgs {i} 0",
                    instance=imgs[i][0],
                    accepted_types=cv2.VideoCapture,
                )
                corrected_imgs.append(
                    read_frm_of_video(video_path=imgs[i][0], frame_index=imgs[i][1])
                )
            else:
                corrected_imgs.append(imgs[i])
        imgs = corrected_imgs
        del corrected_imgs
        if geometry is not None:
            sliced_imgs = []
            check_instance(
                source=f"{GeometryMixin().geometry_contourcomparison.__name__} geometry",
                instance=geometry,
                accepted_types=Polygon,
            )
            for img in imgs:
                sliced_imgs.append(
                    ImageMixin().slice_shapes_in_img(img=img, geometries=[geometry])[0]
                )
            imgs = sliced_imgs
            del sliced_imgs

        return ImageMixin().get_contourmatch(
            img_1=imgs[0], img_2=imgs[1], canny=canny, method=method
        )

    @staticmethod
    def _multifrm_geometry_histocomparison_helper(frm_index: np.ndarray,
                                                  data: np.ndarray,
                                                  video_path: cv2.VideoCapture,
                                                  shape_type: Literal["rectangle", "circle"],
                                                  pixels_per_mm: int,
                                                  parallel_offset: int):

        """Multi-proessing helper for ``multifrm_geometry_histocomparison``"""

        cap = cv2.VideoCapture(video_path)
        results = []
        for frm_range_idx in range(frm_index.shape[0]):
            frm_range = frm_index[frm_range_idx]
            print(f"Analyzing frame {frm_range[1]}...")
            img_1 = read_frm_of_video(video_path=cap, frame_index=frm_range[0])
            img_2 = read_frm_of_video(video_path=cap, frame_index=frm_range[1])
            loc = data[frm_range[0] : frm_range[1], :]
            if shape_type == "circle":
                shape_1 = GeometryMixin().bodyparts_to_circle(
                    data=loc[0],
                    pixels_per_mm=pixels_per_mm,
                    parallel_offset=parallel_offset,
                )
                shape_2 = GeometryMixin().bodyparts_to_circle(
                    data=loc[1],
                    pixels_per_mm=pixels_per_mm,
                    parallel_offset=parallel_offset,
                )
            elif shape_type == "rectangle":
                loc = loc.reshape(2, int(loc.shape[1] / 2), 2)
                shape_1 = GeometryMixin().bodyparts_to_polygon(
                    data=loc[0],
                    parallel_offset=parallel_offset,
                    pixels_per_mm=pixels_per_mm,
                )
                shape_2 = GeometryMixin().bodyparts_to_polygon(
                    data=loc[1],
                    parallel_offset=parallel_offset,
                    pixels_per_mm=pixels_per_mm,
                )
            else:
                loc = loc.reshape(2, int(loc.shape[1] / 2), 2)
                shape_1 = GeometryMixin().bodyparts_to_line(
                    data=loc[0], buffer=parallel_offset, px_per_mm=pixels_per_mm
                )
                shape_2 = GeometryMixin().bodyparts_to_line(
                    data=loc[1], buffer=parallel_offset, px_per_mm=pixels_per_mm
                )
            intersection_shape = shape_1.intersection(shape_2)
            img_1 = (
                ImageMixin()
                .slice_shapes_in_img(img=img_1, geometries=[intersection_shape])[0]
                .astype(np.uint8)
            )
            img_2 = (
                ImageMixin()
                .slice_shapes_in_img(img=img_2, geometries=[intersection_shape])[0]
                .astype(np.uint8)
            )
            results.append(ImageMixin().get_histocomparison(img_1=img_1, img_2=img_2))
        return results

    def multifrm_geometry_histocomparison(self,
                                          video_path: Union[str, os.PathLike],
                                          data: np.ndarray,
                                          shape_type: Literal["rectangle", "circle", "line"],
                                          lag: Optional[int] = 2,
                                          core_cnt: Optional[int] = -1,
                                          pixels_per_mm: int = 1,
                                          parallel_offset: int = 1) -> np.ndarray:

        """
        Perform geometry histocomparison on multiple video frames using multiprocessing.

        .. note::
           Comparions are made using the intersections of the two image geometries, meaning that the same
           experimental area of the image and arena is used in the comparison and shifts in animal location cannot account for variability.

        :param Union[str, os.PathLike] video_path: Path to the video file.
        :param np.ndarray data: Input data, typically containing coordinates of one or several body-parts.
        :param Literal['rectangle', 'circle'] shape_type: Type of shape for comparison.
        :param Optional[int] lag: Number of frames to lag between comparisons. Default is 2.
        :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. Default is -1 which is all available cores.
        :param Optional[int] pixels_per_mm: Pixels per millimeter for conversion. Default is 1.
        :param Optional[int] parallel_offset: Size of the geometry ROI in millimeters. Default 1.
        :returns np.ndarray: The difference between the successive geometry histograms.

        :example:
        >>> data = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_1.csv', nrows=2100, usecols=['Nose_x', 'Nose_y']).fillna(-1).values.astype(np.int64)
        >>> results = GeometryMixin().multifrm_geometry_histocomparison(video_path='/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4', data= data, shape_type='circle', pixels_per_mm=1, parallel_offset=100)
        >>> data = pd.read_csv('/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/csv/outlier_corrected_movement_location/Example_2.csv', nrows=2100, usecols=['Nose_x', 'Nose_y', 'Tail_base_x' , 'Tail_base_y', 'Center_x' , 'Center_y']).fillna(-1).values.astype(np.int64)
        >>> results = GeometryMixin().multifrm_geometry_histocomparison(video_path='/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/videos/Example_1.mp4', data= data, shape_type='rectangle', pixels_per_mm=1, parallel_offset=1)
        """

        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        split_frm_idx = np.full((data.shape[0] - (lag - 1), 2), -1)
        for cnt, i in enumerate(range(lag, data.shape[0] + 1, 1)):
            split_frm_idx[cnt] = [i - 2, i]
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        chunk_size = len(split_frm_idx) // core_cnt
        remainder = len(split_frm_idx) % core_cnt
        split_frm_idx = [
            split_frm_idx[
                i * chunk_size
                + min(i, remainder) : (i + 1) * chunk_size
                + min(i + 1, remainder)
            ]
            for i in range(core_cnt)
        ]
        results = [[0] * lag]
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin()._multifrm_geometry_histocomparison_helper,
                video_path=video_path,
                data=data,
                shape_type=shape_type,
                pixels_per_mm=pixels_per_mm,
                parallel_offset=parallel_offset,
            )
            for cnt, result in enumerate(
                pool.imap(constants, split_frm_idx, chunksize=1)
            ):
                results.append(result)

        return [item for sublist in results for item in sublist]

    @staticmethod
    def rank_shapes(
        shapes: List[Polygon],
        method: Literal[
            "area",
            "min_distance",
            "max_distance",
            "mean_distance",
            "left_to_right",
            "top_to_bottom",
        ],
        deviation: Optional[bool] = False,
        descending: Optional[bool] = True,
    ) -> List[Polygon]:
        """
        Rank a list of polygon geometries based on a specified method. E.g., order the list of geometries according to sizes or distances to each other or from left to right etc.

        :param List[Polygon] shapes: List of Shapely polygons to be ranked. List has to contain two or more shapes.
        :param Literal['area', 'min_center_distance', 'max_center_distance', 'mean_shape_distance'] method: The ranking method to use.
        :param Optional[bool] deviation: If True, rank based on absolute deviation from the mean. Default: False.
        :param Optional[bool] descending: If True, rank in descending order; otherwise, rank in ascending order. Default: False.
        :return: A input list of Shapely polygons sorted according to the specified ranking method.
        """

        check_instance(
            source=GeometryMixin().rank_shapes.__name__,
            instance=shapes,
            accepted_types=list,
        )
        check_iterable_length(
            source=GeometryMixin().rank_shapes.__name__, val=len(shapes), min=2
        )
        for i, shape in enumerate(shapes):
            check_instance(
                source=f"{GeometryMixin().rank_shapes.__name__} {i}",
                instance=shape,
                accepted_types=Polygon,
            )
        check_if_valid_input(
            name=f"{GeometryMixin().rank_shapes.__name__} method",
            input=method,
            options=GeometryEnum.RANKING_METHODS.value,
        )
        ranking_vals = {}
        if method == "area":
            for shp_cnt, shape in enumerate(shapes):
                ranking_vals[shp_cnt] = int(shape.area)
        elif method == "min_center_distance":
            for shp_cnt_1, shape_1 in enumerate(shapes):
                shape_1_loc, shape_min_distance = shape_1.centroid, np.inf
                for shp_cnt_2, shape_2 in enumerate(shapes):
                    if not shape_2.equals(shape_1):
                        shape_min_distance = min(
                            shape_1.centroid.distance(shape_2.centroid),
                            shape_min_distance,
                        )
                ranking_vals[shp_cnt_1] = shape_min_distance
        elif method == "max_distance":
            for shp_cnt_1, shape_1 in enumerate(shapes):
                shape_1_loc, shape_min_distance = shape_1.centroid, -np.inf
                for shp_cnt_2, shape_2 in enumerate(shapes):
                    if not shape_2.equals(shape_1):
                        shape_min_distance = max(
                            shape_1.centroid.distance(shape_2.centroid),
                            shape_min_distance,
                        )
                ranking_vals[shp_cnt_1] = shape_min_distance
        elif method == "mean_distance":
            for shp_cnt_1, shape_1 in enumerate(shapes):
                shape_1_loc, shape_distances = shape_1.centroid, []
                for shp_cnt_2, shape_2 in enumerate(shapes):
                    if not shape_2.equals(shape_1):
                        shape_distances.append(
                            shape_1.centroid.distance(shape_2.centroid)
                        )
                ranking_vals[shp_cnt_1] = np.mean(shape_distances)
        elif method == "left_to_right":
            for shp_cnt, shape in enumerate(shapes):
                ranking_vals[shp_cnt] = np.array(shape.centroid)[0]
        elif method == "top_to_bottom":
            for shp_cnt, shape in enumerate(shapes):
                ranking_vals[shp_cnt] = np.array(shape.centroid)[1]
        if deviation:
            new_ranking_vals, m = {}, sum(ranking_vals.values()) / len(ranking_vals)
            for k, v in ranking_vals.items():
                new_ranking_vals[k] = abs(v - m)
            ranking_vals = new_ranking_vals

        ranked = sorted(ranking_vals, key=ranking_vals.get, reverse=descending)
        return [shapes[idx] for idx in ranked]

    @staticmethod
    def contours_to_geometries(contours: List[np.ndarray], force_rectangles: Optional[bool] = True) -> List[Polygon]:
        """
        Convert a list of contours to a list of geometries.

        E.g., convert a list of contours detected with ``ImageMixin.find_contours`` to a list of Shapely geometries
        that can be used within the ``GeometryMixin``.

        :param List[np.ndarray] contours: List of contours represented as 2D arrays.
        :param force_rectangles: If True, then force the resulting geometries to be rectangular.
        :return List[Polygon]: List of Shapley Polygons.

        :example:
        >>> video_frm = read_frm_of_video(video_path='/Users/simon/Desktop/envs/platea_featurizer/data/video/3D_Mouse_5-choice_MouseTouchBasic_s9_a4_grayscale.mp4')
        >>> contours = ImageMixin.find_contours(img=video_frm)
        >>> GeometryMixin.contours_to_geometries(contours=contours)
        """

        check_instance(
            source=GeometryMixin.contours_to_geometries.__name__,
            instance=contours,
            accepted_types=(list,),
        )
        for i in contours:
            check_instance(
                source=f"{GeometryMixin.contours_to_geometries.__name__} {i}",
                instance=i,
                accepted_types=(np.ndarray,),
            )
        results = []
        for contour in contours:
            if contour.ndim == 3:
                contour = contour.reshape(contour.shape[0], 2)
            polygon = GeometryMixin.bodyparts_to_polygon(data=contour)
            if force_rectangles:
                polygon = GeometryMixin.minimum_rotated_rectangle(shape=polygon)
            results.append(polygon)
        return results

    @staticmethod
    def adjust_geometry_locations(geometries: List[Polygon],
                                  shift: Tuple[int, int],
                                  pixels_per_mm: Optional[float] = None,
                                  minimum: Optional[Tuple[int, int]] = (0, 0),
                                  maximum: Optional[Tuple[int, int]] = (np.inf, np.inf)) -> List[Polygon]:
        """
        Shift a set of geometries specified distance in the x and/or y-axis.

        .. image:: _static/img/adjust_geometry_locations.png
           :width: 600
           :align: center

        :param  List[Polygon] geometries: List of input polygons to be adjusted.
        :param Tuple[int, int] shift: Tuple specifying the shift distances in the x and y-axis. Interpreted as pixels if ``pixels_per_mm`` is None. Else interpreted as millimeter.
        :param float pixels_per_mm: Pixel per millimeter conversion factor.
        :param Optional[Tuple[int, int]] minimum: Minimim allowed coordinates of Polygon points on x and y axes. Default: (0,0).
        :param Optional[Tuple[int, int]] maximum: Maximum allowed coordinates of Polygon points on x and y axes. Default: (np.inf, np.inf).
        :return List[Polygon]: List of adjusted polygons.

        :example:
        >>> shapes = GeometryMixin().adjust_geometry_locations(geometries=shapes, shift=(0, 333))
        """

        check_valid_tuple(x=shift, source=f"{GeometryMixin.adjust_geometry_locations.__name__} shift", accepted_lengths=(2,), valid_dtypes=(int,))
        check_valid_tuple(x=shift, source=f"{GeometryMixin.adjust_geometry_locations.__name__} minimum", accepted_lengths=(2,), valid_dtypes=(int,))
        check_valid_tuple(x=shift, source=f"{GeometryMixin.adjust_geometry_locations.__name__} maximum", accepted_lengths=(2,), valid_dtypes=(int,))
        check_valid_lst(data=geometries, source=f"{GeometryMixin.adjust_geometry_locations.__name__} geometries", valid_dtypes=(Polygon,), min_len=1)
        if pixels_per_mm is not None:
            check_float(name='pixels_per_mm', value=pixels_per_mm, min_value=10e-6)
            shift = (max(0, int(shift[0] / pixels_per_mm)), (max(0, int(shift[1] / pixels_per_mm))))
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

    @staticmethod
    def bucket_img_into_grid_points(point_distance: int,
                                    px_per_mm: float,
                                    img_size: Tuple[int, int],
                                    border_sites: Optional[bool] = True) -> Dict[Tuple[int, int], Point]:

        """
        Generate a grid of evenly spaced points within an image. Use for creating spatial markers within an arena.

        .. image:: _static/img/bucket_img_into_grid_points.png
           :width: 800
           :align: center

        :param int point_distance: Distance between adjacent points in millimeters.
        :param float px_per_mm: Pixels per millimeter conversion factor.
        :param Tuple[int, int] img_size: Size of the image in pixels (width, height).
        :param Optional[bool] border_sites: If True, includes points on the border of the image. Default is True.
        :returns Dict[Tuple[int, int], Point]: Dictionary where keys are (row, column) indices of the point, and values are Shapely Point objects.

        :example:
        >>> GeometryMixin.bucket_img_into_grid_points(point_distance=20, px_per_mm=4, img_size=img.shape, border_sites=False)
        """

        point_distance = int(point_distance * px_per_mm)
        v_bin_cnt, h_bin_cnt = divmod(img_size[0], point_distance), divmod(img_size[1], point_distance)
        if h_bin_cnt[1] != 0:
            h_bin_cnt = (h_bin_cnt[0] + 1, h_bin_cnt[1])
        if v_bin_cnt[1] != 0:
            v_bin_cnt = (v_bin_cnt[0] + 1, v_bin_cnt[1])

        points = {}
        for h_cnt, i in enumerate(range(h_bin_cnt[0] + 1)):
            for v_cnt, j in enumerate(range(v_bin_cnt[0] + 1)):
                x, y = i * point_distance, j * point_distance
                x, y = min(x, img_size[1]), min(y, img_size[0])
                if not border_sites and (
                    (x == 0) or (y == 0) or (y == img_size[0]) or (x == img_size[1])
                ):
                    continue
                else:
                    point = Point(x, y)
                    if point not in points.values():
                        points[(h_cnt, v_cnt)] = Point(x, y)
        return points

    @staticmethod
    def bucket_img_into_grid_square(img_size: Iterable[int],
                                    bucket_grid_size_mm: Optional[float] = None,
                                    bucket_grid_size: Optional[Iterable[int]] = None,
                                    px_per_mm: Optional[float] = None,
                                    add_correction: Optional[bool] = True,
                                    verbose: Optional[bool] = True) -> Tuple[Dict[Tuple[int, int], Polygon], float]:
        """
        Bucketize an image into squares and return a dictionary of polygons representing the bucket locations.

        .. image:: _static/img/bucket_img_into_grid_square_3.png
           :width: 800
           :align: center

        :param Iterable[int] img_size: 2-value tuple, list or array representing the width and height of the image in pixels.
        :param Optional[float] bucket_grid_size_mm: The width/height of each square bucket in millimeters. E.g., 50 will create 5cm by 5cm squares. If None, then buckets will by defined by ``bucket_grid_size`` argument.
        :param Optional[Iterable[int]] bucket_grid_size: 2-value tuple, list or array representing the grid square in number of horizontal squares x number of vertical squares. If None, then buckets will be defined by the ``bucket_size_mm`` argument.
        :param Optional[float] px_per_mm: Pixels per millimeter conversion factor. Necessery if buckets are defined by ``bucket_size_mm`` argument.
        :param Optional[bool] add_correction: If True, performs correction by adding extra columns or rows to cover any remaining space if using ``bucket_size_mm``. Default True.
        :param Optional[bool] verbose: If True, prints progress / completion information. Default True.

        :example:
        >>> img = cv2.imread('/Users/simon/Desktop/Screenshot 2024-01-21 at 10.15.55 AM.png', 1)
        >>> polygons = GeometryMixin().bucket_img_into_grid_square(bucket_grid_size=(10, 5), bucket_grid_size_mm=None, img_size=(img.shape[1], img.shape[0]), px_per_mm=5.0)
        >>> for k, v in polygons[0].items(): cv2.polylines(img, [np.array(v.exterior.coords).astype(int)], True, (255, 0, 133), 2)
        >>> cv2.imshow('img', img)
        >>> cv2.waitKey()
        """

        timer = SimbaTimer(start=True)
        if bucket_grid_size is not None and bucket_grid_size_mm is not None:
            raise InvalidInputError(msg="bucket_size_mm and bucket_grid_size are both not None. Either provide bucket size in millimeters, OR provide the grid size", source=GeometryMixin().bucket_img_into_grid_square.__name__,)
        check_instance(source=f"{GeometryMixin.bucket_img_into_grid_square.__name__} img_size", instance=img_size, accepted_types=(tuple, np.ndarray),)
        check_iterable_length(source=f"{GeometryMixin.bucket_img_into_grid_square.__name__} img_size", val=len(img_size),exact_accepted_length=2,)
        check_int(name=f"{GeometryMixin.bucket_img_into_grid_square.__name__} img_size height", value=img_size[0],)
        check_int(name=f"{GeometryMixin.bucket_img_into_grid_square.__name__} img_size width", value=img_size[1],)
        check_valid_boolean(value=verbose, source=f"{GeometryMixin.bucket_img_into_grid_square.__name__} verbose")
        polygons = {}
        if bucket_grid_size_mm is not None:
            check_float(name=f"{GeometryMixin.bucket_img_into_grid_square.__name__} bucket_size_mm", value=bucket_grid_size_mm,)
            bin_size_px = int(px_per_mm * bucket_grid_size_mm)
            h_bin_cnt, v_bin_cnt = divmod(img_size[0], bin_size_px), divmod(img_size[1], bin_size_px)
            if (img_size[0] < bin_size_px) or (img_size[1] < bin_size_px):
                raise InvalidInputError(msg=f"The bucket square size {bin_size_px} is larger than the video size in pixels {img_size}")
            if add_correction:
                if h_bin_cnt[1] != 0:
                    h_bin_cnt = (h_bin_cnt[0] + 1, h_bin_cnt[1])
                if v_bin_cnt[1] != 0:
                    v_bin_cnt = (v_bin_cnt[0] + 1, v_bin_cnt[1])
            for i in range(h_bin_cnt[0]):
                for j in range(v_bin_cnt[0]):
                    x1, y1 = i * bin_size_px, j * bin_size_px
                    x2, y2 = x1 + bin_size_px, y1 + bin_size_px
                    polygons[(i, j)] = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            timer.stop_timer()
            if verbose:
                stdout_success(msg="Bucket image into grid squares complete", elapsed_time=timer.elapsed_time_str,)
            return polygons, round((v_bin_cnt[0] / h_bin_cnt[0]), 3)

        if bucket_grid_size is not None:
            check_instance( source=f"{GeometryMixin.__name__} bucket_grid_size", instance=bucket_grid_size, accepted_types=(tuple, np.ndarray, list),)
            check_iterable_length(
                source=f"{GeometryMixin.__name__} bucket_grid_size",
                val=len(bucket_grid_size),
                exact_accepted_length=2,
            )
            check_int(
                name=f"{GeometryMixin.bucket_img_into_grid_square.__name__} bucket_grid_size",
                value=bucket_grid_size[0],
            )
            check_int(
                name=f"{GeometryMixin.bucket_img_into_grid_square.__name__} bucket_grid_size",
                value=bucket_grid_size[1],
            )
            bucket_width, bucket_height = int(img_size[0] / bucket_grid_size[0]), int(
                img_size[1] / bucket_grid_size[1]
            )
            if (img_size[0] < bucket_width) or (img_size[1] < bucket_height):
                raise InvalidInputError(
                    msg=f"The bucket square size ({bucket_width} x {bucket_height}) is larger than the video size in pixels {img_size}"
                )

            for h_cnt in range(bucket_grid_size[0]):
                for w_cnt in range(bucket_grid_size[1]):
                    top_left = ((h_cnt * bucket_width), (w_cnt * bucket_height))
                    top_right = ((top_left[0] + bucket_width), top_left[1])
                    bottom_left = (top_left[0], top_left[1] + bucket_height)
                    bottom_right = (top_right[0], top_right[1] + bucket_height)
                    polygons[(h_cnt, w_cnt)] = Polygon(
                        [top_left, bottom_left, bottom_right, top_right]
                    )
            timer.stop_timer()
            if verbose:
                stdout_success(msg="Bucket image into grid squares complete", elapsed_time=timer.elapsed_time_str)
            return polygons, round((bucket_grid_size[1] / bucket_grid_size[0]), 3)

    @staticmethod
    def bucket_img_into_grid_hexagon(
        bucket_size_mm: float, img_size: Tuple[int, int], px_per_mm: float
    ) -> Tuple[Dict[Tuple[int, int], Polygon], float]:
        """
        Bucketize an image into hexagons and return a dictionary of polygons representing the hexagon locations.

        .. image:: _static/img/bucket_img_into_grid_hexagon.png
           :width: 500
           :align: center

        :param float bucket_size_mm: The width/height of each hexagon bucket in millimeters.
        :param Tuple[int, int] img_size: Tuple representing the width and height of the image in pixels.
        :param float px_per_mm: Pixels per millimeter conversion factor.
        :return Tuple[Dict[Tuple[int, int], Polygon], float]: First value is a dictionary where keys are (row, column) indices of the bucket, and values are Shapely Polygon objects representing the corresponding hexagon buckets. Second value is the aspect ratio of the hexagonal grid.

        :example:
        >>> polygons, aspect_ratio = GeometryMixin().bucket_img_into_grid_hexagon(bucket_size_mm=10, img_size=(800, 600), px_per_mm=5.0, add_correction=True)
        """

        timer = SimbaTimer(start=True)
        check_float(
            name=f"bucket_img_into_grid_hexagon bucket_size_mm", value=bucket_size_mm
        )
        check_int(
            name=f"bucket_img_into_grid_hexagon img_size height", value=img_size[0]
        )
        check_int(
            name=f"bucket_img_into_grid_hexagon img_size width", value=img_size[1]
        )
        check_float(name=f"bucket_img_into_grid_hexagon px_per_mm", value=px_per_mm)

        sqrt_3 = math.sqrt(3)
        hex_width = 3 / 2 * bucket_size_mm * px_per_mm
        hex_height = sqrt_3 * bucket_size_mm * px_per_mm

        h_hex_cnt, v_hex_cnt = divmod(img_size[0], int(hex_height)), divmod(
            img_size[1], int(hex_width)
        )
        if h_hex_cnt[1] != 0:
            h_hex_cnt = (h_hex_cnt[0] + 1, h_hex_cnt[1])
        if v_hex_cnt[1] != 0:
            v_hex_cnt = (v_hex_cnt[0] + 1, v_hex_cnt[1])

        polygons = {}
        for i in range(h_hex_cnt[0]):
            for j in range(v_hex_cnt[0] + (i % 2) * 1):
                x = i * 3 / 2 * hex_width
                y = j * sqrt_3 * hex_height + (i % 2) * sqrt_3 * hex_height / 2
                vertices = []
                for k in range(6):
                    angle = (math.pi / 3) * k
                    vertices.append(
                        (
                            x + hex_width * math.cos(angle),
                            y + hex_height * math.sin(angle),
                        )
                    )

                polygons[(i, j)] = Polygon(vertices)

        timer.stop_timer()
        stdout_success(
            msg="Bucket image into hexagon grid complete",
            elapsed_time=timer.elapsed_time_str,
        )
        return polygons, round((v_hex_cnt[0] / h_hex_cnt[0]), 3)

    @staticmethod
    def _cumsum_coord_geometries_helper(data: np.ndarray, geometries: Dict[Tuple[int, int], Polygon], verbose: bool
    ):
        data_point = Point(data[1:])
        if verbose:
            print(f"Processing animal grid square location in frame {int(data[0])}...")
        for k, r in geometries.items():
            if r.contains(data_point):
                return (int(data[0]), k[0], k[1])
        return (int(data[0]), -1, -1)

    def cumsum_coord_geometries(self,
                                data: np.ndarray,
                                geometries: Dict[Tuple[int, int], Polygon],
                                fps: Optional[int] = None,
                                core_cnt: Optional[int] = -1,
                                verbose: Optional[bool] = True):

        """
        Compute the cumulative time a body-part has spent inside a grid of geometries using multiprocessing.

        :param np.ndarray data: Input data array where rows represent frames and columns represent body-part x and y coordinates.
        :param Dict[Tuple[int, int], Polygon] geometries: Dictionary of polygons representing spatial regions. Created by ``GeometryMixin.bucket_img_into_squares``.
        :param Optional[int] fps: Frames per second (fps) for time normalization. If None, cumulative sum of frame count is returned.

        :example:
        >>> img_geometries = GeometryMixin.bucket_img_into_grid_square(img_size=(640, 640), bucket_grid_size=(10, 10), px_per_mm=1)
        >>> bp_arr = np.random.randint(0, 640, (5000, 2))
        >>> geo_data = GeometryMixin().cumsum_coord_geometries(data=bp_arr, geometries=img_geometries[0], verbose=False, fps=1)
        """

        timer = SimbaTimer(start=True)
        check_instance(
            source=f"{self.__class__.__name__} data",
            instance=data,
            accepted_types=np.ndarray,
        )
        if (data.shape[1] != 2) or (data.ndim != 2):
            raise CountError(
                msg=f"A N x 2 array is required (got {data.shape})",
                source=self.__class__.__name__,
            )
        if fps is not None:
            check_int(name="fps", value=fps, min_value=1)
        else:
            fps = 1
        check_int(name="core_cnt", value=core_cnt, min_value=-1)
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        w, h = 0, 0
        for k in geometries.keys():
            w, h = max(w, k[0]), max(h, k[1])
        frm_id = np.arange(0, data.shape[0]).reshape(-1, 1)
        data = np.hstack((frm_id, data))
        img_arr = np.zeros((data.shape[0], h + 1, w + 1))
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.MAXIMUM_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                self._cumsum_coord_geometries_helper,
                geometries=geometries,
                verbose=verbose,
            )
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                if result[1] != -1:
                    img_arr[result[0], result[2] - 1, result[1] - 1] = 1
        pool.join()
        pool.terminate()
        timer.stop_timer()
        stdout_success(
            msg="Cumulative coordinates in geometries complete",
            elapsed_time=timer.elapsed_time_str,
        )
        if fps is None:
            return np.cumsum(img_arr, axis=0)
        else:
            return np.cumsum(img_arr, axis=0) / fps

    @staticmethod
    def _cumsum_bool_helper(
        data: np.ndarray, geometries: Dict[Tuple[int, int], Polygon]
    ):
        data_point = Point(data[1:3])
        print(
            f"Processing animal grid square location for boolean in frame {int(data[0])}..."
        )
        for k, r in geometries.items():
            if r.contains(data_point):
                return (int(data[0]), k[0], k[1])
        return (int(data[0]), -1, -1)

    def cumsum_bool_geometries(
        self,
        data: np.ndarray,
        geometries: Dict[Tuple[int, int], Polygon],
        bool_data: np.ndarray,
        fps: Optional[float] = None,
        core_cnt: Optional[int] = -1,
    ) -> np.ndarray:
        """
        Compute the cumulative sums of boolean events within polygon geometries over time using multiprocessing.

        E.g., compute the cumulative time of classified events within spatial locations at all time-points of the video.

        :param np.ndarray data: Array containing spatial data with shape (n, 2). E.g., 2D-array with body-part coordinates.
        :param Dict[Tuple[int, int], Polygon] geometries: Dictionary of polygons representing spatial regions. Created by ``GeometryMixin.bucket_img_into_squares``.
        :param np.ndarray bool_data: Boolean array with shape (data.shape[0],) or (data.shape[0], 1) indicating the presence or absence in each frame.
        :param Optional[float] fps: Frames per second. If provided, the result is normalized by the frame rate.
        :param Optional[float] core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which means using all available cores.
        :returns np.ndarray: Array of size (frames x horizontal bins x verical bins) with times in seconds (if fps passed) or frames (if fps not passed)

        :example:
        >>> geometries = GeometryMixin.bucket_img_into_grid_square(bucket_size_mm=50, img_size=(800, 800) , px_per_mm=5.0)[0]
        >>> coord_data = np.random.randint(0, 800, (500, 2))
        >>> bool_data = np.random.randint(0, 2, (500,))
        >>> x = GeometryMixin().cumsum_bool_geometries(data=coord_data, geometries=geometries, bool_data=bool_data, fps=15)
        >>> x.shape
        >>> (500, 4, 4)
        """

        check_valid_array(
            data=data,
            accepted_sizes=[2],
            source=f"{GeometryMixin.cumsum_bool_geometries.__name__} data",
        )
        check_instance(
            source=f"{GeometryMixin.cumsum_bool_geometries.__name__} geometries",
            instance=geometries,
            accepted_types=dict,
        )
        check_valid_array(
            data=bool_data,
            accepted_shapes=[(data.shape[0], 1), (data.shape[0],)],
            source=f"{GeometryMixin.cumsum_bool_geometries.__name__} bool_data",
        )
        if fps is not None:
            check_float(
                name=f"{GeometryMixin.cumsum_bool_geometries.__name__} fps",
                value=fps,
                min_value=1.0,
            )
        check_int(
            name=f"{GeometryMixin.cumsum_bool_geometries.__name__} core_cnt",
            value=core_cnt,
            min_value=-1,
        )
        if not np.array_equal(
            np.sort(np.unique(bool_data)).astype(int), np.array([0, 1])
        ):
            raise InvalidInputError(
                msg=f"Invalid boolean data. Expected {np.array([0, 1])} but found {np.sort(np.unique(bool_data)).astype(int)}",
                source=GeometryMixin.cumsum_bool_geometries.__name__,
            )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        w, h = 0, 0
        for k in geometries.keys():
            w, h = max(w, k[0]), max(h, k[1])
        frm_id = np.arange(0, data.shape[0]).reshape(-1, 1)
        if bool_data.ndim == 1:
            bool_data = bool_data.reshape(-1, 1)
        data = np.hstack((data, bool_data))
        data = np.hstack((frm_id, data))
        img_arr = np.zeros((data.shape[0], h + 1, w + 1))
        data = data[np.argwhere((data[:, 3] == 1))].reshape(-1, 4)
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                self._cumsum_bool_helper, geometries=geometries
            )
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                if result[1] != -1:
                    img_arr[result[0], result[2] - 1, result[1] - 1] = 1
        pool.join()
        pool.terminate()
        if fps is None:
            return np.cumsum(img_arr, axis=0)
        else:
            return np.cumsum(img_arr, axis=0) / fps

    @staticmethod
    def _cumsum_animal_geometries_grid_helper(
        data: np.ndarray,
        grid: Dict[Tuple[int, int], Polygon],
        size: Tuple[int],
        verbose: bool,
    ):

        shape, results = data[1], np.zeros((size[0] + 1, size[1] + 1))
        if verbose:
            print(f"Processing animal grid square location in frame {int(data[0])}...")
        for k, r in grid.items():
            if r.intersects(shape):
                results[k[0], k[1]] = 1
        return results

    def cumsum_animal_geometries_grid(
        self,
        data: List[Polygon],
        grid: Dict[Tuple[int, int], Polygon],
        fps: Optional[int] = None,
        core_cnt: Optional[int] = -1,
        verbose: Optional[bool] = True,
    ):

        timer = SimbaTimer(start=True)
        check_valid_lst(
            data=data,
            source=GeometryMixin.cumsum_animal_geometries_grid.__name__,
            valid_dtypes=(Polygon,),
        )
        check_instance(
            source=GeometryMixin.cumsum_animal_geometries_grid.__name__,
            instance=grid,
            accepted_types=(dict,),
        )
        if fps is not None:
            check_int(name="fps", value=fps, min_value=1)
        check_int(name="core_cnt", value=core_cnt, min_value=-1)
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        w, h = 0, 0
        for k in grid.keys():
            w, h = max(w, k[0]), max(h, k[1])
        frm_id = np.arange(0, len(data)).reshape(-1, 1)
        data = np.hstack((frm_id, np.array(data).reshape(-1, 1)))
        img_arr = np.zeros((data.shape[0], h + 1, w + 1))
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                self._cumsum_animal_geometries_grid_helper,
                grid=grid,
                size=(h, w),
                verbose=verbose,
            )
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                img_arr[cnt] = result

        timer.stop_timer()
        stdout_success(
            msg="Cumulative animal geometries in grid complete",
            elapsed_time=timer.elapsed_time_str,
        )
        if fps is None:
            return np.cumsum(img_arr, axis=0)
        else:
            return np.cumsum(img_arr, axis=0) / fps

    @staticmethod
    def hausdorff_distance(geometries: List[List[Union[Polygon, LineString]]]) -> np.ndarray:
        """
        The Hausdorff distance measure of the similarity between time-series sequential geometries. It is defined as the maximum of the distances
        from each point in one set to the nearest point in the other set.

        Hausdorff distance can be used to measure the similarity of the geometry in one frame relative to the geometry in the next frame.
        Larger values indicate that the animal has a different shape than in the preceding shape.

        :param List[List[Union[Polygon, LineString]]] geometries: List of list where each list has two geometries.
        :return np.ndarray: 1D array of hausdorff distances of geometries in each list.

        :example:
        >>> x = Polygon([[0,1], [0, 2], [1,1]])
        >>> y = Polygon([[0,1], [0, 2], [0,1]])
        >>> GeometryMixin.hausdorff_distance(geometries=[[x, y]])
        >>> [1.]
        """

        check_instance(
            source=GeometryMixin.hausdorff_distance.__name__,
            instance=geometries,
            accepted_types=(list,),
        )
        for i in geometries:
            check_valid_lst(
                source=GeometryMixin.hausdorff_distance.__name__,
                data=i,
                valid_dtypes=(
                    Polygon,
                    LineString,
                ),
                exact_len=2,
            )
        results = np.full((len(geometries)), np.nan)
        for i in range(len(geometries)):
            results[i] = geometries[i][0].hausdorff_distance(geometries[i][1])
        return results

    def multiframe_hausdorff_distance(
        self,
        geometries: List[Union[Polygon, LineString]],
        lag: Optional[int] = 1,
        core_cnt: Optional[int] = -1,
    ) -> List[float]:
        """
        The Hausdorff distance measure of the similarity between sequential time-series  geometries.

        :example:
        >>> df = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/outlier_corrected_movement_location/SI_DAY3_308_CD1_PRESENT.csv', file_type='csv')
        >>> cols = [x for x in df.columns if not x.endswith('_p')]
        >>> data = df[cols].values.reshape(len(df), -1 , 2).astype(np.int)
        >>> geometries = GeometryMixin().multiframe_bodyparts_to_polygon(data=data, pixels_per_mm=1, parallel_offset=1, verbose=False, core_cnt=-1)
        >>> hausdorff_distances = GeometryMixin.multiframe_hausdorff_distance(geometries=geometries)
        """

        check_valid_lst(
            source=GeometryMixin.multiframe_hausdorff_distance.__name__,
            data=geometries,
            valid_dtypes=(
                Polygon,
                LineString,
            ),
            min_len=1,
        )
        check_int(
            name=f"{GeometryMixin.multiframe_hausdorff_distance.__name__} CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        check_int(
            name=f"{GeometryMixin.multiframe_hausdorff_distance.__name__} LAG",
            value=lag,
            min_value=-1,
            max_value=len(geometries) - 1,
            raise_error=True,
        )
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]
        reshaped_geometries = []
        for i in range(lag):
            reshaped_geometries.append([[geometries[i], geometries[i]]])
        for i in range(lag, len(geometries)):
            reshaped_geometries.append([[geometries[i - lag], geometries[i]]])
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, mp_return in enumerate(
                pool.imap(
                    GeometryMixin.hausdorff_distance, reshaped_geometries, chunksize=1
                )
            ):
                results.append(mp_return[0])
        return results

    @staticmethod
    def locate_line_point(path: Union[LineString, np.ndarray],
                          geometry: Union[LineString, Polygon, Point],
                          px_per_mm: Optional[float] = 1,
                          fps: Optional[float] = 1,
                          core_cnt: Optional[int] = -1,
                          distance_min: Optional[bool] = True,
                          time_prior: Optional[bool] = True) -> Dict[str, float]:

        """
        Compute the time and distance travelled along a path to reach the most proximal point in reference to a second geometry.

        .. note::
           (i) To compute the time and distance travelled to along a path to reach the most distal point to a second geometry, pass ``distance_min = False``.

           (ii) To compute the time and distance travelled along a path **after** reaching the most distal or proximal point to a second geometry, pass ``time_prior = False``.

        .. image:: _static/img/locate_line_point.png
           :width: 600
           :align: center

        :example:
        >>> line = LineString([[10, 10], [7.5, 7.5], [15, 15], [7.5, 7.5]])
        >>> polygon = Polygon([[0, 5], [0, 0], [5, 0], [5, 5]])
        >>> GeometryMixin.locate_line_point(path=line, geometry=polygon)
        >>> {'distance_value': 3.5355339059327378, 'distance_travelled': 3.5355339059327378, 'time_travelled': 1.0, 'distance_index': 1}
        """

        check_instance(
            source=GeometryMixin.locate_line_point.__name__,
            instance=path,
            accepted_types=(LineString, np.ndarray),
        )
        check_instance(
            source=GeometryMixin.locate_line_point.__name__,
            instance=geometry,
            accepted_types=(LineString, Polygon, Point),
        )
        check_int(
            name="CORE COUNT",
            value=core_cnt,
            min_value=-1,
            max_value=find_core_cnt()[0],
            raise_error=True,
        )
        check_float(
            name="PIXELS PER MM", value=px_per_mm, min_value=0.1, raise_error=True
        )
        check_float(name="FPS", value=fps, min_value=1, raise_error=True)
        if core_cnt == -1:
            core_cnt = find_core_cnt()[0]

        if isinstance(path, np.ndarray):
            check_valid_array(
                data=path,
                accepted_axis_1_shape=(2,),
                accepted_dtypes=(np.float32, np.float64, np.int64, np.int32),
            )
            path = LineString(path)
        if isinstance(geometry, Point):
            geometry = np.array(geometry.coords)
            distances = FeatureExtractionMixin.framewise_euclidean_distance_roi(
                location_1=np.array(path.coords),
                location_2=geometry,
                px_per_mm=px_per_mm,
            )
        else:
            points = [Point(x) for x in np.array(path.coords)]
            geometry = [geometry for x in range(len(points))]
            distances = GeometryMixin().multiframe_shape_distance(
                shape_1=points,
                shape_2=geometry,
                pixels_per_mm=px_per_mm,
                core_cnt=core_cnt,
            )

        if distance_min:
            distance_idx = np.argmin(distances)
        else:
            distance_idx = np.argmax(distances)
        if time_prior:
            dist_travelled = (
                np.sum(np.abs(np.diff(distances[: distance_idx + 1]))) / px_per_mm
            )
            time_travelled = distance_idx / fps
        else:
            dist_travelled = (
                np.sum(np.abs(np.diff(distances[distance_idx:]))) / px_per_mm
            )
            time_travelled = (distances - distance_idx) / fps
        dist_val = distances[distance_idx] / px_per_mm
        raw_distances = np.array(distances) / px_per_mm

        return {
            "distance_value": dist_val,
            "distance_travelled": dist_travelled,
            "time_travelled": time_travelled,
            "distance_index": distance_idx,
            "raw_distances": raw_distances
        }

    @staticmethod
    @njit("(float32[:,:], float32[:,:], int64)")
    def linear_frechet_distance(
        x: np.ndarray, y: np.ndarray, sample: int = 100
    ) -> float:
        """
        Compute the Linear Fréchet Distance between two trajectories.

        The Fréchet Distance measures the dissimilarity between two continuous
        curves or trajectories represented as sequences of points in a 2-dimensional
        space.

        :param ndarray data: First 2D array of size len(frames) representing body-part coordinates x and y.
        :param ndarray data: Second 2D array of size len(frames) representing body-part coordinates x and y.
        :param int sample: The downsampling factor for the trajectories (default is 100If sample > 1, the trajectories are downsampled by selecting every sample-th point.

        .. note::
           Slightly modified from `João Paulo Figueira <https://github.com/joaofig/discrete-frechet/blob/ff5629e5a43cfad44d5e962f4105dd25c90b9289/distances/discrete.py#L67>`_

        :example:
        >>> x = np.random.randint(0, 100, (10000, 2)).astype(np.float32)
        >>> y = np.random.randint(0, 100, (10000, 2)).astype(np.float32)
        >>> distance = GeometryMixin.linear_frechet_distance(x=x, y=y, sample=100)

        """
        if sample > 1:
            x, y = x[::sample], y[::sample]
        n_p, n_q = x.shape[0], y.shape[0]
        ca = np.full((n_p, n_q), 0.0)
        for i in prange(n_p):
            for j in range(n_q):
                d = x[i] - y[j]
                d = np.sqrt(np.dot(d, d))
                if i > 0 and j > 0:
                    ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
                elif i > 0 and j == 0:
                    ca[i, j] = max(ca[i - 1, 0], d)
                elif i == 0 and j > 0:
                    ca[i, j] = max(ca[0, j - 1], d)
                else:
                    ca[i, j] = d
        return ca[n_p - 1, n_q - 1]

    @staticmethod
    def simba_roi_to_geometries(rectangles_df: Optional[pd.DataFrame] = None,
                                circles_df: Optional[pd.DataFrame] = None,
                                polygons_df: Optional[pd.DataFrame] = None,
                                color: Optional[bool] = False) -> dict:

        """
        Convert SimBA dataframes holding ROI geometries to nested dictionary holding Shapley polygons.

        :example:
        >>> #config_path = '/Users/simon/Desktop/envs/simba/troubleshooting/spontenous_alternation/project_folder/project_config.ini'
        >>> #config = ConfigReader(config_path=config_path)
        >>> #config.read_roi_data()
        >>> #GeometryMixin.simba_roi_to_geometries(rectangles_df=config.rectangles_df, circles_df=config.circles_df, polygons_df=config.polygon_df)
        """

        results_roi, results_clr = {}, {}
        if rectangles_df is not None:
            check_instance(source=GeometryMixin.simba_roi_to_geometries.__name__, instance=rectangles_df, accepted_types=(pd.DataFrame,))
            for video_name in rectangles_df["Video"].unique():
                if video_name not in results_roi.keys():
                    results_roi[video_name] = {}
                    results_clr[video_name] = {}
                video_shapes = rectangles_df[["Tags", "Name", "Color BGR"]][rectangles_df["Video"] == video_name]
                for shape_name in video_shapes["Name"].unique():
                    shape_data = video_shapes[video_shapes["Name"] == shape_name].reset_index(drop=True)
                    tags, name = (list(shape_data["Tags"].values[0].values()), shape_data["Name"].values[0])
                    results_roi[video_name][name] = Polygon(Polygon(tags).convex_hull.exterior.coords)
                    results_clr[video_name][name] = shape_data["Color BGR"].values[0]

        if polygons_df is not None:
            check_instance(source=GeometryMixin.simba_roi_to_geometries.__name__, instance=polygons_df, accepted_types=(pd.DataFrame,))
            for video_name in polygons_df["Video"].unique():
                if video_name not in results_roi.keys():
                    results_roi[video_name] = {}
                    results_clr[video_name] = {}
                video_shapes = polygons_df[["Tags", "Name", "Color BGR"]][ polygons_df["Video"] == video_name]
                for shape_name in video_shapes["Name"].unique():
                    shape_data = video_shapes[video_shapes["Name"] == shape_name].reset_index(drop=True)
                    tags, name = (list(shape_data["Tags"].values[0].values()), shape_data["Name"].values[0])
                    results_roi[video_name][name] = Polygon(Polygon(tags).convex_hull.exterior.coords)
                    results_clr[video_name][name] = shape_data["Color BGR"].values[0]

        if circles_df is not None:
            check_instance(source=GeometryMixin.simba_roi_to_geometries.__name__, instance=circles_df, accepted_types=(pd.DataFrame,))
            for video_name in circles_df["Video"].unique():
                if video_name not in results_roi.keys():
                    results_roi[video_name] = {}
                    results_clr[video_name] = {}
                video_shapes = circles_df[["Tags", "Name", "Color BGR", 'radius']][circles_df["Video"] == video_name]
                for shape_name in video_shapes["Name"].unique():
                    shape_data = video_shapes[video_shapes["Name"] == shape_name].reset_index(drop=True)
                    tags, name, radius = shape_data["Tags"].values[0], shape_data["Name"].values[0], shape_data["radius"].values[0]
                    results_roi[video_name][name] = Point(tags["Center tag"]).buffer(distance=radius)
                    results_clr[video_name][name] = shape_data["Color BGR"].values[0]
        if not color:
            return results_roi, None
        else:
            return results_roi, results_clr

    @staticmethod
    def filter_low_p_bps_for_shapes(x: np.ndarray, p: np.ndarray, threshold: float):
        """
        Filter body-part data for geometry construction while maintaining valid geometry arrays.

        Having a 3D array representing body-parts across time, and a second 3D array representing probabilities of those
        body-parts across time, we want to "remove" body-parts with low detection probabilities whilst also keeping the array sizes
        intact and suitable for geometry construction. To do this, we find body-parts with detection probabilities below the threshold, and replace these with a body-part
        that doesn't fall below the detection probability threshold within the same frame. However, to construct a geometry, we need >= 3 unique key-point locations.
        Thus, no substitution can be made to when there are less than three unique body-part locations within a frame that falls above the threshold.

        :example:
        >>> x = np.random.randint(0, 500, (18000, 7, 2))
        >>> p = np.random.random(size=(18000, 7, 1))
        >>> x = GeometryMixin.filter_low_p_bps_for_shapes(x=x, p=p, threshold=0.1)
        >>> x = x.reshape(x.shape[0], int(x.shape[1] * 2))
        """

        results = np.copy(x)
        for i in range(x.shape[0]):
            below_p_idx = np.argwhere(p[i].flatten() < threshold).flatten()
            above_p_idx = np.argwhere(p[i].flatten() >= threshold).flatten()
            if (below_p_idx.shape[0] > 0) and (above_p_idx.shape[0] >= 3):
                for j in below_p_idx:
                    new_val = x[i][above_p_idx[0]]
                    results[i][j] = new_val
        return results








# data = np.array([[[364, 308], [383, 323], [403, 335], [423, 351]],
#                  [[356, 307], [376, 319], [396, 331], [419, 347]],
#                  [[364, 308], [383, 323], [403, 335], [423, 351]],
#                  [[364, 308], [3, 323], [403, 335], [423, 351]],
#                  [[364, 308], [383, 323], [54, 335], [423, 351]],
#                  [[12, 308], [383, 34], [403, 335], [423, 351]],
#                  [[364, 308], [383, 323], [403, 335], [100, 351]]])
# GeometryMixin().multiframe_bodyparts_to_polygon(data=data)

#
# polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
# polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
# polygon_1 = [polygon_1 for x in range(100)]
# polygon_2 = [polygon_2 for x in range(100)]
# data = np.column_stack((polygon_1, polygon_2))
# GeometryMixin.compute_pct_shape_overlap(shapes=data)
#
#
#
# data = np.array_split(data, 8)
#
#
#
# results = GeometryMixin().multiframe_compute_pct_shape_overlap(shape_1=polygon_1, shape_2=polygon_2,)
#
# np.

# polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[0, 100],[100, 100],[0, 0],[100, 0]]))
# polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 75],[75, 75],[25, 25],[75, 25]]))
# y = GeometryMixin().compute_pct_shape_overlap(shapes=[polygon_1, polygon_2], denominator='shape_2')
