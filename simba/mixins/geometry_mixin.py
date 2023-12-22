import numpy as np
from typing import Optional, List, Union, Tuple, Iterable
from typing_extensions import Literal
import pandas as pd
from shapely.geometry import (Polygon,
                              LineString,
                              Point,
                              MultiPolygon,
                              MultiLineString,
                              GeometryCollection,
                              MultiPoint)
from shapely.ops import (unary_union,
                         linemerge,
                         split,
                         triangulate)
from copy import deepcopy
import cv2
import imutils
import multiprocessing
import functools
import itertools
from numba import prange, njit

from simba.utils.checks import (check_instance,
                                check_if_valid_input,
                                check_iterable_length,
                                check_float,
                                check_if_2d_array_has_min_unique_values)
from simba.utils.enums import Defaults
from simba.utils.lookups import get_color_dict
from simba.utils.errors import (InvalidInputError,
                                CountError)
from simba.utils.read_write import (find_core_cnt,
                                    find_max_vertices_coordinates,
                                    stdout_success,
                                    SimbaTimer)
from simba.utils.checks import check_int

CAP_STYLE_MAP = {'round': 1, 'square': 2, 'flat': 3}
MAX_TASK_PER_CHILD = 750

class GeometryMixin(object):

    """
    Methods to perform geometry transformation of pose-estimation data. This includes creating bounding boxes,
    line objects, circles etc. from pose-estimated body-parts and computing relationships between created shapes.

    Very much wip and relies heavily on `shapley <https://shapely.readthedocs.io/en/stable/manual.html>`_.
    """

    def __init__(self):
        pass

    @staticmethod
    def bodyparts_to_polygon(data: np.ndarray,
                             cap_style: Literal['round', 'square', 'flat'] = 'round',
                             parallel_offset: int = 1,
                             simplify_tolerance: float = 2,
                             preserve_topology: bool = True) -> Polygon:
        """
        .. image:: _static/img/bodyparts_to_polygon.png
           :width: 400
           :align: center

        :example:
        >>> data = [[364, 308],[383, 323],[403, 335],[423, 351]]
        >>> GeometryMixin().bodyparts_to_polygon(data=data)
        """

        if not check_if_2d_array_has_min_unique_values(data=data, min=3):
            return Polygon([(0, 0), (0, 0), (0, 0)])
        else:
            return Polygon(LineString(data.tolist()).buffer(distance=parallel_offset, cap_style=CAP_STYLE_MAP[cap_style]).simplify(tolerance=simplify_tolerance, preserve_topology=preserve_topology).convex_hull)

    @staticmethod
    def bodyparts_to_circle(data: np.ndarray,
                            parallel_offset: int = 1) -> Polygon:

        """
        .. image:: _static/img/bodyparts_to_circle.png
           :width: 400
           :align: center

        :example:
        >>> data = np.array([364, 308])
        >>> polygon = GeometryMixin().bodyparts_to_circle(data=data)
        """

        if data.shape != (2,):
            raise InvalidInputError(msg=f'Cannot create circle data is not a (2,) array: '
                                        f'{data.shape}', source=GeometryMixin.bodyparts_to_circle.__name__)

        return Point(data).buffer(parallel_offset)


    @staticmethod
    def bodyparts_to_multistring_skeleton(data: np.ndarray) -> MultiLineString:

        """
        .. image:: _static/img/bodyparts_to_multistring_skeleton.png
           :width: 400
           :align: center

        :example:
        >>> skeleton = np.array([[[5, 5], [1, 10]], [[5, 5], [9, 10]], [[9, 10], [1, 10]], [[9, 10], [9, 25]], [[1, 10], [1, 25]], [[9, 25], [5, 50]], [[1, 25], [5, 50]]])
        >>> shape_multistring = GeometryMixin().bodyparts_to_multistring_skeleton(bodyparts=skeleton, shape=True)
        """


        if data.ndim != 3:
            raise InvalidInputError(msg=f'Body-parts to skeleton expects a 3D array, got {data.ndim}', source=GeometryMixin.bodyparts_to_line.__name__)
        shape_skeleton = []
        for i in data: shape_skeleton.append(GeometryMixin().bodyparts_to_line(data=i))
        shape_skeleton = linemerge(MultiLineString(shape_skeleton))

        return shape_skeleton

    @staticmethod
    def buffer_shape(shape: Union[Polygon, LineString],
                     size_mm: int,
                     pixels_per_mm: float,
                     cap_style: Literal['round', 'square', 'flat'] = 'round') -> Polygon:

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

        check_instance(source=GeometryMixin.buffer_shape.__name__, instance=shape, accepted_types=(LineString, Polygon))
        check_int(name='BUFFER SHAPE size_mm', value=size_mm)
        check_float(name='BUFFER SHAPE pixels_per_mm', value=pixels_per_mm, min_value=1)
        return shape.buffer(distance=int(size_mm / pixels_per_mm), cap_style=CAP_STYLE_MAP[cap_style])

    @staticmethod
    def compute_pct_shape_overlap(shapes: List[Union[Polygon, LineString]]) -> float:

        """
        Compute the percentage of overlap between two shapes.

        .. image:: _static/img/compute_pct_shape_overlap.png
           :width: 400
           :align: center

        :param List[Union[LineString, Polygon]] shapes: A list of two input Polygon or LineString shapes.
        :return float: The percentage of overlap between the two shapes.

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
        >>> GeometryMixin().compute_pct_shape_overlap(shapes=[polygon_1, polygon_2])
        >>> 37.96
        """

        for shape in shapes:
            check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape, accepted_types=(LineString, Polygon))
        check_iterable_length(source=GeometryMixin.compute_pct_shape_overlap.__name__, val=len(shapes), exact_accepted_length=2)
        if shapes[0].intersects(shapes[1]):
            intersection = shapes[0].intersection(shapes[1])
            return round((intersection.area / ((shapes[0].area + shapes[1].area) - intersection.area) * 100), 2)
        else:
            return 0.0

    @staticmethod
    def compute_shape_overlap(shapes: List[Union[Polygon, LineString]]) -> int:
        """
        Computes if two geometrical shapes (Polygon or LineString) overlaps or are disjoint.

        .. note::
           Only returns if two shapes are overlapping or not overlapping. If the amount of overlap is required, use
           ``GeometryMixin().compute_pct_shape_overlap()``.

        .. image:: _static/img/compute_overlap.png
           :width: 400
           :align: center

        :param List[Union[LineString, Polygon]] shapes: A list of two input Polygon or LineString shapes.
        :return float: Returns 1 if the two shapes overlap, otherwise returns 0.
        """

        for shape in shapes:
            check_instance(source=GeometryMixin.compute_shape_overlap.__name__, instance=shape, accepted_types=(LineString, Polygon))
        check_iterable_length(source=GeometryMixin.compute_shape_overlap.__name__, val=len(shapes), exact_accepted_length=2)
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

        check_iterable_length(source=GeometryMixin.compute_pct_shape_overlap.__name__, val=len(shapes), exact_accepted_length=2)
        for shape in shapes:
            check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape, accepted_types=LineString)
        return shapes[0].crosses(shapes[1])

    @staticmethod
    def is_shape_covered(shape: Union[LineString, Polygon, MultiPolygon], other_shape: Union[LineString, Polygon, MultiPolygon]) -> bool:
        """
        Check if one geometry fully covers another.

        .. image:: _static/img/is_line_covered.png
           :width: 400
           :align: center

        :param Union[LineString, Polygon, MultiPolygon] shape: The first geometry to be checked for coverage.
        :param Union[LineString, Polygon, MultiPolygon] other_shape: The second geometry that is potentially covered by the first.
        :return bool: True if the first geometry fully covers the second, False otherwise.

        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 25], [25, 75], [90, 25], [90, 75]]))
        >>> GeometryMixin().is_shape_covered(shape=polygon_1, other_shape=polygon_2)
        >>> True

        """
        check_instance(source=GeometryMixin.is_shape_covered.__name__, instance=shape, accepted_types=(LineString, Polygon, MultiPolygon))
        check_instance(source=GeometryMixin.is_shape_covered.__name__, instance=other_shape, accepted_types=(LineString, Polygon, MultiPolygon))

        return shape.covers(other_shape)

    @staticmethod
    def area(shape: Union[MultiPolygon, Polygon], pixels_per_mm: float):
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

        check_instance(source=f'{GeometryMixin().area.__name__} shape', instance=shape, accepted_types=(MultiPolygon, Polygon))
        check_float(name=f'{GeometryMixin().area.__name__} shape', value=pixels_per_mm, min_value=0.01)

        return shape.area / pixels_per_mm

    @staticmethod
    def shape_distance(shapes: List[Union[LineString, Polygon]],
                       pixels_per_mm: float,
                       unit: Literal['mm', 'cm', 'dm', 'm'] = 'mm') -> float:
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

        check_if_valid_input(name='UNIT', input=unit, options=['mm', 'cm', 'dm', 'm'])
        for shape in shapes:
            check_instance(source=GeometryMixin.shape_distance.__name__, instance=shape, accepted_types=(LineString, Polygon))
        check_iterable_length(source=GeometryMixin.shape_distance.__name__, val=len(shapes), exact_accepted_length=2)

        D = shapes[0].distance(shapes[1]) / pixels_per_mm
        if unit == 'cm':
            D = D / 10
        elif unit == 'dm':
            D = D / 100
        elif unit == 'm':
            D = D / 1000
        return D


    @staticmethod
    def bodyparts_to_line(data: np.ndarray):

        """

        .. image:: _static/img/bodyparts_to_line.png
           :width: 400
           :align: center

        :example:
        >>> data = np.array([[364, 308],[383, 323], [403, 335],[423, 351]])
        >>> line = GeometryMixin().bodyparts_to_line(data=data)
        """

        if data.ndim != 2:
            raise InvalidInputError(msg=f'Body-parts to linestring expects a 2D array, got {data.ndim}', source=GeometryMixin.bodyparts_to_line.__name__)
        return LineString(data.tolist())

    @staticmethod
    def get_center(shape: Union[LineString, Polygon, MultiPolygon]) -> np.ndarray:
        """
        .. image:: _static/img/get_center.png
           :width: 500
           :align: center

        :example:
        >>> multipolygon = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
        >>> GeometryMixin().get_center(shape=multipolygon)
        >>> [33.96969697, 62.32323232]

        """
        check_instance(source=GeometryMixin.get_center.__name__, instance=shape, accepted_types=(MultiPolygon, LineString, Polygon))
        return np.array(shape.centroid)

    @staticmethod
    def is_touching(shapes = List[Union[LineString, Polygon]]) -> bool:
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
            check_instance(source=GeometryMixin.is_touching.__name__, instance=i, accepted_types=(LineString, Polygon))
        check_iterable_length(source=GeometryMixin.is_touching.__name__, val=len(shapes), exact_accepted_length=2)
        return shapes[0].touches(shapes[1])

    @staticmethod
    def is_containing(shapes = List[Union[LineString, Polygon]]) -> bool:
        """
        .. image:: _static/img/is_containing.png
           :width: 500
           :align: center

        :example:
        """
        for i in shapes:
            check_instance(source=GeometryMixin.get_center.__name__, instance=i, accepted_types=(LineString, Polygon))
        check_iterable_length(source=GeometryMixin.get_center.__name__, val=len(shapes), exact_accepted_length=2)

        return shapes[0].contains(shapes[1])

    @staticmethod
    def difference(shapes = List[Union[LineString, Polygon, MultiPolygon]]) -> Polygon:
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

        check_iterable_length(source=GeometryMixin.difference.__name__, val=len(shapes), min=2)
        for shape in shapes:
            check_instance(source=GeometryMixin.difference.__name__, instance=shape, accepted_types=(LineString, Polygon, MultiPolygon))

        results = deepcopy(shapes[0])
        for overlap_shap in shapes[1:]:
            if isinstance(overlap_shap, MultiPolygon):
                for geo in overlap_shap.geoms:
                    results = results.difference(geo)
            else:
                results = results.difference(overlap_shap)
        return results

    @staticmethod
    def union(shapes: List[Union[LineString, Polygon, MultiPolygon]]) -> Union[MultiPolygon, Polygon, MultiLineString]:
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

        check_iterable_length(source=GeometryMixin.union.__name__, val=len(shapes), min=2)
        for shape in shapes: check_instance(source=GeometryMixin.union.__name__, instance=shape, accepted_types=(LineString, Polygon, MultiPolygon))
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
        for shape in shapes: check_instance(source=GeometryMixin.symmetric_difference.__name__, instance=shape,
                                            accepted_types=(LineString, Polygon, MultiPolygon))
        results = deepcopy(shapes)
        for c in itertools.combinations(list(range(0, len(shapes))), 2):
            results[c[0]] = results[c[0]].convex_hull.difference(results[c[1]].convex_hull)
            results[c[1]] = results[c[1]].convex_hull.difference(results[c[0]].convex_hull)

        results = [geometry for geometry in results if not geometry.is_empty]
        return results

    @staticmethod
    def view_shapes(shapes: List[Union[LineString, Polygon, MultiPolygon, MultiLineString]]) -> np.ndarray:

        """
        Helper function to draw shapes on white canvas. Useful for quick troubleshooting.

        :example:
        >>> multipolygon_1 = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[100, 110],[100, 100],[110, 100],[110, 110]]))
        >>> line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
        >>> img = GeometryMixin.view_shapes(shapes=[line_1, polygon_1, multipolygon_1])
        """

        for i in shapes:
            check_instance(source=GeometryMixin.view_shapes.__name__, instance=i, accepted_types=(LineString, Polygon, MultiPolygon, MultiLineString, Point))
        max_vertices = find_max_vertices_coordinates(shapes=shapes, buffer=50)
        img = np.ones((max_vertices[0], max_vertices[1], 3), dtype=np.uint8) * 255
        colors = list(get_color_dict().values())
        for shape_cnt, shape in enumerate(shapes):
            if isinstance(shape, Polygon):
                cv2.polylines(img, [np.array(shape.exterior.coords).astype(np.int)], True, (colors[shape_cnt][::-1]) , thickness=2)
                interior_coords = [np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2)) for interior in shape.interiors]
                for interior in interior_coords:
                    cv2.polylines(img, [interior], isClosed=True, color=(colors[shape_cnt][::-1]), thickness=2)
            if isinstance(shape, LineString):
                cv2.polylines(img, [np.array(shape.coords, dtype=np.int32)], False, (colors[shape_cnt][::-1]), thickness=2)
            if isinstance(shape, MultiPolygon):
                for polygon_cnt, polygon in enumerate(shape.geoms):
                    polygon_np = np.array((polygon.convex_hull.exterior.coords), dtype=np.int32)
                    cv2.polylines(img, [polygon_np], True, (colors[shape_cnt+ polygon_cnt + 1][::-1]),thickness=2)
            if isinstance(shape, MultiLineString):
                for line_cnt, line in enumerate(shape.geoms):
                    cv2.polylines(img, [np.array(shape[line_cnt].coords, dtype=np.int32)], False, (colors[shape_cnt][::-1]), thickness=2)
            if isinstance(shape, Point):
                cv2.circle(img, (int(np.array(shape.centroid)[0]), int(np.array(shape.centroid)[1])), 0, colors[shape_cnt][::-1], 1)

        return imutils.resize(img, width=800)

    @staticmethod
    def minimum_rotated_rectangle(shape = Polygon) -> bool:
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

        check_instance(source=GeometryMixin.get_center.__name__, instance=shape, accepted_types=Polygon)
        rotated_rectangle = shape.minimum_rotated_rectangle
        if isinstance(rotated_rectangle, Point):
            return Polygon([(0, 0), (0, 0), (0, 0)])
        else:
            return rotated_rectangle

    @staticmethod
    def length(shape: Union[LineString, MultiLineString],
               pixels_per_mm: float,
               unit: Literal['mm', 'cm', 'dm', 'm'] = 'mm') -> float:
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

        check_float(name='line_length pixels_per_mm', value=pixels_per_mm, min_value=0)
        check_instance(source=GeometryMixin.length.__name__, instance=shape, accepted_types=LineString)
        L = shape.length
        if unit == 'cm':
            L = L / 10
        elif unit == 'dm':
            L = L / 100
        elif unit == 'm':
            L = L / 1000

        return L

    def multiframe_bodyparts_to_polygon(self,
                                        data: np.ndarray,
                                        video_name: Optional[str] = None,
                                        animal_name: Optional[str] = None,
                                        verbose: Optional[bool] = False,
                                        cap_style: Literal['round', 'square', 'flat'] = 'round',
                                        parallel_offset: int = 1,
                                        pixels_per_mm: Optional[float] = None,
                                        simplify_tolerance: float = 2,
                                        preserve_topology: bool = True,
                                        core_cnt: int = -1) -> List[Polygon]:
        """
        Convert multidimensional NumPy array representing body part coordinates to a list of Polygons.

        :param np.ndarray data: NumPy array of body part coordinates. Each subarray represents the coordinates of a body part.
        :param Literal['round', 'square', 'flat'] cap_style: Style of line cap for parallel offset. Options: 'round', 'square', 'flat'.
        :param int parallel_offset: Offset distance for parallel lines. Default is 1.
        :param float simplify_tolerance: Tolerance parameter for simplifying geometries. Default is 2.

        :example:
        >>> data = np.array([[[364, 308], [383, 323], [403, 335], [423, 351]],[[356, 307], [376, 319], [396, 331], [419, 347]]])
        >>> GeometryMixin().multiframe_bodyparts_to_polygon(data=data)
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if pixels_per_mm is not None:
            check_float(name='PIXELS PER MM', value=pixels_per_mm, min_value=0.1, raise_error=True)
            parallel_offset = parallel_offset / pixels_per_mm
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.bodyparts_to_polygon,
                                          parallel_offset=parallel_offset,
                                          cap_style=cap_style,
                                          simplify_tolerance=simplify_tolerance,
                                          preserve_topology=preserve_topology)
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                if verbose:
                    if not video_name and not animal_name: print(f'Computing polygon {cnt+1}/{data.shape[0]}...')
                    elif not video_name and animal_name: print(f'Computing polygon {cnt + 1}/{data.shape[0]} (Animal: {animal_name})...')
                    elif video_name and not animal_name: print(f'Computing polygon {cnt + 1}/{data.shape[0]} (Video: {video_name})...')
                    else: print(f'Computing polygon {cnt + 1}/{data.shape[0]} (Video: {video_name}, Animal: {animal_name})...')
                results.append(mp_return)

        timer.stop_timer()
        stdout_success(msg='Polygons complete.', elapsed_time=timer.elapsed_time_str)
        pool.join(); pool.terminate()
        return results

    def multiframe_bodyparts_to_circle(self,
                                       data: np.ndarray,
                                       parallel_offset: int = 1,
                                       core_cnt: int = -1) -> List[Polygon]:
        """
        :example:
        >>> data = np.random.randint(0, 100, (100, 2))
        >>> circles = GeometryMixin().multiframe_bodyparts_to_circle(data=data)
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.bodyparts_to_circle,
                                          parallel_offset=parallel_offset)
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

        pool.join(); pool.terminate()
        return results

    @staticmethod
    def delaunay_triangulate_keypoints(data: np.ndarray) -> List[Polygon]:
        """
        Triangulates a set of 2D keypoints. E.g., use to polygonize animal hull.

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

        :param np.ndarray data: NumPy array of body part coordinates. Each subarray represents the coordinates of a body part.
        :returns  List[Polygon]: A list of `Polygon` objects representing the triangles formed by the Delaunay triangulation.

        :example:
        >>> data = np.array([[126, 122],[152, 116],[136,  85],[167, 172],[161, 206],[197, 193],[191, 237]])
        >>> triangulated_hull = GeometryMixin().delaunay_triangulate_keypoints(data=data)
        """

        check_instance(source=GeometryMixin().delaunay_triangulate_keypoints.__name__, instance=data, accepted_types= np.ndarray)
        if data.ndim != 2: raise InvalidInputError(msg=f'Triangulate requires 2D array, got {data.ndim}', source=GeometryMixin.delaunay_triangulate_keypoints.__name__)
        return triangulate(MultiPoint(data.astype(np.int64)))

    def multiframe_bodyparts_to_line(self,
                                     data: np.ndarray,
                                     core_cnt: Optional[int] = -1) -> List[LineString]:
        """
        Convert multiframe body-parts data to a list of LineString objects using multiprocessing.

        :param np.ndarray data: Input array representing multiframe body-parts data. It should be a 3D array with dimensions (frames, points, coordinates).
        :param Optional[int] core_cnt: Number of CPU cores to use for parallel processing. If set to -1, the function will automatically determine the available core count.
        :return List[LineString]: A list of LineString objects representing the body-parts trajectories.

        :example:
        >>> data = np.random.randint(0, 100, (100, 2))
        >>> data = data.reshape(50,-1, data.shape[1])
        >>> lines = GeometryMixin().multiframe_bodyparts_to_line(data=data)
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if data.ndim != 3:
            raise InvalidInputError(msg=f'Multiframe body-parts to linestring expects a 3D array, got {data.ndim}', source=GeometryMixin.bodyparts_to_line.__name__)
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.bodyparts_to_line, data, chunksize=1)):
                results.append(result)
        pool.join(); pool.terminate()
        return results

    def multiframe_compute_pct_shape_overlap(self,
                                           shape_1: List[Polygon],
                                           shape_2: List[Polygon],
                                           core_cnt: Optional[int] = -1,
                                           video_name: Optional[str] = None,
                                           verbose: Optional[bool] = False,
                                           animal_names: Optional[Tuple[str]] = None) -> List[float]:
        """
        Compute the percentage overlap between corresponding Polygons in two lists.

        :param List[Polygon] shape_1: List of Polygons.
        :param List[Polygon] shape_2: List of Polygons with the same length as shape_1.
        :param int core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        :return List[float]: List of percentage overlap between corresponding Polygons.

        :example:

        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2): raise InvalidInputError(msg=f'shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}', source=GeometryMixin.multiframe_compute_pct_shape_overlap.__name__)
        input_dtypes = list(set([type(x) for x in shape_1] + [type(x) for x in shape_2]))
        if len(input_dtypes) > 1:
            raise InvalidInputError(msg=f'shape_1 and shape_2 contains more than 1 dtype {input_dtypes}', source=GeometryMixin.multiframe_compute_pct_shape_overlap.__name__)
        check_instance(source=GeometryMixin.multiframe_compute_pct_shape_overlap.__name__, instance=shape_1[0], accepted_types=(LineString, Polygon))
        data, results, timer = np.column_stack((shape_1, shape_2)), [], SimbaTimer(start=True)
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.compute_pct_shape_overlap, data, chunksize=1)):
                if verbose:
                    if not video_name and not animal_names: print(f'Computing % overlap {cnt+1}/{data.shape[0]}...')
                    elif not video_name and animal_names: print(f'Computing % overlap {cnt + 1}/{data.shape[0]} (Animals: {animal_names})...')
                    elif video_name and not animal_names: print(f'Computing % overlap {cnt + 1}/{data.shape[0]} (Video: {video_name})...')
                    else: print(f'Computing % overlap {cnt + 1}/{data.shape[0]} (Video: {video_name}, Animals: {animal_names})...')
                results.append(result)
        timer.stop_timer()
        stdout_success(msg='Compute overlap complete.', elapsed_time=timer.elapsed_time_str)
        pool.join(); pool.terminate()
        return results

    def multiframe_compute_shape_overlap(self,
                                         shape_1: List[Polygon],
                                         shape_2: List[Polygon],
                                         core_cnt: Optional[int] = -1,
                                         verbose: Optional[bool] = False,
                                         names: Optional[Tuple[str]] = None) -> List[int]:
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

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2): raise InvalidInputError(msg=f'shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}', source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__)
        input_dtypes = list(set([type(x) for x in shape_1] + [type(x) for x in shape_2]))
        if len(input_dtypes) > 1:
            raise InvalidInputError(msg=f'shape_1 and shape_2 contains more than 1 dtype {input_dtypes}', source=GeometryMixin.multiframe_compute_shape_overlap.__name__)
        check_instance(source=GeometryMixin.multiframe_compute_shape_overlap.__name__, instance=shape_1[0], accepted_types=(LineString, Polygon))
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.compute_shape_overlap, data, chunksize=1)):
                if verbose:
                    if not names:
                        print(f'Computing overlap {cnt + 1}/{data.shape[0]}...')
                    else:
                        print(f'Computing overlap {cnt + 1}/{data.shape[0]} (Shape 1: {names[0]}, Shape 2: {names[1]}, Video: {names[2]}...)')
                results.append(result)

        pool.join(); pool.terminate()
        return results

    def multiframe_shape_distance(self,
                                  shape_1: List[Union[LineString, Polygon]],
                                  shape_2: List[Union[LineString, Polygon]],
                                  pixels_per_mm: float,
                                  unit: Literal['mm', 'cm', 'dm', 'm'] = 'mm',
                                  core_cnt = -1) -> List[float]:
        """
        Compute shape distances between corresponding shapes in two lists of LineString or Polygon geometries for multiple frames.

        :param List[Union[LineString, Polygon]] shape_1: List of LineString or Polygon geometries.
        :param List[Union[LineString, Polygon]] shape_2: List of LineString or Polygon geometries with the same length as shape_1.
        :param float pixels_per_mm: Conversion factor from pixels to millimeters.
        :param Literal['mm', 'cm', 'dm', 'm'] unit: Unit of measurement for the result. Options: 'mm', 'cm', 'dm', 'm'. Default: 'mm'.
        :param core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        :return List[float]: List of shape distances between corresponding shapes in passed unit.
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        check_float(name='PIXELS PER MM', value=pixels_per_mm, min_value=0.0)
        check_if_valid_input(name='UNIT', input=unit, options=['mm', 'cm', 'dm', 'm'])
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2): raise InvalidInputError(msg=f'shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}', source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__)
        check_float(name='pixels_per_mm', value=pixels_per_mm, min_value=0.0)
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.shape_distance,
                                          pixels_per_mm=pixels_per_mm,
                                          unit=unit)
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(result)

        pool.join(); pool.terminate()
        return results

    def multiframe_minimum_rotated_rectangle(self,
                                             shapes: List[Polygon],
                                             video_name: Optional[str] = None,
                                             verbose: Optional[bool] = False,
                                             animal_name: Optional[bool] = None,
                                             core_cnt: int = -1) -> List[Polygon]:
        """
        Compute the minimum rotated rectangle for each Polygon in a list using mutiprocessing.

        :param List[Polygon] shapes: List of Polygons.
        :param core_cnt: Number of CPU cores to use for parallel processing. Default is -1, which uses all available cores.
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.minimum_rotated_rectangle, shapes, chunksize=1)):
                if verbose:
                    if not video_name and not animal_name: print(f'Rotating polygon {cnt+1}/{len(shapes)}...')
                    elif not video_name and animal_name: print(f'Rotating polygon {cnt + 1}/{len(shapes)} (Animal: {animal_name})...')
                    elif video_name and not animal_name: print(f'Rotating polygon {cnt + 1}/{len(shapes)} (Video: {video_name})...')
                    else: print(f'Rotating polygon {cnt + 1}/{len(shapes)} (Video: {video_name}, Animal: {animal_name})...')
                results.append(result)

        timer.stop_timer()
        stdout_success(msg='Rotated rectangles complete.', elapsed_time=timer.elapsed_time_str)
        pool.join(); pool.terminate()
        return results

    @staticmethod
    @njit('(float32[:,:,:], float64[:])')
    def static_point_lineside(lines: np.ndarray,
                              point: np.ndarray) -> np.ndarray:

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
            v = ((lines[i][1][0] - lines[i][0][0]) * (point[1] - lines[i][0][1]) - (lines[i][1][1] - lines[i][0][1]) * (
                        point[0] - lines[i][0][0]))
            if v >= threshhold:
                results[i] = 2
            elif v <= -threshhold:
                results[i] = 1
            else:
                results[i] = 0
        return results

    @staticmethod
    @njit('(float32[:,:,:], float32[:, :])')
    def point_lineside(lines: np.ndarray,
                       points: np.ndarray) -> np.ndarray:
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
            v = ((line[1][0] - line[0][0]) * (point[1] - line[0][1]) - (line[1][1] - line[0][1]) * (
                        point[0] - line[0][0]))
            if v >= threshhold:
                results[i] = 2
            elif v <= -threshhold:
                results[i] = 1
            else:
                results[i] = 0
        return results


    @staticmethod
    @njit('(int64[:,:], int64[:])')
    def extend_line_to_bounding_box_edges(line_points: np.ndarray,
                                          bounding_box: np.ndarray) -> np.ndarray:

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
            intersection_points = np.array([[x1, max(min_y, 0)], [x1, min(max_y, min_y)]]).astype(np.float32)
        elif y1 == y2:
            intersection_points = np.array([[min_x, y1], [max_x, y1]]).astype(np.float32)
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Calculate intersection points with the bounding box boundaries
            x_min_intersection = (min_y - intercept) / slope
            x_max_intersection = (max_y - intercept) / slope

            # Clip the intersection points to ensure they are within the valid range
            # x_min_intersection = np.clip(x_min_intersection, min_x, max_x)
            # x_max_intersection = np.clip(x_max_intersection, min_x, max_x)

            intersection_points = np.array([[x_min_intersection, min_y],
                                            [x_max_intersection, max_y]]).astype(np.float32)

        return intersection_points


    @staticmethod
    def line_split_bounding_box(intersections: np.ndarray,
                                bounding_box: np.ndarray) -> GeometryCollection:

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
        original_polygon = Polygon([(bounding_box[0], bounding_box[1]),
                                    (bounding_box[2], bounding_box[1]),
                                    (bounding_box[2], bounding_box[3]),
                                    (bounding_box[0], bounding_box[3])])

        return split(original_polygon, extended_line)

    def multiframe_length(self,
                          shapes: List[Union[LineString, MultiLineString]],
                          pixels_per_mm: float,
                          core_cnt: int = -1,
                          unit: Literal['mm', 'cm', 'dm', 'm'] = 'mm') -> List[float]:

        """
        :example:
        >>> data = np.random.randint(0, 100, (5000, 2))
        >>> data = data.reshape(2500,-1, data.shape[1])
        >>> lines = GeometryMixin().multiframe_bodyparts_to_line(data=data)
        >>> lengths = GeometryMixin().multiframe_length(shapes=lines, pixels_per_mm=1.0)
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        check_float(name='PIXELS PER MM', value=pixels_per_mm, min_value=0.0)
        check_if_valid_input(name='UNIT', input=unit, options=['mm', 'cm', 'dm', 'm'])
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.length,
                                          pixels_per_mm=pixels_per_mm,
                                          unit=unit)
            for cnt, result in enumerate(pool.imap(constants, shapes, chunksize=1)):
                results.append(result)
        pool.join(); pool.terminate()
        return results

    def multiframe_union(self,
                         shapes: Iterable[Union[LineString, MultiLineString]],
                         core_cnt: int = -1) -> Iterable[Union[LineString, MultiLineString]]:
        """
        :example:
        >>> data_1 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
        >>> data_2 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
        >>> polygon_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_1)
        >>> polygon_2 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_2)
        >>> data = np.array([polygon_1, polygon_2]).T
        >>> unions = GeometryMixin().multiframe_union(shapes=data)
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin().union, shapes, chunksize=1)):
                results.append(result)
        pool.join(); pool.terminate()
        return results

    def multiframe_symmetric_difference(self,
                                        shapes: Iterable[Union[LineString, MultiLineString]],
                                        core_cnt: int = -1):
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
        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin().symmetric_difference, shapes, chunksize=1)):
                results.append(result)
        pool.join(); pool.terminate()
        return results


    def multiframe_delaunay_triangulate_keypoints(self,
                                                  data: np.ndarray,
                                                  core_cnt: int = -1) -> List[List[Polygon]]:

        """
        >>> data_path = '/Users/simon/Desktop/envs/troubleshooting/Rat_NOR/project_folder/csv/machine_results/08102021_DOT_Rat7_8(2).csv'
        >>> data = pd.read_csv(data_path, index_col=0).head(1000).iloc[:, 0:21]
        >>> data = data[data.columns.drop(list(data.filter(regex='_p')))]
        >>> animal_data = data.values.reshape(len(data), -1, 2).astype(int)
        >>> tri = GeometryMixin().multiframe_delaunay_triangulate_keypoints(data=animal_data)
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        check_instance(source=GeometryMixin().multiframe_delaunay_triangulate_keypoints.__name__, instance=data, accepted_types=np.ndarray)
        if data.ndim != 3:
            raise InvalidInputError(msg=f'Multiframe delaunay triangulate keypointstriangulate keypoints expects a 3D array, got {data.ndim}', source=GeometryMixin.multiframe_delaunay_triangulate_keypoints.__name__)
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin().delaunay_triangulate_keypoints, data, chunksize=1)):
                results.append(result)

        pool.join(); pool.terminate()
        return results

    def multiframe_difference(self,
                              shapes: Iterable[Union[LineString, Polygon, MultiPolygon]],
                              core_cnt: Optional[int] = -1,
                              verbose: Optional[bool] = False,
                              animal_names: Optional[str] = None,
                              video_name: Optional[str] = None) -> List[Union[Polygon, MultiPolygon]]:
        """
        Compute the multi-frame difference for a collection of shapes using parallel processing.

        :param Iterable[Union[LineString, Polygon, MultiPolygon]] shapes: A collection of shapes, where each shape is a list containing two geometries.
        :param int core_cnt: The number of CPU cores to use for parallel processing. Default is -1, which automatically detects the available cores.
        :param Optional[bool] verbose: If True, print progress messages during computation. Default is False.
        :param Optional[str] animal_names: Optional string representing the names of animals for informative messages.
        :param Optional[str]video_name: Optional string representing the name of the video for informative messages.
        :return List[Union[Polygon, MultiPolygon]]: A list of geometries representing the multi-frame difference.
        """

        check_instance(source=f'{GeometryMixin().multiframe_difference.__name__} shapes', instance=shapes, accepted_types=list)
        for i in shapes:
            check_instance(source=f'{GeometryMixin().multiframe_difference.__name__} shapes {i}', instance=i, accepted_types=list)
            check_iterable_length(f'{GeometryMixin().multiframe_difference.__name__} shapes {i}', val=len(i), exact_accepted_length=2)
            for j in i:
                check_instance(source=f'{GeometryMixin().multiframe_difference.__name__} shapes', instance=j, accepted_types=(LineString, Polygon, MultiPolygon))
        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin().difference, shapes, chunksize=1)):
                if verbose:
                    if not video_name and not animal_names: print(f'Computing geometry difference {cnt + 1}/{len(shapes)}...')
                    elif not video_name and animal_names: print(f'Computing geometry difference {cnt + 1}/{len(shapes)} (Animals: {animal_names})...')
                    elif video_name and not animal_names: print(f'Computing geometry difference {cnt + 1}/{len(shapes)} (Video: {video_name})...')
                    else: print(f'Computing geometry difference {cnt + 1}/{len(shapes)} (Video: {video_name}, Animals: {animal_names})...')
                results.append(result)

        timer.stop_timer()
        stdout_success(msg='Multi-frame difference compute complete', elapsed_time=timer.elapsed_time_str)
        pool.join(); pool.terminate()
        return results

    def multiframe_area(self,
                        shapes: List[Union[MultiPolygon, Polygon]],
                        pixels_per_mm: float,
                        core_cnt: Optional[int] = -1,
                        verbose: Optional[bool] = False,
                        video_name: Optional[bool] = None,
                        animal_names: Optional[bool] = None) -> np.ndarray:

        check_instance(source=f'{GeometryMixin().multiframe_area.__name__} shapes', instance=shapes, accepted_types=list)
        for i in shapes:
            check_instance(source=f'{GeometryMixin().multiframe_difference.__name__} shapes {i}', instance=i, accepted_types=(MultiPolygon, Polygon))
        check_float(name=f'{self.__class__.__name__} pixels_per_mm', value=pixels_per_mm, min_value=0.01)
        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results, timer = [], SimbaTimer(start=True)
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.area,
                                          pixels_per_mm=pixels_per_mm)
            for cnt, result in enumerate(pool.imap(constants, shapes, chunksize=1)):
                if verbose:
                    if not video_name and not animal_names:
                        print(f'Computing area {cnt + 1}/{len(shapes)}...')
                    elif not video_name and animal_names:
                        print(f'Computing % area {cnt + 1}/{len(shapes)} (Animals: {animal_names})...')
                    elif video_name and not animal_names:
                        print(f'Computing % area {cnt + 1}/{len(shapes)} (Video: {video_name})...')
                    else:
                        print(f'Computing % area {cnt + 1}/{len(shapes)} (Video: {video_name}, Animals: {animal_names})...')
                results.append(result)

        timer.stop_timer()
        stdout_success(msg='Multi-frame area compute complete', elapsed_time=timer.elapsed_time_str)
        pool.join(); pool.terminate()
        return results

# polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
# GeometryMixin().area(shape=polygon_1, pixels_per_mm=4.9)



# point_of_interest = Point(0, 0)
# obstacle1 = LineString([(5, 0), (5, 10), (2, 2), (2, 1)])
# visible_space = GeometryMixin().rotational_sweep_visibility(point=point_of_interest, obstacle=[obstacle1])
#
# img = GeometryMixin.view_shapes(shapes=[obstacle1])
# # #
# cv2.imshow('ssadf', img)
# cv2.waitKey(50000)

#data = np.array([[10, 10],[20, 10],[30, 10],[40, 10]])
#line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 10],[20, 10],[30, 10],[40, 10]]))
# data_1 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
# data_2 = np.random.randint(0, 100, (5000, 2)).reshape(1000,-1, 2)
# polygon_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_1)
# polygon_2 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_2)
# data = np.array([polygon_1, polygon_2]).T
# unions = GeometryMixin().multiframe_symmetric_difference(shapes=data)
# len(unions)









# data = data.reshape(1000,-1, data.shape[1])
# lines_1 = GeometryMixin().multiframe_bodyparts_to_line(data=data)
# data = np.random.randint(0, 100, (5000, 2))
# data = data.reshape(2500,-1, data.shape[1])
# lines_2 = GeometryMixin().multiframe_bodyparts_to_line(data=data)
# lines_1.append(lines_2)
#
# data = lines_1.reshape(2500,-1, lines_1.shape[1])
#

# lengths = GeometryMixin().multiframe_length(shapes=lines, pixels_per_mm=1.0)
#






# skeleton = np.array([[[5, 5], [1, 10]],
#                      [[5, 5], [9, 10]],
#                      [[9, 10], [1, 10]],
#                      [[9, 10], [9, 25]],
#                      [[1, 10], [1, 25]],
#                      [[9, 25], [5, 50]],
#                      [[1, 25], [5, 50]]])
#
# shape, _ = GeometryMixin().bodyparts_to_skeleton(bodyparts=skeleton, shape=True)
# img = GeometryMixin.view_shapes(shapes=[shape])
# # #
# cv2.imshow('ssadf', img)
# cv2.waitKey(50000)
#


#multipolygon_1 = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
#GeometryMixin().get_center(shape=multipolygon_1)


#line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
# buffered_polygon = GeometryMixin().buffer_shape(shape=line_1, size_mm=5, pixels_per_mm=1)
# img = GeometryMixin.view_shapes(shapes=[line_1, buffered_polygon])
# # #
# cv2.imshow('ssadf', img)
# cv2.waitKey(50000)

# line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 10], [10, 20], [10, 30], [10, 40]]))
# line_2 = GeometryMixin().bodyparts_to_line(np.array([[10, 50], [10, 60], [10, 70], [10, 80]]))
# covered = GeometryMixin().union(shapes=[line_1, line_2])
#
# covered





#
# img = GeometryMixin.view_shapes(shapes=[line_1, line_2])
# # #
# cv2.imshow('ssadf', img)
# cv2.waitKey(50000)
#



#img = GeometryMixin.view_shapes(shapes=[line_1, polygon_1, multipolygon_1])




# line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
# length = GeometryMixin().length(shape=line_1, pixels_per_mm=1.0)

# def symmetric_difference(shapes: List[Union[LineString, Polygon, MultiPolygon]]) -> Union[Polygon, MultiPolygon]:
#     check_iterable_length(source=GeometryMixin.union.__name__, val=len(shapes), min=2)
#     for shape in shapes: check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape, accepted_types=(LineString, Polygon, MultiPolygon))
#     results = deepcopy(shapes)
#     for c in itertools.combinations(list(range(0, len(shapes))), 2):
#         results[c[0]] = results[c[0]].convex_hull.difference(results[c[1]].convex_hull)
#         results[c[1]] = results[c[1]].convex_hull.difference(results[c[0]].convex_hull)
#
#     return results


    # union_geometry = ops.cascaded_union(shapes)
    # result_geometry = ops.cascaded_union(shapes).symmetric_difference(union_geometry)
    # #result_geometries = [geometry for geometry in result_geometry if not geometry.is_empty]
    # return result_geometry
    # #return result_geometry
    # # union_geometry = unary_union(shapes)
    # # result_geometries = [geometry.difference(union_geometry.intersection(geometry)) for geometry in shapes]
    # # result_geometries = [geometry for geometry in result_geometries if not geometry.is_empty]
    # # return result_geometries
#
#
#
#
# polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
#
# polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 25],
#                                                            [25, 75],
#                                                            [90, 25],
#                                                            [90, 75]]))
#
#
# polygon_3 = GeometryMixin().bodyparts_to_polygon(np.array([[1, 25],
#                                                            [1, 75],
#                                                            [110, 25],
#                                                            [110, 75]]))
#
# symmetric_difference = symmetric_difference(shapes=[polygon_1, polygon_3])
#
# symmetric_difference.exterior.convex_hull.a
#
# # symmetric_difference = polygon_1.exterior.symmetric_difference(polygon_3)
#
# # import shapely.ops
# #
# # #symmetric_difference_all([polygon_3])
# #
# # #diff = GeometryMixin().difference(shape = polygon_1, overlap_shapes=[polygon_2, polygon_3])
# #
# # #
# img = GeometryMixin.view_shapes(shapes=[symmetric_difference[1]])
# # #
# cv2.imshow('ssadf', img)
# cv2.waitKey(50000)
#
