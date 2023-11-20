import functools
import itertools
import multiprocessing
from copy import deepcopy
from typing import List, Optional, Union

import cv2
import numpy as np
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)
from shapely.ops import unary_union
from typing_extensions import Literal

from simba.utils.checks import (check_float, check_if_valid_input,
                                check_instance, check_int,
                                check_iterable_length)
from simba.utils.enums import Defaults
from simba.utils.errors import InvalidInputError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import find_core_cnt, find_max_vertices_coordinates

CAP_STYLE_MAP = {"round": 1, "square": 2, "flat": 3}


class GeometryMixin(object):

    """
    Methods to perform geometry transformation of pose-estimation data. This includes creating bounding boxes,
    line objects, circles etc. from pose-estimated body-parts and computing relationships between created shapes.

    Very much wip and relies hevaily on `shapley <https://shapely.readthedocs.io/en/stable/manual.html>`_.
    """

    def __init__(self):
        pass

    @staticmethod
    def bodyparts_to_polygon(
        data: np.ndarray,
        cap_style: Literal["round", "square", "flat"] = "round",
        parallel_offset: int = 1,
        simplify_tolerance: float = 2,
        preserve_topology: bool = True,
    ) -> Polygon:
        """
        .. image:: _static/img/bodyparts_to_polygon.png
           :width: 400
           :align: center

        :example:
        >>> data = [[364, 308],[383, 323],[403, 335],[423, 351]]
        >>> GeometryMixin().bodyparts_to_polygon(data=data)
        """

        polygon = Polygon(
            LineString(data.tolist())
            .buffer(distance=parallel_offset, cap_style=CAP_STYLE_MAP[cap_style])
            .simplify(tolerance=simplify_tolerance, preserve_topology=preserve_topology)
            .convex_hull
        )
        return polygon

    @staticmethod
    def bodyparts_to_circle(data: np.ndarray, parallel_offset: int = 1) -> Polygon:
        """
        .. image:: _static/img/bodyparts_to_circle.png
           :width: 400
           :align: center

        :example:
        >>> data = np.array([364, 308])
        >>> polygon = GeometryMixin().bodyparts_to_circle(data=data)
        """

        if data.shape != (2,):
            raise InvalidInputError(
                msg=f"Cannot create circle data is not a (2,) array: " f"{data.shape}",
                source=GeometryMixin.bodyparts_to_circle.__name__,
            )

        return Point(data).buffer(parallel_offset)

    @staticmethod
    def buffer_shape(
        shape: Union[Polygon, LineString],
        size_mm: int,
        pixels_per_mm: float,
        cap_style: Literal["round", "square", "flat"] = "round",
    ) -> Polygon:
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
        return shape.buffer(
            distance=int(size_mm / pixels_per_mm), cap_style=CAP_STYLE_MAP[cap_style]
        )

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
            check_instance(
                source=GeometryMixin.compute_pct_shape_overlap.__name__,
                instance=shape,
                accepted_types=(LineString, Polygon),
            )
        check_iterable_length(
            source=GeometryMixin.compute_pct_shape_overlap.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )
        if shapes[0].intersects(shapes[1]):
            intersection = shapes[0].intersection(shapes[1])
            return round(
                (
                    intersection.area
                    / ((shapes[0].area + shapes[1].area) - intersection.area)
                    * 100
                ),
                2,
            )
        else:
            return 0.0

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
            source=GeometryMixin.compute_pct_shape_overlap.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )
        for shape in shapes:
            check_instance(
                source=GeometryMixin.compute_pct_shape_overlap.__name__,
                instance=shape,
                accepted_types=LineString,
            )
        return shapes[0].crosses(shapes[1])

    @staticmethod
    def is_shape_covered(
        shape: Union[LineString, Polygon, MultiPolygon],
        other_shape: Union[LineString, Polygon, MultiPolygon],
    ) -> bool:
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
        check_instance(
            source=GeometryMixin.is_shape_covered.__name__,
            instance=shape,
            accepted_types=(LineString, Polygon, MultiPolygon),
        )
        check_instance(
            source=GeometryMixin.is_shape_covered.__name__,
            instance=other_shape,
            accepted_types=(LineString, Polygon, MultiPolygon),
        )

        return shape.covers(other_shape)

    @staticmethod
    def shape_distance(
        shapes: List[Union[LineString, Polygon]],
        pixels_per_mm: float,
        unit: Literal["mm", "cm", "dm", "m"] = "mm",
    ) -> float:
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
                source=GeometryMixin.compute_pct_shape_overlap.__name__,
                instance=shape,
                accepted_types=(LineString, Polygon),
            )
        check_iterable_length(
            source=GeometryMixin.compute_pct_shape_overlap.__name__,
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
            raise InvalidInputError(
                msg=f"Body-parts to linestring expects a 2D array, got {data.ndim}",
                source=GeometryMixin.bodyparts_to_line.__name__,
            )
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
        check_instance(
            source=GeometryMixin.get_center.__name__,
            instance=shape,
            accepted_types=(MultiPolygon, LineString, Polygon),
        )
        return np.array(shape.centroid)

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
        .. image:: _static/img/is_containing.png
           :width: 500
           :align: center

        :example:
        """
        for i in shapes:
            check_instance(
                source=GeometryMixin.get_center.__name__,
                instance=i,
                accepted_types=(LineString, Polygon),
            )
        check_iterable_length(
            source=GeometryMixin.get_center.__name__,
            val=len(shapes),
            exact_accepted_length=2,
        )

        return shapes[0].contains(shapes[1])

    @staticmethod
    def difference(
        shape=Union[LineString, Polygon, MultiPolygon],
        overlap_shapes=List[Union[LineString, Polygon, MultiPolygon]],
    ) -> Polygon:
        """
        Calculate the difference between a shape and one or more potentially overlapping shapes.

        .. image:: _static/img/difference.png
           :width: 400
           :align: center

        .. image:: _static/img/difference_1.png
           :width: 400
           :align: center

        :param Union[LineString, Polygon, MultiPolygon] shape: The main geometry from which the difference is calculated.
        :param List[Union[LineString, Polygon, MultiPolygon]] overlap_shapes: A list of LineString, Polygon, or MultiPolygon geometries to subtract from the main shape.
        :return: The resulting geometry after subtracting the overlapping shapes.

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[10, 10], [10, 100], [100, 10], [100, 100]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[25, 25],[25, 75],[90, 25],[90, 75]]))
        >>> polygon_3 = GeometryMixin().bodyparts_to_polygon(np.array([[1, 25],[1, 75],[110, 25],[110, 75]]))
        >>> difference = GeometryMixin().difference(shape = polygon_1, overlap_shapes=[polygon_2, polygon_3])
        """

        check_iterable_length(
            source=GeometryMixin.difference.__name__, val=len(overlap_shapes), min=1
        )
        for overlap_shap in overlap_shapes:
            check_instance(
                source=GeometryMixin.difference.__name__,
                instance=overlap_shap,
                accepted_types=(LineString, Polygon, MultiPolygon),
            )
        check_instance(
            source=GeometryMixin.difference.__name__,
            instance=shape,
            accepted_types=(LineString, Polygon),
        )

        for overlap_shap in overlap_shapes:
            if isinstance(overlap_shap, MultiPolygon):
                for geo in overlap_shap.geoms:
                    shape = shape.difference(geo)
            else:
                shape = shape.difference(overlap_shap)
        return shape

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
    def symmetric_difference(
        shapes: List[Union[LineString, Polygon, MultiPolygon]]
    ) -> List[Union[Polygon, MultiPolygon]]:
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
        check_iterable_length(
            source=GeometryMixin.union.__name__, val=len(shapes), min=2
        )
        for shape in shapes:
            check_instance(
                source=GeometryMixin.compute_pct_shape_overlap.__name__,
                instance=shape,
                accepted_types=(LineString, Polygon, MultiPolygon),
            )
        results = deepcopy(shapes)
        for c in itertools.combinations(list(range(0, len(shapes))), 2):
            results[c[0]] = results[c[0]].convex_hull.difference(
                results[c[1]].convex_hull
            )
            results[c[1]] = results[c[1]].convex_hull.difference(
                results[c[0]].convex_hull
            )

        results = [geometry for geometry in results if not geometry.is_empty]
        return results

    @staticmethod
    def view_shapes(
        shapes: List[Union[LineString, Polygon, MultiPolygon]]
    ) -> np.ndarray:
        """
        Helper function to draw shapes on white canvas .

        :example:
        >>> multipolygon_1 = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[100, 110],[100, 100],[110, 100],[110, 110]]))
        >>> line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
        >>> img = GeometryMixin.view_shapes(shapes=[line_1, polygon_1, multipolygon_1])
        """

        for i in shapes:
            check_instance(
                source=GeometryMixin.view_shapes.__name__,
                instance=i,
                accepted_types=(LineString, Polygon, MultiPolygon),
            )
        max_vertices = find_max_vertices_coordinates(shapes=shapes, buffer=50)
        img = np.ones((max_vertices[0], max_vertices[1], 3), dtype=np.uint8) * 255
        colors = list(get_color_dict().values())
        for shape_cnt, shape in enumerate(shapes):
            if isinstance(shape, Polygon):
                cv2.polylines(
                    img,
                    [np.array(shape.exterior.coords).astype(np.int)],
                    True,
                    (colors[shape_cnt][::-1]),
                    thickness=2,
                )
                interior_coords = [
                    np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                    for interior in shape.interiors
                ]
                for interior in interior_coords:
                    cv2.polylines(
                        img,
                        [interior],
                        isClosed=True,
                        color=(colors[shape_cnt][::-1]),
                        thickness=2,
                    )
            if isinstance(shape, LineString):
                cv2.polylines(
                    img,
                    [np.array(shape.coords, dtype=np.int32)],
                    False,
                    (colors[shape_cnt][::-1]),
                    thickness=2,
                )
            if isinstance(shape, MultiPolygon):
                for polygon_cnt, polygon in enumerate(shape.geoms):
                    print(np.array(polygon.convex_hull.exterior.coords))
                    polygon_np = np.array(
                        (polygon.convex_hull.exterior.coords), dtype=np.int32
                    )
                    cv2.polylines(
                        img,
                        [polygon_np],
                        True,
                        (colors[shape_cnt + polygon_cnt + 4][::-1]),
                        thickness=2,
                    )
        return img

    @staticmethod
    def minimum_rotated_rectangle(shape=Polygon) -> bool:
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
            source=GeometryMixin.get_center.__name__,
            instance=shape,
            accepted_types=Polygon,
        )
        return shape.minimum_rotated_rectangle

    @staticmethod
    def length(
        shape: LineString,
        pixels_per_mm: float,
        unit: Literal["mm", "cm", "dm", "m"] = "mm",
    ) -> float:
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
        check_instance(
            source=GeometryMixin.compute_pct_shape_overlap.__name__,
            instance=shape,
            accepted_types=LineString,
        )
        L = shape.length
        if unit == "cm":
            L = L / 10
        elif unit == "dm":
            L = L / 100
        elif unit == "m":
            L = L / 1000

        return L

    def multiframe_bodyparts_to_polygon(
        self,
        data: np.ndarray,
        cap_style: Literal["round", "square", "flat"] = "round",
        parallel_offset: int = 1,
        simplify_tolerance: float = 2,
        preserve_topology: bool = True,
        core_cnt: int = -1,
    ) -> List[Polygon]:
        """
        :example:
        >>> data = np.array([[[364, 308], [383, 323], [403, 335], [423, 351]],[[356, 307], [376, 319], [396, 331], [419, 347]]])
        >>> GeometryMixin().multiframe_bodyparts_to_polygon(data=data)
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
            core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.bodyparts_to_polygon,
                parallel_offset=parallel_offset,
                cap_style=cap_style,
                simplify_tolerance=simplify_tolerance,
                preserve_topology=preserve_topology,
            )
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

        pool.join()
        pool.terminate()
        return results

    def multiframe_bodyparts_to_circle(
        self, data: np.ndarray, parallel_offset: int = 1, core_cnt: int = -1
    ) -> List[Polygon]:
        """
        :example:
        >>> data = np.random.randint(0, 100, (100, 2))
        >>> circles = GeometryMixin().multiframe_bodyparts_to_circle(data=data)
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
            core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value
        ) as pool:
            constants = functools.partial(
                GeometryMixin.bodyparts_to_circle, parallel_offset=parallel_offset
            )
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

        pool.join()
        pool.terminate()
        return results

    def multiframe_bodyparts_to_line(
        self, data: np.ndarray, core_cnt: int = -1
    ) -> List[LineString]:
        """ """

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
        results = []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin.bodyparts_to_line, data, chunksize=1)
            ):
                results.append(result)

        return results

    def multifrm_compute_pct_shape_overlap(
        self, shape_1: List[Polygon], shape_2: List[Polygon], core_cnt=-1
    ) -> List[float]:
        """ """

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
                source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__,
            )
        input_dtypes = list(
            set([type(x) for x in shape_1] + [type(x) for x in shape_2])
        )
        if len(input_dtypes) > 1:
            raise InvalidInputError(
                msg=f"shape_1 and shape_2 contains more than 1 dtype {input_dtypes}",
                source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__,
            )
        check_instance(
            source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__,
            instance=shape_1[0],
            accepted_types=(LineString, Polygon),
        )
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin.compute_pct_shape_overlap, data, chunksize=1)
            ):
                results.append(result)

        pool.join()
        pool.terminate()
        return results

    def multiframe_shape_distance(
        self,
        shape_1: List[Union[LineString, Polygon]],
        shape_2: List[Union[LineString, Polygon]],
        pixels_per_mm: float,
        unit: Literal["mm", "cm", "dm", "m"] = "mm",
        core_cnt=-1,
    ) -> List[float]:
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
            raise InvalidInputError(
                msg=f"shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}",
                source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__,
            )
        check_float(name="pixels_per_mm", value=pixels_per_mm, min_value=0.0)
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(
            core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value
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
        self, shapes: List[Polygon], core_cnt=-1
    ) -> List[Polygon]:
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
            core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value
        ) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin.minimum_rotated_rectangle, shapes, chunksize=1)
            ):
                results.append(result)

        pool.join()
        pool.terminate()
        return results


# multipolygon_1 = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
# GeometryMixin().get_center(shape=multipolygon_1)


# line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
# buffered_polygon = GeometryMixin().buffer_shape(shape=line_1, size_mm=5, pixels_per_mm=1)
# img = GeometryMixin.view_shapes(shapes=[line_1, buffered_polygon])
# # #
# cv2.imshow('ssadf', img)
# cv2.waitKey(50000)

line_1 = GeometryMixin().bodyparts_to_line(
    np.array([[10, 10], [10, 20], [10, 30], [10, 40]])
)
line_2 = GeometryMixin().bodyparts_to_line(
    np.array([[10, 50], [10, 60], [10, 70], [10, 80]])
)
covered = GeometryMixin().union(shapes=[line_1, line_2])

covered


#
# img = GeometryMixin.view_shapes(shapes=[line_1, line_2])
# # #
# cv2.imshow('ssadf', img)
# cv2.waitKey(50000)
#


# img = GeometryMixin.view_shapes(shapes=[line_1, polygon_1, multipolygon_1])


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

# for geom in diff.geoms:
#     print(geom)


# GeometryMixin.view_shapes(shapes=[line_1, polygon_1, multipolygon_1])


# GeometryMixin.view_shapes(shapes=[polygon_1])


# polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
# line_1 = GeometryMixin().bodyparts_to_line(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
# GeometryMixin().difference(shapes=[polygon_1, polygon_2])
# from shapely.geometry import Polygon, LineString, Point
# data_path = '/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/termites/project_folder/csv/outlier_corrected_movement_location/termite_test.csv'
#
#
#
# df = pd.read_csv(data_path, index_col=0).fillna(0).astype(np.int64)
# df = df[df.columns.drop(list(df.filter(regex='_p')))]
#
# data_1 = df.iloc[:, 0:8].values.reshape(len(df),-1,  2)
# data_2 = df.iloc[:, 8:16].values.reshape(len(df), -1,  2)
# #
# shapes_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_1)
# shapes_2 = GeometryMixin().multiframe_bodyparts_to_polygon(data=data_2)
#
# #GeometryMixin().multiframe_shape_distance(shape_1=shapes_1, shape_2=shapes_2, pixels_per_mm=1.0, unit='mm')
# GeometryMixin().multiframe_minimum_rotated_rectangle(shapes=shapes_1)


#
#
#
# data_1 = df.iloc[1, 0:8].values.reshape(-1, 2)
# line1 = GeometryMixin().bodyparts_to_line(data=data_1)
# polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
# polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
# #polygon_1 = GeometryMixin().bodyparts_to_polygon(data=data_1)
# # polygon_2 = GeometryMixin().bodyparts_to_polygon(data=rectangle_2)
# line_1 = GeometryMixin().bodyparts_to_line(np.array([[356, 307], [376, 319], [396, 331], [419, 347]]))
#
# polygon = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
# rectangle = GeometryMixin().minimum_rotated_rectangle(shape=polygon)
#
# polygon_1.minimum_rotated_rectangle
#
# diff = GeometryMixin().difference(shapes=[line_1, polygon_1])
#
# GeometryMixin.compute_pct_shape_overlap(shapes=[polygon_1, polygon_2])
#
#

# rectangle_2 = Polygon(np.array([[20, 20], [30, 30], [20, 30], [30, 20]]))
# GeometryMixin().is_containing(shapes=[rectangle_1, rectangle_2])
#
# rectangle_1.contains()

# polygon_1 = GeometryMixin().bodyparts_to_polygon(data=rectangle_1)
# polygon_2 = GeometryMixin().bodyparts_to_polygon(data=rectangle_2)
#
#
# polygon_1.contains()
#
# GeometryMixin().is_touching(shapes=[polygon_1, polygon_2])

# polygon_1.
# polygon_1.mi

#
#
# GeometryMixin().get_center(shape=polygon_1)


#
# polygon_1.distance(polygon_2)
#
# from shapely.geometry import Polygon
#
# rectangle_1 = Polygon([(0, 0),
#                        (10, 10),
#                        (0, 10),
#                        (10, 0)])
#
# rectangle_2 = Polygon([(10, 20),
#                        (30, 30),
#                        (20, 30),
#                        (30, 20)])
#
# distance = rectangle_1.distance(rectangle_2)
# print(distance)


# test = GeometryMixin()
# data = np.array([[[364, 308], [383, 323], [403, 335], [423, 351]],[[356, 307], [376, 319], [396, 331], [419, 347]]])
# test.multiframe_bodyparts_to_line(data=data)


# polygon_1 = BoundingBoxMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
# polygon_2 = BoundingBoxMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
# BoundingBoxMixin().compute_pct_shape_overlap(shapes=[polygon_1, polygon_2])
# # #
# #test = GeometryMixin()
# # #
# data_path = '/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/termites/project_folder/csv/outlier_corrected_movement_location/termite_test.csv'
# test = GeometryMixin()
# df = pd.read_csv(data_path, index_col=0).fillna(0).astype(np.int64)
# df = df[df.columns.drop(list(df.filter(regex='_p')))]
# data_1 = df.iloc[1, 0:8].values.reshape(-1, 2)
# line1 = test.bodyparts_to_line(data=data_1)
# # # #
# # # data_2 = df.iloc[2, 0:8].values.reshape(-1, 2)
# # #
# # # test.bodyparts_to_circle(data=data_1[0])
# #
# #
# # test.bodyparts_to_line(data=data_1)

#
# data_1 = df.iloc[:, 0:8].values.reshape(len(df),-1,  2)
# data_2 = df.iloc[:, 8:16].values.reshape(len(df), -1,  2)
# #
# polygons_1 = test.multiframe_bodyparts_to_polygon(data=data_1)
# polygons_2 = test.multiframe_bodyparts_to_polygon(data=data_2)
#
# polygons_2 = [polygons_1[-1]] + polygons_1[:-1]
# test.mulifrm_compute_pct_shape_overlap(shape_1=polygons_1, shape_2=polygons_2)
#

#
# data_2 = np.random.randint(0, 100, (100, 2))
# data_1 = np.random.randint(0, 100, (100, 2))
#
#
#
# data = df.iloc[:, 0:2].values
# circles = test.multiframe_bodyparts_to_circle(data=data)
#
#

# polygon_1 = test.bodyparts_to_polygon(data=data_1)
# polygon_2 = test.bodyparts_to_polygon(data=data_2)

# test.compute_pct_pylygon_overlap(shape_1=polygon_1, shape_2=polygon_2)

# polygon_1 = LineString([[364, 100],[383, 800],[403, 335],[423, 351]])
# polygon_2 = LineString([[364, 308],[383, 323],[403, 600],[423, 1]])
# test.compute_pct_pylygon_overlap(shape_1=polygon_1, shape_2=polygon_2)

# circle = test.bodyparts_to_circle(data=data[0])

# test.multiframe_bodyparts_to_polygon(data=data)


# .values.reshape(-1, 2)
