import numpy as np
from typing import Optional, List, Union, Tuple, Iterable
from typing_extensions import Literal
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString
from shapely.ops import unary_union, linemerge
from copy import deepcopy
import cv2
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import find_core_cnt, find_max_vertices_coordinates
from simba.utils.checks import check_int
import multiprocessing
import functools
import itertools
from simba.utils.checks import check_instance, check_if_valid_input, check_iterable_length, check_float
from simba.utils.enums import Defaults
from simba.utils.lookups import get_color_dict

CAP_STYLE_MAP = {'round': 1, 'square': 2, 'flat': 3}

class GeometryMixin(object):

    """
    Methods to perform geometry transformation of pose-estimation data. This includes creating bounding boxes,
    line objects, circles etc. from pose-estimated body-parts and computing relationships between created shapes.

    Very much wip and relies hevaily on `shapley <https://shapely.readthedocs.io/en/stable/manual.html>`_.
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

        polygon = Polygon(LineString(data.tolist()).buffer(distance=parallel_offset,
                                                   cap_style=CAP_STYLE_MAP[cap_style]).simplify(tolerance=simplify_tolerance,
                                                                                                preserve_topology=preserve_topology).convex_hull)
        return polygon

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
            check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape, accepted_types=(LineString, Polygon))
        check_iterable_length(source=GeometryMixin.compute_pct_shape_overlap.__name__, val=len(shapes), exact_accepted_length=2)

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
    def difference(shape = Union[LineString, Polygon, MultiPolygon],
                   overlap_shapes = List[Union[LineString, Polygon, MultiPolygon]]) -> Polygon:
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

        check_iterable_length(source=GeometryMixin.difference.__name__, val=len(overlap_shapes), min=1)
        for overlap_shap in overlap_shapes:
            check_instance(source=GeometryMixin.difference.__name__, instance=overlap_shap, accepted_types=(LineString, Polygon, MultiPolygon))
        check_instance(source=GeometryMixin.difference.__name__, instance=shape, accepted_types=(LineString, Polygon))

        for overlap_shap in overlap_shapes:
            if isinstance(overlap_shap, MultiPolygon):
                for geo in overlap_shap.geoms:
                    shape = shape.difference(geo)
            else:
                shape = shape.difference(overlap_shap)
        return shape

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
        for shape in shapes: check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape,
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
        Helper function to draw shapes on white canvas .

        :example:
        >>> multipolygon_1 = MultiPolygon([Polygon([[200, 110],[200, 100],[200, 100],[200, 110]]), Polygon([[70, 70],[70, 60],[10, 50],[1, 70]])])
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[100, 110],[100, 100],[110, 100],[110, 110]]))
        >>> line_1 = GeometryMixin().bodyparts_to_line(np.array([[10, 70],[20, 60],[30, 50],[40, 70]]))
        >>> img = GeometryMixin.view_shapes(shapes=[line_1, polygon_1, multipolygon_1])
        """

        for i in shapes:
            check_instance(source=GeometryMixin.view_shapes.__name__, instance=i, accepted_types=(LineString, Polygon, MultiPolygon, MultiLineString))
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

        return img

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
        return shape.minimum_rotated_rectangle


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
        check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape, accepted_types=LineString)
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
                                        cap_style: Literal['round', 'square', 'flat'] = 'round',
                                        parallel_offset: int = 1,
                                        simplify_tolerance: float = 2,
                                        preserve_topology: bool = True,
                                        core_cnt: int = -1) -> List[Polygon]:
        """
        :example:
        >>> data = np.array([[[364, 308], [383, 323], [403, 335], [423, 351]],[[356, 307], [376, 319], [396, 331], [419, 347]]])
        >>> GeometryMixin().multiframe_bodyparts_to_polygon(data=data)
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.bodyparts_to_polygon,
                                          parallel_offset=parallel_offset,
                                          cap_style=cap_style,
                                          simplify_tolerance=simplify_tolerance,
                                          preserve_topology=preserve_topology)
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.bodyparts_to_circle,
                                          parallel_offset=parallel_offset)
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

        pool.join(); pool.terminate()
        return results

    def multiframe_bodyparts_to_line(self,
                                     data: np.ndarray,
                                     core_cnt: Optional[int] = -1) -> List[LineString]:
        """
        Convert multiframe body-parts data to a list of LineString objects in parallel.

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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.bodyparts_to_line, data, chunksize=1)):
                results.append(result)
        pool.join(); pool.terminate()
        return results

    def multifrm_compute_pct_shape_overlap(self,
                                          shape_1: List[Polygon],
                                          shape_2: List[Polygon],
                                          core_cnt = -1) -> List[float]:
        """
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2): raise InvalidInputError(msg=f'shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}', source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__)
        input_dtypes = list(set([type(x) for x in shape_1] + [type(x) for x in shape_2]))
        if len(input_dtypes) > 1:
            raise InvalidInputError(msg=f'shape_1 and shape_2 contains more than 1 dtype {input_dtypes}', source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__)
        check_instance(source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__, instance=shape_1[0], accepted_types=(LineString, Polygon))
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.compute_pct_shape_overlap, data, chunksize=1)):
                results.append(result)

        pool.join(); pool.terminate()
        return results

    def multiframe_shape_distance(self,
                                  shape_1: List[Union[LineString, Polygon]],
                                  shape_2: List[Union[LineString, Polygon]],
                                  pixels_per_mm: float,
                                  unit: Literal['mm', 'cm', 'dm', 'm'] = 'mm',
                                  core_cnt = -1) -> List[float]:

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        check_float(name='PIXELS PER MM', value=pixels_per_mm, min_value=0.0)
        check_if_valid_input(name='UNIT', input=unit, options=['mm', 'cm', 'dm', 'm'])
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2): raise InvalidInputError(msg=f'shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}', source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__)
        check_float(name='pixels_per_mm', value=pixels_per_mm, min_value=0.0)
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            constants = functools.partial(GeometryMixin.shape_distance,
                                          pixels_per_mm=pixels_per_mm,
                                          unit=unit)
            for cnt, result in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(result)

        pool.join(); pool.terminate()
        return results

    def multiframe_minimum_rotated_rectangle(self,
                                             shapes: List[Polygon],
                                             core_cnt: int = -1) -> List[Polygon]:

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.minimum_rotated_rectangle, shapes, chunksize=1)):
                results.append(result)

        pool.join(); pool.terminate()
        return results


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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin().union, shapes, chunksize=1)):
                results.append(result)
        pool.join(); pool.terminate()
        return results

    def multiframe_symmetric_difference(self,
                                        shapes: Iterable[Union[LineString, MultiLineString]],
                                        core_cnt: int = -1):
        """
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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.MAX_TASK_PER_CHILD.value) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin().symmetric_difference, shapes, chunksize=1)):
                results.append(result)
        pool.join(); pool.terminate()
        return results











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
