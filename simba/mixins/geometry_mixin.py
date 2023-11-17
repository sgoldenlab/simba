import numpy as np
from typing import Optional, List, Union
from typing_extensions import Literal
import pandas as pd
from shapely.geometry import Polygon, LineString, Point
from simba.utils.errors import InvalidInputError
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.read_write import find_core_cnt
from simba.utils.checks import check_int
import multiprocessing
import functools
import itertools
from simba.utils.checks import check_instance, check_if_valid_input

CAP_STYLE_MAP = {'round': 1, 'square': 2, 'flat': 3}

class GeometryMixin(object):

    """
    Methods to perform geometry transformation of pose-estimation data. This includes creating bounding boxes,
    line objects, circles etc. from pose-estimated body-parts and computing intersections.
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
           :width: 500
           :align: center

        :example:
        >>> data = [[364, 308],[383, 323],[403, 335],[423, 351]]
        >>> GeometryMixin().bodyparts_to_polygon(data=data)
        """

        polygon = LineString(data.tolist()).buffer(distance=parallel_offset,
                                                   cap_style=CAP_STYLE_MAP[cap_style]).simplify(tolerance=simplify_tolerance,
                                                                                                preserve_topology=preserve_topology)
        return polygon


    @staticmethod
    def bodyparts_to_circle(data: np.ndarray,
                            parallel_offset: int = 1) -> Polygon:

        """
        .. image:: _static/img/bodyparts_to_circle.png
           :width: 500
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
    def buffer_shape(shape: Polygon,
                     size_mm: int,
                     pixels_per_mm: float,
                     cap_style: Literal['round', 'square', 'flat'] = 'round') -> Polygon:

        """

        .. image:: _static/img/buffer_shape.png
           :width: 500
           :align: center

        """

        check_instance(source=GeometryMixin.buffer_shape.__name__, instance=shape, accepted_types=(LineString, Polygon))
        return shape.buffer(distance=int(size_mm / pixels_per_mm), cap_style=CAP_STYLE_MAP[cap_style])

    @staticmethod
    def compute_pct_shape_overlap(shapes: List[Polygon]) -> float:

        """

        .. image:: _static/img/compute_pct_shape_overlap.png
           :width: 500
           :align: center

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
        >>> GeometryMixin().compute_pct_shape_overlap(shapes=[polygon_1, polygon_2])
        >>> 37.96
        """

        for shape in shapes:
            check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape, accepted_types=(LineString, Polygon))
        if shapes[0].intersects(shapes[1]):
            intersection = shapes[0].intersection(shapes[1])
            return round((intersection.area / ((shapes[0].area + shapes[1].area) - intersection.area) * 100), 2)
        else:
            return 0.0

    @staticmethod
    def are_lines_crossing(lines: List[LineString]):
        """
        .. image:: _static/img/are_lines_crossing.png
           :width: 500
           :align: center
        """
        for line in lines:
            check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=line, accepted_types=LineString)

        return lines[0].crosses(lines[1])

    @staticmethod
    def is_shape_covered(lines: List[Union[LineString, Polygon]]) -> bool:
        """
        .. image:: _static/img/is_line_covered.png
           :width: 500
           :align: center

           TODO: multiframe method
        """
        pass

    @staticmethod
    def shape_distance(shapes: List[Union[LineString, Polygon]],
                       pixels_per_mm: float,
                       unit: Literal['mm', 'cm', 'dm', 'm'] = 'mm') -> float:
        """
        .. image:: _static/img/shape_distance.png
           :width: 500
           :align: center


        >>> shape_1 = Polygon([(0, 0), 10, 10), 0, 10), 10, 0)])
        >>> shape_2 = Polygon([(0, 0), 10, 10), 0, 10), 10, 0)])
        >>> GeometryMixin.shape_distance(shapes=[shape_1, shape_2], pixels_per_mm=1)
        >>> 10
        TODO: multiframe
        """
        check_if_valid_input(name='UNIT', input=unit, options=['mm', 'cm', 'dm', 'm'])
        for shape in shapes:
            check_instance(source=GeometryMixin.compute_pct_shape_overlap.__name__, instance=shape, accepted_types=(LineString, Polygon))
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
           :width: 500
           :align: center

        :example:
        >>> data = np.array([[364, 308],[383, 323], [403, 335],[423, 351]])
        >>> line = GeometryMixin().bodyparts_to_line(data=data)
        """

        if data.ndim != 2:
            raise InvalidInputError(msg=f'Body-parts to linestring expects a 2D array, got {data.ndim}', source=GeometryMixin.bodyparts_to_line.__name__)
        return LineString(data.tolist())

    @staticmethod
    def get_center(shape: Union[LineString, Polygon]) -> np.ndarray:
        """
        .. image:: _static/img/get_center.png
           :width: 500
           :align: center
        """
        check_instance(source=GeometryMixin.get_center.__name__, instance=shape, accepted_types=(LineString, Polygon))
        return np.array(shape.centroid)

    @staticmethod
    def is_touching(self):

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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
            constants = functools.partial(GeometryMixin.bodyparts_to_circle,
                                          parallel_offset=parallel_offset)
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

        pool.join(); pool.terminate()
        return results

    def multiframe_bodyparts_to_line(self,
                                     data: np.ndarray,
                                     core_cnt: int = -1) -> List[LineString]:
        """
        """

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if data.ndim != 3:
            raise InvalidInputError(msg=f'Multiframe body-parts to linestring expects a 3D array, got {data.ndim}', source=GeometryMixin.bodyparts_to_line.__name__)
        results = []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.bodyparts_to_line, data, chunksize=1)):
                results.append(result)

        return results

    def multifrm_compute_pct_shape_overlap(self,
                                          shape_1: List[Polygon],
                                          shape_2: List[Polygon],
                                          core_cnt = -1) -> List[float]:

        check_int(name='CORE COUNT', value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
        if core_cnt == -1: core_cnt = find_core_cnt()[0]
        if len(shape_1) != len(shape_2):
            raise InvalidInputError(msg=f'shape_1 and shape_2 are unequal sizes: {len(shape_1)} vs {len(shape_2)}', source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__)
        input_dtypes = list(set([type(x) for x in shape_1] + [type(x) for x in shape_2]))
        if len(input_dtypes) > 1:
            raise InvalidInputError(msg=f'shape_1 and shape_2 contains more than 1 dtype {input_dtypes}', source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__)
        check_instance(source=GeometryMixin.multifrm_compute_pct_shape_overlap.__name__, instance=shape_1[0], accepted_types=(LineString, Polygon))
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
            for cnt, result in enumerate(pool.imap(GeometryMixin.compute_pct_shape_overlap, data, chunksize=1)):
                results.append(result)

        pool.join(); pool.terminate()
        return results



#
#
#
# from shapely.geometry import Polygon, LineString, Point
# import numpy as np
# rectangle_1 = np.array([[0, 0], [10, 10], [0, 10], [10, 0]])
# rectangle_2 = np.array([[20, 20], [30, 30], [20, 30], [30, 20]])
#
# polygon_1 = GeometryMixin().bodyparts_to_polygon(data=rectangle_1)
# polygon_1.
# polygon_1.mi
# polygon_2 = GeometryMixin().bodyparts_to_polygon(data=rectangle_2)
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

#test.compute_pct_pylygon_overlap(shape_1=polygon_1, shape_2=polygon_2)

#polygon_1 = LineString([[364, 100],[383, 800],[403, 335],[423, 351]])
#polygon_2 = LineString([[364, 308],[383, 323],[403, 600],[423, 1]])
#test.compute_pct_pylygon_overlap(shape_1=polygon_1, shape_2=polygon_2)

#circle = test.bodyparts_to_circle(data=data[0])

#test.multiframe_bodyparts_to_polygon(data=data)



#.values.reshape(-1, 2)


