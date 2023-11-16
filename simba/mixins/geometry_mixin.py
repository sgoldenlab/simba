import functools
import itertools
import multiprocessing
from typing import List, Optional

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from typing_extensions import Literal

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_int
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import find_core_cnt

CAP_STYLE_MAP = {"round": 1, "square": 2, "flat": 3}


class GeometryMixin(object):

    """
    Methods to perform geometry transformation of pose-estimation data. This includes creating bounding boxes,
    line objects, circles etc. from pose-estimated body-parts and computing intersections.
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
           :width: 700
           :align: center

        :example:
        >>> data = [[364, 308],[383, 323],[403, 335],[423, 351]]
        >>> GeometryMixin().bodyparts_to_polygon(data=data)
        """

        polygon = (
            LineString(data.tolist())
            .buffer(distance=parallel_offset, cap_style=CAP_STYLE_MAP[cap_style])
            .simplify(tolerance=simplify_tolerance, preserve_topology=preserve_topology)
        )
        return polygon

    @staticmethod
    def bodyparts_to_circle(data: np.ndarray, parallel_offset: int = 1) -> Polygon:
        """
        .. image:: _static/img/bodyparts_to_circle.png
           :width: 700
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
        shape: Polygon,
        size_mm: int,
        pixels_per_mm: float,
        cap_style: Literal["round", "square", "flat"] = "round",
    ) -> Polygon:
        """

        .. image:: _static/img/compute_pct_shape_overlap.png
           :width: 700
           :align: center

        """

        return shape.buffer(
            distance=int(size_mm / pixels_per_mm), cap_style=CAP_STYLE_MAP[cap_style]
        )

    @staticmethod
    def compute_pct_shape_overlap(shapes: List[Polygon]) -> float:
        """

        .. image:: _static/img/compute_pct_shape_overlap.png
           :width: 700
           :align: center

        :example:
        >>> polygon_1 = GeometryMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
        >>> polygon_2 = GeometryMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
        >>> GeometryMixin().compute_pct_shape_overlap(shapes=[polygon_1, polygon_2])
        >>> 37.96
        """

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
        >>> data = [[[364, 308], [383, 323], [403, 335], [423, 351]],[[356, 307], [376, 319], [396, 331], [419, 347]]])
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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
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
        with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
            constants = functools.partial(
                GeometryMixin.bodyparts_to_circle, parallel_offset=parallel_offset
            )
            for cnt, mp_return in enumerate(pool.imap(constants, data, chunksize=1)):
                results.append(mp_return)

        pool.join()
        pool.terminate()
        return results

    def mulifrm_compute_pct_shape_overlap(
        self, shape_1: List[Polygon], shape_2: List[Polygon], core_cnt=-1
    ) -> List[float]:
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
                source=GeometryMixin.mulifrm_compute_pct_shape_overlap.__name__,
            )
        input_dtypes = list(
            set([type(x) for x in shape_1] + [type(x) for x in shape_2])
        )
        if len(input_dtypes) > 1:
            raise InvalidInputError(
                msg=f"shape_1 and shape_2 contains more than 1 dtype {input_dtypes}",
                source=GeometryMixin.mulifrm_compute_pct_shape_overlap.__name__,
            )
        elif not isinstance(shape_1[0], Polygon):
            raise InvalidInputError(
                msg=f"The shape input is not Polygons: {input_dtypes}",
                source=GeometryMixin.mulifrm_compute_pct_shape_overlap.__name__,
            )
        data, results = np.column_stack((shape_1, shape_2)), []
        with multiprocessing.Pool(core_cnt, maxtasksperchild=10) as pool:
            for cnt, result in enumerate(
                pool.imap(GeometryMixin.compute_pct_shape_overlap, data, chunksize=1)
            ):
                results.append(result)

        pool.join()
        pool.terminate()
        return results


# polygon_1 = BoundingBoxMixin().bodyparts_to_polygon(np.array([[364, 308],[383, 323],[403, 335],[423, 351]]))
# polygon_2 = BoundingBoxMixin().bodyparts_to_polygon(np.array([[356, 307],[376, 319],[396, 331],[419, 347]]))
# BoundingBoxMixin().compute_pct_shape_overlap(shapes=[polygon_1, polygon_2])
# #
# test = BoundingBoxMixin()
# #
# data_path = '/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/termites/project_folder/csv/outlier_corrected_movement_location/termite_test.csv'
# df = pd.read_csv(data_path, index_col=0).fillna(0).astype(np.int64)
# df = df[df.columns.drop(list(df.filter(regex='_p')))]
# data_1 = df.iloc[1, 0:8].values.reshape(-1, 2)
# #
# data_2 = df.iloc[2, 0:8].values.reshape(-1, 2)
#
# test.bodyparts_to_circle(data=data_1[0])
# #
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
