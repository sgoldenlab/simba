from shapely.geometry import Polygon, LineString, MultiPolygon
from typing import List, Optional
import numpy as np
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import check_valid_lst, check_int
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import find_core_cnt
import multiprocessing
from simba.utils.enums import Defaults



def multiframe_is_shape_covered(self,
                                shape_1: List[Polygon],
                                shape_2: List[Polygon],
                                core_cnt: Optional[int] = -1) -> List[bool]:
    """
    For each shape in time-series of shapes, check if another shape in the same time-series fully covers the
    first shape.

    .. image:: _static/img/multiframe_is_shape_covered.png
       :width: 600
       :align: center


    :example:
    >>> shape_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=np.random.randint(0, 200, (100, 6, 2)))
    >>> shape_2 = [Polygon([[0, 0], [20, 20], [20, 10], [10, 20]]) for x in range(len(shape_1))]
    >>> multiframe_is_shape_covered(shape_1=shape_1, shape_2=shape_2, core_cnt=3)
    """
    check_valid_lst(data=shape_1, source=multiframe_is_shape_covered.__name__, valid_dtypes=(LineString, Polygon, MultiPolygon,))
    check_valid_lst(data=shape_2, source=multiframe_is_shape_covered.__name__, valid_dtypes=(LineString, Polygon, MultiPolygon,))
    if len(shape_1) != len(shape_2):
        raise InvalidInputError(msg=f'shape_1 ({len(shape_1)}) and shape_2 ({len(shape_2)}) are unequal length', source=multiframe_is_shape_covered.__name__)
    check_int(name="CORE COUNT", value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
    if core_cnt == -1: core_cnt = find_core_cnt()[0]
    shapes = [list(x) for x in zip(shape_1, shape_2)]
    results = []
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
        for cnt, mp_return in enumerate(pool.imap(GeometryMixin.is_shape_covered, shapes, chunksize=1)):
            results.append(mp_return)
    pool.join()
    pool.terminate()
    return results


shape_1 = GeometryMixin().multiframe_bodyparts_to_polygon(data=np.random.randint(0, 200, (100, 6, 2)))
shape_2 = [Polygon([[0, 0], [20, 20], [20, 10], [10, 20]]) for x in range(len(shape_1))]
multiframe_is_shape_covered(shape_1=shape_1, shape_2=shape_2, core_cnt=3)

