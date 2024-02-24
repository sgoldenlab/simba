import functools
import multiprocessing
from typing import List, Optional, Union

import numpy as np
from shapely.geometry import Point

from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.checks import check_int, check_valid_array
from simba.utils.enums import Defaults
from simba.utils.read_write import find_core_cnt


def multiframe_bodypart_to_point(
    data: np.ndarray,
    core_cnt: Optional[int] = -1,
    buffer: Optional[int] = None,
    px_per_mm: Optional[int] = None,
) -> Union[List[Point], List[List[Point]]]:
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
    >>> point_lst_of_lst = multiframe_bodypart_to_point(data=data)
    """

    check_valid_array(
        data=data, accepted_dtypes=(np.int64, np.int32, np.int8), accepted_ndims=(2, 3)
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


# data = np.random.randint(0, 100, (10, 2))
data = np.random.randint(0, 100, (10, 10, 2))
points = multiframe_bodypart_to_point(data=data)
