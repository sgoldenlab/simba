from shapely.geometry import Polygon, LineString
from typing import List, Union, Optional
from simba.utils.checks import check_valid_lst, check_instance, check_int
import numpy as np
from simba.utils.read_write import read_df, find_core_cnt
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.enums import Defaults

import multiprocessing

def hausdorff_distance(geometries: List[List[Union[Polygon, LineString]]]) -> np.ndarray:
    """
    The Hausdorff distance measure of the similarity between time-series sequential geometries. It is defined as the maximum of the distances
    from each point in one set to the nearest point in the other set.

    Hausdorff distance can be used to measure the similarity of the geometry in one frame relative to the geometry in the next frame.
    Large values indicate that the animal has a different shape than in the preceding shape.

    :param List[List[Union[Polygon, LineString]]] geometries: List of list where each list has two geometries.
    :return np.ndarray: 1D array of hausdorff distances of geometries in each list.

    :example:
    >>> x = Polygon([[0,1], [0, 2], [1,1]])
    >>> y = Polygon([[0,1], [0, 2], [0,1]])
    >>> hausdorff_distance(geometries=[[x, y]])
    >>> [1.]
    """

    check_instance(source=hausdorff_distance.__name__, instance=geometries, accepted_types=(list,))
    for i in geometries:
        check_valid_lst(source=hausdorff_distance.__name__, data=i, valid_dtypes=(Polygon, LineString,), exact_len=2)
    results = np.full((len(geometries)), np.nan)
    for i in range(len(geometries)):
        results[i] = geometries[i][0].hausdorff_distance(geometries[i][1])
    return results

def multiframe_hausdorff_distance(geometries: List[Union[Polygon, LineString]],
                                  lag: Optional[int] = 1,
                                  core_cnt: Optional[int] = -1) -> List[float]:
    """
    The Hausdorff distance measure of the similarity between sequential time-series  geometries.

    :example:
    >>> df = read_df(file_path='/Users/simon/Desktop/envs/simba/troubleshooting/mouse_open_field/project_folder/csv/outlier_corrected_movement_location/SI_DAY3_308_CD1_PRESENT.csv', file_type='csv')
    >>> cols = [x for x in df.columns if not x.endswith('_p')]
    >>> data = df[cols].values.reshape(len(df), -1 , 2).astype(np.int)
    >>> geometries = GeometryMixin().multiframe_bodyparts_to_polygon(data=data, pixels_per_mm=1, parallel_offset=1, verbose=False, core_cnt=-1)
    >>> hausdorff_distances = multiframe_hausdorff_distance(geometries=geometries)
    """
    check_valid_lst(source=multiframe_hausdorff_distance.__name__, data=geometries, valid_dtypes=(Polygon, LineString,), min_len=1)
    check_int(name=f"{multiframe_hausdorff_distance.__name__} CORE COUNT", value=core_cnt, min_value=-1, max_value=find_core_cnt()[0], raise_error=True)
    check_int(name=f"{multiframe_hausdorff_distance.__name__} LAG", value=lag, min_value=-1, max_value=len(geometries)-1, raise_error=True)
    if core_cnt == -1: core_cnt = find_core_cnt()[0]
    reshaped_geometries = []
    for i in range(lag): reshaped_geometries.append([[geometries[i], geometries[i]]])
    for i in range(lag, len(geometries)): reshaped_geometries.append([[geometries[i-lag], geometries[i]]])
    results = []
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
        for cnt, mp_return in enumerate(pool.imap(hausdorff_distance, reshaped_geometries, chunksize=1)):
            results.append(mp_return[0])
    return results








#
#

#
#
# hausdorff(geometries=data)
#
#




