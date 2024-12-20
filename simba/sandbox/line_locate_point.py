import numpy as np
from shapely.geometry import LineString, Polygon, Point
from typing import Union, Optional
from simba.utils.checks import check_instance, check_valid_array, check_float, check_int
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.geometry_mixin import GeometryMixin
from simba.utils.read_write import find_core_cnt

def locate_line_point(path: Union[LineString, np.ndarray],
                      geometry: Union[LineString, Polygon, Point],
                      px_per_mm: Optional[float] = 1,
                      fps: Optional[float] = 1,
                      core_cnt: Optional[int] = -1,
                      distance_min: Optional[bool] = True,
                      time_prior: Optional[bool] = True):

    """
    Compute the time and distance travelled to along a path to reach the most proximal point in reference to a second geometry.

    .. note::
       (i) To compute the time and distance travelled to along a path to reach the most distal point to a second geometry, pass ``distance_min = False``.

       (ii) To compute the time and distance travelled along a path **after** reaching the most distal or proximal point to a second geometry, pass ``time_prior = False``.

    .. image:: _static/img/locate_line_point.png
       :width: 600
       :align: center

    :example:
    >>> line = LineString([[10, 10], [7.5, 7.5], [15, 15], [7.5, 7.5]])
    >>> polygon = Polygon([[0, 5], [0, 0], [5, 0], [5, 5]])
    >>> locate_line_point(path=line, geometry=polygon)
    >>> {'distance_value': 3.5355339059327378, 'distance_travelled': 3.5355339059327378, 'time_travelled': 1.0, 'distance_index': 1}
    """



    check_instance(source=locate_line_point.__name__, instance=path, accepted_types=(LineString, np.ndarray))
    check_instance(source=locate_line_point.__name__, instance=geometry, accepted_types=(LineString, Polygon, Point))
    check_int(name="CORE COUNT",value=core_cnt,min_value=-1,max_value=find_core_cnt()[0],raise_error=True,)
    check_float(name="PIXELS PER MM",value=px_per_mm,min_value=0.1,raise_error=True)
    check_float(name="FPS", value=fps, min_value=1, raise_error=True)
    if core_cnt == -1: core_cnt = find_core_cnt()[0]

    if isinstance(path, np.ndarray):
        check_valid_array(data=path, accepted_axis_1_shape=(2,), accepted_dtypes=(np.float32, np.float64, np.int64, np.int32))
        path = LineString(path)
    if isinstance(geometry, Point):
        geometry = np.array(geometry.coords)
        distances = FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=np.array(path.coords), location_2=geometry, px_per_mm=px_per_mm)
    else:
        points = [Point(x) for x in np.array(path.coords)]
        geometry = [geometry for x in range(len(points))]
        distances = GeometryMixin().multiframe_shape_distance(shape_1=points, shape_2=geometry, pixels_per_mm=px_per_mm, core_cnt=core_cnt)

    if distance_min:
        distance_idx = np.argmin(distances)
    else:
        distance_idx = np.argmax(distances)
    if time_prior:
        dist_travelled = np.sum(np.abs(np.diff(distances[:distance_idx + 1]))) / px_per_mm
        time_travelled = distance_idx / fps
    else:
        dist_travelled = np.sum(np.abs(np.diff(distances[distance_idx:]))) / px_per_mm
        time_travelled = (distances - distance_idx) / fps
    dist_val = distances[distance_idx] / px_per_mm

    return {'distance_value': dist_val,
            'distance_travelled': dist_travelled,
            'time_travelled': time_travelled,
            'distance_index': distance_idx}