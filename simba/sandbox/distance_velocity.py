import numpy as np
from typing import Optional, Tuple
from simba.utils.checks import check_float, check_valid_array

def distance_and_velocity(x: np.array,
                          fps: float, pixels_per_mm: float,
                          centimeters: Optional[bool] = True) -> Tuple[float, float]:
    """
    Calculate total movement and mean velocity from a sequence of position data.

    :param x: Array containing movement data. For example, created by ``simba.mixins.FeatureExtractionMixin.framewise_euclidean_distance``.
    :param fps: Frames per second of the data.
    :param pixels_per_mm: Conversion factor from pixels to millimeters.
    :param Optional[bool] centimeters: If True, results are returned in centimeters. Defaults to True.
    :return Tuple[float, float]: A tuple containing total movement and mean velocity.
    """

    check_valid_array(data=x, source=distance_and_velocity.__name__, accepted_ndims=(1,), accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, int, float), min_axis_0=1)
    check_float(name=f'{distance_and_velocity.__name__} fps', value=fps, min_value=1)
    check_float(name=f'{distance_and_velocity.__name__} pixels_per_mm', value=pixels_per_mm, min_value=10e-6)
    movement = (np.sum(x) / pixels_per_mm)
    v = []
    for i in range(0, x.shape[0], int(fps)):
        w = x[i: (i+fps)]
        v.append((np.sum(w) / pixels_per_mm) * (1 / (w.shape[0] / int(fps))))
    if centimeters:
        v = [vi/ 10 for vi in v]
        movement = movement / 10
    return movement, np.mean(v)


# x = np.random.randint(0, 10, (20,))
# distance_and_velocity(x=x, fps=10, pixels_per_mm=99, centimeters=True)