
import numpy as np

from simba.utils.checks import check_str, check_valid_array

try:
    from typing import Literal
except:
    from typing_extensions import Literal

    from simba.utils.enums import Formats

def symmetry_index(x: np.ndarray, y: np.ndarray, agg_type: Literal['mean', 'median'] = 'mean') -> float:

    """
    Calculate the Symmetry Index (SI) between two arrays of measurements, `x` and `y`, over a given time series.
    The Symmetry Index quantifies the relative difference between two measurements at each time point, expressed as a percentage.
    The function returns either the mean or median Symmetry Index over the entire series, based on the specified aggregation type.

    Zero indicates perfect symmetry. Positive values pepresent increasing asymmetry between the two measurements.

    :param np.ndarray x: A 1-dimensional array of measurements from one side (e.g., left side), representing a time series or sequence of measurements.
    :param np.ndarray y: A 1-dimensional array of measurements from the other side (e.g., right side), of the same length as `x`.
    :param Literal['mean', 'median'] agg_type: The aggregation method used to summarize the Symmetry Index across all time points.
    :return: The aggregated Symmetry Index over the series, either as the mean or median SI.
    :rtype: float

    :example:
    >>> x = np.random.randint(0, 155, (100,))
    >>>y = np.random.randint(0, 155, (100,))
    >>> symmetry_index(x=x, y=y)
    """

    check_valid_array(data=x, source=f'{symmetry_index.__name__} x', accepted_ndims=(1,), min_axis_0=1, accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=x, source=f'{symmetry_index.__name__} y', accepted_ndims=(1,), min_axis_0=1, accepted_axis_0_shape=[x.shape[0]], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_str(name=f'{symmetry_index.__name__} agg_type', value=agg_type, options=('mean', 'median'))
    si_values = np.abs(x - y) / (0.5 * (x + y)) * 100
    if agg_type == 'mean':
        return np.float32(np.nanmean(si_values))
    else:
        return np.float32(np.nanmedian(si_values))








# x = np.random.randint(0, 155, (100,))
# y = np.random.randint(0, 155, (100,))
# symmetry_index(x=x, y=y)