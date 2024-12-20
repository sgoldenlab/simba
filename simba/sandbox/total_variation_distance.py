from typing import Optional
try:
    from typing import Literal
except:
    from typing_extensions import Literal
import numpy as np

from simba.utils.checks import check_str, check_valid_array
from simba.utils.enums import Options
from simba.mixins.statistics_mixin import Statistics
from simba.utils.data import bucket_data
from numba import jit



def total_variation_distance(x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = "auto"):
    """
    Calculate the total variation distance between two probability distributions.

    :param np.ndarray x: A 1-D array representing the first sample.
    :param np.ndarray y: A 1-D array representing the second sample.
    :param Optional[str] bucket_method: The method used to determine the number of bins for histogram computation. Supported methods are 'fd' (Freedman-Diaconis), 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', and 'sqrt'. Defaults to 'auto'.
    :return float: The total variation distance between the two distributions.

    .. math::

       TV(P, Q) = 0.5 \sum_i |P_i - Q_i|

    where :math:`P_i` and :math:`Q_i` are the probabilities assigned by the distributions :math:`P` and :math:`Q`
    to the same event :math:`i`, respectively.

    :example:
    >>> total_variation_distance(x=np.array([1, 5, 10, 20, 50]), y=np.array([1, 5, 10, 100, 110]))
    >>> 0.3999999761581421
    """

    check_valid_array(data=x, source=total_variation_distance.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, np.int8, np.float32, np.float64, int, float))
    check_valid_array(data=y, source=total_variation_distance.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, np.int8, np.float32, np.float64, int, float))
    check_str(name=f"{total_variation_distance.__name__} method", value=bucket_method, options=Options.BUCKET_METHODS.value)
    bin_width, bin_count = bucket_data(data=x, method=bucket_method)
    s1_h = Statistics._hist_1d(data=x, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]), normalize=True)
    s2_h = Statistics._hist_1d(data=y, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]), normalize=True)
    return 0.5 * np.sum(np.abs(s1_h - s2_h))