import numpy as np

from simba.mixins.circular_statistics import CircularStatisticsMixin
from simba.utils.checks import check_valid_array
from simba.utils.data import get_mode
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError


def preferred_turning_direction(x: np.ndarray) -> int:
    """
    Determines the preferred turning direction from a 1D array of circular directional data.

    .. note::
       The input ``x`` can be created using any of the following methods:
       - :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_two_bps`
       - :func:`simba.data_processors.cuda.circular_statistics.direction_from_two_bps`
       - :func:`simba.data_processors.cuda.circular_statistics.direction_from_three_bps`
       - :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_three_bps`

    :param np.ndarray x: 1D array of circular directional data (values between 0 and 360, inclusive). The array represents angular directions measured in degrees.
    :return:
        The most frequent turning direction from the input data:
        - `0`: No change in the angular value between consecutive frames.
        - `1`: An increase in the angular value (rotation in the positive direction, counterclockwise).
        - `2`: A decrease in the angular value (rotation in the negative direction, clockwise).
    :rtype: int

    :example:
    >>> x = np.random.randint(0, 361, (200,))
    >>> preferred_turning_direction(x=x)
    """

    check_valid_array(data=x, source=preferred_turning_direction.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if np.max(x) > 360 or np.min(x) < 0:
        raise InvalidInputError(msg='x has to be values between 0 and 360 inclusive', source=preferred_turning_direction.__name__)
    rotational_direction = CircularStatisticsMixin.rotational_direction(data=x.astype(np.float32))
    return get_mode(x=rotational_direction)




