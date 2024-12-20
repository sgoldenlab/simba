import numpy as np

from simba.mixins.circular_statistics import CircularStatisticsMixin
from simba.utils.checks import check_float, check_valid_array
from simba.utils.data import get_mode
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError


def sliding_preferred_turning_direction(x: np.ndarray,
                                        time_window: float,
                                        sample_rate: float) -> np.ndarray:
    """
    Computes the sliding preferred turning direction over a given time window from a 1D array of circular directional data.

    Calculates the most frequent turning direction (mode) within a sliding window  of a specified duration.

    :param np.ndarray x: A 1D array of circular directional data (values between 0 and 360, inclusive).  Each value represents an angular direction in degrees.
    :param float time_window:  The duration of the sliding window in seconds.
    :param float sample_rate: The sampling rate of the data in Hz (samples per second) or FPS (frames per seconds)
    :return:
        A 1D array of integers indicating the preferred turning direction for each window:
        - `0`: No change in angular values within the window.
        - `1`: An increase in angular values (counterclockwise rotation).
        - `2`: A decrease in angular values (clockwise rotation).
        For indices before the first full window, the value is `-1`.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 361, (213,))
    >>> sliding_preferred_turning_direction(x=x, time_window=1, sample_rate=10)
    """
    check_valid_array(data=x, source=sliding_preferred_turning_direction.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if np.max(x) > 360 or np.min(x) < 0:
        raise InvalidInputError(msg='x has to be values between 0 and 360 inclusive', source=sliding_preferred_turning_direction.__name__)
    check_float(name=f'{sliding_preferred_turning_direction} time_window', value=time_window)
    check_float(name=f'{sliding_preferred_turning_direction} sample_rate', value=sample_rate)
    rotational_directions = CircularStatisticsMixin.rotational_direction(data=x.astype(np.float32))
    window_size = np.int64(np.max((1.0, (time_window*sample_rate))))
    results = np.full(shape=(x.shape[0]), fill_value=-1, dtype=np.int32)
    for r in range(window_size, x.shape[0]+1):
        l = r-window_size
        sample = rotational_directions[l:r]
        results[r-1] = get_mode(x=sample)
    return results.astype(np.int32)


    #return

# x = np.random.randint(0, 361, (213,))
# sliding_preferred_turning_direction(x=x, time_window=1, sample_rate=10)