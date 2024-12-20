import numpy as np

from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats


def mean_squared_jerk(x: np.ndarray,
                      time_step: float,
                      sample_rate: float) -> float:

    """
    Calculate the Mean Squared Jerk (MSJ) for a given set of 2D positions over time.

    The Mean Squared Jerk is a measure of the smoothness of movement, calculated as the mean of
    squared third derivatives of the position with respect to time. It provides an indication of
    how abrupt or smooth a trajectory is, with higher values indicating more erratic movements.

    :param np.ndarray x: A 2D array where each row represents the [x, y] position at a time step.
    :param float time_step: The time difference between successive positions in seconds.
    :param float sample_rate: The rate at which the positions are sampled (samples per second).
    :return: The computed Mean Squared Jerk for the input trajectory data.
    :rtype: float

    :example I:
    >>> x = np.random.randint(0, 500, (100, 2))
    >>> mean_squared_jerk(x=x, time_step=1.0, sample_rate=30)
    """

    check_float(name=f'{mean_squared_jerk.__name__} time_step', min_value=10e-6, value=time_step)
    check_float(name=f'{mean_squared_jerk.__name__} sample_rate', min_value=10e-6, value=sample_rate)
    check_valid_array(data=x, source=f'{mean_squared_jerk.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2,], accepted_dtypes=Formats.NUMERIC_DTYPES.value)

    frame_step = int(max(1.0, time_step * sample_rate))
    V = np.diff(x, axis=0) / frame_step
    A = np.diff(V, axis=0) / frame_step
    jerks = np.diff(A, axis=0) / frame_step
    squared_jerks = np.sum(jerks ** 2, axis=1)
    return np.mean(squared_jerks)


def sliding_mean_squared_jerk(x: np.ndarray,
                              window_size: float,
                              sample_rate: float) -> np.ndarray:
    """
    Calculates the mean squared jerk (rate of change of acceleration) for a position path in a sliding window.

    Jerk is the derivative of acceleration, and this function computes the mean squared jerk over sliding windows
    across the entire path. High jerk values indicate abrupt changes in acceleration, while low values indicate
    smoother motion.

    :param np.ndarray x: An (N, M) array representing the path of an object, where N is the number of samples (time steps) and M is the number of spatial dimensions (e.g., 2 for 2D motion). Each row represents the position at a time step.
    :param float window_size: The size of each sliding window in seconds. This defines the interval over which the mean squared jerk is calculated.
    :param float sample_rate: The sampling rate in Hz (samples per second), which is used to convert the window size from seconds to frames.
    :return: A 1D array of length N, containing the mean squared jerk for each sliding window that ends at each time step. The first `frame_step` values will be NaN, as they do not have enough preceding data points to compute jerk over the full window.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 500, (12, 2))
    >>> sliding_mean_squared_jerk(x=x, window_size=1.0, sample_rate=2)

    :example II:
    >>> jerky_path = np.zeros((100, 2))
    >>> jerky_path[::10] = np.random.randint(0, 500, (10, 2))
    >>> non_jerky_path = np.linspace(0, 500, 100).reshape(-1, 1)
    >>> non_jerky_path = np.hstack((non_jerky_path, non_jerky_path))
    >>> jerky_jerk_result = sliding_mean_squared_jerk(jerky_path, 1.0, 10)
    >>> non_jerky_jerk_result = sliding_mean_squared_jerk(non_jerky_path, 1.0, 10)
    """

    V = np.diff(x, axis=0)
    A = np.diff(V, axis=0)
    frame_step = int(max(1.0, window_size * sample_rate))
    results = np.full(x.shape[0], fill_value=0, dtype=np.int64)
    for r in range(frame_step, x.shape[0]):
        l =  r - frame_step
        V_a = A[l:r, :]
        jerks = np.diff(V_a, axis=0)
        if jerks.shape[0] == 0:
            results[r] = 0
        else:
            results[r] = np.sum(jerks ** 2) / jerks.shape[0]

    return results

x = np.random.randint(0, 500, (12, 2))
sliding_mean_squared_jerk(x=x, window_size=1.0, sample_rate=2)


jerky_path = np.zeros((100, 2))
jerky_path[::10] = np.random.randint(0, 500, (10, 2))
non_jerky_path = np.linspace(0, 500, 100).reshape(-1, 1)
non_jerky_path = np.hstack((non_jerky_path, non_jerky_path))
jerky_jerk_result = sliding_mean_squared_jerk(jerky_path, 1.0, 10)
non_jerky_jerk_result = sliding_mean_squared_jerk(non_jerky_path, 1.0, 10)


# Parameters
window_size = 1.0  # seconds
sample_rate = 10   # samples per second

# Apply the function to both paths


# Print results
print("Jerky Path Mean Squared Jerk:", np.nanmean(jerky_jerk_result))
print("Non-Jerky Path Mean Squared Jerk:", np.nanmean(non_jerky_jerk_result))

#sliding_mean_squared_jerk()