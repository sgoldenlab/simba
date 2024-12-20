import numpy as np
import pandas as pd

from simba.utils.checks import check_float, check_int, check_valid_array
from simba.utils.enums import Formats


def entropy_of_directional_changes(x: np.ndarray, bins: int = 16) -> float:
    """
    Computes the Entropy of Directional Changes (EDC) of a path represented by an array of points.

    The output value ranges from 0 to log2(bins).

    The Entropy of Directional Changes quantifies the unpredictability or randomness of the directional
    changes in a given path. Higher entropy indicates more variation in the directions of the movement,
    while lower entropy suggests more linear or predictable movement.

    The function works by calculating the change in direction between consecutive points, discretizing
    those changes into bins, and then computing the Shannon entropy based on the probability distribution
    of the directional changes.

    :param np.ndarray x: A 2D array of shape (N, 2) representing the path, where N is the number of points and each point has two spatial coordinates (e.g., x and y for 2D space). The path should be in the form of an array of consecutive (x, y) points.
    :param int bins: The number of bins to discretize the directional changes. Default is 16 bins for angles between 0 and 360 degrees. A larger number of bins will increase the precision of direction change measurement.
    :return: The entropy of the directional changes in the path. A higher value indicates more unpredictable or random direction changes, while a lower value indicates more predictable or linear movement.
    :rtype: float

    :example:
    >>> x = np.random.randint(0, 500, (100, 2))
    >>> TimeseriesFeatureMixin.entropy_of_directional_changes(x, 3)
    """

    check_int(name=f'{entropy_of_directional_changes.__name__} bins', value=bins)
    check_valid_array(data=x, source=f'{entropy_of_directional_changes.__name__} x',
                      accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    direction_vectors = np.diff(x, axis=0)
    angles = np.arctan2(direction_vectors[:, 1], direction_vectors[:, 0]) * (180 / np.pi)
    angles = (angles + 360) % 360
    angle_bins = np.linspace(0, 360, bins + 1)
    digitized_angles = np.digitize(angles, angle_bins) - 1
    hist, _ = np.histogram(digitized_angles, bins=bins, range=(0, bins))
    hist = hist / hist.sum()
    return np.max((0.0, -np.sum(hist * np.log2(hist + 1e-10))))


def sliding_entropy_of_directional_changes(x: np.ndarray,
                                           bins: int,
                                           window_size: float,
                                           sample_rate: float) -> np.ndarray:
    """
    Computes a sliding window Entropy of Directional Changes (EDC) over a path represented by an array of points.

    This function calculates the entropy of directional changes within a specified window, sliding across the entire path.
    By analyzing the changes in direction over shorter segments (windows) of the path, it provides a dynamic view of
    movement unpredictability or randomness along the path. Higher entropy within a window indicates more varied directional
    changes, while lower entropy suggests more consistent directional movement within that segment.

    :param np.ndarray x: A 2D array of shape (N, 2) representing the path, where N is the number of points and each point has two spatial coordinates (e.g., x and y for 2D space). The path should be in the form of an array of consecutive (x, y) points.
    :param int bins: The number of bins to discretize the directional changes. Default is 16 bins for angles between 0 and 360 degrees. A larger number of bins will increase the precision of direction change measurement.
    :param float window_size: The duration of the sliding window, in seconds, over which to compute the entropy.
    :param float sample_rate: The sampling rate (in frames per second) of the path data. This parameter converts `window_size` from seconds into frames, defining the number of consecutive points in each sliding window.
    :return: A 1D numpy array of length N, where each element contains the entropy of directional changes for each frame, computed over the specified sliding window. Frames before the first full window contain NaN values.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 100, (400, 2))
    >>> results = sliding_entropy_of_directional_changes(x=x, bins=16, window_size=5.0, sample_rate=30)
    >>> x = pd.read_csv(r"C:\troubleshooting\two_black_animals_14bp\project_folder\csv\input_csv\Together_1.csv")[['Ear_left_1_x', 'Ear_left_1_y']].values
    >>> results = sliding_entropy_of_directional_changes(x=x, bins=16, window_size=5.0, sample_rate=30)
    """

    direction_vectors = np.diff(x, axis=0)
    angles = np.arctan2(direction_vectors[:, 1], direction_vectors[:, 0]) * (180 / np.pi)
    angles = (angles + 360) % 360
    angle_bins = np.linspace(0, 360, bins + 1)
    frame_step = int(max(1.0, window_size * sample_rate))
    results = np.full(shape=(x.shape[0]), fill_value=np.nan, dtype=np.float64)
    for r in range(frame_step, direction_vectors.shape[0]+1):
        l = r - frame_step
        sample_angles = angles[l:r]
        digitized_angles = np.digitize(sample_angles, angle_bins) - 1
        hist, _ = np.histogram(digitized_angles, bins=bins, range=(0, bins))
        hist = hist / hist.sum()
        results[r] = np.max((0.0, -np.sum(hist * np.log2(hist + 1e-10))))

    return results

x = pd.read_csv(r"C:\troubleshooting\two_black_animals_14bp\project_folder\csv\input_csv\Together_1.csv")[['Ear_left_1_x', 'Ear_left_1_y']].values
# x = np.random.randint(0, 100, (400, 2))
bins = 16
window_size = 5
sample_rate = 30
results = sliding_entropy_of_directional_changes(x=x, bins=bins, window_size=window_size, sample_rate=2)
#entropy_of_directional_changes(x=x)



