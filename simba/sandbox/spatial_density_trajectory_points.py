import numpy as np

from simba.utils.checks import check_float, check_int, check_valid_array
from simba.utils.enums import Formats


def spatial_density(x: np.ndarray,
                    radius: float,
                    pixels_per_mm: float) -> float:

    """
    Computes the spatial density of trajectory points in a 2D array, based on the number of neighboring points
    within a specified radius for each point in the trajectory.

    Spatial density provides insights into the movement pattern along a trajectory. Higher density values indicate
    areas where points are closely packed, which can suggest slower movement, lingering, or frequent changes in
    direction. Lower density values suggest more spread-out points, often associated with faster, more linear movement.

    :param np.ndarray x: A 2D array of shape (N, 2), where N is the number of points and each point has two spatial coordinates.
    :param float radius: The radius within which to count neighboring points around each point. Defines the area of interest around each trajectory point.
    :return: A single float value representing the average spatial density of the trajectory.
    :rtype: float

    :example:
    >>> x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [1, 0.5], [1.5, 1.5]])
    >>> density = spatial_density(x, pixels_per_mm=2.5, radius=5)
    >>> high_density_points = np.array([[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], [0, 0.5], [0.5, 0.5], [1, 0.5], [1.5, 0.5], [2, 0.5]])
    >>> low_density_points = np.array([[0, 0], [5, 5], [10, 10], [15, 15], [20, 20]])
    >>> high = spatial_density(x=high_density_points,radius=1, pixels_per_mm=1)
    >>> low = spatial_density(x=low_density_points,radius=1, pixels_per_mm=1)
    """

    check_valid_array(data=x, source=f'{spatial_density.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{spatial_density.__name__} radius', value=radius)
    check_float(name=f'{spatial_density.__name__} radius', value=pixels_per_mm)
    pixel_radius = np.ceil(max(1.0, (radius * pixels_per_mm)))
    n_points = x.shape[0]
    total_neighbors = 0

    for i in range(n_points):
        distances = np.linalg.norm(x - x[i], axis=1)
        neighbors = np.sum(distances <= pixel_radius) - 1
        total_neighbors += neighbors

    return total_neighbors / n_points



def sliding_spatial_density(x: np.ndarray,
                            radius: float,
                            pixels_per_mm: float,
                            window_size: float,
                            sample_rate: float) -> np.ndarray:

    """
    Computes the sliding spatial density of trajectory points in a 2D array, based on the number of neighboring points
    within a specified radius, considering the density over a moving window of points. This function accounts for the
    spatial scale in pixels per millimeter, providing a density measurement that is adjusted for the physical scale
    of the trajectory.

    :param np.ndarray x: A 2D array of shape (N, 2), where N is the number of points and each point has two spatial coordinates (x, y). The array represents the trajectory path of points in a 2D space (e.g., x and y positions in space).
    :param float radius: The radius (in millimeters) within which to count neighboring points around each trajectory point. Defines the area of interest around each point.
    :param float pixels_per_mm: The scaling factor that converts the physical radius (in millimeters) to pixel units for spatial density calculations.
    :param float window_size: The size of the sliding window (in seconds or points) to compute the density of points. A larger window size will consider more points in each density calculation.
    :param float sample_rate: The rate at which to sample the trajectory points (e.g., frames per second or samples per unit time). It adjusts the granularity of the sliding window.
    :return: A 1D numpy array where each element represents the computed spatial density for the trajectory at the corresponding point in time (or frame). Higher values indicate more densely packed points within the specified radius, while lower values suggest more sparsely distributed points.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 20, (100, 2))  # Example trajectory with 100 points in 2D space
    >>> results = sliding_spatial_density(x=x, radius=5.0, pixels_per_mm=10.0, window_size=1, sample_rate=31)
    """

    pixel_radius = np.ceil(max(1.0, (radius * pixels_per_mm)))
    frame_window_size = int(np.ceil(max(1.0, (window_size * sample_rate))))
    results = np.full(shape=(x.shape[0]), fill_value=np.nan, dtype=np.float32)
    for r in range(frame_window_size, x.shape[0]+1):
        l = r - frame_window_size
        sample_x = x[l:r]
        n_points, total_neighbors = sample_x.shape[0], 0
        for i in range(n_points):
            distances = np.linalg.norm(sample_x - sample_x[i], axis=1)
            neighbors = np.sum(distances <= pixel_radius) - 1
            total_neighbors += neighbors
        results[r-1] = total_neighbors / n_points

    return results




# high_density_points = np.array([[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], [0, 0.5], [0.5, 0.5], [1, 0.5], [1.5, 0.5], [2, 0.5]])
# low_density_points = np.array([[0, 0], [5, 5], [10, 10], [15, 15], [20, 20]])
# high = spatial_density(x=high_density_points,radius=1, pixels_per_mm=1)
# low = spatial_density(x=low_density_points,radius=1, pixels_per_mm=1)
