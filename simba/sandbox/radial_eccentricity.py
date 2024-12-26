import numpy as np

from simba.mixins.statistics_mixin import Statistics
from simba.utils.checks import check_valid_array
from simba.utils.enums import Formats


def radial_eccentricity(x: np.ndarray, reference_point: np.ndarray):
    """
    Compute the radial eccentricity of a set of points relative to a reference point.

    Radial eccentricity quantifies the degree of elongation in the spatial distribution
    of points. The value ranges between 0 and 1, where: - 0 indicates a perfectly circular distribution. - Values approaching 1 indicate a highly elongated or linear distribution.

    :param np.ndarray x: 2-dimensional numpy array representing the input data with shape (n, m), where n is the number of frames and m is the coordinates.
    :param np.ndarray data: A 1D array of shape (n_dimensions,) representing the reference point with  respect to which the radial eccentricity is calculated.

    :example:
    >>> points = np.random.randint(0, 1000, (100000, 2))
    >>> reference_point = np.mean(points, axis=0)
    >>> radial_eccentricity(x=points, reference_point=reference_point)
    """

    check_valid_array(data=x, source=f"{radial_eccentricity.__name__} x", accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=reference_point, source=f"{radial_eccentricity.__name__} reference_point", accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    centered_points = x - reference_point
    cov_matrix = Statistics.cov_matrix(data=centered_points.astype(np.float32))
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.sqrt(1 - eigenvalues[1] / eigenvalues[0])






