import numpy as np

from simba.mixins.statistics_mixin import Statistics
from simba.utils.checks import check_valid_array
from simba.utils.enums import Formats


def radial_dispersion_index(x: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute the Radial Dispersion Index (RDI) for a set of points relative to a reference point.

    The RDI quantifies the variability in radial distances of points from the reference
    point, normalized by the mean radial distance.

    :param np.ndarray x: 2-dimensional numpy array representing the input data with shape (n, m), where n is the number of frames and m is the coordinates.
    :param np.ndarray reference_point: A 1D array of shape (n_dimensions,) representing the reference point with  respect to which the radial dispertion index is calculated.
    :rtype: float
    """

    check_valid_array(data=x, source=f"{radial_dispersion_index.__name__} x", accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=reference_point, source=f"{radial_dispersion_index.__name__} reference_point", accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    radial_distances = np.linalg.norm(x - reference_point, axis=1)
    return np.std(radial_distances) / np.mean(radial_distances)
