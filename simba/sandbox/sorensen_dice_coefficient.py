import numpy as np
from simba.utils.checks import check_valid_array

def sorensen_dice_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Sørensen's Similarity Index between two communities/clusters.

    The Sørensen similarity index, also known as the overlap index, quantifies the overlap between two populations by comparing the number of shared categories to the total number of categories in both populations. It ranges from zero, indicating no overlap, to one, representing perfect overlap

    Sørensen's Similarity Index (S) is calculated using the formula:

    .. math::

        S = \\frac{2 \times |X \cap Y|}{|X| + |Y|}

    where:
    - \( S \) is Sørensen's Similarity Index,
    - \( X \) and \( Y \) are the sets representing the categories in the first and second communities, respectively,
    - \( |X \cap Y| \) is the number of shared categories between the two communities,
    - \( |X| \) and \( |Y| \) are the total number of categories in the first and second communities, respectively.


    :param x: 1D numpy array with ordinal values for the first cluster/community.
    :param y: 1D numpy array with ordinal values for the second cluster/community.
    :return: Sørensen's Similarity Index between x and y.

    :example:
    >>> x = np.random.randint(0, 10, (100,))
    >>> y = np.random.randint(0, 10, (100,))
    >>> sorensen_dice_coefficient(x=x, y=y)
    """

    check_valid_array(source=f"{sorensen_dice_coefficient.__name__} x", accepted_ndims=(1,), data=x, accepted_dtypes=(np.int32, np.int64, np.int8, int), min_axis_0=2)
    check_valid_array(source=f"{sorensen_dice_coefficient.__name__} y", accepted_ndims=(1,), data=y, accepted_dtypes=(np.int32, np.int64, np.int8, int), min_axis_0=2)
    x, y = set(x), set(y)
    return 2 * len(x.intersection(y)) / (len(x) + len(y))