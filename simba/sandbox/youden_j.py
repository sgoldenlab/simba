import numpy as np
from simba.utils.checks import check_valid_array




def youden_j(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
    """
    Calculate Youden's J statistic from two binary arrays.

    :param sample_1: The first binary array.
    :param sample_2: The second binary array.
    :return float: Youden's J statistic.
    """
    check_valid_array(data=sample_1, source=f'{youden_j.__name__} sample_1', accepted_ndims=(1,), accepted_values=[0, 1])
    check_valid_array(data=sample_2, source=f'{youden_j.__name__} sample_2', accepted_ndims=(1,), accepted_shapes=[(sample_1.shape)], accepted_values=[0, 1])
    tp = np.sum((sample_1 == 1) & (sample_2 == 1))
    tn = np.sum((sample_1 == 0) & (sample_2 == 0))
    fp = np.sum((sample_1 == 0) & (sample_2 == 1))
    fn = np.sum((sample_1 == 1) & (sample_2 == 0))
    if tp + fn == 0 or tn + fp == 0:
        return np.nan
    else:
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity + specificity - 1




sample_1 = np.random.randint(0, 2, (100))
sample_2 = np.random.randint(0, 2, (100))




youden_j(sample_1=sample_1, sample_2=sample_2)