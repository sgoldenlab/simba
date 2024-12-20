import numpy as np
from simba.utils.checks import check_valid_array
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, adjusted_mutual_info_score

def adjusted_rand(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Adjusted Rand Index (ARI) between two clusterings.

    :param np.ndarray x: 1D array representing the labels of the first model.
    :param np.ndarray y: 1D array representing the labels of the second model.
    :return float: 1 indicates perfect clustering agreement, 0 indicates random clustering, and negative values indicate disagreement between the clusterings.

    :example:
    >>> x = np.array([0, 0, 0, 0, 0])
    >>> y = np.array([1, 1, 1, 1, 1])
    >>> adjusted_rand(x=x, y=y)
    >>> 1.0
    """

    check_valid_array(data=x, source=adjusted_rand_score.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, int), min_axis_0=1)
    check_valid_array(data=y, source=adjusted_rand_score.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, int), accepted_shapes=[(x.shape[0],)])
    return adjusted_rand_score(labels_true=x, labels_pred=y)

def fowlkes_mallows(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Fowlkes-Mallows Index (FMI) between two clusterings.

    :param np.ndarray x: 1D array representing the labels of the first model.
    :param np.ndarray y: 1D array representing the labels of the second model.
    :return float: Score between 0 and 1. 1 indicates perfect clustering agreement, 0 indicates random clustering.
    """
    check_valid_array(data=x, source=adjusted_rand_score.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, int), min_axis_0=1)
    check_valid_array(data=y, source=adjusted_rand_score.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, int), accepted_shapes=[(x.shape[0],)])
    return fowlkes_mallows_score(labels_true=x, labels_pred=y)


def adjusted_mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Adjusted Mutual Information (AMI) between two clusterings as a meassure of similarity.
    clusterings.

    :param np.ndarray x: 1D array representing the labels of the first model.
    :param np.ndarray y: 1D array representing the labels of the second model.
    :return float: Score between 0 and 1, where 1 indicates perfect clustering agreement.

    """
    check_valid_array(data=x, source=adjusted_rand_score.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, int), min_axis_0=1)
    check_valid_array(data=y, source=adjusted_rand_score.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, int), accepted_shapes=[(x.shape[0],)])
    return adjusted_mutual_info_score(labels_true=x, labels_pred=y)








#
#
# x = np.array([0, 0, 1, 1, 2])
# y = np.array([1, 1, 2, 1, 3])
#
# #x = np.random.randint(0, 5, (100,))
#
#
# adjusted_mutual_info(x=x, y=y)



