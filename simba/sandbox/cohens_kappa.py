import numpy as np
from numba import njit
from sklearn.metrics import cohen_kappa_score
import time
im

@njit("(int64[:],int64[:])")
def cohens_kappa(sample_1: np.ndarray, sample_2: np.ndarray):
    """
    Jitted compute Cohen's Kappa coefficient for two binary samples.

    Cohen's Kappa coefficient between classification sand ground truth taking into account agreement between classifications and ground truth occurring by chance.

    :example:
    >>> sample_1 = np.random.randint(0, 2, size=(10000,))
    >>> sample_2 = np.random.randint(0, 2, size=(10000,))
    >>> cohens_kappa(sample_1=sample_1, sample_2=sample_2))
    """
    sample_1 = np.ascontiguousarray(sample_1)
    sample_2 = np.ascontiguousarray(sample_2)
    data = np.hstack((sample_1.reshape(-1, 1), sample_2.reshape(-1, 1)))
    tp = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 1)).flatten())
    tn = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 0)).flatten())
    fp = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 0)).flatten())
    fn = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 1)).flatten())
    data = np.array(([tp, fp], [fn, tn]))
    sum0 = data.sum(axis=0)
    sum1 = data.sum(axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w_mat = np.full(shape=(2, 2),fill_value=1)
    w_mat[0, 0] = 0
    w_mat[1, 1] = 0
    return 1 - np.sum(w_mat * data) / np.sum(w_mat * expected)


def cohens_kappa_one_against_all(data: pd.DataFrame, labels: np.ndarray):
    results = {}
    for lbl in np.unique(labels):
        cluster_data, non_cluster_data = split_specific_cluster_data(data=data, labels=labels, label=lbl)
        results[lbl] = {}
        for field in cluster_data.columns:
            sample_1 = cluster_data[field].values
            sample_2 = non_cluster_data[field].values
            data = np.hstack((sample_1.reshape(-1, 1), sample_2.reshape(-1, 1)))
            tp = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 1)).flatten())
            tn = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 0)).flatten())
            fp = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 0)).flatten())
            fn = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 1)).flatten())
            data = np.array(([tp, fp], [fn, tn]))
            sum0 = data.sum(axis=0)
            sum1 = data.sum(axis=1)
            expected = np.outer(sum0, sum1) / np.sum(sum0)
            w_mat = np.full(shape=(2, 2), fill_value=1)
            w_mat[0, 0] = 0
            w_mat[1, 1] = 0
            results[lbl][field] = 1 - np.sum(w_mat * data) / np.sum(w_mat * expected)






#confusion_matrix()
