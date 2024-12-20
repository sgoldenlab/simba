import numpy as np
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score)


def cluster_comparison(x: np.ndarray, y: np.ndarray, method: str = 'adjusted_rand_score'):
    if method == 'adjusted_mutual_info_score':
        return adjusted_mutual_info_score(labels_true=x, labels_pred=y)
    elif method == 'fowlkes_mallows_score':
        return fowlkes_mallows_score(labels_true=x, labels_pred=y)
    else:
        return adjusted_rand_score(labels_true=x, labels_pred=y)




def adjusted_rand(x: np.ndarray, y: np.ndarray):
    n_samples = np.int64(x.shape[0])
    classes, class_idx = np.unique(x, return_inverse=True)
    clusters, cluster_idx = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    contingency = coo_matrix((np.ones(class_idx.shape[0]), (class_idx, cluster_idx)), shape=(n_classes, n_clusters), dtype=np.int64)
    print(contingency)


x = np.array([0, 1, 0, 0, 0])
y = np.array([1, 1, 1, 1, 1])
adjusted_rand(x=x, y=y)



    #(tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)