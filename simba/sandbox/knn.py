import numpy as np

def knn(data: np.ndarray,
        k: int,
        target: np.ndarray):

    norm = np.full(data.shape, np.nan)
    for i in range(data.shape[1]):
        norm[:, i] = (data[:, i]-np.min(data[:, i]))/(np.max(data[:, i])-np.min(data[:, i]))

    results = {}
    for target in np.unique(target):
        congruent_idx = np.argwhere(bool_target == target).flatten()
        incongruent_idx = np.argwhere(bool_target != target).flatten()
        for i in congruent_idx:
            results[i] = {}
            for j in incongruent_idx:
                n_dist = 0
                for k in range(norm[i, :].shape[0]):
                    n_dist += np.abs(norm[i, k] - norm[j, k])
                results[i][j] = n_dist






    # for i in range(norm.shape[0]):
    #     sum[i] = np.sum(norm[i])
    # for i in np.unique(bool_target):
    #     idx = np.argwhere(bool_target == i).flatten()
    #     n_idx = np.argwhere(bool_target != i).flatten()
    #     for j in idx:
    #         results[j] = {}
    #         for k in n_idx:
    #             results[j][k] = np.abs(sum[j] - sum[k])[0]
    # for k, v in results.items():
    #     keys = list(v.keys())
    #     values = list(v.values())
    #     sorted_value_index = np.argsort(values)
    #     sorted_dict = [keys[i] for i in sorted_value_index[:k]]
    #     print(sorted_dict)
    #


    #keys = list(dict.keys())
    #values = list()




    # for i in range(sum.shape[0]):
    #     for j in range(i, sum.shape[0]):
    #         dist = np.abs(sum[i] - sum[j])
    #         dist_matrix[i, j] = dist
    #         dist_matrix[j, i] = dist
    # for i in range(dist_matrix.)


       #x = np.linalg.norm()
      # print(x)




data = np.array([[1, 5],
                 [2, 4],
                 [3, 3],
                 [4, 2],
                 [0, 0]])

bool_target = np.array([0, 0, 0, 1, 1])

knn(data=data, k=4, target=bool_target)












