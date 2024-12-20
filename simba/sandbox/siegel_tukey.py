import numpy as np
#from simba.utils.data import rank

def siegel_tukey(x: np.ndarray, y: np.ndarray):
    """
    Siegel-Tukey test, also known as the Tukey one-degree-of-freedom test for nonadditivity.

    Non-parametric test to check if the variability between two groups (e.g., features) and similar.

    :example:
    >>> x = np.random.random((10,))
    >>> y = np.random.random((10,))
    >>> siegel_tukey(x=x, y=y)
    """
    x = np.hstack((np.zeros(x.shape[0]).reshape(-1, 1), x.reshape(-1, 1)))
    y = np.hstack((np.ones(y.shape[0]).reshape(-1, 1), y.reshape(-1, 1)))
    data = np.vstack((x, y))
    sorted_data = data[data[:, 1].argsort()]
    results = np.full((sorted_data.shape[0], 3), -1.0)
    results[0, 0:2] = sorted_data[0, :]
    results[0, 2] = 1
    top, bottom = np.array_split(sorted_data[1:, ], 2)
    bottom = bottom[::-1]
    start, end, c_rank = 1, 5, 2
    for i in range(1, max(bottom.shape[0], top.shape[0]), 2):
        b_ = bottom[i-1:i+1]
        b_ = np.hstack((b_, np.full((b_.shape[0], 1), c_rank)))
        c_rank += 1
        t_ = top[i - 1:i + 1]
        t_ = np.hstack((t_, np.full((t_.shape[0], 1), c_rank)))
        c_rank += 1
        results[start:end, :] = np.vstack((b_, t_))
        start, end = end, end+4

    w_a = np.sum(results[np.argwhere(results[:, 0] == 0)][:, -1][:, -1])
    w_b = np.sum(results[np.argwhere(results[:, 0] == 1)][:, -1][:, -1])

    u_a = np.abs(w_a - (x.shape[0] * (x.shape[0] + 1) / 2))
    u_b = np.abs(w_b - (y.shape[0] * (y.shape[0] + 1) / 2))

    return u_a, u_b, (np.max((u_b, u_a)) - np.min((u_b, u_a))) - x.shape[0]










    #print(sorted_data)
    # rank_0 = np.full((sorted_data.shape[0], 1), 0)
    # visited = []
    # t, b, c_rank, dir, pos = 0, sorted_data.shape[0]-1, 1, 1, 0
    # while len(visited) < sorted_data.shape[0]:
    #     rank_0[pos] = c_rank
    #     visited.append(pos)
    #     if dir == 1:
    #         if sorted_data[pos+dir] == sorted_data[pos]:
    #             pos += 1
    #             t += 1
    #         else:
    #             pos = b
    #             dir = -1
    #     elif dir == -1:
    #         if sorted_data[pos + dir] == sorted_data[pos]:
    #             pos -= 1
    #             b -= 1
    #         else:
    #             pos = t
    #             dir = 1
    #



        #print(visited, pos)







    #sorted_data = [i[::-1] for i in sorted_data[::-1]]
    #print(sorted_data)

    #np.sort(data)




x = np.random.randint(0, 100, (10,))
y = np.random.randint(50, 10000, (10,))
siegel_tukey(x=y, y=x)
