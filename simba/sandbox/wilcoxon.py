import numpy as np
from simba.utils.data import fast_mean_rank


def wilcoxon(x: np.ndarray, y: np.ndarray):
    data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    n = data.shape[0]
    diff = np.diff(data).flatten()
    diff_abs = np.abs(diff)
    rank_w_ties = fast_mean_rank(data=diff_abs, descending=False)
    signed_rank_w_ties = np.full((rank_w_ties.shape[0]), np.nan)
    t_plus, t_minus = 0, 0

    for i in range(diff.shape[0]):
        if diff[i] < 0:
            signed_rank_w_ties[i] = -rank_w_ties[i]
            t_minus += np.abs(rank_w_ties[i])
        else:
            signed_rank_w_ties[i] = rank_w_ties[i]
            t_plus += np.abs(rank_w_ties[i])
    print(t_minus, t_plus, n)
    u_w = (n * (n + 1)) / 4
    std_correction = 0
    for i in range(signed_rank_w_ties.shape[0]):
        same_rank_n = np.argwhere(signed_rank_w_ties == signed_rank_w_ties[i]).flatten().shape[0]
        if same_rank_n > 1:
            std_correction += (((same_rank_n**3) - same_rank_n) / 2)

    std = np.sqrt(((n * (n + 1)) * ((2 * n) + 1) - std_correction) / 24)
    W = np.min((t_plus, t_minus))
    z = (W - u_w) / std
    r = (z / np.sqrt(n))
    return z, r






x = np.random.random(20,)
y = np.random.random(20,)

wilcoxon(x=x, y=y)