import numpy as np
from scipy.stats import norm




def wald_wolfowitz_runs_test(data: np.ndarray):
    diffs = np.diff(data)
    trans = np.count_nonzero(diffs != 0)
    r = trans + 1
    m = np.argwhere(data == 0).shape[0]
    n = np.argwhere(data == 1).shape[0]

    nominator = r - ((2*m*n) / (m+n)) + 1

    denominator = ((2*m*n)*(2*m*n-(m+n))) / (np.square(m+n)) * ((m+n) -1)

    z = nominator / denominator
    p = 2 * (1 - norm.cdf(abs(z)))
    print(z, p)



    # n = data.shape[0]
    # mean_runs = (2 * n - 1) / 3
    # var_runs = (16 * n - 29) / 90
    # z_score = (runs - mean_runs) / (var_runs ** 0.5)
    # p_value = 2 * (1 - norm.cdf(abs(z_score)))
    # return runs, p_value

# data = np.array((0))
# for i in range(5):
#     data = np.array([1,0,1,0,1,0,1,0,1,0])
#     v

data_1 = np.random.randint(0, 2, (100,))
data = np.array([1,1,1,1,1,1,1,0,0,0,1,0, 0, 0, 1])

wald_wolfowitz_runs_test(data=data_1)