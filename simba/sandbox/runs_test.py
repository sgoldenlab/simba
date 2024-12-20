import numpy as np
from typing import Optional
from scipy import stats

def runs_test_one_sample(x: np.ndarray):
    cutoff = np.mean(x)
    xindicator = (x >= cutoff).astype(int)
    runstart = np.nonzero(np.diff(np.r_[[-np.inf], xindicator, [np.inf]]))[0]
    runs = np.diff(runstart)
    runs_sign = x[runstart[:-1]]
    runs_pos = runs[runs_sign == 1]
    runs_neg = runs[runs_sign == 0]
    n_runs = len(runs)
    npo = runs_pos.sum()
    nne = runs_neg.sum()
    n = npo + nne
    npn = npo * nne
    rmean = 2. * npn / n + 1
    rvar = 2. * npn * (2. * npn - n) / n ** 2. / (n - 1.)
    rstd = np.sqrt(rvar)
    rdemean = n_runs - rmean
    z = rdemean

    z /= rstd
    return z



x = np.random.random_integers(0, 2, (1000,))
#runs_test_one_sample(x=x)


x = np.zeros((100,))
x = np.concatenate((x, np.ones((100,))))
for i in range(1, x.shape[0], 2): x[i] = 1
runs_test_one_sample(x=x)