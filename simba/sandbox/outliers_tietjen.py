from typing import List, Union
import numpy as np

def outliers_tietjen(x: Union[List, np.ndarray], k: int = 2, hypo: bool = False, alpha: float = 0.05) -> Union[np.ndarray, bool]:
    arr = np.copy(x)
    n = arr.size

    def tietjen(x_, k_):
        x_mean = x_.mean()
        r = np.abs(x_ - x_mean)
        z = x_[r.argsort()]
        E = np.sum((z[:-k_] - z[:-k_].mean()) ** 2) / np.sum((z - x_mean) ** 2)
        return E

    e_x = tietjen(arr, k)
    e_norm = np.zeros(10000)

    for i in np.arange(10000):
        norm = np.random.normal(size=n)
        e_norm[i] = tietjen(norm, k)

    CV = np.percentile(e_norm, alpha * 100)
    result = e_x < CV

    if hypo:
        return result
    else:
        if result:
            ind = np.argpartition(np.abs(arr - arr.mean()), -k)[-k:]
            return np.delete(arr, ind)
        else:
            return arr


x = np.random.randint(0, 100, (100, ))

x = np.random.normal(100, 2, 100)
outlier = 10  # Value of the outlier
x = np.append(x, outlier)


d = outliers_tietjen(x=x)

d.shape

