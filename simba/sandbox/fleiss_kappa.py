import numpy as np
from simba.utils.checks import check_valid_array
from statsmodels.stats.inter_rater import fleiss_kappa


def fleiss_kappa_(data: np.ndarray):
    check_valid_array(data=data, source=f'{fleiss_kappa_.__name__} data', accepted_ndims=(2,))
    col_sums = []
    for i in range(data.shape[1]):
        pass




    n_rat = data.sum(axis=1).max()
    p_cat = data.sum(axis=0) / data.sum()




    data_2 = data * data
    p_rat = (data_2.sum(axis=1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()
    p_mean_exp = (p_cat*p_cat).sum()
    return (p_mean - p_mean_exp) / (1 - p_mean_exp)




data = np.array([[0, 1, 2, 3], [0, 3, 1, 2]])

#data = np.random.randint(0, 2, (100, 4))
x = fleiss_kappa_(data=data)
y = fleiss_kappa(data)
print(x, y)