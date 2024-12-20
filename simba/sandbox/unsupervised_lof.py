from typing import List, Union
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

import pandas as pd
from simba.utils.checks import check_valid_lst, check_valid_array
from simba.utils.errors import InvalidInputError


def embedding_local_outliers(data: List[np.ndarray], k: Union[int, float] = 5, contamination: float = 1e-10):
    check_valid_lst(data=data, source=embedding_local_outliers.__name__, valid_dtypes=(np.ndarray,), min_len=1)
    for i in data: check_valid_array(data=i, source=embedding_local_outliers.__name__, accepted_ndims=(2,))
    if not isinstance(k, (int, float)):
        raise InvalidInputError(msg=f'k is invalid dtype. Found {type(k)}, accepted: int, float', source=embedding_local_outliers.__name__)
    for i in data:
        if isinstance(k, float):
            K = int(i.shape[0] * k)
        else:
            K = k
        if K > i.shape[0]:
            K = i.shape[0]
        lof_model = LocalOutlierFactor(n_neighbors=K, contamination=contamination)
        _ = lof_model.fit_predict(i)
        results = -lof_model.negative_outlier_factor_.astype(np.float32)
        print(results)









x = np.random.random(size=(500, 2))
y = np.array([[99, 100]]).reshape(-1, 2)

x = np.vstack([x, y])
embedding_local_outliers(data=[x], k=200)




#
# class LOF():
#     def __init__(self):
#         pass

