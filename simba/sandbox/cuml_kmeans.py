import numpy as np
from typing import Optional, Tuple


from simba.utils.read_write import read_img_batch_from_video_gpu
from simba.mixins.image_mixin import ImageMixin
try:
   from cuml.cluster import KMeans
except:
    from sklearn.cluster import KMeans

from simba.utils.checks import check_int, check_valid_array
from simba.utils.enums import Formats


def kmeans_cuml(data: np.ndarray,
                k: int = 2,
                max_iter: int = 300,
                output_type: Optional[str] = None,
                sample_n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:

    """CRAP, SLOWER THAN SCIKIT"""

    check_valid_array(data=data, source=f'{kmeans_cuml.__name__} data', accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name=f'{kmeans_cuml.__name__} k', value=k, min_value=1)
    check_int(name=f'{kmeans_cuml.__name__} max_iter', value=max_iter, min_value=1)
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    if sample_n is not None:
        check_int(name=f'{kmeans_cuml.__name__} sample', value=sample_n, min_value=1)
        sample = min(sample_n, data.shape[0])
        data_idx = np.random.choice(np.arange(data.shape[0]), sample)
        mdl = kmeans.fit(data[data_idx])
    else:
        mdl = kmeans.fit(data)

    return  (mdl.cluster_centers_, mdl.predict(data))

import time
for i in [1000000, 2000000]:
    data = np.random.randint(0, 500, (i, 400)).astype(np.int32)
    start = time.perf_counter()
    results = kmeans_cuml(data=data)
    elapsed = time.perf_counter() - start
    print(i, elapsed)