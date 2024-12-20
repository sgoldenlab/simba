import pandas as pd
import numpy as np
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import List
from simba.utils.checks import check_float, check_instance, check_int, check_valid_dataframe, check_str
from typing import Optional, Tuple, Any
from simba.utils.errors import InvalidInputError
from simba.mixins.statistics_mixin import Statistics
from itertools import combinations
from simba.utils.read_write import read_pickle

def find_collinear_features(df: pd.DataFrame,
                           threshold: float,
                           method: Optional[Literal['pearson', 'spearman', 'kendall']] = 'pearson',
                           verbose: Optional[bool] = False) -> List[str]:
    """
    Identify collinear features in the dataframe based on the specified correlation method and threshold.

    :param pd.DataFrame df: Input DataFrame containing features.
    :param float threshold: Threshold value to determine collinearity.
    :param Optional[Literal['pearson', 'spearman', 'kendall']] method: Method for calculating correlation. Defaults to 'pearson'.
    :return: Set of feature names identified as collinear. Returns one feature for every feature pair with correlation value above specified threshold.

    :example:
    >>> x = pd.DataFrame(np.random.randint(0, 100, (100, 100)))
    >>> names = find_collinear_features(df=x, threshold=0.2, method='pearson', verbose=True)
    """

    check_valid_dataframe(df=df, source=find_collinear_features.__name__, valid_dtypes=(float, int, np.float32, np.float64, np.int32, np.int64), min_axis_1=1, min_axis_0=1)
    check_float(name=find_collinear_features.__name__, value=threshold, max_value=1.0, min_value=0.0)
    check_str(name=find_collinear_features.__name__, value=method, options=('pearson', 'spearman', 'kendall'))
    feature_names = set()
    feature_pairs = list(combinations(list(df.columns), 2))

    for cnt, i in enumerate(feature_pairs):
        if verbose:
            print(f'Analyzing feature pair {cnt+1}/{len(feature_pairs)}...')
        if (i[0] not in feature_names) and (i[1] not in feature_names):
            sample_1, sample_2 = df[i[0]].values.astype(np.float32), df[i[1]].values.astype(np.float32)
            if method == 'pearson':
                r = Statistics.pearsons_r(sample_1=sample_1, sample_2=sample_2)
            elif method == 'spearman':
                r = Statistics.spearman_rank_correlation(sample_1=sample_1, sample_2=sample_2)
            else:
                r = Statistics.kendall_tau(sample_1=sample_1, sample_2=sample_2)[0]
            if abs(r) > threshold:
                feature_names.add(i[0])
    if verbose:
        print('Collinear analysis complete.')
    return feature_names



data_path = '/Users/simon/Desktop/envs/NG_Unsupervised/project_folder/embeddings/amazing_rubin.pickle'
data = read_pickle(data_path)['DATA']['FRAME_FEATURES'] #['BOUTS_FEATURES']

x = pd.DataFrame(np.random.randint(0, 100, (100, 100)))
names = find_collinear_features(df=x, threshold=0.2, method='pearson', verbose=True)

data['Mouse_1_poly_area'].corr(data['Low_prob_detections_0.75'])

Statistics.pearsons_r(sample_1=data['Mouse_1_poly_area'].values, sample_2=data['Low_prob_detections_0.75'].values)




# l = find_colinear_features(df=data, threshold=0.7, method='pearson')
#




remain = list(set(list(data.columns)) - set(names))

print(len(remain))

#

#check_valid_dataframe(df=df, valid_dtypes=(float, int,),)











