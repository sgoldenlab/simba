import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import itertools
from typing import List
from simba.utils.checks import check_int, check_instance, check_that_column_exist, check_str, check_valid_lst, check_int
try:
    from typing import Literal
except:
    from typing_extensions import Literal


def granger_tests(data: pd.DataFrame,
                  variables: List[str],
                  lag: int,
                  test: Literal['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'] = 'ssr_chi2test') -> pd.DataFrame:
    """
    Perform Granger causality tests between pairs of variables in a DataFrame.

    This function computes Granger causality tests between pairs of variables in a DataFrame
    using the statsmodels library. The Granger causality test assesses whether one time series
    variable (predictor) can predict another time series variable (outcome). This test can help
    determine the presence of causal relationships between variables.

    .. note::
       Modified from `Selva Prabhakaran <https://www.machinelearningplus.com/time-series/granger-causality-test-in-python/>`_.

    :example:
    >>> x = np.random.randint(0, 50, (100, 2))
    >>> data = pd.DataFrame(x, columns=['r', 'k'])
    >>> granger_tests(data=data, variables=['r', 'k'], lag=4, test='ssr_chi2test')
    """

    check_instance(source=granger_tests.__name__, instance=data, accepted_types=(pd.DataFrame,))
    check_valid_lst(data=variables, source=granger_tests.__name__, valid_dtypes=(str,), min_len=2)
    check_that_column_exist(df=data, column_name=variables, file_name='')
    check_str(name=granger_tests.__name__, value=test, options=('ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'))
    check_int(name=granger_tests.__name__, value=lag, min_value=1)
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c, r in itertools.product(df.columns, df.index):
        result = grangercausalitytests(data[[r, c]], maxlag=[lag], verbose=False)
        print(result)
        p_val = min([round(result[lag][0][test][1], 4) for i in range(1)])
        df.loc[r, c] = p_val
    return df


x = np.random.randint(0, 50, (100, 2))
data = pd.DataFrame(x, columns=['r', 'k'])
granger_tests(data=data, variables=['r', 'k'], lag=4, test='ssr_chi2test')



#
# for c in df.columns:
#     for r in df.index:
#         test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
#         p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
#         if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
#         min_p_value = np.min(p_values)
#         df.loc[r, c] = min_p_value
# df.columns = [var + '_x' for var in variables]
# df.index = [var + '_y' for var in variables]






#results = grangercausalitytests(x=x, maxlag=[4], verbose=False)