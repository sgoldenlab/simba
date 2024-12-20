import math
import numpy as np
from scipy.stats import norm
from typing import Optional


def d_prime(x: np.ndarray, y: np.ndarray):
    target_idx = np.argwhere(y == 1).flatten()
    hit_rate = np.sum(x[np.argwhere(y == 1)]) / target_idx.shape[0]
    false_alarm_rate = np.sum(x[np.argwhere(y == 0)]) / target_idx.shape[0]
    q = math.sqrt(-2.0 * math.log(hit_rate))
    q2 = math.sqrt(-2.0 * math.log(false_alarm_rate))
    zHR = q - ((q + 0.044715 * math.pow(q, 3)) / (1.0 + 0.196854 * math.pow(q, 2) + 0.056415 * math.pow(q, 3) + 0.004298 * math.pow(q, 4)))
    zFA = q - ((q2 + 0.044715 * math.pow(q2, 3)) / (1.0 + 0.196854 * math.pow(q2, 2) + 0.056415 * math.pow(q2, 3) + 0.004298 * math.pow(q2, 4)))

    return zHR - zFA



def d_prime(x: np.ndarray,
            y: np.ndarray,
            lower_limit: Optional[float] = 0.0001,
            upper_limit: Optional[float] = 0.9999) -> float:
    """
    Computes d-prime from two Boolean 1d arrays.

    :param np.ndarray x: Boolean 1D array of response values, where 1 represents presence, and 0 representing absence.
    :param np.ndarray y: Boolean 1D array of ground truth, where 1 represents presence, and 0 representing absence.
    :param Optional[float] lower_limit: Lower limit to bound hit and false alarm rates. Defaults to 0.0001.
    :param Optional[float] upper_limit: Upper limit to bound hit and false alarm rates. Defaults to 0.9999.
    :return float: The calculated d' (d-prime) value.

    :example:
    >>> x = np.random.randint(0, 2, (1000,))
    >>> y = np.random.randint(0, 2, (1000,))
    >>> d_prime(x=x, y=y)
    """

    target_idx = np.argwhere(y == 1).flatten()
    hit_rate = np.sum(x[np.argwhere(y == 1)]) / target_idx.shape[0]
    false_alarm_rate = np.sum(x[np.argwhere(y == 0)]) / target_idx.shape[0]
    if hit_rate < lower_limit: hit_rate = lower_limit
    elif hit_rate > upper_limit: hit_rate = upper_limit
    if false_alarm_rate < lower_limit: false_alarm_rate = lower_limit
    elif false_alarm_rate > upper_limit: false_alarm_rate = upper_limit
    return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)


#q = math.sqrt(-2.0 * math.log(p))
#  return q - ((q + 0.044715 * math.pow(q, 3)) / (1.0 + 0.196854 * math.pow(q, 2) + 0.056415 * math.pow(q, 3) + 0.004298 * math.pow(q, 4)))

# x = np.random.randint(0, 2, (1000,))
# y = np.random.randint(0, 2, (1000,))
# print(d_prime(x=x, y=y))
