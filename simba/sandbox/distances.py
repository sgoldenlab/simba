import numpy as np
from scipy import stats
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from simba.utils.checks import check_valid_array, check_str
from simba.utils.data import bucket_data
from simba.utils.enums import Options
from simba.mixins.statistics_mixin import Statistics


class DistanceStatistics(object):
    """
    Computes distances between probability distributions. Useful for (i) measure drift in datasets, and (ii) featurization of distribution shifts across.

    :examples:
    >>> x = np.array([1, 5, 10, 20, 50]).astype(np.float32)
    >>> y = np.array([1, 5, 10, 100, 110]).astype(np.float32)
    >>> distance_statistics = DistanceStatistics()
    >>> wave_hedges = distance_statistics.wave_hedges_distance(x=x, y=y, normalize=True)
    >>> gower = distance_statistics.gower_distance(x=x, y=y, normalize=True, bucket_method='auto')
    >>> wasserstein = distance_statistics.wasserstein_distance(x=x, y=y, normalize=True, bucket_method='auto')
    >>> jensen_shannon_divergence = distance_statistics.jensen_shannon_divergence(x=x, y=y, normalize=True, bucket_method='auto')
    >>> kullback_leibler_divergence = distance_statistics.kullback_leibler_divergence(x=x, y=y, normalize=True, bucket_method='auto', fill_value=20)
    >>> total_variation_distance = distance_statistics.total_variation_distance(x=x, y=y, normalize=False)
    >>> population_stability_index = distance_statistics.population_stability_index(x=x, y=y, normalize=True, bucket_method='auto', fill_value=1)
    >>> normalized_google_distance = distance_statistics.normalized_google_distance(x=x, y=y, normalize=False, bucket_method='auto', fill_value=1)
    >>> jeffreys_divergence = distance_statistics.jeffreys_divergence(x=x, y=y, normalize=True, bucket_method='auto', fill_value=1)


    References
    ----------
    .. [1] `statistical-distances <https://github.com/aziele/statistical-distance/blob/main/distance.py>`_.

    """
    def __init__(self):
        pass

    @staticmethod
    def distance_discretizer(func):
        def wrapper(self, x: np.ndarray, y: np.ndarray, bucket_method: Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = 1):
            check_valid_array(data=x, source=func.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, np.int8, np.float32, np.float64, int, float))
            check_valid_array(data=y, source=func.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, np.int8, np.float32, np.float64, int, float))
            check_str(name=f"{func.__name__} method", value=bucket_method, options=Options.BUCKET_METHODS.value)
            bin_width, bin_count = bucket_data(data=x, method=bucket_method)
            x_h = Statistics._hist_1d(data=x, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]), normalize=normalize)
            y_h = Statistics._hist_1d(data=y, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]), normalize=normalize)
            return func(self, x=x, y=y, bucket_method=bucket_method, bin_width=bin_width, bin_count=bin_count, normalize=normalize, x_h=x_h, y_h=y_h, fill_value=fill_value)

        return wrapper

    @distance_discretizer.__get__('')
    def wave_hedges_distance(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute Wave Hedges distance between two distributions.
        """

        return 0.5 * np.sum(np.abs(x_h - y_h))

    @distance_discretizer.__get__('')
    def gower_distance(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute Gower distance between two probability distributions.
        """
        return np.sum(np.abs(x_h - y_h)) / x_h.size

    @distance_discretizer.__get__('')
    def wasserstein_distance(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute Wasserstein distance between two distributions.

        .. note::
           Uses ``stats.wasserstein_distance``. I have tried to move ``stats.wasserstein_distance`` to jitted method extensively,
           but this doesn't give significant runtime improvement. Rate-limiter appears to be the _hist_1d.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: Wasserstein distance between ``sample_1`` and ``sample_2``
        """
        return stats.wasserstein_distance(u_values=x_h, v_values=y_h)

    @distance_discretizer.__get__('')
    def jensen_shannon_divergence(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute Jensen-Shannon divergence between two distributions. Useful for (i) measure drift in datasets, and (ii) featurization of distribution shifts across
        sequential time-bins.

        .. note::
           JSD = 0: Indicates that the two distributions are identical.
           0 < JSD < 1: Indicates a degree of dissimilarity between the distributions, with values closer to 1 indicating greater dissimilarity.
           JSD = 1: Indicates that the two distributions are maximally dissimilar.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators.
        :returns float: Jensen-Shannon divergence between ``sample_1`` and ``sample_2``
        """
        mean_hist = np.mean([x_h, y_h], axis=0)
        kl_sample_1, kl_sample_2 = stats.entropy(pk=x_h, qk=mean_hist), stats.entropy(pk=y_h, qk=mean_hist)
        return (kl_sample_1 + kl_sample_2) / 2

    @distance_discretizer.__get__('')
    def kullback_leibler_divergence(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute Kullback-Leibler divergence between two distributions.

        .. note::
           Empty bins (0 observations in bin) in is replaced with passed ``fill_value``.

           Its range is from 0 to positive infinity. When the KL divergence is zero, it indicates that the two distributions are identical. As the KL divergence increases, it signifies an increasing difference between the distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Optional[int] fill_value: Optional pseudo-value to use to fill empty buckets in ``sample_2`` histogram
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: Kullback-Leibler divergence between ``sample_1`` and ``sample_2``
        """
        x_h[x_h == 0] = fill_value
        y_h[y_h == 0] = fill_value
        return stats.entropy(pk=x_h, qk=y_h)

    @distance_discretizer.__get__('')
    def population_stability_index(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute Population Stability Index (PSI) comparing two distributions.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``. The PSI value ranges from 0 to positive infinity.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Optional[int] fill_value: Empty bins (0 observations in bin) in is replaced with ``fill_value``. Default 1.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: PSI distance between ``sample_1`` and ``sample_2``

        :example:
        >>> sample_1, sample_2 = np.random.randint(0, 100, (100,)), np.random.randint(0, 10, (100,))
        >>> Statistics().population_stability_index(sample_1=sample_1, sample_2=sample_2, fill_value=1, bucket_method='auto')
        >>> 3.9657026867553817
        """

        x_h[x_h == 0] = fill_value
        y_h[y_h == 0] = fill_value
        x_h, y_h = x_h / np.sum(x_h), y_h / np.sum(y_h)
        samples_diff = y_h - x_h
        log = np.log(y_h / x_h)
        return np.sum(samples_diff * log)

    @distance_discretizer.__get__('')
    def total_variation_distance(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Calculate the total variation distance between two probability distributions.

        :param np.ndarray x: A 1-D array representing the first sample.
        :param np.ndarray y: A 1-D array representing the second sample.
        :param Optional[str] bucket_method: The method used to determine the number of bins for histogram computation. Supported methods are 'fd' (Freedman-Diaconis), 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', and 'sqrt'. Defaults to 'auto'.
        :return float: The total variation distance between the two distributions.

        .. math::

           TV(P, Q) = 0.5 \sum_i |P_i - Q_i|

        where :math:`P_i` and :math:`Q_i` are the probabilities assigned by the distributions :math:`P` and :math:`Q`
        to the same event :math:`i`, respectively.

        :example:
        >>> DistanceStatistics.total_variation_distance(x=np.array([1, 5, 10, 20, 50]), y=np.array([1, 5, 10, 100, 110]))
        >>> 0.3999999761581421
        """
        return 0.5 * np.sum(np.abs(x_h - y_h))

    @distance_discretizer.__get__('')
    def normalized_google_distance(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute the normalized google distance between two probability distributions represented by histograms.
        """
        x, y = float(np.sum(x_h)), float(np.sum(y_h))
        return (max([x, y]) - float(np.sum(np.minimum(x_h, y_h)))) / ((x + y) - min([x, y]))

    @distance_discretizer.__get__('')
    def jeffreys_divergence(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = 'auto', normalize: Optional[bool] = True, fill_value: Optional[int] = None, bin_width=None, bin_count=None, x_h=None, y_h=None):
        """
        Compute the Jeffreys divergence between two probability distributions represented by histograms.
        """
        x_h[x_h == 0] = fill_value
        y_h[y_h == 0] = fill_value
        return np.sum((x_h - y_h) * np.log(x_h / y_h))