import numpy as np
from pathlib import Path
import pytest
import pickle
import os

from simba.mixins.statistics_mixin import Statistics
from simba.utils.enums import TestPaths, Paths

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.fixture(scope="session")
def sample_data_dir():
    return Path(__file__).resolve().parent / "data" / "sample_data"

def test_rolling_independent_sample_t():
    data = np.array([1, 3, 4, 7, 10, 6, 10, 12, 239, 34, 21]).astype(np.float32)
    results = Statistics().rolling_independent_sample_t(data=data, time_window=1.0, fps=2)
    expected_results = np.array([-1. , -1. , -2.74562589, -2.74562589, -1.41421356, -1.41421356, -1.8973666 , -1.8973666 , -1.73146689, -1.73146689, 0.92005224])
    assert np.allclose(results, expected_results)


@pytest.mark.parametrize("critical_values, expected_results", [(True, (-2.5266047334683974, True)), (False, (-2.5266047334683974, None))])
def test_independent_samples_t(critical_values, expected_results):
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10]).astype(np.float32)
    sample_2 = np.array([2, 5, 10, 4, 8, 10, 7, 10, 7, 10, 10]).astype(np.float32)
    if critical_values:
        critical_values = pickle.load(open(Paths.CRITICAL_VALUES.value, 'rb'))['independent_t_test']['one_tail'].values
        results = Statistics().independent_samples_t(sample_1=sample_1, sample_2=sample_2, critical_values=critical_values)
    else:
        results = Statistics().independent_samples_t(sample_1=sample_1, sample_2=sample_2)
    assert results == expected_results

def test_cohens_d():
    sample_1 = np.array([2, 4, 7, 3, 7, 35, 8, 9]).astype(np.float64)
    sample_2 = np.array([4, 8, 14, 6, 14, 70, 16, 18]).astype(np.float64)
    results = Statistics().cohens_d(sample_1=sample_1, sample_2=sample_2)
    assert results == -0.5952099775170546

def test_rolling_cohens_d():
    data = np.array([11.66098032, 11.52882001,  8.28137675,  9.63358521, 10.55202715, 8.57905361, 14.06717964,  8.85201707]).astype(np.float64)
    results = Statistics().rolling_cohens_d(data=data, time_windows=np.array([1.0]), fps=4.0)
    expected_results = np.array([[0], [0], [0], [0], [-0.12864096], [-0.12864096], [-0.12864096], [-0.12864096]]).astype(np.float64)
    assert np.allclose(results, expected_results)

def test_rolling_two_sample_ks():
    data = np.array([4., 0., 4., 4., 2., 2., 3., 0., 0., 3., 0., 1., 0., 1., 2.], dtype=np.float32)
    expected_results = np.array([-1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,  1., 1.,  1.])
    results = Statistics().rolling_two_sample_ks(data=data, time_window=1, fps=1)
    assert np.allclose(results, expected_results)

def test_two_sample_ks():
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10]).astype(np.float32)
    sample_2 = np.array([10, 5, 10, 4, 8, 10, 7, 10, 7, 10, 10]).astype(np.float32)
    results = Statistics().two_sample_ks(sample_1=sample_1, sample_2=sample_2)
    assert results == (0.7272727272727273, None)

def test_one_way_anova():
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
    sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
    critical_values = pickle.load(open(Paths.CRITICAL_VALUES.value, 'rb'))['f']['one_tail'].values
    results = Statistics().one_way_anova(sample_1=sample_2, sample_2=sample_1, critical_values=critical_values)
    assert results == (5.8483754512635375, True)

def test_rolling_one_way_anova():
    sample = np.array([11.50357579,  9.32686126,  9.63068002,  9.6988871 , 11.40003894, 9.42987609,  9.28868644,  9.77685886, 11.19658191,  8.86118912], dtype=np.float32)
    results = Statistics().rolling_one_way_anova(data=sample, time_windows=np.array([1.0]), fps=2)
    expected_results = np.array([[0.        ], [0.        ], [0.47496008], [0.47496008], [0.57924093], [0.57924093], [0.75560836], [0.75560836], [0.17295212], [0.17295212]])
    assert np.allclose(results, expected_results)

@pytest.mark.parametrize('bucket_method, expected_results', [("fd", 0.59), ("doane", 0.37), ("auto", 0.53), ("scott", 1.22), ("stone", 0.03), ("rice", 0.53), ("sturges", 0.53), ("sqrt", 0.66), ("auto",   0.53)])
def test_kullback_leibler_divergence(bucket_method, expected_results):
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
    sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
    results = Statistics().kullback_leibler_divergence(sample_1=sample_1, sample_2=sample_2, fill_value=1, bucket_method=bucket_method)
    assert np.round(results, 2) == expected_results

@pytest.mark.parametrize('bucket_method, expected_results', [("fd", 0.13), ("doane", 0.32), ("auto", 0.3), ("scott", 0.25), ("stone", 0.43), ("rice", 0.3), ("sturges", 0.3), ("sqrt", 0.28)])
def test_jensen_shannon_divergence(bucket_method, expected_results):
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
    sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
    results = Statistics().jensen_shannon_divergence(sample_1=sample_1, sample_2=sample_2, bucket_method=bucket_method)
    assert np.round(results, 2) == expected_results


@pytest.mark.parametrize('bucket_method, expected_results', [("fd", 0.05), ("doane", 0.03), ("auto", 0.03), ("scott", 0.12), ("stone", 0.0), ("rice", 0.03), ("sturges", 0.03), ("sqrt", 0.04)])
def test_wasserstein_distance(bucket_method, expected_results):
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
    sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
    results = Statistics().wasserstein_distance(sample_1=sample_1, sample_2=sample_2, bucket_method=bucket_method)
    assert np.round(results, 2) == expected_results

@pytest.mark.parametrize('bucket_method, expected_results', [("fd", 1.13), ("doane", 0.72), ("auto", 1.0), ("scott", 2.26), ("stone", 0.06), ("rice", 1.0), ("sturges", 1.0), ("sqrt", 1.24)])
def test_population_stability_index(bucket_method, expected_results):
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
    sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
    results = Statistics().population_stability_index(sample_1=sample_1, sample_2=sample_2, bucket_method=bucket_method)
    assert np.round(results, 2) == expected_results

def test_kruskal_wallis():
    sample_1 = np.array([1, 1, 3, 4, 5]).astype(np.float64)
    sample_2 = np.array([6, 7, 8, 9, 10]).astype(np.float64)
    results = Statistics().kruskal_wallis(sample_1=sample_1, sample_2=sample_2)
    assert np.equal(results, 39.4)

def test_mann_whitney():
    sample_1 = np.array([1, 1, 3, 4, 5]).astype(np.float64)
    sample_2 = np.array([1, 3, 2, 1, 6]).astype(np.float64)
    results = Statistics().mann_whitney(sample_1=sample_1, sample_2=sample_2)
    assert np.equal(results, 11.5)

def test_brunner_munzel():
    sample_1 = np.array([1, 1, 3, 4, 5]).astype(np.float64)
    sample_2 = np.array([1, 3, 2, 1, 6]).astype(np.float64)
    results = Statistics().brunner_munzel(sample_1=sample_1, sample_2=sample_2)
    assert np.equal(np.round(results, 2), -0.31)

def test_pearsons_r():
    sample_1 = np.array([7, 2, 9, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    sample_2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]).astype(np.float32)
    results = Statistics().pearsons_r(sample_1=sample_1, sample_2=sample_2)
    assert np.equal(np.round(results, 2), 0.47)

def test_spearman_rank_correlation():
    sample_1 = np.array([7, 2, 9, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    sample_2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]).astype(np.float32)
    results = Statistics().spearman_rank_correlation(sample_1=sample_1, sample_2=sample_2)
    assert np.equal(np.round(results, 2), 0.53)

def test_rolling_mann_whitney():
    sample_1 = np.array([2., 0., 0., 1., 0., 1., 0., 0., 1., 1.], dtype=np.float32)
    results = Statistics().rolling_mann_whitney(data=sample_1, time_windows=np.array([1.0]), fps=1)
    expected_results = np.array([[0. ], [0. ], [0.5], [0. ], [0. ], [0. ], [0. ], [0.5], [0. ], [0.5]])
    assert np.allclose(results, expected_results)


def test_sliding_spearman_rank_correlation():
    sample_1 = np.array([9, 10, 13, 22, 15, 18, 15, 19, 32, 11]).astype(np.float32)
    sample_2 = np.array([11, 12, 15, 19, 21, 26, 19, 20, 22, 19]).astype(np.float32)
    expected_results = np.array([[ 0. ], [ 0. ], [ 0. ], [ 0. ], [ 0.9], [ 0.7], [ 0.3], [-0.5], [ 0.3], [ 0.6]])
    results = Statistics().sliding_spearman_rank_correlation(sample_1=sample_1, sample_2=sample_2, time_windows=np.array([0.5]), fps=10)
    assert np.allclose(results, expected_results)

def test_sliding_autocorrelation():
    data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 10, 11, 12, 13, 14]).astype(np.float32)
    expected_results = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.        ,  0.        ,  0.        ,  0.        , -3.68623543,-2.02938271, -1.32312334, -1.75346696, -3.8074882 , -4.63441181])
    results = Statistics().sliding_autocorrelation(data=data, max_lag=0.5, time_window=1.0, fps=10)
    np.allclose(results, expected_results)

def test_kendall_tau():
    sample_1 = np.array([4, 2, 3, 4, 5, 7]).astype(np.float32)
    sample_2 = np.array([1, 2, 3, 4, 5, 7]).astype(np.float32)
    results = Statistics().kendall_tau(sample_1=sample_1, sample_2=sample_2)
    assert results == (0.7333333333333333, 2.0665401605809928)

def test_sliding_kendall_tau():
    sample_1 = np.array([4, 2, 3, 4, 5, 7]).astype(np.float32)
    sample_2 = np.array([1, 2, 3, 4, 5, 7]).astype(np.float32)
    time_windows = np.array([1.0, 1.5])
    expected_results = np.array([[ 0.        ,  0.        ],[ 0.        ,  0.        ],[-1.        ,  0.        ],[ 1.        , -0.33333333],[ 1.        ,  1.        ],[ 1.        ,  1.        ]])
    results = Statistics().sliding_kendall_tau(sample_1=sample_1, sample_2=sample_2, time_windows=time_windows, fps=2)
    assert np.allclose(expected_results, results)

@pytest.mark.parametrize('sample_data, expected_results', [('8x100_f32.npy', (8, ))])
def test_local_outlier_factor(sample_data, expected_results, sample_data_dir):
    data_path = os.path.join(sample_data_dir, sample_data)
    data = np.load(data_path)
    results = Statistics().local_outlier_factor(data=data, k=5)
    assert results.shape == expected_results
    assert np.issubdtype(results.dtype, np.number)

@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="descretizier.")
@pytest.mark.parametrize('sample_data, expected_results', [('8x100_f32.npy', (8, ))])
def test_hbos(sample_data, expected_results, sample_data_dir):
    data_path = os.path.join(sample_data_dir, sample_data)
    data = np.load(data_path)
    results = Statistics().hbos(data=data)
    assert results.shape == expected_results
    assert np.issubdtype(results.dtype, np.number)

def test_cohens_h():
    sample_1 = np.array([1, 0, 0, 1])
    sample_2 = np.array([1, 1, 1, 0])
    result = Statistics().cohens_h(sample_1=sample_1, sample_2=sample_2)
    assert np.round(result, 2) == -0.52

@pytest.mark.parametrize('time_windows', [(1.0, 2.0), (1.0), (1.5)])
def test_sliding_skew(time_windows):
    time_windows = np.array([time_windows]).flatten()
    data = np.random.randint(0, 100, (10,))
    results = Statistics().sliding_skew(data=data.astype(np.float32), time_windows=time_windows, sample_rate=2)
    assert np.issubdtype(results.dtype, np.number)
    assert results.shape[1] == time_windows.shape[0]

@pytest.mark.parametrize('time_windows', [(1.0, 2.0), (1.0), (1.5)])
def test_sliding_kurtosis(time_windows):
    time_windows = np.array([time_windows]).flatten()
    data = np.random.randint(0, 100, (10,))
    results = Statistics().sliding_kurtosis(data=data.astype(np.float32), time_windows=time_windows, sample_rate=2)
    assert np.issubdtype(results.dtype, np.number)
    assert results.shape[1] == time_windows.shape[0]

@pytest.mark.parametrize('k, data_size, medians', [(3, 100, True), (10, 200, False)])
def test_kmeans_1d(k, data_size, medians):
    data = np.random.randint(0, 100, (data_size,)).astype(np.float64)
    centroids, labels, median_res = Statistics().kmeans_1d(data=data, k=k, max_iters=10, calc_medians=medians)
    assert np.unique(centroids).shape[0] == k
    assert np.unique(labels).shape[0] == k
    assert labels.shape[0] == data_size
    if medians:
        assert len(list(median_res.keys())) == k

def test_hamming_distance():
    x, y = np.random.randint(0, 2, (10,)).astype(np.int8), np.random.randint(0, 2, (10,)).astype(np.int8)
    results = Statistics().hamming_distance(x=x, y=y)
    assert type(results) == float
    assert 0.0 <= results <= 1.0

@pytest.mark.parametrize('data_size,', [(100, 23), (50, 100), (1000, 10)])
def test_bray_curtis_dissimilarity(data_size):
    x = np.random.random(size=data_size).astype(np.float32)
    results = Statistics().bray_curtis_dissimilarity(x=x)
    assert np.issubdtype(results.dtype, np.number)
    assert results.shape[1] == data_size[0]
    assert results.shape[0] == data_size[0]

@pytest.mark.parametrize('bucket_method, data_size,', [("fd", 100), ("doane", 10), ("auto", 50), ("scott", 100), ("stone", 232), ("rice", 100), ("sturges", 20), ("sqrt", 40)])
def test_hellinger_distance(bucket_method, data_size):
    x = np.random.randint(-100, 100, (data_size))
    y = np.random.randint(-100, 100, (data_size))
    results = Statistics().hellinger_distance(x=x, y=y, bucket_method=bucket_method)
    assert type(results) == float