import pytest
import numpy as np
from simba.mixins.timeseries_features_mixin import TimeseriesFeatureMixin

def test_local_maxima_minima_1():
    data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
    results = TimeseriesFeatureMixin().local_maxima_minima(data=data, maxima=False)
    assert np.array_equal(results, np.array([[0. , 3.9], [2. , 4.2], [5. , 3.9]], dtype=np.float32))
    results = TimeseriesFeatureMixin().local_maxima_minima(data=data, maxima=True)
    assert np.array_equal(results, np.array([[1. , 7.5], [4. , 7.5], [9. , 9.5]], dtype=np.float32))

@pytest.mark.parametrize('size, maxima,', [(1000, True), (3000, (False))])
def test_local_maxima_minima_2(size, maxima):
    data = np.random.random(size=(size,)).astype(np.float32)
    results = TimeseriesFeatureMixin().local_maxima_minima(data=data, maxima=maxima)
    assert np.issubdtype(results.dtype, np.number)
    assert results.shape[1] == 2

def test_crossings_1():
    data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
    results = TimeseriesFeatureMixin().crossings(data=data, val=7)
    assert results == 5
    results = TimeseriesFeatureMixin().crossings(data=data, val=8)
    assert results == 1

@pytest.mark.parametrize('size, threshold,', [(1000, 0.5), (3000, 0.2)])
def test_crossings_2(size, threshold):
    data = np.random.random(size=(size,)).astype(np.float32)
    results = TimeseriesFeatureMixin().crossings(data=data, val=threshold)
    assert type(results) == int
    assert results > size

def test_sliding_crossings_1():
    data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
    results = TimeseriesFeatureMixin().sliding_crossings(data=data, time_windows=np.array([1.0]), fps=2.0, val=7.0)
    expected_results = np.array([[-1], [ 1], [ 1], [ 0], [ 1], [ 1], [ 0], [ 0], [ 1], [ 0]], dtype=np.int32)
    assert np.array_equal(results, expected_results)
    results = TimeseriesFeatureMixin().sliding_crossings(data=data, time_windows=np.array([1.0]), fps=4.0, val=7.0)
    expected_results = np.array([[-1], [-1], [-1], [ 2], [ 2], [ 2], [ 2], [ 1], [ 1], [ 1]], dtype=np.int32)
    assert np.array_equal(results, expected_results)

@pytest.mark.parametrize('size, time_windows,', [(1000, (0.5, 1.0)), (3000, (2.0, 3.0))])
def test_sliding_crossings_2(size, time_windows):
    time_windows = np.array([time_windows]).flatten()
    data = np.random.random(size=(size,)).astype(np.float32)
    results = TimeseriesFeatureMixin().sliding_crossings(data=data, time_windows=time_windows, fps=4.0, val=0.5)
    assert np.issubdtype(results.dtype, np.int)
    assert results.shape[0] == data.shape[0]
    assert results.shape[1] == time_windows.shape[0]

def test_percentile_difference():
    data = np.array([3.9, 7.5, 4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
    result = TimeseriesFeatureMixin().percentile_difference(data=data, upper_pct=95, lower_pct=5)
    assert np.equal(np.round(result, 2), 0.74)
    data = np.array([-100, 10, 10, 10, 10, 10, 10, 10, 10, 10, 100]).astype(np.float32)
    result = TimeseriesFeatureMixin().percentile_difference(data=data, upper_pct=90, lower_pct=10)
    assert np.equal(result, 0.0)

@pytest.mark.parametrize('size, upper_pct, lower_pct', [(1000, 90, 10), (3000, 70, 40), (10000, 99, 1)])
def test_sliding_percentile_difference(size, upper_pct, lower_pct):
    data = np.random.random(size=(size,)) * 100
    results = TimeseriesFeatureMixin().sliding_percentile_difference(data=data.astype(np.float32), upper_pct=upper_pct, lower_pct=lower_pct, window_sizes=np.array([1.0]), fps=15)
    assert np.issubdtype(results.dtype, np.float)
    assert results.shape == (data.shape[0], 1)

def test_percent_beyond_n_std():
    data = np.array([3.9, 7.5, 4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float64)
    result = TimeseriesFeatureMixin().percent_beyond_n_std(data=data, n=1)
    assert result == 0.1
    data = np.random.normal(loc=50.0, scale=1.0, size=10000000).astype(np.float64)
    result = TimeseriesFeatureMixin().percent_beyond_n_std(data=data, n=1.0)
    assert np.round(result, 3) == 0.159
    result = TimeseriesFeatureMixin().percent_beyond_n_std(data=data, n=2.0)
    assert np.round(result, 3) == 0.023

@pytest.mark.parametrize('data, expected', [(np.array([-100, 10, 10, 10, 10, 10, 10, 10, 10, 100]).astype(np.float32), 0.8)])
def test_percent_in_percentile_window(data, expected):
    results = TimeseriesFeatureMixin().percent_in_percentile_window(data=data, upper_pct=90.0, lower_pct=10.0)
    assert results == expected

def test_permutation_entropy():
    data_sample = np.arange(0, 1000, 2).astype(np.float32)
    results_1 = TimeseriesFeatureMixin.permutation_entropy(data=data_sample, delay=1, dimension=3)
    np.random.shuffle(data_sample)
    results_2 = TimeseriesFeatureMixin.permutation_entropy(data=data_sample, delay=1, dimension=3)
    assert results_2 > results_1

def test_line_length():
    data = np.arange(0, 1002, 2).astype(np.float32)
    results = TimeseriesFeatureMixin.line_length(data=data)
    assert results == 1000.0

def test_sliding_variance():
    data = np.array([1, 2, 3, 1, 2, 9, 17, 2, 10, 4]).astype(np.float32)
    results = TimeseriesFeatureMixin().sliding_variance(data=data, window_sizes=np.array([0.5]), sample_rate=10)
    expected = np.array([[-1.], [-1.], [-1.], [-1.], [ 1.], [ 8.], [36.], [37.], [32.], [27.]], dtype=np.float32)
    assert np.array_equal(np.rint(results), expected)

@pytest.mark.parametrize('data, above, expected', [(np.array([1, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32), True, 2),
                                                   (np.array([1, 8, 2, 10, 8, 6, 8, 2, 2, 2]).astype(np.float32), False, 3)])
def test_longest_strike(data, above, expected):
    result = TimeseriesFeatureMixin.longest_strike(data=data, threshold=7, above=above)
    assert result == expected

def test_time_since_previous_threshold():
    data = np.array([1, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
    results = TimeseriesFeatureMixin().time_since_previous_threshold(data=data, threshold=7.0, above=True, fps=2.0)
    expected = np.array([-1. ,  0. ,  0.5,  0. ,  0. ,  0.5,  0. ,  0.5,  1. ,  1.5])
    assert np.array_equal(results, expected)
    results = TimeseriesFeatureMixin().time_since_previous_threshold(data=data, threshold=7.0, above=False, fps=2.0)
    expected = np.array([0. , 0.5, 0. , 0.5, 1. , 0. , 0.5, 0. , 0. , 0. ])
    assert np.array_equal(results, expected)

def test_time_since_previous_target_value():
    data = np.array([8, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
    results = TimeseriesFeatureMixin().time_since_previous_target_value(data=data, value=8.0, inverse=False, sample_rate=2.0)
    expected = np.array([0., 0., 0.5, 1., 0., 0.5, 0., 0.5, 1., 1.5])
    assert np.array_equal(results, expected)
    results = TimeseriesFeatureMixin().time_since_previous_target_value(data=data, value=8.0, inverse=True, sample_rate=2.0)
    expected = np.array([-1., -1., 0., 0., 0.5, 0., 0.5, 0., 0., 0.])
    assert np.array_equal(results, expected)

def test_benford_distribution():
    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
    data = np.random.choice(np.arange(1, 10), size=1000000, p=benford_distribution).astype(np.float32)
    results = TimeseriesFeatureMixin().benford_correlation(data=data)
    assert np.round(results, 1) == 1.0




