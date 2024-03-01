import pytest
import numpy as np
from simba.mixins.circular_statistics import CircularStatisticsMixin

def test_mean_resultant_vector_length():
    data = np.array([50, 90, 70, 60, 20, 90]).astype(np.float32)
    results = CircularStatisticsMixin().mean_resultant_vector_length(data=data)
    assert np.allclose(results, 0.9132277170817057)

@pytest.mark.parametrize('data_size, time_windows,', [(100, (1.0)), (200, (1.5, 2.0)), (10, (3.0, 1.0))])
def test_sliding_mean_resultant_vector_length(data_size, time_windows):
    time_windows = np.array([time_windows]).flatten()
    data = np.random.normal(loc=45, scale=1, size=data_size).astype(np.float32)
    results = CircularStatisticsMixin().sliding_mean_resultant_vector_length(data=data, time_windows=time_windows, fps=10)
    assert np.issubdtype(results.dtype, np.number)
    assert results.shape[0] == data.shape[0]
    assert results.shape[1] == time_windows.shape[0]

def test_circular_mean_1():
    data = np.array([315, 45, 270, 90]).astype(np.float32)
    assert CircularStatisticsMixin().circular_mean(data=data) == 0.0
    data = np.array([135, 45, 180, 0]).astype(np.float32)
    assert CircularStatisticsMixin().circular_mean(data=data) == 90.0

@pytest.mark.parametrize('data_size,', [(100), (150), (75)])
def test_circular_mean_1(data_size):
    data = np.random.randint(0, 361, size=(data_size,)).astype(np.float32)
    results = CircularStatisticsMixin().circular_mean(data=data)
    assert 0.0 <= results <= 360.0

def test_sliding_circular_mean_1():
    data = np.array([0, 10, 20, 30, 40, 50, 60, 70]).astype(np.float32)
    results = CircularStatisticsMixin().sliding_circular_mean(data=data, time_windows=np.array([1.0]), fps=2)
    expected_results = np.array([[-1.], [5.], [15], [25.], [35.], [45.], [55.], [65.]])
    assert np.allclose(results, expected_results)

@pytest.mark.parametrize('data_size, time_windows', [(100, (1.0, 1.5)), (150, 1.0), (75, 1.5)])
def test_sliding_circular_mean_2(data_size, time_windows):
    time_windows = np.array([time_windows]).flatten()
    data = np.random.randint(0, 361, size=(data_size,)).astype(np.float32)
    results = CircularStatisticsMixin().sliding_circular_mean(data=data, time_windows=time_windows, fps=2)
    assert np.issubdtype(results.dtype, np.number)
    assert results.shape[0] == data.shape[0]
    assert results.shape[1] == time_windows.shape[0]

@pytest.mark.parametrize('mean, std, size', [(45, 10.0, 500), (360, 2.0, 1000)])
def test_circular_std(mean, std, size):
    data = np.random.normal(loc=mean, scale=std, size=size).astype(np.float32)
    results = np.floor(CircularStatisticsMixin().circular_std(data=data))
    assert 0 <= results <= 360

def test_sliding_circular_std_1():
    data = np.array([0, 10, 20, 30 , 40 , 50, 60]).astype(np.float32)
    results = CircularStatisticsMixin.sliding_circular_std(data=data, fps=2, time_windows=np.array([1.0])).astype(np.int8)
    expected_results = np.array([[0], [5], [5], [5], [5], [5], [5]], dtype=np.int8)
    assert np.allclose(results, expected_results)

@pytest.mark.parametrize('size, time_windows', [(10, (1.5, 2.0)), (100, (1.0)), (1000, (2.0, 5.0))])
def test_sliding_circular_std_2(size, time_windows):
    data = np.random.randint(0, 361, size).astype(np.float32)
    time_windows = np.array([time_windows]).flatten().astype(np.float64)
    results = CircularStatisticsMixin.sliding_circular_std(data=data, fps=2, time_windows=time_windows)
    assert 0 <= np.min(results) <= 360
    assert 0 <= np.max(results) <= 360
    assert results.shape[0] == size
    assert results.shape[1] == time_windows.shape[0]

def test_instantaneous_angular_velocity():
    data = np.array([10, 20, 30, 40]).astype(np.float32)
    results = CircularStatisticsMixin.instantaneous_angular_velocity(data=data, bin_size=1)
    expected_results = np.array([-1,  10, 10, 10])
    np.equal(results, expected_results)

def test_degrees_to_cardinal():
    data = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360]).astype(np.float32)
    results = list(CircularStatisticsMixin().degrees_to_cardinal(data=data))
    assert results == ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

def test_circular_correlation_1():
    sample_1 = np.array([50, 90, 20, 60, 20, 90]).astype(np.float32)
    sample_2 = np.array([50, 90, 70, 60, 20, 90]).astype(np.float32)
    results = CircularStatisticsMixin().circular_correlation(sample_1=sample_1, sample_2=sample_2)
    assert np.allclose(results, 0.7649115920066833)

@pytest.mark.parametrize('size, ', [(100), (1000), (10000)])
def test_circular_correlation_1(size):
    sample_1 = np.random.randint(0, 361, size).astype(np.float32)
    sample_2 = np.random.randint(0, 361, size).astype(np.float32)
    result = CircularStatisticsMixin().circular_correlation(sample_1=sample_1, sample_2=sample_2)
    assert 0.0 <= result <= 1.0

def test_sliding_circular_correlation_1():
    pass
    sample_1 = np.array([50, 90, 20, 60, 20, 90]).astype(np.float32)
    sample_2 = np.array([50, 90, 70, 60, 20, 90]).astype(np.float32)
    results = CircularStatisticsMixin().sliding_circular_correlation(sample_1=sample_1, sample_2=sample_2, time_windows=np.array([1.0]), fps=2.0)
    expected_results = np.array([[np.nan], [1], [1], [1], [ 1], [1]], dtype=np.float32)
    assert np.allclose(results.astype(np.int8), expected_results.astype(np.int8))

@pytest.mark.parametrize('size, time_windows', [(100, (1.5, 2.0)), (1000, (0.5, 10.0)), (10000, (1.0))])
def test_sliding_circular_correlation_2(size, time_windows):
    sample_1 = np.random.randint(0, 361, size).astype(np.float32)
    sample_2 = np.random.randint(0, 361, size).astype(np.float32)
    time_windows = np.array([time_windows]).flatten().astype(np.float64)
    results = CircularStatisticsMixin().sliding_circular_correlation(sample_1=sample_1, sample_2=sample_2, time_windows=time_windows, fps=10.0)
    assert np.all((results >= 0.0) & (results <= 1.0))
    assert results.shape[0] == size
    assert results.shape[1] == time_windows.shape[0]

def test_sliding_angular_diff_1():
    data = np.array([1.0, 20.0, 40.0, 360.0]).astype(np.float32)
    results = CircularStatisticsMixin().sliding_angular_diff(data=data, time_windows=np.array([1.0]), fps=1.0)
    assert np.allclose(results, np.array([[ 0], [19], [20], [40]], dtype=np.int8))
    results = CircularStatisticsMixin().sliding_angular_diff(data=data, time_windows=np.array([1.0]), fps=3.0)
    assert np.allclose(results, np.array([[ 0], [0], [0], [1]], dtype=np.int8))

@pytest.mark.parametrize('size, time_windows', [(100, (1.5, 2.0)), (1000, (0.5, 10.0)), (10000, (1.0))])
def test_sliding_angular_diff_1(size, time_windows):
    data = np.random.randint(0, 361, size).astype(np.float32)
    time_windows = np.array([time_windows]).flatten().astype(np.float64)
    results = CircularStatisticsMixin().sliding_angular_diff(data=data, time_windows=time_windows, fps=5.0)
    assert np.all((results >= 0.0) & (results <= 180.0))
    assert results.shape[0] == size
    assert results.shape[1] == time_windows.shape[0]

def test_agg_angular_diff_timebins_1():
    data = np.array([0, 0, 0, 0, 0, 5, 5, 5, 5, 5]).astype(np.float32)
    expected_output = np.array([[0.], [0.], [0.], [0.], [0.], [5.], [5.], [5.], [5.], [5.]], dtype=np.int)
    results = CircularStatisticsMixin().agg_angular_diff_timebins(data=data, time_windows=np.array([1.0]), fps=5)
    assert np.array_equal(expected_output.astype(np.int32), results.astype(np.int32))

@pytest.mark.parametrize('size, time_windows', [(100, (1.5, 2.0)), (1000, (0.5, 10.0)), (10000, (1.0))])
def test_agg_angular_diff_timebins_2(size, time_windows):
    data = np.random.randint(0, 361, size).astype(np.float32)
    time_windows = np.array([time_windows]).flatten().astype(np.float64)
    results = CircularStatisticsMixin().agg_angular_diff_timebins(data=data, time_windows=time_windows, fps=5)
    assert np.all((results >= 0.0) & (results <= 180.0))
    assert results.shape[0] == size
    assert results.shape[1] == time_windows.shape[0]

def test_circular_range_1():
    data = np.array([180, 360, 0]).astype(np.float32)
    results = CircularStatisticsMixin().circular_range(data=data)
    assert results == 180.0
    data = np.array([180.0, 270.0, 225.0]).astype(np.float32)
    results = CircularStatisticsMixin().circular_range(data=data)
    assert results == 90.0

@pytest.mark.parametrize('size,', [(1000,), (500, ), (10000,)])
def test_circular_range_2(size):
    data = np.random.randint(0, 361, size).astype(np.float32)
    results = CircularStatisticsMixin().circular_range(data=data)
    assert 0.0 <= results <= 360

def test_sliding_circular_range_1():
    data = np.array([260, 280, 300, 340, 360, 0, 10, 350, 0, 15]).astype(np.float32)
    results = CircularStatisticsMixin().sliding_circular_range(data=data, time_windows=np.array([1.0]), fps=1)
    expected_results = np.array([[0],[20],[20],[40],[20],[0],[10],[20],[10],[15]])
    assert np.array_equal(expected_results.astype(np.int32), results.astype(np.int32))

@pytest.mark.parametrize('size, time_windows', [(100, (1.5, 2.0)), (1000, (0.5, 10.0)), (10000, (1.0))])
def test_sliding_circular_range_2(size, time_windows):
    data = np.random.randint(0, 361, size).astype(np.float32)
    time_windows = np.array([time_windows]).flatten().astype(np.float64)
    results = CircularStatisticsMixin().sliding_circular_range(data=data, time_windows=time_windows, fps=15)
    assert np.all((results >= 0.0) & (results <= 360.0))
    assert results.shape[0] == size
    assert results.shape[1] == time_windows.shape[0]

def test_circular_hotspots():
    data = np.array([270, 360, 10, 90, 91, 180, 185, 260]).astype(np.float32)
    bins = np.array([[270, 90], [91, 268]])
    results = CircularStatisticsMixin().circular_hotspots(data=data, bins=bins)
    assert np.array_equal(results, np.array([0.5, 0.5]))
    bins = np.array([[270, 0], [1, 90], [91, 180], [181, 269]])
    results = CircularStatisticsMixin().circular_hotspots(data=data, bins=bins)
    assert np.array_equal(results, np.array([0.25, 0.25, 0.25, 0.25]))

@pytest.mark.parametrize('size, time_windows', [(10, (np.array([[0, 90], [91, 180], [181, 270], [271, 359]]))), (10, (np.array([[0, 45], [46, 90], [91, 135], [136, 180], [181, 225], [225, 270], [271, 315], [316, 359]])))])
def test_sliding_circular_hotspots(size, time_windows):
    data = np.random.randint(0, 361, size=(size,)).astype(np.float32)
    results = CircularStatisticsMixin().sliding_circular_hotspots(data=data, bins=time_windows, time_window=1.0, fps=2)
    assert results.shape[0] == data.shape[0]
    assert results.shape[1] == time_windows.shape[0]
    assert np.all((results >= 0.0) & (results <= 1.0))

def test_rotational_direction_1():
    data = np.array([45, 50, 35, 50, 80, 350, 350, 0 , 180]).astype(np.float32)
    results = CircularStatisticsMixin().rotational_direction(data=data, stride=1)
    expected_result = np.array([0,  1,  2,  1,  1,  2,  0,  1,  2], dtype=np.int8)
    assert np.array_equal(results.astype(np.int8), expected_result.astype(np.int8))
    results = CircularStatisticsMixin().rotational_direction(data=data, stride=2)
    expected_result = np.array([0.0, 0.0,  2,  0,  1,  2,  2,  1,  2], dtype=np.int8)
    assert np.array_equal(results.astype(np.int8), expected_result.astype(np.int8))

@pytest.mark.parametrize('size, stride', [(100000, 2), (500000, 4)])
def test_rotational_direction_2(size, stride):
    data = np.random.randint(0, 361, size=(10000,)).astype(np.float32)
    results = CircularStatisticsMixin().rotational_direction(data=data, stride=2)
    assert np.array_equal(np.unique(results), np.array([0, 1, 2], dtype=np.int8))
    assert results.shape[0] == data.shape[0]

def test_fit_circle_1():
    data = np.array([[[5, 10], [10, 5], [15, 10], [10, 15]]]).astype(np.float64)
    results = CircularStatisticsMixin().fit_circle(data=data, max_iterations=300)
    expected_results = np.array([[10, 10,  5]])
    assert np.array_equal(expected_results, results)

@pytest.mark.parametrize('size, iterations', [(500, 100), (100, 200)])
def test_fit_circle_2(size, iterations):
    data = np.random.randint(0, 500, size=(size, 10, 10)).astype(np.float64)
    results = CircularStatisticsMixin().fit_circle(data=data, max_iterations=iterations)
    assert results.shape[0] == data.shape[0]
    assert np.issubdtype(results.dtype, np.number)
    assert results.shape[1] == 3

def test_rao_spacing():
    results_1 = CircularStatisticsMixin().rao_spacing(data=np.random.randint(0, 360, (5000,)).astype(np.float32))
    results_2 = CircularStatisticsMixin().rao_spacing(data=np.random.randint(45, 90, (5000,)).astype(np.float32))
    results_3 = CircularStatisticsMixin().rao_spacing(data=np.random.randint(0, 10, (5000,)).astype(np.float32))
    assert results_1 > results_2 > results_3 >= 0

@pytest.mark.parametrize('low, high, time_windows', [(0, 10, (1.5, 2.0)), (0, 319, (1.5))])
def test_sliding_rao_spacing(low, high, time_windows):
    data = np.random.randint(low=low, high=high, size=(500,)).astype(np.float32)
    time_windows = np.array([time_windows]).flatten().astype(np.float64)
    result = CircularStatisticsMixin().sliding_rao_spacing(data=data, time_windows=time_windows, fps=10)
    assert np.min(result) >= 0.0
    assert np.issubdtype(result.dtype, np.number)
    assert result.shape[0] == data.shape[0]
    assert result.shape[1] == time_windows.shape[0]

