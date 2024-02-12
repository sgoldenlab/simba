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
    results = np.round(CircularStatisticsMixin().circular_std(data=data), 0)
    assert np.allclose(results, std)
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




