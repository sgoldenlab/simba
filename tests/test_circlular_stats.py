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
