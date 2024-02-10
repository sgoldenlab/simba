import numpy as np
import pytest
import pickle

from simba.mixins.statistics_mixin import Statistics
from simba.utils.enums import TestPaths


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
        critical_values = pickle.load(open(TestPaths.CRITICAL_VALUES.value, 'rb'))['independent_t_test']['one_tail'].values
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
    critical_values = pickle.load(open(TestPaths.CRITICAL_VALUES.value, 'rb'))['f']['one_tail'].values
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


@pytest.mark.parametrize('bucket_method, expected_results', [("fd", 0.05),
                                                             ("doane", 0.03),
                                                             ("auto", 0.03),
                                                             ("scott", 0.12),
                                                             ("stone", 0.0),
                                                             ("rice", 0.03),
                                                             ("sturges", 0.03),
                                                             ("sqrt", 0.04)])
def test_wasserstein_distance(bucket_method, expected_results):
    sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
    sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
    results = Statistics().wasserstein_distance(sample_1=sample_1, sample_2=sample_2, bucket_method=bucket_method)
    assert np.round(results, 2) == expected_results