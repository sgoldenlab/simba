import pytest
import numpy as np
from simba.mixins.circular_statistics import CircularStatisticsMixin

def test_mean_resultant_vector_length():
    data = np.array([50, 90, 70, 60, 20, 90]).astype(np.float32)
    results = CircularStatisticsMixin().mean_resultant_vector_length(data=data)
    assert np.allclose(results, 0.9132277170817057)