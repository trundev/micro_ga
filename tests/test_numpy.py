"""Test `numpy` array integration"""
import numpy as np
import pytest
import micro_ga

@pytest.fixture(params=[2,3])
def layout(request):
    return micro_ga.Cl(request.param)

def test_ndarray(layout):
    """Test integration with `numpy.ndarray`"""
    res = np.arange(10) + layout.scalar
    np.testing.assert_equal(res, np.arange(10) + 1)
