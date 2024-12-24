"""Test `numpy` array integration"""
import pytest
import numpy as np
from . import pos_sig, layout   # pylint: disable=W0611

@pytest.fixture
def neg_sig():
    """Skip tests with basis-vectors of negative signature"""
    return 0

@pytest.fixture
def zero_sig():
    """Skip tests with basis-vectors of zero signature"""
    return 0

def test_ndarray(layout):
    """Test integration with `numpy.ndarray`"""
    res = np.arange(10) + layout.scalar
    np.testing.assert_equal(res, np.arange(10) + 1)
