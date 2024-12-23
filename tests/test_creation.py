"""Test algebra/layout creation"""
import numpy as np
import pytest
import micro_ga

@pytest.fixture(params=[2, 3])
def pos_sig(request):
    return request.param

@pytest.fixture(params=[0, 1])
def neg_sig(request):
    return request.param

@pytest.fixture(params=[0, 1])
def zero_sig(request):
    return request.param

def test_dimensions(pos_sig, neg_sig, zero_sig):
    """Test signature vs. dimensions, blade number"""
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig)
    assert layout.sig.size == pos_sig + neg_sig + zero_sig
    assert layout.dims == layout.sig.size
    assert layout.gaDims == 1<<layout.dims
    assert len(layout.blades) == layout.gaDims

@pytest.fixture
def layout(pos_sig, neg_sig, zero_sig):
    return micro_ga.Cl(pos_sig, neg_sig, zero_sig)

def test_blades(layout):
    """Test blade names"""
    _ = layout.scalar
    _ = layout.I
    for k, v in layout.blades.items():
        mv = getattr(layout, k) if k else layout.scalar
        assert mv is v, 'The layout attribute and blade value must be identical'
        assert sum(v.value) == 1 and np.count_nonzero(v.value) == 1, \
               'Blade must have a single value set to 1'

def test_repr():
    """Test `repr()` and `str()` results"""
    layout = micro_ga.Cl(3)
    assert str(layout.scalar + 1e-12) == '+1.0'
    assert repr(layout.scalar + 1e-12) == '+1.000000000001'
    assert str(layout.scalar - layout.scalar) == '0'
