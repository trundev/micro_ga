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
    layout = micro_ga.multivector.Cl(pos_sig, neg_sig, zero_sig)
    # Test signature vs. dimensions
    assert layout.sig.size == pos_sig + neg_sig + zero_sig
    assert layout.dims == layout.sig.size
    assert layout.gaDims == 1<<layout.dims
    assert len(layout.blades) == layout.gaDims

@pytest.fixture
def layout(pos_sig, neg_sig, zero_sig):
    return micro_ga.multivector.Cl(pos_sig, neg_sig, zero_sig)

def test_blades(layout):
    # Test blade names
    layout.scalar
    layout.I
    for k, v in layout.blades.items():
        mv = getattr(layout, k) if k else layout.scalar
        assert mv is v, 'The layout attribute and blade value must be identical'
        assert sum(v.value) == 1 and np.count_nonzero(v.value) == 1, \
               'Blade must have a single value set to 1'
