"""Test algebra/layout creation"""
from typing import Any
import numpy as np
import micro_ga
from . import pos_sig, neg_sig, zero_sig, layout, dtype # pylint: disable=W0611


def test_dimensions(pos_sig, neg_sig, zero_sig):
    """Test signature vs. dimensions, blade number"""
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig)
    assert layout.sig.size == pos_sig + neg_sig + zero_sig
    assert layout.dims == layout.sig.size
    assert layout.gaDims == 1<<layout.dims
    assert len(layout.blades) == layout.gaDims

def test_null_sig():
    """Test scalar-only algebra"""
    test_dimensions(0, 0, 0)
    layout = micro_ga.Cl(0)
    assert layout.scalar is layout.I
    assert layout.blades == {'': layout.scalar}

def test_blades(layout):
    """Test blade names"""
    _ = layout.scalar
    _ = layout.I
    for k, v in layout.blades.items():
        mv = getattr(layout, k) if k else layout.scalar
        assert mv is v, 'The layout attribute and blade value must be identical'
        assert sum(v.value) == 1 and np.count_nonzero(v.value) == 1, \
               'Blade must have a single value set to 1'

def test_comparison(pos_sig, neg_sig):
    """Test algebra and multi-vector comparison operators"""
    layout = micro_ga.Cl(pos_sig, neg_sig)
    assert layout == micro_ga.Cl(pos_sig, neg_sig)
    assert layout != 'BAD'
    assert layout.scalar != 'BAD'
    # Different layout with different dimensions
    layout2 = micro_ga.Cl(neg_sig, pos_sig+1)
    assert layout != layout2
    assert layout.scalar != layout2.scalar

def test_repr(dtype):
    """Test `repr()` and `str()` results"""
    layout = micro_ga.Cl(3, dtype=dtype)
    if dtype is object:
        exp_dtype = int         # Default `micro_ga` type
    else:
        exp_dtype = dtype
    # Algebra representation
    assert str(layout) == f'Cl(sig={[1]*layout.dims}, dtype={exp_dtype.__name__})'
    # Basic multi-vector representations
    py_type = exp_dtype
    if issubclass(py_type, np.number):
        # String representation works on python type, but not on `numpy` type
        py_type: Any = type(np.zeros(1, dtype=dtype).item(0))
    assert str(layout.scalar) == f'{py_type(1)}'
    assert str(layout.scalar - layout.scalar) == '0'
    assert str(-layout.scalar + layout.I) == f'{py_type(-1)} + {py_type(1)}*e123'
    assert repr(-layout.scalar + layout.I) == f'MVector({py_type(-1)!r} + {py_type(1)!r}*e123)'
