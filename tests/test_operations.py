"""Test arithmetic operations"""
import operator
import fractions
import decimal
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

@pytest.fixture
def layout(pos_sig, neg_sig, zero_sig):
    return micro_ga.Cl(pos_sig, neg_sig, zero_sig)

@pytest.fixture(params=[
        operator.add,
        operator.sub,
        #operator.mul,
        #operator.xor,  # outer product
        #operator.or_,  # inner product
    ])
def operation(request):
    return request.param

def test_operation(layout, operation):
    """Compare results from arithmetic operations with scalar vs. python integer"""
    # Pick the middle blade
    blade = tuple(layout.blades.values())[layout.gaDims//2]
    assert operation(layout.scalar, blade) == operation(1, blade)
    assert operation(blade, layout.scalar) == operation(blade, 1)
    # Unsupported operand type (right and left side)
    with pytest.raises(TypeError):
        _ = operation(blade, None)
    with pytest.raises(TypeError):
        _ = operation(None, blade)

@pytest.fixture(params=[ np.int32, np.float64, np.complex64, object,
                         fractions.Fraction, decimal.Decimal])
def dtype(request):
    return request.param

def test_blade_dtype(dtype):
    """Check the internal `numpy` array `dtype` of all blades"""
    layout = micro_ga.Cl(3, dtype=dtype)
    # Note: `dtype('O') == object`
    assert layout.scalar.value.dtype == dtype, 'Internal numpy array must use requested dtype'
    for blade in layout.blades.values():
        assert blade.value.dtype == dtype
    # Check type of individual values from the scalar-blade
    exp_t = int if dtype is object else dtype
    for v in layout.scalar.value:
        assert type(v) is exp_t, 'Individual values must be of requested type'

    # Check if both string representations work
    _ = repr(layout.scalar)
    _ = str(layout.scalar)

def test_operation_dtype(operation, dtype):
    """Check the internal `numpy` array `dtype` of operation result"""
    layout = micro_ga.Cl(3, dtype=dtype)
    mv = operation(1, layout.scalar)
    exp_dt = np.result_type(layout.scalar.value.dtype, dtype)
    assert mv.value.dtype is exp_dt, 'Result dtype must come from numpy conversion rules'
    # Check type of individual values
    exp_t = int if dtype is object else dtype
    for v in mv.value:
        assert type(v) is exp_t, 'Individual values of result must be of requested type'

def test_unbounded_int():
    """Test python unbounded `int` operation"""
    # With `object`, `numpy` falls-back to original python unbounded operation
    layout = micro_ga.Cl(2, dtype=object)
    mv = layout.scalar + (1<<100)
    assert (mv.value[0] - (1<<100)) == 1

    # With `int`, `numpy` uses `int64`, which is 64-bit only
    layout = micro_ga.Cl(2, dtype=int)
    with pytest.raises(OverflowError):
        mv = layout.scalar + (1<<100)
