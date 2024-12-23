"""Test arithmetic operations"""
import operator
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

@pytest.mark.parametrize('func', [
        operator.add,
        operator.sub,
        #operator.mul,
        #operator.xor,  # outer product
        #operator.or_,  # inner product
    ])
def test_operation(layout, func):
    """Compare results from arithmetic operations with scalar vs. python integer"""
    # Pick the middle blade
    blade = tuple(layout.blades.values())[layout.gaDims//2]
    assert func(layout.scalar, blade) == func(1, blade)
    assert func(blade, layout.scalar) == func(blade, 1)

@pytest.mark.parametrize('dtype', [np.int32, np.float64, np.complex64, object])
def test_dtypes(dtype):
    """Check the internal `numpy` array `dtype` of all blades and in operation result"""
    layout = micro_ga.Cl(3, dtype=dtype)
    # Note: `dtype('O') == object`
    assert layout.scalar.value.dtype == dtype, 'Internal numpy array must use requested dtype'
    for blade in layout.blades.values():
        assert blade.value.dtype == dtype

    mv = 1 + layout.scalar
    exp = np.result_type(layout.scalar.value.dtype, dtype)
    assert mv.value.dtype == exp, 'Result dtype must come from numpy conversion rules'

    # Check if string representation works
    _ = repr(layout.scalar)

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
