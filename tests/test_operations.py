"""Test arithmetic operations"""
import numpy as np
import pytest
import micro_ga
from . import rng, pos_sig, layout, operation, dtype, mvector_gen   # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture
def neg_sig():
    """Skip tests with basis-vectors of negative signature"""
    return 0

@pytest.fixture
def zero_sig():
    """Skip tests with basis-vectors of zero signature"""
    return 0

def test_operation(layout, operation, mvector_gen):
    """Compare results from arithmetic operations with scalar vs. python integer"""
    # Iterate over some picked values
    for mv_val in mvector_gen(layout):
        assert operation(layout.scalar, mv_val) == operation(1, mv_val)
        assert operation(mv_val, layout.scalar) == operation(mv_val, 1)
    # Unsupported operand type (right and left side)
    with pytest.raises(TypeError):
        _ = operation(layout.scalar, None)
    with pytest.raises(TypeError):
        _ = operation(None, layout.scalar)

def test_astype(dtype):
    """Check conversion of internal `numpy` array `dtype`"""
    layout = micro_ga.Cl(3)
    # Check type of individual blades before and after type conversion
    exp_type = int if dtype is object else dtype
    for blade in layout.blades.values():
        assert isinstance(blade.value[0], np.integer), 'Internal blade numpy array must use int'
        blade = blade.astype(dtype)
        # Note: `dtype('O') == object`
        assert blade.value.dtype == dtype, 'Internal numpy array must use requested dtype'
        assert blade.subtype == exp_type, 'Reported subtype must match'
    # Check type of individual values from the scalar-blade
    scalar = layout.scalar.astype(dtype)
    for v in scalar.value:
        # pylint: disable=C0123     # Here we expect exactly the same type
        assert type(v) is exp_type, 'Individual values must be of requested type'

def test_operation_dtype(operation, dtype):
    """Check the internal `numpy` array `dtype` of operation result"""
    layout = micro_ga.Cl(3)
    mv = operation(layout.mvector(12345).astype(dtype), layout.scalar)
    exp_dt = np.result_type(dtype)
    assert mv.value.dtype is exp_dt, 'Result dtype must match requested type'
    # Check type of individual values
    exp_t = int if dtype is object else dtype
    for v in mv.value:
        # pylint: disable=C0123     # Here we expect exactly the same type
        assert type(v) is exp_t, 'Individual values of result must be of requested type'

def test_unbounded_int():
    """Test python unbounded `int` operation"""
    layout = micro_ga.Cl(2)
    # When converted to `object`, `numpy` falls-back to original python unbounded operation
    scalar = layout.scalar.astype(object)
    mv = scalar + (1<<100)
    assert (mv.value[0] - (1<<100)) == 1

    # Default type promotes to `numpy.int64` (64-bit only)
    layout = micro_ga.Cl(2)
    with pytest.raises(OverflowError):
        mv = layout.scalar + (1<<100)
    mv = layout.scalar + (1<<40)

def test_round(dtype):
    """Test `round` operation"""
    layout = micro_ga.Cl(2)
    if dtype is object:
        exp_type = int          # Default `micro_ga` type
    else:
        exp_type = dtype
    # Pick a convenient number, which after rounding has finite binary representation
    val = exp_type(1.2456) + layout.I
    val = round(val, 2)
    assert val == exp_type(1.25) + layout.I
