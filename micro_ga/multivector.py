"""Geometric algebra multi-vector basic implementation"""
import numbers
from typing import Union, Callable, ForwardRef, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
# Avoid circular imports, while type checking works
if TYPE_CHECKING:
    from .layout import Cl  # pragma: no cover # pylint: disable=C0401
else:
    Cl = ForwardRef('Cl')

# Multi-vector operation `other` argument type
OtherArg = Union['MVector', int, complex, numbers.Number]

class MVector:
    """Multi-vector representation"""
    layout: Cl
    value: npt.NDArray

    def __init__(self, layout: Cl, value: npt.NDArray) -> None:
        assert layout.gaDims == value.size, 'The value and layout signature must match'
        self.layout = layout
        self.value = value.copy()

    @property
    def subtype(self) -> type:
        """Type of underlying objects"""
        # Start with native `numpy` data-type
        subtype = self.value.dtype.type
        if subtype == np.object_:
            # Type of underlying object
            subtype = type(self.value.item(0))
        return subtype

    def _to_string(self, str_fn: Callable, *, tol: float=0, mult_sym='*') -> str:
        """String representation, strips near-zero blades"""
        vals = self.value
        nz_mask = np.abs(vals) > tol
        if not nz_mask.any():
            return '0'
        vals = vals[nz_mask]

        # Start with strings to join blades (`object` is to allow `+=`)
        el_strs = np.full(vals.shape, ' + ', dtype=object)
        el_strs[0] = ''
        if issubclass(self.subtype, (int, float, np.integer, np.floating)):
            # Only primitive scalar types are joined using their sign
            el_strs[1:][vals[1:] < 0] = ' - '
            vals[1:] = abs(vals[1:])

        # Convert each coefficient to string using `str_fn`
        el_strs += np.vectorize(str_fn, otypes=[str])(vals)

        # Add blade-names
        blade_strs = np.array(tuple(self.layout.blades.keys()))[nz_mask]
        el_strs += np.where(blade_strs, mult_sym, '') + blade_strs
        return el_strs.sum()

    def __str__(self) -> str:
        return self._to_string(str)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._to_string(repr)})'

    def __round__(self, ndigits: int=0) -> 'MVector':
        """Implement built-in round(), esp. for `dtype=object`"""
        vals = self.value
        #HACK: `numpy.round()` crashes if `dtype=object` and underlying object has no `rint` method
        # But, `round()` does NOT work with complex: "type complex doesn't define __round__ method"
        if vals.dtype == object:
            vals = np.vectorize(round, otypes=[object])(vals, ndigits=ndigits)
        else:
            vals = vals.round(decimals=ndigits)
        return MVector(self.layout, vals)

    def _get_other_value(self, other: OtherArg) -> npt.NDArray:
        """Convert values of an operation argument to match ours"""
        # Check if it is scalar
        if isinstance(other, numbers.Number):
            return self.layout.scalar.value * other
        if not isinstance(other, MVector):
            return NotImplemented
        if self.layout != other.layout:
            return NotImplemented
        return other.value

    def __eq__(self, other) -> bool:
        """Multi-vector comparison"""
        value = self._get_other_value(other)
        if value is NotImplemented:
            return NotImplemented
        return (self.value == value).all()

    def __add__(self, other: OtherArg) -> 'MVector':
        """Left-side addition"""
        value = self._get_other_value(other)
        if value is NotImplemented:
            return NotImplemented
        return MVector(self.layout, self.value + value)

    __radd__ = __add__

    def __neg__(self) -> 'MVector':
        """Negation"""
        return MVector(self.layout, -self.value)

    def __sub__(self, other: OtherArg) -> 'MVector':
        """Left-side subtraction"""
        return self.__add__(-other)

    def __rsub__(self, other: OtherArg) -> 'MVector':
        """Right-side subtraction"""
        return self.__neg__().__add__(other)
