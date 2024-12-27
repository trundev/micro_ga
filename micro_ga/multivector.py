"""Geometric algebra multi-vector basic implementation"""
import numbers
from typing import Union, Callable
import numpy as np
import numpy.typing as npt

#
# Geometric algebra signature elements (0, +1, or -1)
# Use minimum if 8-bits
#
SigType = np.int8
#
# Multiplication/sign table (0, +1, or -1)
# Combined signature and euclidean-sign swap
#
MultTableType = np.int8
#
# Bit-mask to represent the basis-vectors, included in a multi-vector blade
# 16-bits allow max of 16 basis-vectors (2**16 == 65536 blades) - ought to be enough!
#
BasisBitmaskType = np.uint16

NDSigType = npt.NDArray[SigType]
NDMultTableType = npt.NDArray[MultTableType]
NDResultIdxType = npt.NDArray[np.int_]

# Multi-vector operation `other` argument type
OtherArg = Union['MVector', int, complex, numbers.Number]

class Cl:
    """Clifford algebra generator (similar to `clifford.Cl()`)"""
    #
    # Algebra signature, similar to `clifford.Layout.sig`
    #
    sig: NDSigType
    # Basis-vector dimensions, similar to `clifford.Layout.dims`
    dims: int
    # Multi-vector dimensions, similar to `clifford.Layout.gaDims`
    gaDims: int
    # Blade-name to multi-vector map
    blades: dict[str, 'MVector']
    # Individual blades, also include 'e1', 'e2', etc.
    scalar: 'MVector'
    I: 'MVector'
    #
    # Bit-masks for basis-vectors in each multi-vector blade
    #
    _blade_basis_masks: npt.NDArray[BasisBitmaskType]

    def __init__(self, pos_sig: int, neg_sig: int=0, zero_sig: int=0, *,
                 dtype: type|None=None) -> None:
        # Build signature
        self.sig = np.array([0] * zero_sig + [1] * pos_sig + [-1] * neg_sig,
                            dtype=SigType)
        self.dims = self.sig.size
        self.gaDims = 1<<self.dims      # pylint: disable=C0103 #HACK: match `clifford` naming
        #
        # Select bit-masks for all available blades
        #
        blade_masks = np.arange(1<<self.dims, dtype=BasisBitmaskType)
        # Sort by grades - number of set-bits, which is the number of basis-vectors
        # like: 000b; 001b, 010b, 100b; 011b, 101b, 110b; 111b
        # Then, by the smallest basis vector: `e14` (mask 9) is before `e23` (mask 6)
        argsort = np.lexsort(list(-(blade_masks & 1<<np.arange(self.dims)[:, np.newaxis]))[::-1]
                             + [np.bitwise_count(blade_masks)])
        self._blade_basis_masks = blade_masks[argsort]
        # Update blade names, add object attributes
        self._add_blades(dtype)

    def _add_blades(self, dtype: type|None) -> None:
        """Assign blade-names as the object attributes"""
        #
        # Select blade names
        #
        blade_names = np.where(
                self._blade_basis_masks[:, np.newaxis] & 1<<np.arange(self.dims),
                np.arange(self.dims) + 1, '').astype(object).sum(-1)
        self.blades = {}
        blade_val = np.empty(shape=self.gaDims, dtype=dtype)
        # Create 0 and 1 objects of type `dtype`
        zero, one = (0, 1) if dtype in (None, object) else (dtype(0), dtype(1))
        for idx, n in enumerate(blade_names):
            # Create multi-vector for this blade
            blade_val[...] = zero
            blade_val[idx] = one
            blade_mvec = MVector(self, blade_val)
            # Add to `blades` map, the scalar is ''
            name = 'e'+n if n else ''
            self.blades[name] = blade_mvec
            # Add it as object attribute, the scalar is 'scalar'
            if name == '':
                name = 'scalar'
            setattr(self, name, blade_mvec)
            # Extra pseudo-scalar property from the last blade
            if idx == self.gaDims - 1:
                setattr(self, 'I', blade_mvec)

    def __repr__(self) -> str:
        """String representation"""
        dtype_str = self.scalar.subtype.__name__
        return f'{type(self).__name__}(sig={self.sig.tolist()}, dtype={dtype_str})'

    def __eq__(self, other) -> bool:
        """Algebra comparison"""
        if self is other:   # The algebra-objects are often identical
            return True
        if not isinstance(other, type(self)):
            return False
        return np.array_equal(self.sig, other.sig)

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
