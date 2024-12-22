"""Geometric algebra multi-vector basic implementation"""
import numbers
import numpy as np
import numpy.typing as npt
from typing import ForwardRef

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
# Bitmask to represent the basis-vectors, included in a multi-vector blade
# 16-bits allow max of 16 basis-vectors (2**16 == 65536 blades) - ought to be enought!
#
BasisBitmaskType = np.uint16

NDSigType = npt.NDArray[SigType]
NDMultTableType = npt.NDArray[MultTableType]
NDResultIdxType = npt.NDArray[np.int_]

T_MVector = ForwardRef('MVector')

class Cl:
    """Clifford algebra generator (similar to clifford.Cl())"""
    #
    # Algebra signature, similar to clifford.Layout.sig
    #
    sig: NDSigType
    # Basis-vector dimensions, similar to clifford.Layout.dims
    dims: int
    # Multi-vector dimensions, similar to clifford.Layout.gaDims
    gaDims: int
    # Blade-name to multi-vector map
    blades: dict[str, T_MVector]
    #
    # Bit-masks for basis-vectors in each multi-vector blade
    #
    _blade_basis_masks: npt.NDArray[BasisBitmaskType]
    # Individual blades, also include 'e1', 'e2', etc.
    scalar: T_MVector
    I: T_MVector

    def __init__(self, pos_sig: int, neg_sig: int=0, zero_sig: int=0, *, dtype: type|None=None) -> None:
        # Build signature
        self.sig = np.array([0] * zero_sig + [1] * pos_sig + [-1] * neg_sig,
                            dtype=SigType)
        self.dims = self.sig.size
        self.gaDims = 1<<self.dims
        #
        # Select arange all available blades
        #
        blade_masks = np.arange(1<<self.dims, dtype=BasisBitmaskType)
        # Sort by grades - number of set-bits, which is the number of basis-vectors
        # like: 000b; 001b, 010b, 100b; 011b, 101b, 110b; 111b
        argsort = np.argsort(np.bitwise_count(blade_masks), stable=True)
        blade_masks = blade_masks[argsort]
        self._blade_basis_masks = blade_masks[argsort]
        # Update blade names, add object attributes
        self._add_blades(dtype)

    def _add_blades(self, dtype: np.dtype|None) -> None:
        """Assign blade-names as the object attributes"""
        #
        # Select blade names
        #
        blade_names = np.where(
                self._blade_basis_masks[:, np.newaxis] & 1<<np.arange(self.dims),
                np.arange(self.dims) + 1, '').astype(object).sum(-1)
        self.blades = {}
        blade_val = np.empty(shape=self.gaDims, dtype=dtype)
        for idx, n in enumerate(blade_names):
            # Create multi-vector for this blade
            blade_val[...] = 0
            blade_val[idx] = 1
            blade_mvec = MVector(self, blade_val)
            # Add to `blades` map, the scalar is ''
            name = 'e'+n if n else n
            self.blades['e'+n if n else n] = blade_mvec
            # Add it as object attribute, the scalar is 'scalar'
            if name == '':
                name = 'scalar'
            setattr(self, name, blade_mvec)
            # Extra pseudo-scalar property from the last blade
            if idx == self.gaDims - 1:
                setattr(self, 'I', blade_mvec)

class MVector:
    """Multi-vector representation"""
    layout: Cl
    value: npt.NDArray

    def __init__(self, layout: Cl, value: npt.NDArray) -> None:
        assert layout.gaDims == value.size, 'The value and layout signature must match'
        self.layout = layout
        self.value = value.copy()

    def _to_string(self, *, tol: float=0, round: int|None=None) -> str:
        """String representation, strips near-zero blades"""
        vals = self.value
        if round is not None:
            vals = np.round(vals, round)
        nz_mask = ~np.isclose(vals, 0, atol=tol)
        if not nz_mask.any():
            return '0'
        vals = vals[nz_mask]
        names = np.array(tuple(self.layout.blades.keys()))[nz_mask]
        # Combine individual blades
        return ' '.join(f'{v:+}*{n}' if n else f'{v:+}' for v, n in zip(vals, names))

    def __str__(self) -> str:
        return self._to_string(round=np.get_printoptions()['precision'])

    def __repr__(self) -> str:
        return self._to_string()

    def _get_other_value(self, other: T_MVector | numbers.Number) -> T_MVector:
        """Convert values of an operation argument to match ours"""
        # Check if it is scalar
        if isinstance(other, numbers.Number):
            return self.layout.scalar.value * other

        assert isinstance(other, MVector), 'Must be MVector or Number'
        if self.layout is other.layout:
            return other.value  # The layout is identical
        # Check signature
        np.testing.assert_array_equal(self.layout.sig, other.layout.sig, 'Multi-vector signatures must match')
        return other.value

    def __eq__(self, other) -> bool:
        """Comparison"""
        value = self._get_other_value(other)
        return (self.value == value).all()

    def __add__(self, other: T_MVector) -> T_MVector:
        """Left-side addition"""
        value = self._get_other_value(other)
        return MVector(self.layout, self.value + value)

    __radd__ = __add__

    def __neg__(self) -> T_MVector:
        """Negation"""
        return MVector(self.layout, -self.value)

    def __sub__(self, other: T_MVector) -> T_MVector:
        """Left-side subtraction"""
        return self.__add__(-other)

    def __rsub__(self, other: T_MVector) -> T_MVector:
        """Right-side subtraction"""
        return self.__neg__().__add__(other)
