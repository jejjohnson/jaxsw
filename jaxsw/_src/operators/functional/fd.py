import typing as tp
import finitediffx as fdx
import jax
import jax.numpy as jnp
from jaxtyping import Array
import functools as ft

# TODO: Forward Different Init
# TODO: Backward Difference Init
# TODO: Mixed Difference Init
# TODO: Forward Difference


generate_finitediff_coeffs = fdx._src.utils.generate_finitediff_coeffs
generate_forward_offsets = fdx._src.utils._generate_forward_offsets
generate_central_offsets = fdx._src.utils._generate_central_offsets
generate_backward_offsets = fdx._src.utils._generate_backward_offsets


def difference_slicing(
    x: Array, axis: int, coeffs: tp.Sequence[int], offsets: tp.Iterable[int]
) -> Array:
    size = x.shape[axis]
    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)

    x = sum(
        coeff * sliced(ioffset - offsets[0], size + (ioffset - offsets[-1]))
        for ioffset, coeff in zip(offsets, coeffs)
    )
    return x


def finite_difference(
    x: Array,
    step_size: int,
    axis: int,
    derivative: int,
    coeffs: tp.Iterable[int],
    offsets: tp.Iterable[int],
    padding: tp.Optional[tp.Iterable[int]] = None,
    mode: str = "edge",
) -> Array:
    if padding is not None:
        x = jnp.pad(x, pad_width=padding, mode=mode)

    # slicing with coeffficients
    x = difference_slicing(x, axis=axis, coeffs=coeffs, offsets=offsets)

    # derivative factor
    x = x / (step_size**derivative)

    return x
