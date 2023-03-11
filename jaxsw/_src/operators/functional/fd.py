import typing as tp
import finitediffx as fdx
import jax
import jax.numpy as jnp
from jaxtyping import Array
import functools as ft
from jaxsw._src.operators.functional.padding import (
    generate_backward_padding,
    generate_central_padding,
    generate_forward_padding,
    _add_padding_dims,
)

# TODO: Forward Different Init
# TODO: Backward Difference Init
# TODO: Mixed Difference Init
# TODO: Forward Difference

COEFFS = tp.Tuple[tp.Tuple[int], tp.Tuple[int], tp.Tuple[int]]

generate_finitediff_coeffs = fdx._src.utils.generate_finitediff_coeffs
generate_forward_offsets = fdx._src.utils._generate_forward_offsets
generate_central_offsets = fdx._src.utils._generate_central_offsets
generate_backward_offsets = fdx._src.utils._generate_backward_offsets
check_dims = fdx._src.utils._check_and_return


def generate_central_diff(derivative: int, accuracy: int) -> COEFFS:
    offsets = generate_central_offsets(derivative=derivative, accuracy=accuracy)
    coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
    padding = generate_central_padding(derivative=derivative, accuracy=accuracy)
    return offsets, coeffs, padding


def generate_backward_diff(derivative: int, accuracy: int):
    offsets = generate_backward_offsets(derivative=derivative, accuracy=accuracy)
    coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
    padding = generate_backward_padding(derivative=derivative, accuracy=accuracy)
    return offsets, coeffs, padding


def generate_forward_diff(derivative: int, accuracy: int):
    offsets = generate_forward_offsets(derivative=derivative, accuracy=accuracy)
    coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
    padding = generate_forward_padding(derivative=derivative, accuracy=accuracy)
    return offsets, coeffs, padding


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
    print("x before padding:", x.shape)
    if padding is not None:
        print("Padding before:", padding)
        padding = _add_padding_dims(padding=padding, ndim=x.ndim, axis=axis)
        print("Padding after:", padding)
        x = jnp.pad(x, pad_width=padding, mode=mode)

    print("x after padding:", x.shape)
    # slicing with coeffficients
    x = difference_slicing(x, axis=axis, coeffs=coeffs, offsets=offsets)

    # derivative factor
    x = x / (step_size**derivative)
    print("x after fd:", x.shape)
    return x


def gradient(
    x: Array,
    step_size: int,
    axis: int,
    coeffs: tp.Iterable[int],
    offsets: tp.Iterable[int],
    padding: tp.Optional[tp.Iterable[int]] = None,
    mode: str = "edge",
) -> Array:
    pass
