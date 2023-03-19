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


def difference(
    array,
    *,
    axis: int = 0,
    accuracy: int = 1,
    step_size: jnp.ndarray = 1,
    derivative: int = 1,
    method: str = "forward",
):
    size = array.shape[axis]
    center_offsets = fdx._src.utils._generate_central_offsets(derivative, accuracy + 1)
    center_coeffs = fdx._src.utils.generate_finitediff_coeffs(
        center_offsets, derivative
    )

    left_offsets = fdx._src.utils._generate_forward_offsets(derivative, accuracy)
    left_coeffs = fdx._src.utils.generate_finitediff_coeffs(left_offsets, derivative)

    right_offsets = fdx._src.utils._generate_backward_offsets(derivative, accuracy)
    right_coeffs = fdx._src.utils.generate_finitediff_coeffs(right_offsets, derivative)

    # print(size, len(center_offsets), size >= len(center_offsets))

    if method == "central":
        return _central_difference(
            array,
            axis=axis,
            left_coeffs=left_coeffs,
            center_coeffs=center_coeffs,
            right_coeffs=right_coeffs,
            left_offsets=left_offsets,
            center_offsets=center_offsets,
            right_offsets=right_offsets,
        ) / (step_size**derivative)

    elif method == "forward":
        return _forward_difference(
            array,
            axis=axis,
            left_coeffs=left_coeffs,
            right_coeffs=right_coeffs,
            left_offsets=left_offsets,
            right_offsets=right_offsets,
        ) / (step_size**derivative)

    elif method == "backward":
        return _backward_difference(
            array,
            axis=axis,
            left_coeffs=left_coeffs,
            right_coeffs=right_coeffs,
            left_offsets=left_offsets,
            right_offsets=right_offsets,
        ) / (step_size**derivative)
    else:
        raise ValueError(f"Unrecognized method: {method}!")


def gradient(
    array: Array,
    *,
    accuracy: int = 1,
    step_size: jnp.ndarray = 1,
    method: str = "forward",
) -> Array:
    accuracy = fdx._src.utils._check_and_return(accuracy, array.ndim, "accuracy")
    step_size = fdx._src.utils._check_and_return(step_size, array.ndim, "step_size")

    return jnp.stack(
        [
            difference(
                array,
                accuracy=acc,
                step_size=step,
                derivative=1,
                axis=axis,
                method=method,
            )
            for axis, (acc, step) in enumerate(zip(accuracy, step_size))
        ],
        axis=0,
    )


def jacobian(array, *, accuracy, step_size, method):
    accuracy = fdx._src.utils._check_and_return(accuracy, array.ndim - 1, "accuracy")
    step_size = fdx._src.utils._check_and_return(step_size, array.ndim - 1, "step_size")

    return jnp.stack(
        [
            gradient(xi, accuracy=accuracy, step_size=step_size, method=method)
            for xi in array
        ],
        axis=0,
    )


def divergence(array, *, accuracy, step_size, method, keepdims=True):
    accuracy = fdx._src.utils._check_and_return(accuracy, array.ndim - 1, "accuracy")
    step_size = fdx._src.utils._check_and_return(step_size, array.ndim - 1, "step_size")

    result = sum(
        difference(
            array[axis],
            accuracy=acc,
            step_size=step,
            method=method,
            derivative=1,
            axis=axis,
        )
        for axis, (acc, step) in enumerate(zip(accuracy, step_size))
    )

    if keepdims:
        return jnp.expand_dims(result, axis=0)
    return result


def laplacian(array, *, accuracy, step_size, method):
    accuracy = fdx._src.utils._check_and_return(accuracy, array.ndim, "accuracy")
    step_size = fdx._src.utils._check_and_return(step_size, array.ndim, "step_size")

    return sum(
        difference(
            array, accuracy=acc, step_size=step, derivative=2, axis=axis, method=method
        )
        for axis, (acc, step) in enumerate(zip(accuracy, step_size))
    )


def _central_difference(
    x,
    axis: int,
    left_coeffs,
    center_coeffs,
    right_coeffs,
    left_offsets,
    center_offsets,
    right_offsets,
):
    size = x.shape[axis]
    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)

    # use central difference for interior points
    left_x = sum(
        coeff * sliced(offset, offset - center_offsets[0])
        for offset, coeff in zip(left_offsets, left_coeffs)
    )

    right_x = sum(
        coeff * sliced(size + (offset - center_offsets[-1]), size + (offset))
        for offset, coeff in zip(right_offsets, right_coeffs)
    )

    center_x = sum(
        coeff * sliced(offset - center_offsets[0], size + (offset - center_offsets[-1]))
        for offset, coeff in zip(center_offsets, center_coeffs)
    )

    return jnp.concatenate([left_x, center_x, right_x], axis=axis)


def _backward_difference(
    x,
    *,
    axis,
    left_coeffs,
    right_coeffs,
    left_offsets,
    right_offsets,
):
    size = x.shape[axis]
    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)
    # use central difference for interior points
    left_x = sum(
        coeff * sliced(offset, offset - right_offsets[0])
        for offset, coeff in zip(left_offsets, left_coeffs)
    )

    right_x = sum(
        coeff * sliced(offset - right_offsets[0], size + (offset - right_offsets[-1]))
        for offset, coeff in zip(right_offsets, right_coeffs)
    )

    return jnp.concatenate([left_x, right_x], axis=axis)


def _forward_difference(
    x,
    *,
    axis,
    left_coeffs,
    right_coeffs,
    left_offsets,
    right_offsets,
):
    size = x.shape[axis]
    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)

    # use central difference for interior points
    left_x = sum(
        coeff * sliced(offset - left_offsets[0], size + (offset - left_offsets[-1]))
        for offset, coeff in zip(left_offsets, left_coeffs)
    )

    right_x = sum(
        coeff * sliced(size + (offset - left_offsets[-1]), size + (offset))
        for offset, coeff in zip(right_offsets, right_coeffs)
    )

    return jnp.concatenate([left_x, right_x], axis=axis)
