import typing as tp
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from jaxtyping import Array
import finitediffx as fdx
from jaxsw._src.operators.functional.advection import plusminus


def fd_difference(
    u: Array,
    step_size: Array = 1,
    axis: int = 0,
    method: str = "right",
    accuracy: int = 1,
    derivative: int = 1,
    a: Array = None,
    **kwargs,
) -> Array:
    if method in ["forward", "right"]:
        u = fdx.difference(
            u,
            step_size=step_size,
            axis=axis,
            method="forward",
            derivative=derivative,
            accuracy=accuracy,
        )
        return u
    elif method in ["backward", "left"]:
        u = fdx.difference(
            u,
            step_size=step_size,
            axis=axis,
            method="backward",
            derivative=derivative,
            accuracy=accuracy,
        )
        return u
    elif method in ["central", "inner"]:
        u = fdx.difference(
            u,
            step_size=step_size,
            axis=axis,
            method="central",
            derivative=derivative,
            accuracy=accuracy,
        )
        return u
    elif method == "upwind":
        # get plus-minus
        u_plus, u_minus = plusminus(a if a is not None else u)

        du_b = difference(
            u, step_size=step_size, axis=axis, derivative=derivative, method="backward"
        )
        du_f = difference(
            u, step_size=step_size, axis=axis, derivative=derivative, method="forward"
        )

        return du_f * u_minus + du_b * u_plus
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be: 'forward', 'backward', 'central'."
        raise ValueError(msg)


def ps_difference():
    raise NotImplementedError()


def fv_difference(
    u: Array,
    step_size: Array = 1,
    axis: int = 0,
    method: str = "right",
    derivative: int = 1,
    a: Array = None,
    **kwargs,
) -> Array:
    if method in ["forward", "right", "backward", "left", "central", "inner"]:
        kernel = fd_kernel_init(
            step_size=step_size,
            ndims=u.ndim,
            axis=axis,
            derivative=derivative,
            accuracy=1,
            method=method,
        )

        return convolution(x=u, kernel=kernel)

    elif method == "upwind":
        # get plus-minus
        u_plus, u_minus = plusminus(a if a is not None else u)

        du_b = fv_difference(
            u_plus * u,
            step_size=step_size,
            axis=axis,
            derivative=derivative,
            accuracy=1,
            method="backward",
        )
        du_f = fv_difference(
            u_minus * u,
            step_size=step_size,
            axis=axis,
            derivative=derivative,
            accuracy=1,
            method="forward",
        )

        return du_f + du_b

    elif method in ["lax-wendroff"]:
        raise NotImplementedError()
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be: 'forward', 'backward', 'central'."
        raise ValueError(msg)


def fd_kernel_init(
    step_size: Array = 1,
    ndims: int = 1,
    axis: int = 0,
    derivative: int = 1,
    accuracy: int = 1,
    method: str = "forward",
) -> Array:
    # generate FD offsets
    offsets = generate_fd_offsets(
        derivative=derivative, accuracy=accuracy, method=method
    )

    # generate FD coefficients
    coeffs = fdx._src.utils.generate_finitediff_coeffs(
        offsets=offsets, derivative=derivative
    )

    # add kernel dims
    kernel = jnp.asarray(coeffs, dtype=jnp.float32)
    kernel = _add_kernel_dims(kernel, ndim=ndims, axis=axis)

    # add dx scaling
    kernel = kernel / step_size

    return kernel


def convolution(
    x: Array,
    kernel: Array,
    pad: tp.Optional[tp.Callable] = None,
    reverse: bool = True,
    **kwargs,
) -> Array:
    if pad is not None:
        x = pad(x)

    # flip (convolutions are the opposite)
    if reverse:
        kernel = jnp.flip(kernel, axis=tuple(range(kernel.ndim)))

    # do convolutions
    return jsp.signal.convolve(x, kernel, mode="valid")


def _add_kernel_dims(kernel: Array, ndim: int, axis: int) -> Array:
    if ndim > 1:
        for _ in range(ndim - 1):
            kernel = jnp.expand_dims(kernel, axis=0)

        # move kernel to correct axis
        kernel = jnp.moveaxis(kernel, -1, axis)

    return kernel


check_dims = fdx._src.utils._check_and_return


def generate_fd_offsets(derivative: int, accuracy: int, method: str):
    if method == "central":
        return fdx._src.utils._generate_central_offsets(
            derivative=derivative, accuracy=accuracy + 1
        )
    if method == "forward":
        return fdx._src.utils._generate_forward_offsets(
            derivative=derivative, accuracy=accuracy
        )
    if method == "backward":
        return fdx._src.utils._generate_backward_offsets(
            derivative=derivative, accuracy=accuracy
        )
    else:
        raise ValueError(f"Unrecognized method: {method}")
