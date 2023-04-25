import typing as tp
from jaxtyping import Array
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange


def fd_kernel_init(dims: tp.Tuple[int], coeffs: tp.Tuple[int], axis: int = 0) -> Array:
    # add kernel dims
    kernel = jnp.asarray(coeffs, dtype=jnp.float32)

    kernel = _add_kernel_dims(kernel, ndim=len(dims), axis=axis)

    # add dx
    kernel = kernel / dims[axis]

    # flip (convolutions are the opposite)
    kernel = jnp.flip(kernel, axis=tuple(range(kernel.ndim)))

    return kernel


def fd_convolution(
    x: Array, kernel: Array, pad: tp.Optional[tp.Tuple[int]] = None, mode: str = "edge"
) -> Array:
    if pad is not None:
        x = jnp.pad(x, pad_width=pad, mode=mode)

    return jsp.signal.convolve(x, kernel, mode="valid")
    # return jax.lax.conv(x, kernel, window_strides=(1,), padding="VALID")


def _add_kernel_dims(kernel: Array, ndim: int, axis: int) -> Array:
    if ndim > 1:
        for _ in range(ndim - 1):
            kernel = rearrange(kernel, "... -> ... 1")

        # move kernel to correct axis
        kernel = jnp.moveaxis(kernel, -1, axis)

    return kernel
