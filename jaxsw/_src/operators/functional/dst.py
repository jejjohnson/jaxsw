from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array


def dstI1D(x, norm="ortho"):
    """1D type-I discrete sine transform."""
    num_dims = x.ndim
    N = x.shape
    padding = ((0,0),) * (num_dims-1) + ((1,1),) 
    x = jnp.pad(x, pad_width=padding, mode="constant", constant_values=0.0)
    x = jnp.fft.irfft(-1j * x, axis=-1, norm=norm)
    x = jax.lax.slice_in_dim(x, 1, N[-1]+1, axis=-1)
    return x


def dstI2D(x, norm="ortho"):
    """2D type-I discrete sine transform."""
    x = dstI1D(x, norm=norm)
    x = jnp.transpose(x, axes=(-1,-2))
    x = dstI1D(x, norm=norm)
    x = jnp.transpose(x, axes=(-1,-2))
    return x


def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    num_dims = f.ndim
    padding = ((0,0),) * (num_dims-2) + ((1,1),(1,1)) 
    x = dstI2D(f) / operator_dst
    # print(x)
    
    return jnp.pad(dstI2D(x), pad_width=padding, mode="constant", constant_values=0.0)


def laplacian_dst(nx, ny, dx, dy, mean: bool = True) -> Array:
    if mean:
        dx = dy = jnp.mean(jnp.asarray([dx, dy]))

    x, y = jnp.meshgrid(
        jnp.arange(1, nx+1, dtype=dx.dtype),
        jnp.arange(1, ny+1, dtype=dx.dtype),
        indexing="ij"
    )

    return (
        2 * (jnp.cos(jnp.pi / nx * x) - 1) / dx**2
        + 2 * (jnp.cos(jnp.pi / ny * y) - 1) / dy**2
    )


def helmholtz_dst(
    nx: int,
    ny: int,
    dx: Union[float, Array],
    dy: Union[float, Array],
    alpha: float = 1.0,
    beta: float = 0.0,
    mean: bool = True,
) -> Array:
    laplace_op = laplacian_dst(nx=nx, ny=ny, dx=dx, dy=dy, mean=mean)
    return alpha * laplace_op - beta


def inverse_elliptical_dst_solver(
    q: Array,
    nx: int,
    ny: int,
    dx: Union[float, Array],
    dy: Union[float, Array],
    alpha: float = 1.0,
    beta: float = 0.0,
    mean: bool = True,
) -> Array:
    """Solves the Poisson Equation
    with Dirichlet Boundaries using the Discrete Sine
    transform
    """
    assert q.shape == (nx - 2, ny - 2)

    operator = helmholtz_dst(
        nx=nx, ny=ny, dx=dx, dy=dy, mean=mean, alpha=alpha, beta=beta
    )

    # print(q.shape, operator.shape)

    return inverse_elliptic_dst(q, operator)
