from typing import Union

import jax.numpy as jnp
from jaxtyping import Array


def dstI1D(x, norm="ortho"):
    """1D type-I discrete sine transform."""
    return jnp.fft.irfft(-1j * jnp.pad(x, (1, 1)), axis=-1, norm=norm)[
        1 : x.shape[0] + 1, 1 : x.shape[1] + 1
    ]


def dstI2D(x, norm="ortho"):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).T, norm=norm).T


def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    return dstI2D(dstI2D(f) / operator_dst)


def laplacian_dist(nx, ny, dx, dy, mean: bool = True) -> Array:
    if mean:
        dx = dy = jnp.mean(jnp.asarray([dx, dy]))

    x, y = jnp.meshgrid(
        jnp.arange(1, nx - 1, dtype=dx.dtype),
        jnp.arange(1, ny - 1, dtype=dx.dtype),
    )

    return (
        2 * (jnp.cos(jnp.pi / (nx - 1) * x) - 1) / dx**2
        + 2 * (jnp.cos(jnp.pi / (ny - 1) * y) - 1) / dy**2
    )


def helmholtz_dist(
    nx: int,
    ny: int,
    dx: Union[float, Array],
    dy: Union[float, Array],
    alpha: float = 1.0,
    beta: float = 0.0,
    mean: bool = True,
) -> Array:
    laplace_op = laplacian_dist(nx=nx, ny=ny, dx=dx, dy=dy, mean=mean)
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

    operator = helmholtz_dist(
        nx=nx, ny=ny, dx=dx, dy=dy, mean=mean, alpha=alpha, beta=beta
    ).T

    # print(q.shape, operator.shape)

    return inverse_elliptic_dst(q, operator)
