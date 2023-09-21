import typing as tp

import finitediffx as fdx
import jax.numpy as jnp
from jaxtyping import Array

from jaxsw._src.operators.functional.utils import identity


def laplacian_matvec(
    u: Array,
    alpha: float = 1.0,
    bc_fn: tp.Callable = identity,
    **kwargs,
):
    return helmholtz_matvec(u=u, bc_fn=bc_fn, alpha=alpha, beta=0.0, **kwargs)


def helmholtz_matvec(
    u: Array,
    alpha: float = 1.0,
    beta: float = 0.0,
    bc_fn: tp.Callable = identity,
    **kwargs,
) -> Array:
    u_lap = fdx.laplacian(u, **kwargs)
    u_helmholtz = alpha * u_lap - beta * u
    return bc_fn(u_helmholtz)


def laplacian_dst(nx, ny, dx, dy, mean: bool = True, dtype=jnp.float32) -> Array:
    if mean:
        dx = dy = jnp.mean(jnp.asarray([dx, dy]))

    x, y = jnp.meshgrid(
        jnp.arange(1, nx - 1, dtype=dtype),
        jnp.arange(1, ny - 1, dtype=dtype),
        indexing="ij",
    )

    return (
        2 * (jnp.cos(jnp.pi / (nx - 1) * x) - 1) / dx**2
        + 2 * (jnp.cos(jnp.pi / (ny - 1) * y) - 1) / dy**2
    )


def helmholtz_dst(
    nx: int,
    ny: int,
    dx: tp.Union[float, Array],
    dy: tp.Union[float, Array],
    alpha: float = 1.0,
    beta: float = 0.0,
    mean: bool = True,
) -> Array:
    laplace_op = laplacian_dst(nx=nx, ny=ny, dx=dx, dy=dy, mean=mean)
    return alpha * laplace_op - beta


def dstI1D(x, norm="ortho"):
    """1D type-I discrete sine transform."""
    return jnp.fft.irfft(-1j * jnp.pad(x, (1, 1)), axis=-1, norm=norm)[
        ..., 1 : x.shape[0] + 1, 1 : x.shape[1] + 1
    ]


def dstI2D(x, norm="ortho"):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).T, norm=norm).T


def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    return dstI2D(dstI2D(f) / operator_dst)
