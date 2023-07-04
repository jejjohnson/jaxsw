import typing as tp

import finitediffx as fdx
from jaxtyping import Array


def laplacian_matvec(
    u: Array,
    bc_fn: tp.Callable,
    **kwargs,
):
    return helmholtz_matvec(u=u, bc_fn=bc_fn, alpha=1.0, beta=0.0, **kwargs)


def helmholtz_matvec(
    u: Array,
    bc_fn: tp.Callable,
    alpha: float = 1.0,
    beta: float = 0.0,
    **kwargs,
) -> Array:
    u_lap = fdx.laplacian(u, **kwargs)
    u_helmholtz = alpha * u_lap - beta * u
    return bc_fn(u_helmholtz)
