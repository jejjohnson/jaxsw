import jax.numpy as jnp
from jaxtyping import Array

from jaxsw._src.domain.base import Domain


def init_hat(domain: Domain) -> Array:
    dx, dy = domain.dx[0], domain.dx[0]
    nx, ny = domain.size[0], domain.size[1]

    u = jnp.ones((nx, ny))

    u = u.at[int(0.5 / dx) : int(1 / dx + 1), int(0.5 / dy) : int(1 / dy + 1)].set(2.0)

    return u


def fin_bump(x: Array) -> Array:
    if x <= 0 or x >= 1:
        return 0
    else:
        return 100 * jnp.exp(-1.0 / (x - jnp.power(x, 2.0)))


def init_smooth(domain: Domain) -> Array:
    dx, dy = domain.dx[0], domain.dx[0]
    nx, ny = domain.size[0], domain.size[1]

    u = jnp.ones((nx, ny))

    for ix in range(nx):
        for iy in range(ny):
            x = ix * dx
            y = iy * dy
            u = u.at[ix, iy].set(fin_bump(x / 1.5) * fin_bump(y / 1.5) + 1.0)

    return u
