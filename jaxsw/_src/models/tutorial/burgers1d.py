import typing as tp
import functools as ft
import jax
import jax.numpy as jnp
import equinox as eqx
import finitediffx as fdx
from jaxtyping import Float, Array
from jaxsw._src.domain.base_v2 import Domain
from jaxsw._src.models.pde import DynamicalSystem


# class Params(eqx.Module):
#     bc_fn: tp.Callable
#     nu: float = eqx.field(static=True)
#     domain: Domain = eqx.field(static=True)


class Params(tp.NamedTuple):
    bc_fn: tp.Callable
    nu: float
    domain: Domain


class Burgers1D(DynamicalSystem):
    @staticmethod
    def equation_of_motion(t: float, u: Array, params: Params):
        u = params.bc_fn(u)

        # advection 1d
        rhs_adv = u * fdx.difference(
            u,
            step_size=params.domain.dx[0],
            axis=0,
            method="backward",
            derivative=1,
            accuracy=2,
        )

        # diffusion 1D
        rhs_diff = params.nu * fdx.difference(
            u,
            step_size=params.domain.dx[0],
            axis=0,
            method="central",
            derivative=2,
            accuracy=5,
        )

        return rhs_diff - rhs_adv


def phi(x: Array, t: float, nu: float) -> Array:
    denominator = 4 * nu * (t + 1)
    t1 = jnp.exp(-((x - 4 * t) ** 2) / denominator)
    t2 = jnp.exp(-((x - 4 * t - 2 * jnp.pi) ** 2) / denominator)
    return t1 + t2


def init_u(x: Array, t: float, nu: float) -> Array:
    c = phi(x, t, nu)

    u = -((2 * nu) / c) * dphi_dx(x, t, nu) + 4

    return u.squeeze()


init_u_batch: tp.Callable = jax.vmap(init_u, in_axes=(0, None, None))


def bc_fn(u: Float[Array, "D"]) -> Float[Array, "D"]:
    u = u.at[0].set(u[-1])
    return u


dphi_dx: tp.Callable = jax.grad(lambda x, t, nu: phi(x, t, nu).squeeze(), argnums=0)
