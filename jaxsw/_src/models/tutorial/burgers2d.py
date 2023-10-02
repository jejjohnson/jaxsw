import typing as tp
import functools as ft
import jax
import jax.numpy as jnp
import equinox as eqx
import finitediffx as fdx
from jaxtyping import Float, Array
from jaxsw._src.domain.base_v2 import Domain
from jaxsw._src.models.pde import DynamicalSystem


def bc_fn(u: Array) -> Array:
    u = u.at[0, :].set(1.0)
    u = u.at[-1, :].set(1.0)
    u = u.at[:, 0].set(1.0)
    u = u.at[:, -1].set(1.0)
    return u


class State(tp.NamedTuple):
    u: Array
    v: Array


class Params(tp.NamedTuple):
    bc_fn: tp.Callable
    nu: float
    domain: Domain


class Burgers2D(DynamicalSystem):
    @staticmethod
    def equation_of_motion(t: float, state: State, params: Params):
        u, v = state.u, state.v
        u = params.bc_fn(u)
        v = params.bc_fn(v)

        # advection 1d
        rhs_u = rhs_pde(u=u, a=u, b=u, params=params)
        rhs_v = rhs_pde(u=v, a=u, b=v, params=params)

        state = State(u=rhs_u, v=rhs_v)

        return state


def rhs_pde(u: Array, a: Array, b: Array, params: Params) -> Array:
    # advection 2d
    du_dx = fdx.difference(
        u,
        step_size=params.domain.dx[0],
        axis=0,
        method="backward",
        derivative=1,
        accuracy=2,
    )
    du_dy = fdx.difference(
        u,
        step_size=params.domain.dx[1],
        axis=1,
        method="backward",
        derivative=1,
        accuracy=2,
    )

    rhs_adv = a * du_dx + b * du_dy

    # diffusion 2D
    d2u_dx2 = params.nu * fdx.difference(
        u,
        step_size=params.domain.dx[0],
        axis=0,
        method="central",
        derivative=2,
        accuracy=5,
    )

    d2u_dy2 = params.nu * fdx.difference(
        u,
        step_size=params.domain.dx[1],
        axis=1,
        method="central",
        derivative=2,
        accuracy=5,
    )

    rhs_diff = params.nu * (d2u_dx2 + d2u_dy2)

    return rhs_diff - rhs_adv


def init_u(domain):
    """Initial condition from grid"""
    u = jnp.ones(domain.Nx, dtype=jnp.float64)
    u = u.at[
        int(0.5 / domain.dx[0]) : int(1 / domain.dx[0] + 1),
        int(0.5 / domain.dx[1]) : int(1 / domain.dx[1] + 1),
    ].set(2.0)
    return u
