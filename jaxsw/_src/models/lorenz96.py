from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from equinox import static_field
from jax.random import PRNGKeyArray
from jaxtyping import Array, PyTree

from .base import DynamicalSystem


class L96Params(eqx.Module):
    F: Array = static_field()


class L96State(NamedTuple):
    x: Array

    @classmethod
    def init_state(
        cls,
        ndim: int = 10,
        noise: float = 0.01,
        batchsize: int = 1,
        F: float = 8.0,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        if batchsize > 1:
            x0 = F * jnp.ones(shape=(batchsize, ndim))
            perturb = noise * jrandom.normal(key, shape=(batchsize,))
        else:
            x0 = F * jnp.ones(shape=(ndim,))
            perturb = noise * jrandom.normal(key, shape=())

        x0 = x0.at[..., 0].set(x0[..., 0] + perturb)

        return cls(x=x0)

    @staticmethod
    def init_state_and_params(
        ndim: int = 10,
        noise: float = 0.01,
        F: float = 8,
        batchsize: int = 1,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        return L96State.init_state(
            ndim=ndim, noise=noise, batchsize=batchsize, F=F, key=key
        ), L96Params(F=F)


class Lorenz96(DynamicalSystem):
    advection: bool

    def __init__(self, advection: bool = True):
        self.advection = advection

    def equation_of_motion(self, t: float, state: L96State, args: PyTree) -> PyTree:
        # compute RHS
        x = state.x
        F = args.F

        x_dot = rhs_lorenz_96(x=x, F=F, advection=self.advection)

        # update state
        state = eqx.tree_at(lambda x: x.x, state, x_dot)

        return state


def rhs_lorenz_96(x: Array, F: float = 8, advection: bool = True) -> Array:
    x_plus_1 = jnp.roll(x, -1)
    x_minus_2 = jnp.roll(x, 2)
    x_minus_1 = jnp.roll(x, 1)

    if advection:
        return (x_plus_1 - x_minus_2) * x_minus_1 - x + F
    else:
        return -x + F
