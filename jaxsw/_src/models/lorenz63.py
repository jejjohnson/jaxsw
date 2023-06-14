from typing import NamedTuple, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from equinox import static_field
from jax.random import PRNGKeyArray
from jaxtyping import Array, PyTree

from .base import DynamicalSystem


class L63Params(eqx.Module):
    s: Array = static_field()
    r: Array = static_field()
    b: Array = static_field()


class L63State(NamedTuple):
    x: Array
    y: Array
    z: Array

    @classmethod
    def init_state(
        cls,
        noise: float = 0.01,
        s: float = 8,
        r: float = 28,
        b: float = 8.0 / 3.0,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        x0, y0, z0 = jnp.ones((3,))

        perturb = noise * jrandom.normal(key, shape=())

        return cls(x=x0 + perturb, y=y0, z=z0), L63Params(s=s, r=r, b=b)

    @classmethod
    def init_state_batch(
        cls,
        batchsize: int = 10,
        noise: float = 0.01,
        s: float = 8,
        r: float = 28,
        b: float = 8.0 / 3.0,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        x0, y0, z0 = jnp.array_split(jnp.ones((batchsize, 3)), 3, axis=-1)

        perturb = noise * jrandom.normal(key, shape=(batchsize, 1))

        return cls(x=x0 + perturb, y=y0, z=z0), L63Params(s=s, r=r, b=b)


class Lorenz63(DynamicalSystem):
    def equation_of_motion(
        self, t: float, state: L63State, args: Optional[PyTree] = None
    ) -> L63State:
        # compute RHS
        x, y, z = state.x, state.y, state.z
        s, r, b = args.s, args.r, args.b

        x_dot, y_dot, z_dot = rhs_lorenz_63(x=x, y=y, z=z, s=s, r=r, b=b)

        # update state
        state = eqx.tree_at(lambda u: u.x, state, x_dot)
        state = eqx.tree_at(lambda u: u.y, state, y_dot)
        state = eqx.tree_at(lambda u: u.z, state, z_dot)

        return state


def rhs_lorenz_63(
    x: Array,
    y: Array,
    z: Array,
    s: float = 10,
    r: float = 28,
    b: float = 2.667,
) -> Tuple[Array, Array, Array]:
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot
