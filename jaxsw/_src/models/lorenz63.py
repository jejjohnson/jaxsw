from typing import NamedTuple, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from equinox import static_field
from jax.random import PRNGKeyArray
from jaxtyping import Array, PyTree

from .base import DynamicalSystem


class L63Params(eqx.Module):
    sigma: Array = static_field()
    rho: Array = static_field()
    beta: Array = static_field()


class L63State(NamedTuple):
    x: Array
    y: Array
    z: Array

    @classmethod
    def init_state(
        cls,
        noise: float = 0.01,
        batchsize: int = 1,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        msg = f"batchsize not >= 1, {batchsize}"
        assert batchsize >= 1, msg

        if batchsize > 1:
            x0, y0, z0 = jnp.array_split(jnp.ones((batchsize, 3)), 3, axis=-1)
            perturb = noise * jrandom.normal(key, shape=(batchsize, 1))
        else:
            x0, y0, z0 = jnp.array_split(jnp.ones((3,)), 3, axis=-1)
            perturb = noise * jrandom.normal(key, shape=())

        return cls(x=x0 + perturb, y=y0, z=z0)

    @staticmethod
    def init_state_and_params(
        noise: float = 0.01,
        batchsize: int = 1,
        sigma: float = 8,
        rho: float = 28,
        beta: float = 8.0 / 3.0,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        return L63State.init_state(
            noise=noise, batchsize=batchsize, key=key
        ), L63Params(sigma=sigma, rho=rho, beta=beta)

    @property
    def array(self):
        return jnp.stack([self.x, self.y, self.z], axis=1).squeeze()


class Lorenz63(DynamicalSystem):
    def equation_of_motion(
        self, t: float, state: L63State, args: Optional[PyTree] = None
    ) -> L63State:
        x, y, z = state.x, state.y, state.z
        sigma, rho, beta = args.sigma, args.rho, args.beta
        # import jax
        # jax.debug.print("x={x} | y={y} | z={z}", x=x, y=y, z=z)

        # compute RHS
        x_dot, y_dot, z_dot = rhs_lorenz_63(
            x=x, y=y, z=z, sigma=sigma, rho=rho, beta=beta
        )

        # update state
        state = eqx.tree_at(lambda u: u.x, state, x_dot)
        state = eqx.tree_at(lambda u: u.y, state, y_dot)
        state = eqx.tree_at(lambda u: u.z, state, z_dot)

        return state


def rhs_lorenz_63(
    x: Array,
    y: Array,
    z: Array,
    sigma: float = 10,
    rho: float = 28,
    beta: float = 2.667,
) -> Tuple[Array, Array, Array]:
    x_dot = sigma * (y - x)
    # y_dot = rho * x - y - x * z
    y_dot = x * (rho - z) - y
    # x[0]*(rho-x[2])-x[1]
    z_dot = x * y - beta * z

    return x_dot, y_dot, z_dot
