from typing import NamedTuple, Optional, Tuple

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
from equinox import static_field
from jax.random import PRNGKeyArray
from jaxtyping import Array, PyTree

from .base import DynamicalSystem


class L63State(NamedTuple):
    x: Array
    y: Array
    z: Array

    @classmethod
    def init_state(
        cls,
        noise: float = 0.01,
        key: jrandom.PRNGKey = jrandom.PRNGKey(123),
    ):
        x0, y0, z0 = jnp.ones((3,))

        perturb = noise * jrandom.normal(key, shape=())

        return cls(x=x0 + perturb, y=y0, z=z0)

    @classmethod
    def init_state_batch(
        cls,
        batchsize: int = 10,
        noise: float = 0.01,
        key: jrandom.PRNGKey = jrandom.PRNGKey(123),
    ):
        x0, y0, z0 = jnp.array_split(jnp.ones((batchsize, 3)), 3, axis=-1)

        perturb = noise * jrandom.normal(key, shape=(batchsize, 1))

        return cls(x=x0 + perturb, y=y0, z=z0)

    @staticmethod
    def update_state(state, **kwargs):
        return L63State(
            x=kwargs.get("x", state.x),
            y=kwargs.get("y", state.y),
            z=kwargs.get("z", state.z),
        )


class Lorenz63(DynamicalSystem):
    # observe_every: int = static_field()
    s: float = static_field()
    r: float = static_field()
    b: float = static_field()
    key: jrandom.PRNGKey = jrandom.PRNGKey(0)

    def __init__(
        self,
        tmin: float,
        tmax: float,
        s: float = 8,
        r: float = 28,
        b: float = 8.0 / 3.0,
        key: PRNGKeyArray = jrandom.PRNGKey(0),
        solver: dfx.AbstractSolver = dfx.Euler(),
        stepsize_controller: dfx.PIDController = dfx.ConstantStepSize(),
    ):
        super().__init__(
            tmin=tmin,
            tmax=tmax,
            solver=solver,
            stepsize_controller=stepsize_controller,
        )
        self.s = s
        self.r = r
        self.b = b
        # self.observe_every = observe_every
        self.key = key

    def equation_of_motion(
        self, t: float, state: L63State, args: Optional[PyTree] = None
    ) -> L63State:
        # compute RHS
        x_dot, y_dot, z_dot = rhs_lorenz_63(state=state, s=self.s, r=self.r, b=self.b)

        # update state
        state = state.update_state(state, x=x_dot, y=y_dot, z=z_dot)

        return state

    # def observe(self, x: Float[Array, " dim"], n_steps: int):
    #     t = jnp.asarray([n * self.dt for n in range(n_steps)])
    #     return x[:: self.observe_every], t[:: self.observe_every]


def rhs_lorenz_63(
    state: Tuple[Array, Array, Array],
    s: float = 10,
    r: float = 28,
    b: float = 2.667,
) -> Tuple[Array, Array, Array]:
    x, y, z = state
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot
