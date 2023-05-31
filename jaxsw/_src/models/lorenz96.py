from typing import NamedTuple, Optional

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from equinox import static_field
from jax.random import PRNGKeyArray
from jaxtyping import Array, PyTree

from .base import DynamicalSystem


class L96Params(eqx.Module):
    F: float = static_field()


class L96State(NamedTuple):
    x: Array

    @classmethod
    def init_state(
        cls,
        ndim: int = 10,
        F: float = 8,
        noise: float = 0.01,
        key: jrandom.PRNGKey = jrandom.PRNGKey(123),
    ):
        x0 = F * jnp.ones(shape=(ndim,))

        perturb = noise * jrandom.normal(key, shape=())

        x0 = x0.at[0].set(x0[0] + perturb)

        return cls(x=x0), L96Params(F=F)

    @classmethod
    def init_state_batch(
        cls,
        ndim: int = 10,
        batchsize: int = 5,
        F: float = 8,
        noise: float = 0.01,
        key: jrandom.PRNGKey = jrandom.PRNGKey(123),
    ):
        x0 = F * jnp.ones(shape=(batchsize, ndim))

        perturb = noise * jrandom.normal(key, shape=(batchsize,))

        x0 = x0.at[..., 0].set(x0[..., 0] + perturb)

        return cls(x=x0), L96Params(F=F)


class Lorenz96(DynamicalSystem):
    # observe_every: int = static_field()
    key: jrandom.PRNGKey = jrandom.PRNGKey(0)

    def __init__(
        self,
        tmin: float,
        tmax: float,
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
        # self.observe_every = observe_every
        self.key = key

    def equation_of_motion(
        self, t: float, state: L96State, args: Optional[PyTree] = 8
    ) -> L96State:
        # compute RHS
        x = state.x
        F = args.F

        x_dot = rhs_lorenz_96(x=x, F=F)

        # update state
        state = eqx.tree_at(lambda x: x.x, state, x_dot)

        return state


# @pytc.treeclass
# class Lorenz96(DynamicalSystem):
#     grid_size: int = pytc.field(nondiff=True)
#     observe_every: int = pytc.field(nondiff=True)
#     F: float = pytc.field(nondiff=True)
#     key: jrandom.PRNGKey = jrandom.PRNGKey(0)

#     def __init__(
#         self,
#         dt: Float,
#         grid_size: Int,
#         observe_every: Int = 1,
#         F: Int = 8,
#         key: PyTree = jrandom.PRNGKey(0),
#     ):
#         self.dt = dt
#         self.grid_size = grid_size
#         self.observe_every = observe_every
#         self.F = F
#         self.key = key

#     @property
#     def state_dim(self):
#         return (self.grid_size,)

#     def equation_of_motion(
#         self, x: Float[Array, " dim"], t: float
#     ) -> Float[Array, " dim"]:
#         return rhs_lorenz_96(x=x, t=t, F=self.F)

#     def observe(self, x: Float[Array, " dim"], n_steps: int):
#         t = jnp.asarray([n * self.dt for n in range(n_steps)])
#         return x[:: self.observe_every], t[:: self.observe_every]

#     def init_x0(self, noise: float = 0.01):
#         x0 = self.F * jnp.ones(self.grid_size)
#         perturb = noise * jrandom.normal(self.key, shape=())
#         return x0.at[0].set(x0[0] + perturb)

#     def init_x0_batch(self, batchsize: int, noise: float = 0.01):
#         # initial state (equilibrium)
#         x0 = self.F * jnp.ones((batchsize, self.grid_size))

#         # small perturbation
#         perturb = noise * jrandom.normal(self.key, shape=(batchsize,))

#         return x0.at[..., 0].set(x0[..., 0] + perturb)


def rhs_lorenz_96(x: Array, F: float = 8) -> Array:
    x_plus_1 = jnp.roll(x, -1)
    x_minus_2 = jnp.roll(x, 2)
    x_minus_1 = jnp.roll(x, 1)
    return (x_plus_1 - x_minus_2) * x_minus_1 - x + F
