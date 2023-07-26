from typing import Any, NamedTuple, Tuple, Union

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox import static_field
from jax.random import PRNGKeyArray
from jaxtyping import Array, PyTree

from .base import DynamicalSystem


class L96TParams(eqx.Module):
    """
    Args:
    -----
        F (Array) : Forcing Term
        h (Array) : coupling coefficient
        b (Array) : ratio of amplitudes
        c (Array) : time-scale ratio
    """

    F: Array = static_field()
    h: Array = static_field()
    b: Array = static_field()
    c: Array = static_field()


class L96TState(NamedTuple):
    x: Array
    y: Array

    @classmethod
    def init_state(
        cls,
        ndims: Tuple[int] | int = (10, 20),
        noise: Tuple[float] | float = 0.01,
        batchsize: int = 1,
        b: float = 10.0,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        # check dims
        noise: Tuple[float] = check_dims(value=noise, ndim=2, name="noise")
        ndims: Tuple[int] = check_dims(value=ndims, ndim=2, name="ndims")

        keyx, keyy = jrandom.split(key=key, num=2)
        if batchsize > 1:
            x0 = b * noise[0] * jrandom.normal(key=keyx, shape=(batchsize, ndims[0]))
            y0 = noise[1] * jrandom.normal(
                key=keyy, shape=(batchsize, ndims[1] * ndims[0])
            )

        else:
            x0 = b * noise[0] * jrandom.normal(key=keyx, shape=(ndims[0],))
            y0 = noise[1] * jrandom.normal(key=keyy, shape=(ndims[1] * ndims[0],))

        return cls(x=x0, y=y0)

    @staticmethod
    def init_state_and_params(
        ndims: Tuple[int] | int = (10, 20),
        noise: Tuple[float] | float = 0.01,
        batchsize: int = 1,
        b: float = 10.0,
        c: float = 10.0,
        h: float = 1.0,
        F: float = 18.0,
        key: PRNGKeyArray = jrandom.PRNGKey(123),
    ):
        return L96TState.init_state(
            ndims=ndims,
            noise=noise,
            batchsize=batchsize,
            b=b,
            key=key,
        ), L96TParams(F=F, b=b, c=c, h=h)


class Lorenz96t(DynamicalSystem):
    advection: bool

    def __init__(self, advection: bool = True):
        self.advection = advection

    def equation_of_motion(
        self, t: float, state: L96TState, args: L96TParams
    ) -> PyTree:
        # compute RHS
        x, y = state.x, state.y
        F, h, b, c = args.F, args.h, args.b, args.c

        x_dot, y_dot, coupling = rhs_lorenz_96t(
            x=x, y=y, F=F, h=h, c=c, b=b, advection=self.advection, return_coupling=True
        )

        # update state
        state = eqx.tree_at(lambda x: x.x, state, x_dot)
        state = eqx.tree_at(lambda x: x.y, state, y_dot)

        return state


def rhs_lorenz_96t(
    x: Array,
    y: Array,
    F: Array | float = 18.0,
    h: Array | float = 1.0,
    b: Array | float = 10.0,
    c: Array | float = 10.0,
    advection: bool = True,
    return_coupling: bool = True,
) -> Union[Array, Array, Array]:
    x_dims = x.shape[0]
    xy_dims = y.shape[0]
    y_dims = xy_dims // x_dims

    assert xy_dims == x_dims * y_dims, "X n Y have incompatible dims"

    hcb = (h * c) / b

    # compute x-term
    # X[i-1] ( X[i+1]  - X[i-2])
    x_minus_1 = jnp.roll(x, 1)
    x_plus_1 = jnp.roll(x, -1)
    x_minus_2 = jnp.roll(x, 2)

    x_advection = x_minus_1 * (x_plus_1 - x_minus_2)

    # sum_j Y[j,i]
    y_summed = einops.rearrange(y, "(Dy Dx) -> Dy Dx", Dy=y_dims, Dx=x_dims)
    y_summed = einops.reduce(y_summed, "Dy Dx -> Dx", reduction="sum")

    if advection:
        x_rhs = x_advection - x + F - hcb * y_summed
    else:
        x_rhs = -x + F - hcb * y_summed

    # compute y-term
    # Y[j+1] ( Y[j+2] - Y[j-1] )
    y_plus_1 = jnp.roll(y, -1)
    y_plus_2 = jnp.roll(y, -2)
    y_minus_1 = jnp.roll(y, 1)

    y_advection = y_plus_1 * (y_plus_2 - y_minus_1)

    x_repeat = einops.repeat(x, "Dx -> (Dx Dy)", Dx=x_dims, Dy=y_dims)

    y_rhs = -b * c * y_advection - c * y + hcb * x_repeat

    #
    if return_coupling:
        return x_rhs, y_rhs, -hcb * y_summed
    else:
        return x_rhs, y_rhs


def check_dims(value: Any, ndim: int, name: str):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, float):
        return (value,) * ndim
    elif isinstance(value, jax.Array):
        return jnp.repeat(value, ndim)
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    raise ValueError(f"Expected int or tuple for {name}, got {value}.")
