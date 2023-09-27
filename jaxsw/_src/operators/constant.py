import typing as tp

import equinox as eqx

# from .functional.conv import fd_convolve
import jax.numpy as jnp

from jaxsw._src.domain.base import Domain
from jaxsw._src.fields.base import Field


class Constant(eqx.Module):
    constant: tp.Union[float, jnp.ndarray] = eqx.static_field()

    def __init__(
        self,
        constant: tp.Union[float, jnp.ndarray] = 1.0,
    ):
        self.constant = constant

    def binop(self, other, fn: tp.Callable):
        # check discretization
        values = fn(self.constant, other.values)
        return other.__class__(values, other.domain)

    def single_op(self, fn: tp.Callable):
        # check discretization
        return Constant(fn(self.constant))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __neg__(self):
        return self.single_op(lambda x: -x)

    def __pow__(self, power):
        return self.single_op(lambda x: x**power)

    def __rpow__(self, power):
        return self.single_op(lambda x: power**x)
