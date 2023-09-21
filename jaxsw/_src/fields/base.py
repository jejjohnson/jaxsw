import typing as tp

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from ..domain.base import Domain
from .utils import check_discretization

# TODO: add power operation
# TODO: add negative operation


class Field(eqx.Module):
    """A Field for a discrete domain

    Attributes:
        values (Array): An arbitrary sized array
        domain (Domain): the domain for the array
    """

    values: Float
    domain: Domain

    def __init__(self, values: Array, domain: Domain):
        """
        Args:
            values (Array): An arbitrary sized array
            domain (Domain): the domain for the array
        """
        values = jnp.atleast_1d(values)
        msg = "Incorrect dimentions\n"
        msg += f"Values: {values.shape} | Grid: {domain.grid_axis[0].shape}"
        assert values.shape == domain.grid_axis[0].shape, msg

        self.values = values
        self.domain = domain

    @classmethod
    def init_from_fn(cls, domain: Domain, fn: tp.Callable, **kwargs):
        # vectorize coordinate values

        values = jax.vmap(fn)(domain.coords, **kwargs)
        # reshape to match grid size
        values = jnp.reshape(values, domain.grid_axis[0].shape)

        return cls(values=values, domain=domain)

    @classmethod
    def init_from_array(cls, values: Array):
        return cls(values=values, domain=Domain.from_array(u=values))

    def replace_values(self, values):
        return Field(values=values, domain=self.domain)

    @property
    def shape(self) -> tp.Tuple[int]:
        return self.values.shape

    def binop(self, other, fn: tp.Callable):
        # check discretization
        check_discretization(self.domain, other.domain)
        values = fn(self.values, other.values)
        return Field(values, self.domain)

    def __getitem__(self, idx):
        return self.values[idx]

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.binop(other, lambda x, y: y + x)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.binop(other, lambda x, y: y * x)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

    # def __bool__(self):
    #     return eqx.tree_at(lambda x: x.values, self, self.values.__bool__())

    def __neg__(self):
        return eqx.tree_at(lambda x: x.values, self, -self.values)

    def __pow__(self, power):
        return eqx.tree_at(lambda x: x.values, self, self.values**power)

    def __rpow__(self, power):
        return eqx.tree_at(lambda x: x.values, self, power**self.values)

    def __truediv__(self, other):
        return self.binop(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self.binop(other, lambda x, y: y / x)

    def inverse(self):
        return eqx.tree_at(lambda x: x.values, self, 1 / self.values)

    def min(self):
        return self.values.min()

    def max(self):
        return self.values.max()
