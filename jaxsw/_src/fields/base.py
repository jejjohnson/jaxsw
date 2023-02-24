import equinox as eqx
import jax.numpy as jnp
from ..domain.base import Domain
from jaxtyping import Array, Float


class Field(eqx.Module):
    domain: Domain
    values: Array

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __neg__(self):
        pass

    def __pow__(self, other):
        pass

    def __rpow__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass
