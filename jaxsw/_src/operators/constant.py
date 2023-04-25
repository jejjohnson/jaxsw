import typing as tp

import equinox as eqx

# from .functional.conv import fd_convolve
import jax.numpy as jnp

from jaxsw._src.domain.base import Domain
from jaxsw._src.fields.base import Field


class Constant(eqx.Module):
    domain: Domain = eqx.static_field()
    constant: tp.Union[float, jnp.ndarray] = eqx.static_field()

    def __init__(
        self,
        domain: Domain,
        constant: tp.Union[float, jnp.ndarray] = 1.0,
    ):
        self.domain = domain
        self.constant = constant

    def __call__(self, u: Field) -> Field:
        u = eqx.tree_at(lambda x: x.values, u, self.constant * u.values)

        return u
