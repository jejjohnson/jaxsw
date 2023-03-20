from typing import Callable
import equinox as eqx
from jaxsw._src.fields.base import Field


class FuncOperator(eqx.Module):
    f: Callable = eqx.static_field()

    def __init__(self, f):
        self.f = f

    def __call__(self, u: Field) -> Field:

        u = eqx.tree_at(lambda x: x.values, u, self.f(u.values))

        return u
