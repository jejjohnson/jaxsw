from typing import Callable

import equinox as eqx

from jaxsw._src.fields.base import Field


class FuncOperator(eqx.Module):
    fn: Callable

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, u: Field, *args, **kwargs) -> Field:
        # operate on values
        u_values = self.fn(u.values, *args, **kwargs)

        # replace values
        u = u.replace_values(u_values)

        return u
