from typing import Optional

import equinox as eqx
from jaxtyping import PyTree


class DynamicalSystem(eqx.Module):
    def init_u0(self, domain: PyTree):
        raise NotImplementedError()

    def boundary(self, state: PyTree):
        raise NotImplementedError()

    def equation_of_motion(
        self, t: float, state: PyTree, args: Optional[PyTree] = None
    ) -> PyTree:
        raise NotImplementedError()
