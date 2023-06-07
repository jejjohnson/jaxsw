from typing import Callable, NamedTuple, Union

import autoroot  # noqa: F401, I001
from jaxtyping import Array

from jaxsw._src.domain.base import Domain


class State(NamedTuple):
    eta: Array
    psi: Array
    q: Array
    domain: Domain
    f0: Union[float, Array]
    beta: Union[float, Array]
    c1: Union[float, Array]

    @classmethod
    def init_state(
        cls,
        domain: Domain,
        f0: float,
        beta: float,
        c1: float,
        init_fn: Callable,
    ):
        eta = init_fn(domain, "eta")
        psi = init_fn(domain, "psi")
        q = init_fn(domain, "q")
        f0 = init_fn(domain, "f0")
        beta = init_fn(domain, "beta")
        c1 = init_fn(domain, "c1")

        return cls(eta=eta, psi=psi, q=q, c1=c1, domain=domain, f0=f0, beta=beta)

    def update_state(state, **kwargs):
        return State(
            domain=kwargs.get("domain", state.domain),
            eta=kwargs.get("eta", state.eta),
            psi=kwargs.get("psi", state.psi),
            q=kwargs.get("q", state.q),
            f0=kwargs.get("f0", state.f0),
            beta=kwargs.get("beta", state.beta),
            c1=kwargs.get("c1", state.c1),
        )
