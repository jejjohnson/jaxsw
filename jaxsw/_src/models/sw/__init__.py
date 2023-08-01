import typing as tp
from jaxsw._src.domain.base import Domain
import jax.numpy as jnp
from jaxtyping import Array


class Params(tp.NamedTuple):
    domain: Domain
    depth: float
    gravity: float
    coriolis_f0: float  # or ARRAY
    coriolis_beta: float  # or ARRAY

    @property
    def phase_speed(self):
        return jnp.sqrt(self.gravity * self.depth)

    def rossby_radius(self, domain):
        return self.phase_speed / self.coriolis_param(domain).mean()
        # return self.phase_speed / self.coriolis_f0

    def coriolis_param(self, domain):
        return self.coriolis_f0 + domain.grid[..., 1] * self.coriolis_beta

    def lateral_viscosity(self, domain):
        return 1e-3 * self.coriolis_f0 * domain.dx[0] ** 2


class State(tp.NamedTuple):
    u: Array
    v: Array
    h: Array

    @classmethod
    def init_state(cls, params, init_h=None, init_v=None, init_u=None):
        h = init_h(params) if init_h is not None else State.zero_init(params.domain)
        v = init_v(params) if init_v is not None else State.zero_init(params.domain)
        u = init_u(params) if init_u is not None else State.zero_init(params.domain)
        return cls(u=u, v=v, h=h)

    @staticmethod
    def zero_init(domain):
        return jnp.zeros_like(domain.grid[..., 0])
