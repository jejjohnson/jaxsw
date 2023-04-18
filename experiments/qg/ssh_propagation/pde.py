import autoroot
import xarray as xr
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxsw._src.domain.base import Domain
from jaxsw._src.domain.latlon import LatLonMeanDomain
from jaxsw._src.operators.functional.advection import upwind_2D
from jaxsw._src.operators.functional.geostrophic import uv_velocity
from jaxsw._src.operators.functional.geostrophic import (
    streamfn_to_pvort,
    pvort_to_streamfn,
    ssh_to_streamfn,
    streamfn_to_ssh,
)
from jaxsw._src.boundaries.helmholtz import enforce_boundaries_helmholtz
from typing import NamedTuple
from jaxsw._src.models.pde import DynamicalSystem
from jaxsw._src.domain.time import TimeDomain


class StateParams(NamedTuple):
    domain: Domain
    f0: float
    beta: float
    c1: float
    eta: Array


class AdvectionParams(NamedTuple):
    accuracy: int
    method: str
    way: int


class StreamFnParams(NamedTuple):
    accuracy: int
    method: str


class State(NamedTuple):
    q: Array

    @classmethod
    def init_state(cls, da: xr.DataArray, c1: float = 1.5):
        da = da.transpose("lon", "lat")
        lon = da.lon.values
        lat = da.lat.values
        c1 = jnp.asarray(c1)

        domain = LatLonMeanDomain(lat=lat, lon=lon)

        # initialize parameters
        eta = jnp.asarray(da.values)
        f0 = jnp.asarray(domain.f0)
        beta = domain.beta

        # ssh --> stream function
        psi = ssh_to_streamfn(eta, f0=f0)

        # stream function --> potential vorticity
        q = streamfn_to_pvort(
            psi, dx=domain.dx_mean, dy=domain.dx_mean, f0=f0, c1=c1, accuracy=1
        )
        q = enforce_boundaries_helmholtz(q, psi, beta=(f0 / c1) ** 2)

        # initialize state parameters
        state_params = StateParams(c1=c1, domain=domain, f0=f0, beta=beta, eta=eta)

        return cls(q=q), state_params

    @staticmethod
    def update_state(state, **kwargs):
        return State(
            q=kwargs.get("q", state.q),
        )


class QG(DynamicalSystem):
    @staticmethod
    def equation_of_motion(t: float, state: State, args):
        """Quasi-Geostrophic Equations

        Equation:
            ∂q/∂t + det J(Ψ,q) = -β ∂Ψ/∂x
            q = ∇²Ψ - (f₀²/c₁²) Ψ
            Ψ = (f₀/g) η
        """
        # parse params
        params = args
        dx = dy = params.domain.dx_mean
        f0, beta, c1, eta = params.f0, params.beta, params.c1, params.eta

        # jax.debug.print(f0, beta, c1, eta, dx, dy)

        # print("Before:", state.q.min(), state.q.max())

        # parse state
        q = state.q

        # ssh -> stream function
        psi_bv = ssh_to_streamfn(ssh=eta, f0=f0)

        # potential vorticity -> stream function
        psi = pvort_to_streamfn(
            q,
            psi_bv,
            dx=dx,
            dy=dy,
            f0=f0,
            c1=c1,
            accuracy=1,
            method="central",
        )

        # upwind scheme for advection
        q_rhs = -advection_term_upwind(
            q=q,
            psi=psi,
            dx=dx,
            dy=dy,
            way=-1,
            method="central",
        )

        # beta term
        _, v = uv_velocity(psi, dx=dx, dy=dy, accuracy=1, method="central")

        q_rhs += -beta * v

        # update state
        state = State.update_state(state, q=q_rhs)

        return state

    @staticmethod
    def ssh_from_state(state, params) -> Array:
        domain = params.domain
        dx = dy = domain.dx_mean
        f0, c1, eta = params.f0, params.c1, params.eta
        q = state.q
        assert q.ndim == 2

        psi_bv = ssh_to_streamfn(ssh=eta, f0=f0)

        psi = pvort_to_streamfn(q, psi_bv, dx=dx, dy=dy, f0=f0, c1=c1, accuracy=1)

        return streamfn_to_ssh(psi, f0=f0)


def advection_term_upwind(
    q: Array, psi: Array, dx: Array, dy: Array, **kwargs
) -> Array:
    # u,v schemes
    u, v = uv_velocity(psi, dx=dx, dy=dy, accuracy=kwargs.get("accuracy", 1))

    udq_dx = upwind_2D(q, u, dx=dx, axis=0)

    vdq_dy = upwind_2D(q, v, dx=dy, axis=1)

    rhs = jnp.zeros_like(q)

    rhs = rhs.at[1:-1, 1:-1].set(udq_dx[1:-1, 1:-1] + vdq_dy[1:-1, 1:-1])

    return rhs
