from typing import Optional, NamedTuple

import diffrax as dfx
import equinox as eqx
from jaxtyping import PyTree

from jaxsw._src.domain.time import TimeDomain


class DynamicalSystem(eqx.Module):
    t_domain: TimeDomain
    saveat: dfx.SaveAt

    def __init__(self, t_domain: TimeDomain, saveat: Optional[dfx.SaveAt] = None):
        self.t_domain = t_domain
        self.saveat = saveat if saveat is not None else dfx.SaveAt(t1=True)

    def init_u0(self, domain: PyTree):
        raise NotImplementedError()

    def enforce_boundaries(self, u: PyTree, *args, **kwargs) -> PyTree:
        raise NotImplementedError()

    def equation_of_motion(self, t: float, u: PyTree, args):
        raise NotImplementedError()

    def integrate(
        self, u0: PyTree, params: Optional[PyTree] = None, **kwargs
    ) -> dfx.solution.Solution:
        # extract default arguments
        t0 = kwargs.pop("t0", self.t_domain.tmin)
        t1 = kwargs.pop("t1", self.t_domain.tmax)
        dt0 = kwargs.pop("dt0", self.t_domain.dt)
        saveat = kwargs.pop("saveat", self.saveat)
        solver = kwargs.pop("solver", dfx.Tsit5())
        stepsize_controller = kwargs.pop("stepsize_controller", dfx.ConstantStepSize())

        # do integration
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(self.equation_of_motion),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=u0,
            args=params,
            saveat=saveat,
            solver=solver,
            stepsize_controller=stepsize_controller,
            **kwargs
        )
        return sol

    def integrate_step(self, *args, **kwargs):
        pass
