from typing import Optional
from jaxtyping import PyTree
import equinox as eqx
import diffrax as dfx
from jaxsw._src.domain import TimeDomain


class DynamicalSystem(eqx.Module):
    t_domain: TimeDomain
    solver: dfx.AbstractSolver = dfx.Euler()
    stepsize_controller: Optional[dfx.PIDController] = dfx.ConstantStepSize()
    saveat: Optional[dfx.SaveAt] = None

    def __init__(
        self,
        t_domain,
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        saveat=None
    ):
        self.solver = solver
        self.t_domain = t_domain
        self.saveat = saveat
        self.stepsize_controller = stepsize_controller
        
    def init_u0(self, domain: PyTree):
        raise NotImplementedError()
        
    def boundary(self, u: PyTree):
        raise NotImplementedError()
        
    def equation_of_motion(self, t: float, u: PyTree, args):
        raise NotImplementedError()
        
    def integrate(self, u: PyTree, dt: float, **kwargs) -> PyTree:
        
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(self.equation_of_motion),
            solver=self.solver,
            t0=self.t_domain.tmin,
            t1=self.t_domain.tmax,
            dt0=dt,
            y0=u,
            saveat=self.saveat,
            stepsize_controller=self.stepsize_controller,
            **kwargs
        )
        return sol.ys