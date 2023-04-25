from functools import partial
from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


class DynamicalSystem(eqx.Module):
    tmin: float
    tmax: float
    solver: dfx.AbstractSolver = dfx.Euler()
    stepsize_controller: dfx.PIDController = dfx.ConstantStepSize()

    def __init__(
        self,
        tmin: float,
        tmax: float,
        solver: dfx.AbstractSolver = dfx.Euler(),
        stepsize_controller: dfx.PIDController = dfx.ConstantStepSize(),
    ):
        self.solver = solver
        self.tmin = tmin
        self.tmax = tmax
        self.stepsize_controller = stepsize_controller

    def init_u0(self, domain: PyTree):
        raise NotImplementedError()

    def boundary(self, state: PyTree):
        raise NotImplementedError()

    def equation_of_motion(
        self, t: float, state: PyTree, args: Optional[PyTree] = None
    ) -> PyTree:
        raise NotImplementedError()

    def integrate(self, state: PyTree, dt: float, args=None, **kwargs) -> PyTree:
        ts = self.saveat(dt=kwargs.pop("dtsave", dt))
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(self.equation_of_motion),
            solver=self.solver,
            t0=self.tmin,
            t1=self.tmax,
            dt0=dt,
            y0=state,
            saveat=dfx.SaveAt(t0=self.tmin, t1=self.tmax, ts=ts),
            args=args,
            stepsize_controller=self.stepsize_controller,
            **kwargs
        )
        return sol.ys, sol.ts

    @partial(jax.vmap, in_axes=(0, None, None))
    def batch_integrate(self, state: PyTree, dt: float, args=None, **kwargs) -> PyTree:
        return self.integrate(state=state, dt=dt, args=args, **kwargs)

    def saveat(self, dt=None):
        if dt is None:
            dt = int((self.tmax - self.tmin) / 2)
        ts = jnp.arange(self.tmin, self.tmax + dt, dt)
        return ts


# @pytc.treeclass
# class DynamicalSystem:
#     """
#     Base class to derive a dynamical system
#     """

#     dt: float = pytc.field(nondiff=True)

#     def __init__(self, dt):
#         self.dt = dt

#     def equation_of_motion(
#         self, X: Float[Array, " dim"], t: PyTree[int]
#     ) -> Float[Array, " dim"]:
#         raise NotImplementedError

#     @property
#     def state_dim(self) -> int:
#         raise NotImplementedError

#     def observe(self, x: Float[Array, " dim"], t: int) -> Float[Array, " dim"]:
#         raise NotImplementedError

#     @partial(jax.vmap, in_axes=(None, 0, None))
#     def batch_observe(
#         self, x: Float[Array, " batch dim"], t: int
#     ) -> Float[Array, " batch dim"]:
#         return self.observe(x, t)

#     def integrate(
#         self, x0: Float[Array, " dim"], n_steps: int
#     ) -> Union[Float[Array, " steps dim"], Float[Array, " steps"]]:
#         t = jnp.asarray([n * self.dt for n in range(n_steps)])
#         traj = odeint(self.equation_of_motion, x0, t)
#         return traj, t

#     @partial(jax.vmap, in_axes=(None, 0, None))
#     def batch_integrate(
#         self, x0: Float[Array, " batch dim"], n_steps: int
#     ) -> Union[Float[Array, " batch steps dim"], Float[Array, " batch steps"]]:
#         return self.integrate(x0, n_steps)

#     def warmup(
#         self, x0: Float[Array, " dim"], n_steps: int
#     ) -> Union[Float[Array, " dim"], Float[Array, " dim"]]:
#         return self.integrate(x0=x0, n_steps=n_steps)[0][-1, ...]

#     @partial(jax.vmap, in_axes=(None, 0, None))
#     def batch_warmup(
#         self, x0, n_steps
#     ) -> Union[Float[Array, " batch dim"], Float[Array, " batch dim"]]:
#         return self.warmup(x0=x0, n_steps=n_steps)
