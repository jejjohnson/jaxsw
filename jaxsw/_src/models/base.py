import pytreeclass as pytc
import jax.numpy as jnp
import jax
from jax.experimental.ode import odeint
from functools import partial
from jaxtyping import Array, Int, Float, PyTree
from typing import Union

@pytc.treeclass
class DynamicalSystem:
    """
    Base class to derive a dynamical system
    """
    dt: float = pytc.field(nondiff=True)
    
    def __init__(self, dt):
        self.dt = dt
    
    def equation_of_motion(self, X: Float[Array, "dim"], t: PyTree[int]) -> Float[Array, "dim"]:
        raise NotImplementedError
        
    @property
    def state_dim(self) -> int:
        raise NotImplementedError
        
    
    def observe(self, x: Float[Array, "dim"], t: int) -> Float[Array, "dim"]:
        raise NotImplementedError
        
    @partial(jax.vmap, in_axes=(None, 0, None))
    def batch_observe(self, x: Float[Array, "batch dim"], t: int) -> Float[Array, "batch dim"]:
        return self.observe(x, t)
    
    def integrate(
        self, x0: Float[Array, "dim"], n_steps: int
    ) -> Union[Float[Array, "steps dim"], Float[Array, "steps"]]:
        t = jnp.asarray([n*self.dt for n in range(n_steps)])
        traj = odeint(self.equation_of_motion, x0, t)
        return traj, t
    
    @partial(jax.vmap, in_axes=(None, 0, None))
    def batch_integrate(
            self, x0: Float[Array, "batch dim"], n_steps: int
    ) -> Union[Float[Array, "batch steps dim"], Float[Array, "batch steps"]]:
        return self.integrate(x0, n_steps)
    
    def warmup(
        self, x0: Float[Array, "dim"], n_steps: int
    ) -> Union[Float[Array, "dim"], Float[Array, "dim"]]:
        return self.integrate(x0=x0, n_steps=n_steps)[0][-1, ...]
    
    @partial(jax.vmap, in_axes=(None, 0, None))
    def batch_warmup(
        self, x0, n_steps
    ) -> Union[Float[Array, "batch dim"], Float[Array, "batch dim"]]:
        return self.warmup(x0=x0, n_steps=n_steps)
    
        
    
    
        