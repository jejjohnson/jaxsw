import pytreeclass as pytc
from .base import DynamicalSystem
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Int, Float, PyTree

@pytc.treeclass
class Lorenz96(DynamicalSystem):
    grid_size : int = pytc.field(nondiff=True)
    observe_every: int = pytc.field(nondiff=True)
    F: float = pytc.field(nondiff=True)
    key: jrandom.PRNGKey = jrandom.PRNGKey(0)
    
    def __init__(self, dt: Float, grid_size: Int, observe_every: Int=1, F: Int=8, key: PyTree=jrandom.PRNGKey(0)):
        self.dt = dt
        self.grid_size = grid_size
        self.observe_every = observe_every
        self.F = F
        self.key = key
        
    @property
    def state_dim(self):
        return (self.grid_size,)
    
    def equation_of_motion(self, x: Float[Array, "dim"], t: float) -> Float[Array, "dim"]:
        return rhs_lorenz_96(x=x, t=t, F=self.F)
    
    def observe(self, x: Float[Array, "dim"], n_steps: int):
        t = jnp.asarray([n*self.dt for n in range(n_steps)])
        return x[::self.observe_every], t[::self.observe_every]
    
    def init_x0(self, noise: float=0.01):
        x0 = self.F * jnp.ones(self.grid_size)
        perturb = noise * jrandom.normal(self.key, shape=())
        return x0.at[0].set(x0[0] + perturb)
    
    def init_x0_batch(self, batchsize: int, noise: float=0.01):
        # initial state (equilibrium)
        x0 = self.F * jnp.ones((batchsize, self.grid_size))
        
        # small perturbation
        perturb = noise * jrandom.normal(self.key, shape=(batchsize,))
        
        return x0.at[..., 0].set(x0[..., 0] + perturb)

    
def rhs_lorenz_96(x: Float[Array, "dim"], t: float, F: float=80) -> Float[Array, "dim"]:
    x_plus_1 = jnp.roll(x, -1)
    x_minus_2 = jnp.roll(x, 2)
    x_minus_1 = jnp.roll(x, 1)
    return (x_plus_1 - x_minus_2) * x_minus_1 - x + F