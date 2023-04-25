import jax.numpy as jnp
import jax.random as jrandom
import pytreeclass as pytc
from jax.random import PRNGKeyArray
from jaxtyping import Array, Float, Int, PyTree

from .base import DynamicalSystem


@pytc.treeclass
class Lorenz63(DynamicalSystem):
    observe_every: int = pytc.field(nondiff=True)
    s: float = pytc.field(nondiff=True)
    r: float = pytc.field(nondiff=True)
    b: float = pytc.field(nondiff=True)
    key: jrandom.PRNGKey = jrandom.PRNGKey(0)

    def __init__(
        self,
        dt: float,
        observe_every: int = 1,
        s: float = 8,
        r: float = 28,
        b: float = 8.0 / 3.0,
        key: PRNGKeyArray = jrandom.PRNGKey(0),
    ):
        self.dt = dt
        self.s = s
        self.r = r
        self.b = b
        self.observe_every = observe_every
        self.key = key

    @property
    def state_dim(self):
        return (self.grid_size,)

    def equation_of_motion(
        self, x: Float[Array, "dim"], t: float
    ) -> Float[Array, "dim"]:
        return rhs_lorenz_63(x=x, t=t, s=self.s, r=self.r, b=self.b)

    def observe(self, x: Float[Array, "dim"], n_steps: int):
        t = jnp.asarray([n * self.dt for n in range(n_steps)])
        return x[:: self.observe_every], t[:: self.observe_every]

    def init_x0(self, noise: float = 0.01):
        x0 = jnp.ones((3,))
        perturb = noise * jrandom.normal(self.key, shape=())
        return x0.at[0].set(x0[0] + perturb)

    def init_x0_batch(self, batchsize: int, noise: float = 0.01):
        # initial state (equilibrium)
        x0 = jnp.ones((batchsize, 3))

        # small perturbation
        perturb = noise * jrandom.normal(self.key, shape=(batchsize,))

        return x0.at[..., 0].set(x0[..., 0] + perturb)


def rhs_lorenz_63(
    x: Float[Array, "dim"], t: float, s: float = 10, r: float = 28, b: float = 2.667
) -> Float[Array, "dim"]:
    x, y, z = x
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return jnp.array([x_dot, y_dot, z_dot])
