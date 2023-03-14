import equinox as eqx
import jax.numpy as jnp


class TimeDomain(eqx.Module):
    tmin: float = eqx.static_field()
    tmax: float = eqx.static_field()
    dt: float = eqx.static_field()

    @classmethod
    def from_numpoints(cls, tmin, tmax, nt):
        dt = (tmax - tmin) / (float(nt) - 1)

        return cls(tmin=tmin, tmax=tmax, dt=dt)

    @property
    def coords(self):
        return jnp.arange(self.tmin, self.tmax, self.dt)
