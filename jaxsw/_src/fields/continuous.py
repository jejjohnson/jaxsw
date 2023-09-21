import typing as tp
from jaxsw._src.fields.base import Field
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxsw._src.domain.base import Domain


class ContinuousField(Field):
    fn: tp.Callable
    domain: Domain

    def __init__(self, domain: Domain, fn: tp.Callable):
        """
        Args:
            fn (Callable): An arbitrary function that can query coordinate values.
            domain (Domain): the domain for the array
        """

        self.fn = fn
        self.domain = domain

    def __call__(self, coords: Array, **kwargs) -> Array:
        return self.fn(coords, **kwargs)

    def batch_call(self, coords: Array, **kwargs) -> Array:
        return jax.vmap(self.fn)(coords, **kwargs)

    @property
    def values(self):
        return self.evaluate_grid()

    def evaluate_grid(self, **kwargs) -> Array:
        """Evaluates the function on the domain

        Returns:
            Array: The function evaluated at the coordinates.
        """
        values = self.batch_call(self.domain.coords, **kwargs)

        # handle multi-output dimensions
        if values.ndim > 1:
            output_dims = (values.shape[0],) + self.domain.grid_axis[0].shape
        else:
            output_dims = self.domain.grid_axis[0].shape

        return jnp.reshape(values, output_dims)

    def to_discrete_grid(self):
        return Field(values=self.evaluate_grid(), domain=self.domain)


class ParameterizedField(Field):
    fn: eqx.Module