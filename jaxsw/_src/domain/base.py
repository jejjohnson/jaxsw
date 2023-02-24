from typing import NamedTuple, Iterable, List
from jaxtyping import Array, Float
from functools import reduce
import equinox as eqx
import jax.numpy as jnp


class Domain(eqx.Module):
    """Domain class for a rectangular domain

    Attributes:
        size (Tuple[int]): The size of the domain in absolute units.
        dx (Tuple[float]):
    """

    D: Iterable[int] = eqx.static_field()
    dx: Iterable[int] = eqx.static_field()

    @property
    def size(self) -> List[float]:
        """The length of the grid sides

        Returns:
            List[float]: the size of the domain, in absolute units.
        """
        return list(map(lambda x: x[0] * x[1], zip(self.D, self.dx)))

    @property
    def ndim(self) -> int:
        """The number of dimensions of the domain

        Returns:
            int: the number of dimensions in the domain
        """
        return len(self.D)

    @property
    def cell_volume(self) -> float:
        """The volume of a single cell

        Returns:
            float: the volume of a single cell
        """
        return reduce(lambda x, y: x * y, self.dx)

    @property
    def spatial_axis(self) -> List[Array]:
        axis = [make_axis(n, delta) for n, delta in zip(self.D, self.dx)]
        axis = [ax - jnp.mean(ax) for ax in axis]
        return axis

    @property
    def origin(self) -> Array:
        return jnp.zeros((self.ndim,))

    @property
    def grid(self) -> Array:
        return make_grid_from_axis(self.spatial_axis)


def make_axis(D, delta):
    if D % 2 == 0:
        return jnp.arange(0, D) * delta - delta * D / 2
    else:
        return jnp.arange(0, D) * delta - delta * (D - 1) / 2


def make_grid_from_axis(axis):
    if isinstance(axis, (list, tuple)):
        return jnp.stack(jnp.meshgrid(*axis, indexing="ij"), axis=-1)
    elif isinstance(axis, jnp.ndarray):
        return jnp.stack(jnp.meshgrid(axis, indexing="ij"), axis=-1)
