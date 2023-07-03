import typing as tp
from functools import reduce
from operator import mul

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from finitediffx._src.utils import _check_and_return


def _fix_iterable_input(x, num_iters) -> tp.Iterable:
    if isinstance(x, tp.Iterable):
        pass
    elif isinstance(x, int | float):
        x = (float(x),) * num_iters
    elif isinstance(x, jnp.ndarray | np.ndarray):
        x = (float(x),) * num_iters
    else:
        raise ValueError(f"Improper input...{type(x)}")

    return x


class Domain(eqx.Module):
    """Domain class for a rectangular domain

    Attributes:
        size (Tuple[int]): The size of the domain
        xmin: (Iterable[float]): The min bounds for the input domain
        xmax: (Iterable[float]): The max bounds for the input domain
        coord (List[Array]): The coordinates of the domain
        grid (Array): A grid of the domain
        ndim (int): The number of dimenions of the domain
        size (Tuple[int]): The size of each dimenions of the domain
        cell_volume (float): The total volume of a grid cell
    """

    xmin: tp.Iterable[float] = eqx.static_field()
    xmax: tp.Iterable[float] = eqx.static_field()
    dx: tp.Iterable[float] = eqx.static_field()

    def __init__(self, xmin, xmax, dx):
        """Initializes domain
        Args:
            xmin (Iterable[float]): the min bounds for the input domain
            xmax (Iterable[float]): the max bounds for the input domain
            dx (Iterable[float]): the step size for the input domain
        """
        assert len(xmin) == len(xmax)
        dx = _check_and_return(dx, ndim=len(xmin), name="dx")
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx

    @classmethod
    def from_numpoints(
        cls,
        xmin: tp.Iterable[float],
        xmax: tp.Iterable[float],
        N: tp.Iterable[int],
    ):
        f = lambda xmin, xmax, N: (xmax - xmin) / (float(N) - 1)

        dx = tuple(map(f, xmin, xmax, N))

        return cls(xmin=xmin, xmax=xmax, dx=dx)

    @property
    def coords(self) -> tp.List:
        return list(map(make_coords, self.xmin, self.xmax, self.dx))

    @property
    def grid(self) -> jnp.ndarray:
        return make_grid_from_coords(self.coords)

    @property
    def ndim(self) -> int:
        return len(self.xmin)

    @property
    def size(self) -> tp.Tuple[int]:
        return tuple(map(len, self.coords))

    @property
    def cell_volume(self) -> float:
        return reduce(mul, self.dx)


def make_coords(xmin, xmax, delta):
    return jnp.arange(xmin, xmax, delta)


def make_grid_from_coords(coords: tp.Iterable) -> Float[Array, " D"]:
    if isinstance(coords, tp.Iterable):
        coords = jnp.meshgrid(*coords, indexing="ij")
    elif isinstance(coords, (jnp.ndarray, np.ndarray)):
        coords = jnp.meshgrid(coords, indexing="ij")
    else:
        raise ValueError("Unrecognized dtype for inputs")

    return jnp.stack(coords, axis=-1)
