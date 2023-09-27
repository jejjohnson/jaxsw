import typing as tp
import jax.numpy as jnp
from jaxtyping import Array, Float
import einops


def make_coords(xmin, xmax, nx):
    return jnp.linspace(start=xmin, stop=xmax, num=nx, endpoint=True)


def make_grid_from_coords(coords: tp.Iterable) -> tp.List[Array]:
    if isinstance(coords, tp.Iterable):
        return jnp.meshgrid(*coords, indexing="ij")
    elif isinstance(coords, (jnp.ndarray, np.ndarray)):
        return jnp.meshgrid(coords, indexing="ij")
    else:
        raise ValueError("Unrecognized dtype for inputs")


def make_grid_coords(coords: tp.Iterable) -> Array:
    grid = make_grid_from_coords(coords)

    grid = jnp.stack(grid, axis=0)

    grid = einops.rearrange(grid, "N ... -> (...) N")

    return grid


def create_meshgrid_coordinates(shape):
    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in shape], indexing="ij")
    # create indices
    indices = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

    return indices


def bounds_and_step_to_points(xmin: float, xmax: float, dx: float) -> int:
    return 1 + int(jnp.floor(((float(xmax) - float(xmin)) / float(dx))))


def bounds_to_length(xmin: float, xmax: float) -> float:
    return abs(float(xmax) - float(xmin))


def bounds_and_points_to_step(xmin: float, xmax: float, Nx: float) -> float:
    return (float(xmax) - float(xmin)) / (float(Nx) - 1.0)


def length_and_points_to_step(Lx: float, Nx: float) -> float:
    return float(Lx) / (float(Nx) - 1.0)


def length_and_step_to_points(Lx: float, dx: float) -> int:
    return int(jnp.floor(1.0 + float(Lx) / float(dx)))
