import typing as tp
import jax.numpy as jnp
from jaxtyping import Array, Float
import einops


def make_coords(xmin, xmax, delta):
    return jnp.arange(xmin, xmax + delta, delta)


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
