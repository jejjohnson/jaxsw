import einops
import jax.numpy as jnp
import jax.scipy as jsp
from jaxsw._src.domain.utils import create_meshgrid_coordinates
from jaxsw._src.domain.base import Domain
from jaxtyping import Array
from jaxsw._src.fields.base import Field



def x_average_1D(x: Array) -> Array:
    """[Nx, ...] -> [Nx-1, ...]"""
    return 0.5 * (x[..., 1:] + x[..., :-1])
def x_average_2D(x: Array) -> Array:
    """[Nx, ...] -> [Nx-1, ...]"""
    return 0.5 * (x[..., 1:, :] + x[..., :-1, :])
def y_average_2D(x: Array) -> Array:
    """[Nx, ...] -> [Nx-1, ...]"""
    return 0.5 * (x[..., 1:] + x[..., :-1])

def center_average_2D(u: Array) -> Array:
    """[Nx,Ny] -> [Nx-1, Ny-1]"""
    return 0.25 *(u[...,1:,1:] + u[...,1:,:-1] + u[...,:-1,1:] + u[...,:-1,:-1])