import typing as tp
import jax
import jax.numpy as jnp
from jaxtyping import Array


def plusminus(u: Array, way: int=1) -> tp.Tuple[Array, Array]:
    """Plus Minus Scheme
    It returns the + and - for the array whereby 
    "plus" has all values less than zero equal to zero
    and "minus" has all values greater than zero equal to zero.
    This is useful for advection schemes
    
    Args:
        u (Array): the input field
        way (int): chooses which "way" (default=1)
    
    Returns:
        plus (Array): the postive values
        minus (Array): the negative values
    """
    plus = jnp.where(way * u < 0.0, 0.0, u)
    minus = jnp.where(way * u > 0.0, 0.0, u)
    return plus, minus


def plusminus_fn(
    u: Array, way: int=1, fn: tp.Optional[tp.Callable]=jax.nn.relu
) -> tp.Tuple[Array, Array]:
    """Plus Minus Scheme with a function
    It returns the + and - for the array whereby 
    "plus" has all values less than zero equal to zero
    and "minus" has all values greater than zero equal to zero.
    The function should be something like relu (similar to DL schemes)
    This is useful for advection schemes
    
    Args:
        u (Array): the input field
        way (int): chooses which "way" (default=1)
        fn (Optional[Callable]): the function (default=relu)
    
    Returns:
        plus (Array): the postive values
        minus (Array): the negative values
    """
    u_plus = fn(way * u)
    u_minus = - 1. * fn(- 1. * way * u)
    return u_plus, u_minus


def center_average(u: Array) -> Array:
    """Returns the four-point average at the centres between grid points.
    If psi has shape (nx, ny), returns an array of shape (nx-1, ny-1)."""
    return 0.25*(u[:-1,:-1] + u[:-1,1:] + u[1:, :-1] + u[1:,1:])