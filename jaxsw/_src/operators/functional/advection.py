import typing as tp
import jax
import jax.numpy as jnp
from jaxtyping import Array
import finitediffx as fdx


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
    u_plus = jnp.where(way * u < 0.0, 0.0, u)
    u_minus = jnp.where(way * u > 0.0, 0.0, u)
    return u_plus, u_minus


def plusminus_fn(
    u: Array, 
    way: int=1, 
    fn: tp.Optional[tp.Callable]=jax.nn.relu
) -> tp.Tuple[Array, Array]:
    """Plus Minus Scheme with an "activation" function
    It returns the + and - for the array whereby 
    "plus" has all values less than zero equal to zero
    and "minus" has all values greater than zero equal to zero.
    The function should be something like relu (similar to DL schemes)
    This is useful for advection schemes.
    
    Args:
        u (Array): the input field
        way (int): chooses which "way" (default=1)
        fn (Optional[Callable]): the function (default=relu)
    
    Returns:
        plus (Array): the postive values
        minus (Array): the negative values
        
    Resources:
        See this page for more information on different activation
        functions: https://en.wikipedia.org/wiki/Activation_function
    """
    u_plus = fn(way * u)
    u_minus = - 1. * fn(- 1. * way * u)
    return u_plus, u_minus


def x_average_1D(u: Array) -> Array:
    """Returns the 2-point average at the centres between 1D grid points.
    
    Grid:
        + ------- +
        u    u̅    u
        + ------- + 
    
    Args:
        u (Array): the field [Nx]
    
    Returns:
        ubar (Array): the field averaged [Nx-1]
    
    """
    return 0.5 * (u[:-1] + u[1:])


def x_average_2D(u: Array) -> Array:
    return u


def y_average_2D(u: Array) -> Array:
    return u


def center_average_2D(u: Array) -> Array:
    """Returns the four-point average at the centres between grid points.
    
    Grid:
        u -- ⋅ -- u
        |         | 
        ⋅    u̅    ⋅
        |         |
        u -- ⋅ -- u
    
    Args:
        u (Array): the field [Nx,Ny]
    
    Returns:
        ubar (Array): the field averaged [Nx-1, Ny-1]
    
    """
    return 0.25 * (u[:-1,:-1] + u[:-1,1:] + u[1:, :-1] + u[1:,1:])


def upwind_2D(
    u: Array, 
    a: Array, 
    dx: Array, 
    way: int=1,
    **kwargs
) -> Array:
    """Calculates an advection term using and upwind scheme.
    1. We use a cell-centered average for the factor, a
    2. We clamp the negative values and positive values for the 
        factor, a
    3. We do the backward and foward difference for the field, u
    
    Eqn:
        
        a ∂u/∂x := a̅⁺ D₋[∂u/∂x] + a̅⁻ D₊[∂u/∂x]
    
    where:
        * a̅ : cell-centered average term
        * a⁺, a⁻: clamped values that are < 0.0 and > 0.0 respectively
        * D₋, D₊: backwards and forwards finite difference schemes
    
    Args:
        u (Array): the field
        a (Array): the multiplicative factor on the field
        dx (Array): the step size for the field
        way (int): the direction
    
    Returns:
        u (Array): the field 
    """
    a = jnp.pad(a, pad_width=((1,0),(1,0)), mode="constant")
    
    a_avg = center_average_2D(a)
    
    a_plus, a_minus = plusminus(a_avg, way=way)
    
    du_dx_f = fdx.difference(
        u, 
        step_size=dx,
        method="forward", 
        **kwargs
    )
    du_dx_b = fdx.difference(
        u, 
        step_size=dx,
        method="backward", 
        **kwargs
    )
    
    return a_plus * du_dx_b + a_minus * du_dx_f
