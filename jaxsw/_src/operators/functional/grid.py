import kernex as kex
import jax.numpy as jnp
from jaxtyping import Array


@kex.kmap(kernel_size=(2,))
def x_average_1D(u: Array) -> Array:
    """Returns the two-point average at the centres between grid points.
    
    Grid:
        + -- ⋅ -- +
        u -- u̅ -- u
        + -- ⋅ -- +
    
    Args:
        u (Array): the field [Nx,]
    
    Returns:
        ubar (Array): the field averaged [Nx-1,]
    
    """
    return jnp.mean(u)


@kex.kmap(kernel_size=(2,2))
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
    return jnp.mean(u)


@kex.kmap(kernel_size=(2,1))
def x_average_2D(u: Array) -> Array:
    """Returns the two-point average at the centres between grid points.
    
    Grid:
        u -- u̅ -- u
        |         | 
        ⋅         ⋅
        |         |
        u -- u̅ -- u
        
    Args:
        u (Array): the field [Nx,Ny]
    
    Returns:
        ubar (Array): the field averaged [Nx-1, Ny]
    
    """
    return jnp.mean(u)


@kex.kmap(kernel_size=(1,2), padding=((),()))
def y_average_2D(u: Array) -> Array:
    """Returns the two-point average at the centres between grid points.
    
    Grid:
        u -- ⋅ -- u
        |         | 
        u̅         u̅
        |         |
        u -- ⋅ -- u
    
    Args:
        u (Array): the field [Nx,Ny]
    
    Returns:
        ubar (Array): the field averaged [Nx, Ny-1]
    
    """
    return jnp.mean(u)


def u_at_v(u: Array) -> Array:    
    return center_average_2D(u)[1:-1,:]


def v_at_u(v: Array) -> Array:
    return center_average_2D(v)[:,1:-1]
