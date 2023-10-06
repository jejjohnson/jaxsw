import jax.numpy as jnp
from jaxtyping import Array
from jaxsw._src.operators.functional.finitevol.difference import difference
from jaxsw._src.utils.constants import GRAVITY

def divergence(u: Array, v: Array, dx: float, dy: float) -> Array:
    """Calculates the divergence for a staggered grid
    
    Equation:
        ∇⋅u̅  = ∂x(u) + ∂y(v)
        
    Args:
        u (Array): the input array for the u direction
            Size = [Nx, Ny-1]
        v (Array): the input array for the v direction
            Size = [Nx-1, Ny]
    
    Returns:
        div (Array): the divergence
            Size = [Nx-1,Ny-1]
    
    """
    # ∂xu
    dudx = jnp.diff(u, n=1, axis=-2) / dx
    # ∂yv
    dvdx = jnp.diff(v, n=1, axis=-1) / dy
    
    return dudx + dvdx


def relative_vorticity(u: Array, v: Array, dx: Array, dy: Array, **kwargs) -> Array:
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): the u-velocity
        v (Array): the v-velocity
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all kwargs for the derivatives

    Returns:
        zeta (Array): the geostrophic vorticity
    """
    dv_dx = jnp.diff(v, n=1, axis=-2) / dx

    du_dy = jnp.diff(u, n=1, axis=-1) / dy

    return dv_dx - du_dy


def gradient_perpendicular(u: Array, dx: float, dy: float) -> Array:
    """Calculates the perpendicular gradient for a staggered grid
    
    Equation:
        u = ∂yΨ
        v = ∂xΨ
        
    Args:
        u (Array): the input velocity
            Size = [Nx,Ny]
        dx (float): the stepsize for the x-direction
        dy (float): the stepsize for the y-direction
    
    Returns:
        du_dy (Array): the velocity in the y-direction
            Size = [Nx,Ny-1]
        du_dx (Array): the velocity in the x-direction
            Size = [Nx-1,Ny]
    
    Note:
        for the geostrophic velocity, we need to multiply the 
        derivative in the x-direction by negative 1.
    """

    du_dy = difference(u, axis=-1, step_size=dy)
    du_dx = difference(u, axis=-2, step_size=dx)

    return du_dy, du_dx


def ssh_to_streamfn(ssh: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the ssh to stream function

    Eq:
        η = (g/f₀) Ψ

    Args:
        ssh (Array): the sea surface height [m]
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        psi (Array): the stream function
    """
    return (g / f0) * ssh


def streamfn_to_ssh(psi: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the stream function to ssh

    Eq:
        Ψ = (f₀/g) η

    Args:
        psi (Array): the stream function
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        ssh (Array): the sea surface height [m]
    """
    return (f0 / g) * psi