import typing as tp
from jaxtyping import Array
from jaxsw._src.utils.constants import GRAVITY
import finitediffx as fdx
from jaxsw._src.utils.dst_solver import inverse_elliptical_dst_solver
from jaxsw._src.boundaries.helmholtz import enforce_boundaries_helmholtz


def ssh_to_streamfn(ssh: Array, f0: Array, g=GRAVITY) -> Array:
    """Calculates the ssh to stream function
    
    Eq:
        η = (g/f0) Ψ
    
    Args:
        ssh (Array): the sea surface height [m]
        f0 (Array|float): the coriolis parameter
        g (Array|float): the acceleration due to gravity
    
    Returns:
        psi (Array): the stream function
    """
    return (g / f0) * ssh


def streamfn_to_ssh(psi: Array, f0: Array, g=GRAVITY) -> Array:
    """Calculates the stream function to ssh
    
    Eq:
        Ψ = (f0/g) η
    
    Args:
        psi (Array): the stream function
        f0 (Array|float): the coriolis parameter
        g (Array|float): the acceleration due to gravity
    
    Returns:
        ssh (Array): the sea surface height [m]
    """
    return (f0 / g) * psi


def streamfn_to_pv(
    psi: Array, 
    dx: Array, 
    dy: Array, 
    c1: float=2.7, 
    f0: float=1e-5, 
    **kwargs
) -> Array:
    """Calculates the potential vorticity to the streamfunction
    
    Eq:
        q = ∇²Ψ - (f0²/c1²) Ψ
    
    Args:
        psi (Array): The stream function
        dx (Array): the change in x
        dy (Array): the change in y
        f0 (Array): coriolis parameter at mean latitude
        c1 (Array): the ...
        
    Returns:
        q (Array): The potential vorticity
        
    """
    return fdx.laplacian(psi, step_size=(dx,dy), **kwargs) - (f0/c1)**2 * psi


def pv_to_streamfn(
    q: Array,
    psi_bcs: Array,
    dx: Array,
    dy: Array,
    f0: float=1e-5, c1: float=1.5,
    **kwargs
) -> Array:
    """Does the inverse problem for the PV and SF. We use a trick
    where we decompse the problem into interior and exterior points.
    For this, we use the Discrete Sine Transformation which
    is very fast for inhomogeneous boundary conditions. This function
    works the best when dx=dy but we do leave it open...
    
    Ψ = (∇²-(f0²/c1²))^-1 q
    
    Args:
        q (Array): Potential Vorticity
        psi_bcs (Array): the psi where we enforce the boundaries.
        dx (Array): the change in x
        dy (Array): the change in y
        f0 (Array): the Coriolis parameter
        c1 (Array): the ...
        
    Returns:
        psi (Array): the stream function from the linear solver
    """
    
    
    
    beta = (f0/c1)**2
    
    # calculate the interior potential vorticity
    psi_bcs = psi_bcs.at[1:-1,1:-1].set(0.0)
    q_exterior = streamfn_to_pv(psi_bcs, dx=dx, dy=dy, f0=f0, c1=c1, **kwargs)
    q_exterior = enforce_boundaries_helmholtz(q_exterior, psi_bcs, beta=beta)
    # remove interior
    q_interior = q[1:-1,1:-1] - q_exterior[1:-1,1:-1]
    
    nx, ny = q.shape
    
    # do inverse
    psi_interior = inverse_elliptical_dst_solver(
        q_interior, nx=nx, ny=ny, dx=dx, dy=dy, beta=beta
    )
    
    psi = psi_bcs.at[1:-1,1:-1].set(psi_interior)
    
    return psi


def geostrophic_velocity(
    psi: Array, dx: Array, dy: Array, **kwargs
) -> tp.Tuple[Array, Array]:
    """Calculates the geostrophic velocities from the stream function.
    
    Eqn:
        u = -∂Ψ/∂y
        v =  ∂Ψ/∂x
        
    
    Args:
        psi (Array): the stream function
        dx (Array): the change in x
        dy (Array): the change in y
        **kwargs: all keyword arguments for the fdx.difference operator
    
    Return:
        u (Array): the meridonial velocty [m/s^2]
        v (Array): the zonal velocity [m/s^2]
    
    """
    
    u = - fdx.difference(psi, axis=1, step_size=dy, **kwargs)
    
    v = fdx.difference(psi, axis=0, step_size=dx, **kwargs)
    
    return u, v




