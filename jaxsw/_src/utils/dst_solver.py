from typing import Optional
import jax.numpy as jnp
from jaxtyping import Array


def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return jnp.fft.irfft(-1j*jnp.pad(x, (1,1)), axis=-1, norm=norm)[1:x.shape[0]+1,1:x.shape[1]+1]


def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).T, norm=norm).T

def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    return dstI2D(dstI2D(f) / operator_dst)

def laplacian_dist(nx, ny, dx, dy, mean: bool=True) -> Array:
    
    if mean:
        dx = dy = jnp.mean(jnp.asarray([dx, dy]))
    
    
    x, y = jnp.meshgrid(jnp.arange(1,nx-1,dtype='float64'),
                       jnp.arange(1,ny-1,dtype='float64'))

    return 2 * (jnp.cos(jnp.pi / (nx -1) * x) - 1) / dx ** 2 + \
           2 * (jnp.cos(jnp.pi / (ny -1) * y) - 1) / dy ** 2


def helmholtz_dist(dx, dy, kappa: float=0.0):
    return laplacian_dist(dx, dy) - kappa

def inverse_elliptical_dst_solver(q, nx, ny, dx, dy, kappa: Optional[float]=None, mean:bool=True):
    """Solves the Poisson Equation
    with Dirichlet Boundaries using the Discrete Sine
    transform
    """
    out = jnp.zeros_like(q)
    
    operator = laplacian_dist(nx, ny, dx, dy, mean=mean)
    
    if kappa is not None:
        operator = operator - kappa
    
    return inverse_elliptic_dst(q, operator)

