import typing as tp
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxopt.linear_solve import solve_cg
from jaxtyping import Array
from einops import rearrange
import finitediffx as fdx
from jaxsw._src.utils.constants import R_EARTH, OMEGA, GRAVITY
from jaxsw._src.operators.functional.fd import jacobian


def ekman_number(AV, f0, H):
    """
    Args:
        Av (Array): the vertical eddy viscosity (m^2/s)
        H (Array): the depth (m)
        f0 (Array): Coriolis parameter
    
    Returns:
        kappa (Array): Ekman number (delta/H)
    """
    return jnp.sqrt(Av * f0 / 2.) / H


def boundary_layer_width(kappa, beta):
    """(m)"""
    return kappa / beta

def beta_plane(lat: Array) -> Array:
    lat = jnp.deg2rad(lat)
    return 2 * OMEGA * jnp.cos(lat) / R_EARTH


def coriolis_param(lat: Array) -> Array:
    lat = jnp.deg2rad(lat)
    return 2. * OMEGA * jnp.sin(lat) 


def lat_lon_deltas(lon: Array, lat: Array) -> tp.Tuple[Array, Array]:
    
    lon_grid, lat_grid = jnp.meshgrid(lon, lat, indexing="ij")
    
    f = coriolis_param(lat_grid)
    
    beta = beta_plane(lat_grid) 

    lon_grid = jnp.deg2rad(lon_grid)
    lat_grid = jnp.deg2rad(lat_grid)

    dlon_dx, dlon_dy = jnp.gradient(lon_grid)
    dlat_dx, dlat_dy = jnp.gradient(lat_grid)


    dx = R_EARTH * jnp.hypot(dlon_dx * jnp.cos(lat_grid), dlat_dx)
    dy = R_EARTH * jnp.hypot(dlon_dy * jnp.cos(lat_grid), dlat_dy)
    
    
    return dx, dy, f, beta


def beta_cdiff(f0: Array, dy: Array) -> Array:
    kernel_y = jnp.array([[-1, 0, 1]])
    kernel_y = rearrange(kernel_y, "1 K -> K 1")
    out =  jsp.signal.convolve(f0, kernel_y, mode="same")
    return out / (2 * dy)


def ssh_to_streamfn(ssh: Array, f0: Array, g=GRAVITY) -> Array:
    return (g / f0) * ssh


def streamfn_to_ssh(psi: Array, f0: Array, g=GRAVITY) -> Array:
    return (f0 / g) * psi


def laplacian_2d(array: Array, dx: Array, dy: Array) -> Array:
    lap_kernel_x = jnp.array([[1., -2., 1.]])
    lap_kernel_y = rearrange(lap_kernel_x, "1 K -> K 1")
    d2f_dx2 = jnp.pad(array, pad_width=((0,0),(1,1)), mode="constant")
    d2f_dx2 = jsp.signal.convolve(d2f_dx2, lap_kernel_x, mode="valid") / dx**2
    
    d2f_dy2 = jnp.pad(array, pad_width=((1,1),(0,0)), mode="constant") 
    d2f_dy2 = jsp.signal.convolve(d2f_dy2, lap_kernel_y, mode="valid") / dy**2
    
    return d2f_dx2 + d2f_dy2


# def streamfn_to_pvort(psi: Array, dx: Array, dy: Array, c1: float=1.5) -> Array:
#     return laplacian_2d(psi, dx, dy) - (1./c1**2) * psi

def streamfn_to_pvort(
    psi: Array, 
    dx: Array, 
    dy: Array, 
    c1: float=2.7, 
    f0: float=1e-5, 
    **kwargs
) -> Array:
    return fdx.laplacian(psi, step_size=(dx,dy), **kwargs) - (f0/c1)**2 * psi


def sobel_diff_x(u: Array, dx: Array) -> Array:
    kernel = jnp.asarray([[-0.25,-0.25],[0,0],[0.25,0.25]])
    u = jnp.pad(u, pad_width=((1,1),(1,0)), mode="constant")
    out = jsp.signal.convolve(u, kernel, mode="valid")
    return out / dx


def sobel_diff_y(u: Array, dy: Array) -> Array:
    kernel = jnp.asarray([[-0.25,-0.25],[0,0],[0.25,0.25]])
    kernel = rearrange(kernel, "H W -> W H")
    u = jnp.pad(u, pad_width=((1,0),(1,1)), mode="constant")
    out = jsp.signal.convolve(u, kernel, mode="valid")
    return out / dy

def streamfn_to_velocity(psi, dx, dy):
    
    v = sobel_diff_x(psi, dx)
    
    u = - sobel_diff_y(psi, dy)
    
    return u, v


def diff_x(u: Array) -> Array:
    kernel = jnp.array([[0.5,0.5]])
    u = jnp.pad(u, pad_width=((0,0),(0,1)), mode="constant")
    u = jsp.signal.convolve(u, kernel, mode="valid")
    return u


def diff_y(u: Array) -> Array:
    kernel = jnp.array([[0.5,0.5]])
    kernel = rearrange(kernel, "1 K -> K 1")
    u = jnp.pad(u, pad_width=((0,1),(0,0)), mode="constant")
    u = jsp.signal.convolve(u, kernel, mode="valid")
    return u

def u_plusminus(u: Array, way: int=1) -> tp.Tuple[Array,Array]:
    
    unew = jnp.zeros_like(u)
    u_avg = way*0.5*(u[2:-2,2:-2]+u[2:-2,3:-1])
    uplus, uminus = plusminus(u_avg)
    return uplus, uminus
    # u = diff_x(u)
    # return plusminus(u, way=way)

def v_plusminus(v: Array, way: int=1) -> tp.Tuple[Array,Array]:
    v_avg = way*0.5*(v[2:-2,2:-2]+v[3:-1,2:-2])
    vplus, vminus = plusminus(v_avg)
    return vplus, vminus
    # v = diff_y(v)
    # return plusminus(v, way=way)

def plusminus(u: Array) -> tp.Tuple[Array, Array]:
    
    # u_plus = jnp.where(u < 0.0, 0.0, u)
    # u_minus = jnp.where(u > 0.0, 0.0, u)
    u_plus = jax.nn.relu(u)
    u_minus = - 1. * jax.nn.relu(- 1. * u)
    return u_plus, u_minus


def forward_diff_x(u: Array, dx: Array) -> Array:
    kernel_x = jnp.array([[1/6, -1, 1/2, 1/3]])
    u = jnp.pad(u, pad_width=((0,0),(0,3)), mode="constant")
    u = jsp.signal.convolve(u, kernel_x, mode="valid") 
    return u / dx 
    

def forward_diff_y(u: Array, dy: Array) -> Array:
    kernel_y = jnp.array([[1/6, -1, 1/2, 1/3]])
    kernel_y = rearrange(kernel_y, "1 K -> K 1")
    u = jnp.pad(u, pad_width=((0,3),(0,0)), mode="constant")
    u = jsp.signal.convolve(u, kernel_y, mode="valid")
    return u / dy
    
    
def backward_diff_x(u: Array, dx: Array) -> Array:
    kernel_x = jnp.array([[1/3, 1/2, -1, 1/6]])
    u = jnp.pad(u, pad_width=((0,0),(3,0)), mode="constant")
    u = jsp.signal.convolve(u, kernel_x, mode="valid") 
    return u / dx



def backward_diff_y(u: Array, dy: Array) -> Array:
    kernel_y = jnp.array([[1/3, 1/2, -1, 1/6]])
    kernel_y = rearrange(kernel_y, "1 K -> K 1")
    u = jnp.pad(u, pad_width=((3,0),(0,0)), mode="constant")
    u = jsp.signal.convolve(u, kernel_y, mode="valid")
    return u / dy


def rhs_fn(q, psi, f0, dx, dy, way=1, bc="dirichlet", upwind=True, beta: bool=True):
    
    if upwind:
        out = advection_term_upwind(q, psi, dx, dy, way=way)
    else:
        out = jacobian(p=psi, q=q, dx=dx, dy=dy, bc=bc, pad=True)
        
    if beta:
        u, v = streamfn_to_velocity(psi, dx, dy) 
        out = out + beta_term(f0, dy, v)
        
    if upwind:
        out = out.at[0,:].set(0.0)
        out = out.at[-3:,:].set(0.0)
        out = out.at[:,:2].set(0.0)
        out = out.at[:,-3:].set(0.0)

    return - out
        
    
def beta_term(psi, f, dx, dy):
    
    _, v = streamfn_to_velocity(psi, dx, dy) 
    
    dv_dy = fdx.difference(
        v, axis=1, step_size=dy, 
        accuracy=1, method="central", 
        derivative=1
    )
    
    df_dy = fdx.difference(
        f, step_size=dy,
        accuracy=1, method="central",
        derivative=1
    )
    
    return df_dy * v





def pv_to_streamfn(q, dx, dy, c1=1.5, tol=1e-10, accuracy: int=1):
    
    def matvec_A(x):
        return streamfn_to_pvort(x, dx, dy, c1=c1, accuracy=accuracy)
    
    psi = solve_cg(matvec=matvec_A, b=q, init=q, tol=tol)
    
    return psi


def advection_term_upwind(q, psi, dx, dy, way: int=1):
    
    # u,v schemes
    u, v = streamfn_to_velocity(psi, dx, dy)
    
    term = jnp.zeros_like(q)
    
    v_avg = way * 0.25 * (v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[:-2, 2:])
    v_plus, v_minus = plusminus(v_avg)
    
    u_avg = way * 0.25 * (u[1:-1, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[:-2, 2:])
    u_plus, u_minus = plusminus(u_avg)
    
    # forward methods
    dq_dx_f = fdx.difference(q, step_size=dx, axis=0, derivative=1, accuracy=1, method="forward")
    dq_dy_f = fdx.difference(q, step_size=dy, axis=1, derivative=1, accuracy=1, method="forward")
    # backward methods
    dq_dx_b = fdx.difference(q, step_size=dx, axis=0, derivative=1, accuracy=1, method="backward")
    dq_dy_b = fdx.difference(q, step_size=dy, axis=1, derivative=1, accuracy=1, method="backward")
    
    t1 = u_plus * dq_dx_b[1:-1,1:-1] + u_minus * dq_dx_f[1:-1,1:-1]
    t2 = v_plus * dq_dy_b[1:-1,1:-1] + v_minus * dq_dy_f[1:-1,1:-1]
    
    term = term.at[1:-1,1:-1].set(t1 + t2)
    
    return term