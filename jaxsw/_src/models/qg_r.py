import typing as tp
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jaxopt.linear_solve import solve_cg
from jaxtyping import Array
from einops import rearrange
import finitediffx as fdx
from jaxsw._src.operators.functional.fd import difference


R_EARTH = 6371200.0 # radius of the Earth (m)
GRAVITY = 9.80665 # gravitational acceleration (m/s^2)
OMEGA = 2.*jnp.pi/86164.0  # angular speed of the Earth (7.292e-5) (rad/s)
DEG2M = jnp.pi * R_EARTH / 180.0   # Degrees to Meters
RHO = 1.0e3    # density of water (kg/m^3)


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
    
    f0 = coriolis_param(lat_grid)

    lon_grid = jnp.deg2rad(lon_grid)
    lat_grid = jnp.deg2rad(lat_grid)

    dlon_dx, dlon_dy = jnp.gradient(lon_grid)
    dlat_dx, dlat_dy = jnp.gradient(lat_grid)


    dx = R_EARTH * jnp.hypot(dlon_dx * jnp.cos(lat_grid), dlat_dx)
    dy = R_EARTH * jnp.hypot(dlon_dy * jnp.cos(lat_grid), dlat_dy)
    
    
    return dx, dy, f0


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

def streamfn_to_pvort(psi: Array, dx: Array, dy: Array, c1: float=1.5, accuracy: int=1) -> Array:
    return fdx.laplacian(psi, step_size=(dx,dy), accuracy=accuracy) - (1./c1**2) * psi


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


def plusminus(u: Array, way: int=1) -> tp.Tuple[Array, Array]:
    u_plus = jnp.where(way * u > 0.0, u, 0.0)
    u_minus = jnp.where(way * u < 0.0, u, 0.0)
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


def pad_bc(x, bc="dirichlet", mode: str="constant"):
    
    if bc is not None:
        x = jnp.pad(x, pad_width=((1,1),(1,1)), mode=mode)
        
    if bc == "dirichlet_face":
        # faces
        x = x.at[0,:].set(0.0)
        x = x.at[-1,:].set(0.0)
        x = x.at[:,0].set(0.0)
        x = x.at[:,-1].set(0.0)

        # corners
        x = x.at[0,0].set(0.0)
        x = x.at[-1,0].set(0.0)
        x = x.at[0,-1].set(0.0)
        x = x.at[-1,-1].set(0.0)
    elif bc == "dirichlet":
        x = x.at[0,:].set(-x[1,:])
        
        x = x.at[0,:].set(-x[1,:])
        x = x.at[-1,:].set(-x[-2,:])
        x = x.at[:,0].set(-x[:,1])
        x = x.at[:,-1].set(-x[:,-2])

        # corners
        x =x.at[0,0].set(-x[0,1]   - x[1,0]   - x[1,1])
        x =x.at[-1,0].set(-x[-1,1]  - x[-2,0]  - x[-2,1])
        x =x.at[0,-1].set(-x[1,-1]  - x[0,-2]  - x[1,-2])
        x =x.at[-1,-1].set(-x[-1,-2] - x[-2,-2] - x[-2,-1])
        
    elif bc == "neumann":
        raise NotImplementedError()
    elif bc == "periodic":
        raise NotImplementedError()
    elif bc is None:
        return x
    else:
        raise ValueError(f"Unrecognized boundary condition: {bc}")
        
    return x


# def jacobian(f, g, dx, dy, bc: str="dirichlet", **kwargs):
#     f = pad_bc(f, **kwargs)
#     g = pad_bc(g, **kwargs)
    
#     dx_f = f[2:,:] - f[:-2,:]
#     dx_g = g[2:,:] - g[:-2,:]
#     dy_f = f[...,2:] - f[..., :-2]
#     dy_g = g[...,2:] - g[...,:-2]
    
    
#     jac = (
#             (   dx_f[...,1:-1] * dy_g[1:-1,:] - dx_g[...,1:-1] * dy_f[1:-1,:]  ) +
#             (   (f[2:,1:-1] * dy_g[2:,:] - f[:-2,1:-1] * dy_g[:-2,:]) -
#                 (f[1:-1,2:]  * dx_g[...,2:] - f[1:-1,:-2] * dx_g[...,:-2])     ) +
#             (   (g[1:-1,2:] * dx_f[...,2:] - g[1:-1,:-2] * dx_f[...,:-2]) -
#                 (g[2:,1:-1] * dy_f[2:,:] - g[:-2,1:-1] * dy_f[:-2,:])  )
#            ) 
    
#     return jac / (12. * dx * dy)


def jacobian(p, q, dx, dy, bc: str="dirichlet", **kwargs):
    p = pad_bc(p, **kwargs)
    q = pad_bc(q, **kwargs)
        
    jac = ((q[2:,1:-1]-q[:-2,1:-1])*(p[1:-1,2:]-p[1:-1,:-2]) \
        +(q[1:-1 ,:-2]-q[1:-1 ,2:])*(p[2:, 1:-1]-p[:-2, 1:-1 ]) \
        + q[2:, 1:-1 ]*( p[2:,2: ] - p[2:,:-2 ]) \
        - q[:-2, 1:-1]*( p[:-2,2:] - p[:-2,:-2]) \
        - q[1:-1 ,2:]*( p[2:,2: ] - p[:-2,2: ]) \
        + q[1:-1 ,:-2]*( p[2:,:-2] - p[:-2,:-2]) \
        + p[1:-1 ,2:]*( q[2:,2: ] - q[:-2,2: ]) \
        - p[1:-1 ,:-2]*( q[2:,:-2] - q[:-2,:-2]) \
        - p[2:, 1:-1 ]*( q[2:,2: ] - q[2:,:-2 ]) \
        + p[:-2, 1:-1]*( q[:-2,2:] - q[:-2,:-2]))
    
    
    return jac / (12. * dx * dy)
    # det = jnp.mean(jnp.asarray([jnp.mean(dx),jnp.mean(dy)]))
    # return jac / (12. * det)


def rhs_fn(q, psi, f0, dx, dy, way=1, upwind=True, beta: bool=True):
    
    if upwind:
        out = advection_term_upwind(q, psi, dx, dy, way=way)
    else:
        out = jacobian(psi, q, dx, dy)
        
    if beta:
        u, v = streamfn_to_velocity(psi, dx, dy) 
        out = out + beta_term(f0, dy, v)
        
    if upwind:
        out = out.at[0,:].set(0.0)
        out = out.at[-3:,:].set(0.0)
        out = out.at[:,:2].set(0.0)
        out = out.at[:,-3:].set(0.0)

    return - out
        
    
def beta_term(f0, dy, v):
        
    return beta_cdiff(f0, dy) * diff_y(v)





def pv_to_streamfn(q, dx, dy, c1=1.5, tol=1e-10, accuracy: int=1):
    
    def matvec_A(x):
        return streamfn_to_pvort(x, dx, dy, c1=c1, accuracy=accuracy)
    
    psi = solve_cg(matvec=matvec_A, b=q, init=q, tol=tol)
    
    return psi


def advection_term_upwind(q, psi, dx, dy, way):
    
    # u,v schemes
    u, v = streamfn_to_velocity(psi, dx, dy)
    
    u = diff_x(u)
    v = diff_y(v)

    u_plus, u_minus = plusminus(u, way)
    v_plus, v_minus = plusminus(v, way)
    
    # dq_dx_f = forward_diff_x(q, dx)
    # dq_dy_f = forward_diff_y(q, dy)
    # dq_dx_b = backward_diff_x(q, dx)
    # dq_dy_b = backward_diff_y(q, dy)
    
    # forward methods
    dq_dx_f = difference(q, step_size=dx, axis=0, derivative=1, accuracy=1, method="forward")
    dq_dy_f = difference(q, step_size=dy, axis=1, derivative=1, accuracy=1, method="forward")
    # backward methods
    dq_dx_b = difference(q, step_size=dx, axis=0, derivative=1, accuracy=1, method="backward")
    dq_dy_b = difference(q, step_size=dy, axis=1, derivative=1, accuracy=1, method="backward")
    
    t1 = u_plus * dq_dx_b + u_minus * dq_dx_f
    t2 = v_plus * dq_dy_b + v_minus * dq_dy_f
    
    return t1 + t2