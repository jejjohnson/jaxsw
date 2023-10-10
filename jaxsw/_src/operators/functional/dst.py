from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array

def print_debug_quantity(quantity, name=""):
    size = quantity.shape
    min_ = jnp.min(quantity)
    max_ = jnp.max(quantity)
    mean_ = jnp.mean(quantity)
    median_ = jnp.mean(quantity)
    jax.debug.print(
        f"{name}: {size} | {min_:.6e} | {mean_:.6e} | {median_:.6e} | {max_:.6e}"
    )


def dstI1D(x, norm="ortho"):
    """1D type-I discrete sine transform."""
    num_dims = x.ndim
    N = x.shape
    padding = ((0,0),) * (num_dims-1) + ((1,1),) 
    x = jnp.pad(x, pad_width=padding, mode="constant", constant_values=0.0)
    x = jnp.fft.irfft(-1j * x, axis=-1, norm=norm)
    x = jax.lax.slice_in_dim(x, 1, N[-1]+1, axis=-1)
    return x


def dstI2D(x, norm="ortho"):
    """2D type-I discrete sine transform."""
    # print_debug_quantity(x, "X")
    x = dstI1D(x, norm=norm)
    x = jnp.transpose(x, axes=(-1,-2))
    # print_debug_quantity(x, "DST1D (1)")
    x = dstI1D(x, norm=norm)
    x = jnp.transpose(x, axes=(-1,-2))
    # print_debug_quantity(x, "DST1D (2)")
    return x

def laplacian_dst(nx, ny, dx, dy, mean: bool = True) -> Array:
    if mean:
        dx = dy = jnp.mean(jnp.asarray([dx, dy]))

    x, y = jnp.meshgrid(
        jnp.arange(1, nx+1, dtype=dx.dtype),
        jnp.arange(1, ny+1, dtype=dx.dtype),
        indexing="ij"
    )

    return (
        2 * (jnp.cos(jnp.pi / (nx+1) * x) - 1) / dx**2
        + 2 * (jnp.cos(jnp.pi / (ny+1) * y) - 1) / dy**2
    )


def helmholtz_dst(
    nx: int,
    ny: int,
    dx: Union[float, Array],
    dy: Union[float, Array],
    alpha: float = 1.0,
    beta: float = 0.0,
    mean: bool = True,
) -> Array:
    laplace_op = laplacian_dst(nx=nx, ny=ny, dx=dx, dy=dy, mean=mean)
    return alpha * laplace_op - beta


def helmholtz_fn(u, dx, dy, beta):
    d2u_dx2 = (u[...,2:,1:-1] + u[...,:-2,1:-1] - 2*u[..., 1:-1,1:-1]) / dx**2
    d2u_dy2 = (u[...,1:-1,2:] + u[...,1:-1,:-2] - 2*u[...,1:-1,1:-1]) / dy**2
    return d2u_dx2 + d2u_dy2 - beta * u[..., 1:-1,1:-1]

def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    num_dims = f.ndim
    padding = ((0,0),) * (num_dims-2) + ((1,1),(1,1)) 
    x = dstI2D(f) 
    # print_debug_quantity(x)
    x /= operator_dst
    # print_debug_quantity(x)
    return jnp.pad(dstI2D(x), pad_width=padding, mode="constant", constant_values=0.0).astype(jnp.float64)


def inverse_elliptic_dst_cmm(
    rhs: Array, H_matrix: Array,
    cap_matrices: Array, 
    bounds_xids: Array, bounds_yids: Array,
    mask: Array
) -> Array:
    

    
    # solving the inversion of rhs
    fn = lambda y, x: dstI2D(dstI2D(y) / x)
    sol_rect = jax.vmap(fn)(rhs, H_matrix)
    
    # alpha correction (?)
    alphas = jnp.einsum(
        "...ij, ...j -> ...i", 
        cap_matrices, 
        -sol_rect[..., bounds_xids, bounds_yids]
    )
    
    rhs = rhs.at[..., bounds_xids, bounds_yids].set(alphas)
    
    sol = jax.vmap(fn)(rhs, H_matrix)
    
    sol = jnp.pad(sol, pad_width=((0,0),(1,1),(1,1)), mode="constant", constant_values=0.0)
    
    return sol * mask


def compute_capacitance_matrices(
    H_matrix: Array,
    irrbound_xids: Array,
    irrbound_yids: Array,
) -> Array:
    
    irrbound_xids = irrbound_xids.astype(jnp.int32)
    irrbound_yids = irrbound_yids.astype(jnp.int32)
    
    # make sure it has layers
    assert H_matrix.ndim >= 3
    
    
    num_layers = H_matrix.shape[-3]
    num_queries_true = irrbound_xids.shape[0]
    
    # compute G matrices
    G_matrices = jnp.zeros((num_layers, num_queries_true, num_queries_true))
    rhs = jnp.zeros(H_matrix.shape[-3:])
    fn = lambda rhs, H_matrix: dstI2D(dstI2D(rhs) / H_matrix)
    
    for iquery in range(num_queries_true):
        rhs = rhs.at[:].set(0.0)

        rhs = rhs.at[..., irrbound_xids[iquery], irrbound_yids[iquery]].set(1.0)
        sol = jax.vmap(fn)(rhs, H_matrix)
        G_matrices = G_matrices.at[:,iquery].set(sol[..., irrbound_xids, irrbound_yids])
    capacitance_matrices = jnp.zeros_like(G_matrices)
    
    for ilayer in range(num_layers):
        inv_sol = jnp.linalg.inv(G_matrices[ilayer])
        capacitance_matrices = capacitance_matrices.at[ilayer].set(inv_sol)
    return capacitance_matrices

# def inverse_elliptical_dst_solver(
#     q: Array,
#     nx: int,
#     ny: int,
#     dx: Union[float, Array],
#     dy: Union[float, Array],
#     alpha: float = 1.0,
#     beta: float = 0.0,
#     mean: bool = True,
# ) -> Array:
#     """Solves the Poisson Equation
#     with Dirichlet Boundaries using the Discrete Sine
#     transform
#     """
#     assert q.shape == (nx - 2, ny - 2)

#     operator = helmholtz_dst(
#         nx=nx, ny=ny, dx=dx, dy=dy, mean=mean, alpha=alpha, beta=beta
#     )

#     # print(q.shape, operator.shape)

#     return inverse_elliptic_dst(q, operator)


