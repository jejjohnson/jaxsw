import typing as tp
import jax
import jax.numpy as jnp
import finitediffx as fdx
from jaxtyping import Array, Float


def difference(u: Array, axis: int=0, step_size: float=1.0):

#     du = fdx.difference(
#         u, step_size=step_size, axis=axis, accuracy=1, derivative=1, method="backward"
#     )

#     du = jax.lax.slice_in_dim(du, axis=axis, start_index=1, limit_index=None)

    du = jnp.diff(u, n=1, axis=axis) / step_size

    return du


def laplacian(u: Array, step_size: float=1.0):

    lap_u = fdx.laplacian(u, step_size=step_size, accuracy=1, method="backward")

    return lap_u[1:-1,1:-1]


laplacian_batch = jax.vmap(laplacian, in_axes=(0,None))

# #################
# # TESTING
# ###################
# def gradient_perp(u, dx, dy):
#     """
#     u: [Nx,Ny] -> [Nx,Ny-1]
#     v: [Nx,Ny] -> [Nx-1, Ny]
#     """
#     return (u[...,:-1] - u[...,1:]) / dy, (u[...,1:,:] - u[...,:-1,:]) / dx


# def laplacian_h(u: Array, dx, dy) -> Array:
#     return (
#         (u[...,2:,1:-1] + u[...,:-2,1:-1] - 2*u[...,1:-1,1:-1]) / dx**2 
#     + (u[...,1:-1,2:] + u[...,1:-1,:-2] - 2*u[...,1:-1,1:-1]) / dy**2
#     )