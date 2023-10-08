import typing as tp
import functools as ft
from jaxtyping import Array
import jax
import jax.numpy as jnp
from jax.nn import relu
from jaxsw._src.operators.functional.interp.weno import weno_3pts, weno_3pts_improved
from jaxsw._src.operators.functional.interp.linear import (
    linear_2pts,
    linear_3pts_left,
    linear_3pts_right,
)


def plusminus(u: Array, way: int = 1) -> tp.Tuple[Array, Array]:
    u_pos = relu(float(way) * u)
    u_neg = u - u_pos
    return u_pos, u_neg


def upwind_1pt(q: Array, dim: int) -> tp.Tuple[Array, Array]:
    """creates the stencils for the upwind scheme
    - 1 pts inside domain & boundary
    Args:
        q (Array): input tracer
            shape[dim] = N

    Return:
        qi_left (Array): output tracer left size
            shape[dim] = N-1
        qi_right (Array): output tracer left size
            shape[dim] = N-1
    """
    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    qi_left = dyn_slicer(q, 0, num_pts - 1)
    qi_right = dyn_slicer(q, 1, num_pts - 1)

    return qi_left, qi_right


# def upwind_3pt(q: Array, dim: int, method: str = "weno") -> tp.Tuple[Array]:
#     """creates the stencils for the upwind scheme
#     - 3 pts inside domain
#     - 1 pt near boundaries
#     """

#     # get number of points
#     num_pts = q.shape[dim]

#     # define slicers
#     dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

#     # interior slices
#     q0 = dyn_slicer(q, 0, num_pts - 2)
#     q1 = dyn_slicer(q, 1, num_pts - 2)
#     q2 = dyn_slicer(q, 2, num_pts - 2)

#     if method == "linear":
#         qi_left_interior = linear_3pts_left(q0, q1, q2)
#         qi_right_interior = linear_3pts_right(q0, q1, q2)
#     elif method == "weno":
#         pass
#     elif method == "wenoz":
#         pass
#     else:
#         msg = f"Unrecognized method: {method}"
#         msg += "\nNeeds to be 'linear', 'weno', or 'wenoz'."
#         raise ValueError(msg)

#     # left boundary slices
#     q0 = dyn_slicer(q, 0, 1)
#     q1 = dyn_slicer(q, 1, 1)
#     qi_left_bd = linear_2pts(q0, q1)

#     # right boundary slices
#     q0 = dyn_slicer(q, -1, 1)
#     q1 = dyn_slicer(q, -2, 1)
#     qi_right_bd = linear_2pts(q0, q1)

#     # concatenate each
#     qi_left = jnp.concatenate(
#         [qi_left_bd, dyn_slicer(qi_left_interior, 0, num_pts - 3), qi_right_bd]
#     )

#     qi_right = jnp.concatenate(
#         [qi_left_bd, dyn_slicer(qi_right_interior, 1, num_pts - 3), qi_right_bd]
#     )

#     return qi_left, qi_right


def upwind_3pt(q: Array, dim: int, method: str = "weno") -> tp.Tuple[Array, Array]:
    """creates the stencils for the upwind scheme
    - 3 pts inside domain
    - 1 pt near boundaries
    Args:
        q (Array):
            Size = [Nx,Ny]
        dim (int): ONLY 0 or 1!
    """

    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # interior slices
    q0 = dyn_slicer(q, 0, num_pts - 2)
    q1 = dyn_slicer(q, 1, num_pts - 2)
    q2 = dyn_slicer(q, 2, num_pts - 2)

    # DO WENO Interpolation
    if method == "linear":
        qi_left_interior = linear_3pts_left(q0, q1, q2)
        qi_right_interior = linear_3pts_left(q2, q1, q0)
    elif method == "weno":
        qi_left_interior = weno_3pts(q0, q1, q2)
        qi_right_interior = weno_3pts(q2, q1, q0)
    elif method == "wenoz":
        qi_left_interior = weno_3pts_improved(q0, q1, q2)
        qi_right_interior = weno_3pts_improved(q2, q1, q0)
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be 'linear', 'weno', or 'wenoz'."
        raise ValueError(msg)

    return qi_left_interior, qi_right_interior
