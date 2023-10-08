import jax
import jax.numpy as jnp
import typing as tp
from jaxtyping import Array
import functools as ft
from jaxsw._src.masks import VelocityMask

import einops
from jaxsw._src.domain.mask import Mask
from jaxsw._src.operators.functional.interp.upwind import (
    upwind_1pt,
    upwind_3pt,
    plusminus,
)
from jaxsw._src.operators.functional.interp.linear import (
    linear_3pts_left,
    linear_3pts_right,
    linear_2pts,
)


def tracer_flux_1pt(q: Array, u: Array, dim: int) -> Array:
    qi_left_1pt, qi_right_1pt = upwind_1pt(q=q, dim=dim)
    u_pos, u_neg = plusminus(u)
    flux = u_pos * qi_left_1pt + u_neg * qi_right_1pt
    return flux


def tracer_flux_3pt(q: Array, u: Array, dim: int, method: str = "linear") -> Array:
    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    qi_left_interior, qi_right_interior = upwind_3pt(q=q, dim=dim, method=method)

    qi_left_interior = dyn_slicer(qi_left_interior, 0, num_pts - 3)
    qi_right_interior = dyn_slicer(qi_right_interior, 1, num_pts - 3)

    # left boundary slices
    q0 = dyn_slicer(q, 0, 1)
    q1 = dyn_slicer(q, 1, 1)
    qi_left_bd = linear_2pts(q0, q1)

    # right boundary slices
    q0 = dyn_slicer(q, -1, 1)
    q1 = dyn_slicer(q, -2, 1)
    qi_right_bd = linear_2pts(q0, q1)

    # concatenate
    qi_left = jnp.concatenate([qi_left_bd, qi_left_interior, qi_right_bd], axis=dim)
    qi_right = jnp.concatenate([qi_left_bd, qi_right_interior, qi_right_bd], axis=dim)

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def tracer_flux_3pt_mask(
    q: Array,
    u: Array,
    dim: int,
    u_mask1: Array,
    u_mask2plus: Array,
    method: str = "linear",
):
    # get padding
    if dim == 0:
        pad_left = ((1, 0), (0, 0))
        pad_right = ((0, 1), (0, 0))
    elif dim == 1:
        pad_left = ((0, 0), (1, 0))
        pad_right = ((0, 0), (0, 1))
    else:
        msg = f"Dims should be between 0 and 1!"
        msg += f"\nDims: {dim}"
        raise ValueError(msg)

    # 1 point flux
    qi_left_i_1pt, qi_right_i_1pt = upwind_1pt(q=q, dim=dim)

    # 3 point flux
    qi_left_i_3pt, qi_right_i_3pt = upwind_3pt(q=q, dim=dim, method=method)

    # add padding
    qi_left_i_3pt = jnp.pad(qi_left_i_3pt, pad_width=pad_left)
    qi_right_i_3pt = jnp.pad(qi_right_i_3pt, pad_width=pad_right)

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux_1pt = u_pos * qi_left_i_1pt + u_neg * qi_right_i_1pt
    flux_3pt = u_pos * qi_left_i_3pt + u_neg * qi_right_i_3pt

    # calculate total flux
    flux = flux_1pt * u_mask1 + flux_3pt * u_mask2plus

    return flux


def tracer_flux(
    q: Array, u: Array, dim: int, num_pts: int = 1, masks: tp.Optional[Mask] = None
) -> Array:
    """Flux computation for staggered variables q and u with
    solid boundaries. Typically used for calculating the flux
    Advection Scheme:
        ∇ ⋅ (uq)

    Args:
        q (Array): tracer field to interpolate
            shape[dim] = N
        u (Array): transport velocity
            shape[dim] = N-1
        dim (int): dimension along which computations are done
        num_pts (int): the number of points for the flux computation
            options = (1, 3, 5)

    Returns:
        flux (Array): tracer flux computed on u points
            shape[dim] = N -1

    """

    # print(num_pts)
    # print(masks)

    # calculate flux
    if num_pts == 1:
        if masks is not None:
            raise NotImplementedError()
        else:
            qi_left, qi_right = tracer_flux_1pt(q=1, u=u, dim=dim)
    elif num_pts == 3:
        if masks is not None:
            qi_left, qi_right = interp_3pt(q, dim=dim)
        else:
            qi_left, qi_right = interp_3pt(q, dim=dim)
    elif num_pts == 5:
        msg = "5pt method is not implemented yet"
        raise NotImplementedError(msg)
    else:
        msg = f"Unrecognized method: {num_pts}"
        msg += "\nMust be 1, 3, or 5"
        raise ValueError(msg)

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux = u_pos * qi_left + u_neg * qi_right

    return flux
