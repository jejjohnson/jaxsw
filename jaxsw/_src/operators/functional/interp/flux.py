import jax
import jax.numpy as jnp
import typing as tp
from jaxtyping import Array
import functools as ft
from jaxsw._src.masks import VelocityMask

import einops
from jaxsw._src.masks import Mask
from jaxsw._src.operators.functional.interp.upwind import (
    upwind_1pt,
    upwind_1pt_bnds,
    upwind_2pt_bnds,
    upwind_3pt,
    upwind_3pt_bnds,
    upwind_5pt,
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


def tracer_flux_1pt_mask(q: Array, u: Array, dim: int, u_mask1: Array) -> Array:
    return u_mask1 * tracer_flux_1pt(q=q, u=u, dim=dim)


def tracer_flux_3pt(q: Array, u: Array, dim: int, method: str = "linear") -> Array:
    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    qi_left_interior, qi_right_interior = upwind_3pt(q=q, dim=dim, method=method)

    qi_left_interior = dyn_slicer(qi_left_interior, 0, num_pts - 3)
    qi_right_interior = dyn_slicer(qi_right_interior, 1, num_pts - 3)

    # left-right boundary slices (linear only!)
    qi_left_bd, qi_right_bd = upwind_2pt_bnds(q=q, dim=dim, method="linear")

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


def tracer_flux_5pt(q: Array, u: Array, dim: int, method: str = "linear") -> Array:
    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # 5-pts inside domain
    qi_left_interior, qi_right_interior = upwind_5pt(q=q, dim=dim, method=method)

    # 3pts-near boundary
    qi_left_b, qi_right_b = upwind_3pt_bnds(q, dim=dim, method=method)

    qi_left_b0 = dyn_slicer(qi_left_b, 0, 1)
    qi_left_m = dyn_slicer(qi_left_b, -1, 1)

    qi_right_0 = dyn_slicer(qi_right_b, 0, 1)
    qi_right_bm = dyn_slicer(qi_right_b, -1, 1)

    # 1pts at end-points
    qi_left_0, qi_right_m = upwind_1pt_bnds(q=q, dim=dim)

    # concatenate
    qi_left = jnp.concatenate(
        [qi_left_0, qi_left_b0, qi_left_interior, qi_left_m], axis=dim
    )
    qi_right = jnp.concatenate(
        [qi_right_0, qi_right_interior, qi_right_bm, qi_right_m], axis=dim
    )

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def tracer_flux_5pt_mask(
    q: Array,
    u: Array,
    dim: int,
    u_mask1: Array,
    u_mask2: Array,
    u_mask3plus: Array,
    method: str = "linear",
):
    """Tasks - ++"""
    # get padding
    if dim == 0:
        pad_left_3pt = ((0, 1), (0, 0))
        pad_right_3pt = ((1, 0), (0, 0))
        pad_left_5pt = ((1, 2), (0, 0))
        pad_right_5pt = ((2, 1), (0, 0))
    elif dim == 1:
        pad_left_3pt = ((0, 0), (0, 1))
        pad_right_3pt = ((0, 0), (1, 0))
        pad_left_5pt = ((0, 0), (1, 2))
        pad_right_5pt = ((0, 0), (2, 1))
    else:
        msg = f"Dims should be between 0 and 1!"
        msg += f"\nDims: {dim}"
        raise ValueError(msg)

    # 1 point flux
    qi_left_i_1pt, qi_right_i_1pt = upwind_1pt(q=q, dim=dim)

    # 3 point flux
    qi_left_i_3pt, qi_right_i_3pt = upwind_3pt(q=q, dim=dim, method=method)

    # add padding
    qi_left_i_3pt = jnp.pad(qi_left_i_3pt, pad_width=pad_left_3pt)
    qi_right_i_3pt = jnp.pad(qi_right_i_3pt, pad_width=pad_right_3pt)

    # 5 point flux
    qi_left_i_5pt, qi_right_i_5pt = upwind_5pt(q=q, dim=dim, method=method)

    # add padding
    qi_left_i_5pt = jnp.pad(qi_left_i_5pt, pad_width=pad_left_5pt)
    qi_right_i_5pt = jnp.pad(qi_right_i_5pt, pad_width=pad_right_5pt)

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux_1pt = u_pos * qi_left_i_1pt + u_neg * qi_right_i_1pt
    flux_3pt = u_pos * qi_left_i_3pt + u_neg * qi_right_i_3pt
    flux_5pt = u_pos * qi_left_i_5pt + u_neg * qi_right_i_5pt

    # calculate total flux
    flux = flux_1pt * u_mask1 + flux_3pt * u_mask2 + flux_5pt * u_mask3plus

    return flux


def tracer_flux(
    q: Array, u: Array, dim: int, num_pts: int = 5, method: str = "wenoz", **kwargs
) -> Array:
    return None
