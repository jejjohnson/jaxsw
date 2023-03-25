import typing as tp
import finitediffx as fdx
import jax
import jax.numpy as jnp
from jaxtyping import Array
import functools as ft
from jaxsw._src.boundaries.functional import apply_bcs_2d
from jaxsw._src.operators.functional.padding import (
    generate_backward_padding,
    generate_central_padding,
    generate_forward_padding,
)

# TODO: Forward Different Init
# TODO: Backward Difference Init
# TODO: Mixed Difference Init
# TODO: Forward Difference

COEFFS = tp.Tuple[tp.Tuple[int], tp.Tuple[int], tp.Tuple[int, int]]

generate_finitediff_coeffs = fdx._src.utils.generate_finitediff_coeffs
generate_forward_offsets = fdx._src.utils._generate_forward_offsets
generate_central_offsets = fdx._src.utils._generate_central_offsets
generate_backward_offsets = fdx._src.utils._generate_backward_offsets
check_dims = fdx._src.utils._check_and_return


def generate_central_diff(derivative: int, accuracy: int) -> COEFFS:
    offsets = generate_central_offsets(derivative=derivative, accuracy=accuracy)
    coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
    padding = generate_central_padding(derivative=derivative, accuracy=accuracy)
    return offsets, coeffs, padding


def generate_backward_diff(derivative: int, accuracy: int):
    offsets = generate_backward_offsets(derivative=derivative, accuracy=accuracy)
    coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
    padding = generate_backward_padding(derivative=derivative, accuracy=accuracy)
    return offsets, coeffs, padding


def generate_forward_diff(derivative: int, accuracy: int):
    offsets = generate_forward_offsets(derivative=derivative, accuracy=accuracy)
    coeffs = generate_finitediff_coeffs(offsets=offsets, derivative=derivative)
    padding = generate_forward_padding(derivative=derivative, accuracy=accuracy)
    return offsets, coeffs, padding


def difference_slicing(
    x: Array, axis: int, coeffs: tp.Sequence[int], offsets: tp.Iterable[int]
) -> Array:
    size = x.shape[axis]
    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)

    x = sum(
        coeff * sliced(ioffset - offsets[0], size + (ioffset - offsets[-1]))
        for ioffset, coeff in zip(offsets, coeffs)
    )
    return x


def jacobian(
    p: Array,
    q: Array,
    dx: Array,
    dy: Array,
    bc: str = "dirichlet",
    pad: bool = True,
) -> Array:
    p = apply_bcs_2d(p, bc=bc, pad=pad)
    q = apply_bcs_2d(q, bc=bc, pad=pad)

    jac = (
        (q[2:, 1:-1] - q[:-2, 1:-1]) * (p[1:-1, 2:] - p[1:-1, :-2])
        + (q[1:-1, :-2] - q[1:-1, 2:]) * (p[2:, 1:-1] - p[:-2, 1:-1])
        + q[2:, 1:-1] * (p[2:, 2:] - p[2:, :-2])
        - q[:-2, 1:-1] * (p[:-2, 2:] - p[:-2, :-2])
        - q[1:-1, 2:] * (p[2:, 2:] - p[:-2, 2:])
        + q[1:-1, :-2] * (p[2:, :-2] - p[:-2, :-2])
        + p[1:-1, 2:] * (q[2:, 2:] - q[:-2, 2:])
        - p[1:-1, :-2] * (q[2:, :-2] - q[:-2, :-2])
        - p[2:, 1:-1] * (q[2:, 2:] - q[2:, :-2])
        + p[:-2, 1:-1] * (q[:-2, 2:] - q[:-2, :-2])
    )

    return jac / (12.0 * dx * dy)



    # det = jnp.mean(jnp.asarray([jnp.mean(dx),jnp.mean(dy)]))
    # return jac / (12. * det)


# def jacobian(f, g, dx, dy, bc: str="dirichlet", **kwargs):
#     p = apply_bcs_2d(p, bc=bc, **kwargs)
#     q = apply_bcs_2d(q, bc=bc, **kwargs)

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
