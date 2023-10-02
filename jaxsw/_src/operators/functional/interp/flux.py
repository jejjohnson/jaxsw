import jax
import jax.numpy as jnp
import typing as tp
from jaxtyping import Array
import functools as ft
from jax.nn import relu
import einops

def plusminus(u: Array, way: int=1) -> tp.Union[Array]:
    u_pos = relu(float(way) * u)
    u_neg = u - u_pos
    return u_pos, u_neg

def interp_1pt(q: Array, dim: int) -> tp.Union[Array]:
    """creates the stencils for the upwind scheme
    - 1 pts inside domain & boundary
    """
    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    qi_left = dyn_slicer(q, 0, num_pts-1)
    qi_right = dyn_slicer(q, 1, num_pts-1)
    

    return qi_left, qi_right

def interp_3pt(q: Array, dim: int) -> tp.Union[Array]:
    """creates the stencils for the upwind scheme
    - 3 pts inside domain
    - 1 pt near boundaries
    """

    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # interior slices
    q0 = dyn_slicer(q, 0, num_pts-2)
    q1 = dyn_slicer(q, 1, num_pts-2)
    q2 = dyn_slicer(q, 2, num_pts-2)

    qi_left_interior = linear3_left(q0, q1, q2)
    qi_right_interior = linear3_right(q0, q1, q2)

    # left boundary slices
    q0 = dyn_slicer(q, 0, 1)
    q1 = dyn_slicer(q, 1, 1)
    qi_left_bd = linear2(q0, q1)

    # right boundary slices
    q0 = dyn_slicer(q, -1, 1)
    q1 = dyn_slicer(q, -2, 1)
    qi_right_bd = linear2(q0, q1)

    # concatenate each
    qi_left = jnp.concatenate([
        qi_left_bd,
        dyn_slicer(qi_left_interior, 0, num_pts-3),
        qi_right_bd
    ])

    qi_right = jnp.concatenate([
        qi_left_bd,
        dyn_slicer(qi_right_interior, 1, num_pts-3),
        qi_right_bd
    ])

    return qi_left, qi_right


def tracer_flux(u: Array, a: Array, dim: int, num_pts: int=1) -> Array:
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
    
    # calculate flux
    if num_pts == 1:
        ui_left, ui_right = interp_1pt(u, dim=dim)
    elif num_pts == 3:
        ui_left, ui_right = interp_3pt(u, dim=dim)
    elif num_pts == 5:
        msg = "5pt method is not implemented yet"
        raise NotImplementedError(msg)
    else:
        msg = "Unrecognized method: {num_pts}"
        msg +="\nMust be 1, 3, or 5"
        raise ValueError(msg)
    
    # calculate +ve and -ve points
    a_pos, a_neg = plusminus(a)
    
    # calculate upwind flux
    flux = a_pos * ui_left + a_neg * ui_right
    
    return flux