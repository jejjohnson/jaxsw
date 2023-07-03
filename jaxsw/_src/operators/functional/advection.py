import typing as tp

import finitediffx as fdx
import jax.numpy as jnp
from finitediffx._src.utils import _check_and_return
from jaxtyping import Array


def advection_1D(
    u: Array,
    a: Array,
    step_size: Array,
    axis: int = 0,
    method: str = "backward",
    accuracy: int = 1,
):
    """simple 1D advection scheme using backwards finite
    difference.

        Advection = a ∂u/∂x

    Args:
        u (Array): the field
        a (Array): the field or constant
        step_size (Array): the stepsize for the FD scheme
        axis (int, optional): the axis to operate the FD. Defaults to 0.
        method (str, optional): the method for FD. Defaults to "backward".
        accuracy (int, optional): the accuracy for the FD scheme. Defaults to 1.

    Returns:
        Array: the RHS for the advection term
    """
    du_dx = fdx.difference(
        u,
        axis=axis,
        method=method,
        accuracy=accuracy,
        step_size=step_size,
        derivative=1,
    )

    return a * du_dx


def advection_2D(
    u: Array,
    a: Array,
    b: Array,
    step_size: Array,
    method: str = "backward",
    accuracy: int = 1,
):
    """simple 2D advection scheme using backwards finite
    difference.

        Advection = a ∂u/∂x + b ∂u/∂y

    Args:
        u (Array): the field
        a (Array): the field or constant
        step_size (Array): the stepsize for the FD scheme
        method (str, optional): the method for FD. Defaults to "backward".
        accuracy (int, optional): the accuracy for the FD scheme. Defaults to 1.

    Returns:
        Array: the RHS for the advection term
    """

    u_grad = fdx.gradient(u, method=method, accuracy=accuracy, step_size=step_size)

    return a * u_grad[0] + b * u_grad[1]


def plusminus(
    u: Array, way: int = 1, fn: tp.Optional[tp.Callable] = None
) -> tp.Tuple[Array, Array]:
    """Plus Minus Scheme
    It returns the + and - for the array whereby
    "plus" has all values less than zero equal to zero
    and "minus" has all values greater than zero equal to zero.
    This is useful for advection schemes

    Args:
        u (Array): the input field
        way (int): chooses which "way" (default=1)

    Returns:
        plus (Array): the postive values
        minus (Array): the negative values
    """
    if fn is None:
        u_plus = jnp.where(way * u < 0.0, 0.0, u)
        u_minus = jnp.where(way * u > 0.0, 0.0, u)
    else:
        u_plus = fn(way * u)
        u_minus = -1.0 * fn(-1.0 * way * u)
    return u_plus, u_minus


def advection_upwind_1D(
    u: Array,
    a: Array,
    step_size: Array,
    way: int = 1,
    axis: int = 0,
    accuracy: int = 1,
    fn: tp.Optional[tp.Callable] = None,
) -> Array:
    """Calculates an advection term using and upwind scheme.
    1. We use a cell-centered average for the factor, a
    2. We clamp the negative values and positive values for the
        factor, a
    3. We do the backward and foward difference for the field, u

    Eqn:

        a ∂u/∂x := a̅⁺ D₋[∂u/∂x] + a̅⁻ D₊[∂u/∂x]

    where:
        * a̅ : cell-centered average term
        * a⁺, a⁻: clamped values that are < 0.0 and > 0.0 respectively
        * D₋, D₊: backwards and forwards finite difference schemes

    Args:
        u (Array): the field
        a (Array): the multiplicative factor on the field
        step_size (Array): the step size for the field
        way (int): the direction
        axis (int): the axis for the 1D, default=0
        accuracy (int): the accuracy of the method
        fn (Callable): optional method for the way method, default=None

    Returns:
        u (Array): the field
    """

    a_plus, a_minus = plusminus(a, way=way, fn=fn)

    u_rhs_forward = advection_1D(
        u=u,
        a=a_minus,
        axis=axis,
        step_size=step_size,
        method="forward",
        accuracy=accuracy,
    )

    u_rhs_backward = advection_1D(
        u=u,
        a=a_plus,
        axis=axis,
        step_size=step_size,
        method="backward",
        accuracy=accuracy,
    )

    return u_rhs_backward + u_rhs_forward


def advection_upwind_2D(
    u: Array,
    a: Array,
    b: Array,
    step_size: Array,
    way: int = 1,
    accuracy: int = 1,
    fn: tp.Optional[tp.Callable] = None,
) -> Array:
    """Calculates an advection term using and upwind scheme.
    1. We use a cell-centered average for the factor, a
    2. We clamp the negative values and positive values for the
        factor, a
    3. We do the backward and foward difference for the field, u

    Eqn:
        Advection := a ∂u/∂x + b ∂u/∂y
        a ∂u/∂x := a̅⁺ D₋[∂u/∂x] + a̅⁻ D₊[∂u/∂x]
        b ∂u/∂y := b⁺ D₋[∂u/∂y] + b⁻ D₊[∂u/∂y]

    where:
        * a̅ : cell-centered average term
        * a⁺, a⁻: clamped values that are < 0.0 and > 0.0 respectively
        * D₋, D₊: backwards and forwards finite difference schemes

    Args:
        u (Array): the field
        a (Array): the multiplicative factor on the field
        b (Array): the multiplicative factor on the field
        step_size (Array): the step size for the field
        way (int): the direction
        axis (int): the axis for the 1D, default=0
        accuracy (int): the accuracy of the method
        fn (Callable): optional method for the way method, default=None

    Returns:
        u (Array): the field
    """

    step_size = _check_and_return(value=step_size, ndim=2, name="accuracy")

    a_du_dx = advection_upwind_1D(
        u=u,
        a=a,
        axis=0,
        way=way,
        step_size=step_size[0],
        accuracy=accuracy,
        fn=fn,
    )

    b_du_dy = advection_upwind_1D(
        u=u,
        a=b,
        axis=1,
        way=way,
        step_size=step_size[1],
        accuracy=accuracy,
        fn=fn,
    )

    return a_du_dx + b_du_dy
