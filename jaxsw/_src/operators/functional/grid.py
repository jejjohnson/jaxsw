import typing as tp

import jax
import jax.numpy as jnp
import kernex as kex
from jaxtyping import Array
import finitediffx as fdx

from jaxsw._src.domain.base import Domain
from jaxsw._src.fields.base import Field
from jaxsw._src.operators.functional.advection import plusminus

DEFAULT_PADDING = dict(right=(0, 1), left=(1, 0), inner=(0, 0), outer=(1, 1))
OPERATION_MAP = dict(right=0, left=1, inner=2, outer=3)


def interp(u: Array, axis: int = 0, method: str = "linear", **kwargs) -> Array:

    if method == "linear":
        return interp_linear_constant(u=u, axis=axis)
    elif method == "upwind":
        return interp_upwind_constant(u=u, axis=axis, **kwargs)
    elif method == "geometric":
        raise NotImplementedError()
    elif method == "harmonic":
        raise NotImplementedError()
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nMust be: 'linear'"
        raise ValueError(msg)


def interp_center(u: Array, method: str = "linear") -> Array:

    if method == "linear":
        return 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
    elif method == "upwind":
        raise NotImplementedError()
    elif method == "geometric":
        raise NotImplementedError()
    elif method == "harmonic":
        raise NotImplementedError()


def interp_linear_constant(u: Array, axis: int = 0) -> Array:

    if axis not in [0, 1, 2]:
        msg = f"Unrecongized axis: {axis}"
        msg += "\nAxis must be 0,1, or 2"
        raise ValueError(msg)

    u1 = jax.lax.slice_in_dim(u, None, -1, axis=axis)
    u2 = jax.lax.slice_in_dim(u, 1, None, axis=axis)
    return 0.5 * (u1 + u2)


def interp_upwind_constant(u: Array, axis: int = 0, way: int = 1) -> Array:

    u_plus, u_minus = plusminus(u)
    u_zero = jnp.where(u == 0.0, u, 0.0)

    out = (
        interp_linear_constant(u_plus, axis=axis)
        + interp_linear_constant(u_minus, axis=axis)
        + interp_linear_constant(u_zero, axis=axis)
    )

    return out


def interp_linear_irregular(u: Array, dx: Array, axis: int = 0) -> Array:

    if axis not in [0, 1, 2]:
        msg = f"Unrecongized axis: {axis}"
        msg += "\nAxis must be 0,1, or 2"
        raise ValueError(msg)

    dx1 = jax.lax.slice_in_dim(dx, None, -1, axis=axis)
    u1 = jax.lax.slice_in_dim(u, None, -1, axis=axis)
    dx2 = jax.lax.slice_in_dim(dx, 1, None, axis=axis)
    u2 = jax.lax.slice_in_dim(u, 1, None, axis=axis)

    return (dx2 * u1 + dx1 * u2) / (dx1 + dx2)


def difference(
    u: Array,
    step_size: Array = 1,
    axis: int = 0,
    method: str = "right",
    accuracy: int = 1,
    a: Array = None,
    **kwargs,
) -> Array:

    if method == "right":
        u = fdx.difference(
            u, step_size=step_size, axis=axis, method="backward", accuracy=accuracy
        )
        u = jax.lax.slice_in_dim(u, 1, None, axis=axis)
        return u
    elif method == "left":
        u = fdx.difference(
            u, step_size=step_size, axis=axis, method="forward", accuracy=accuracy
        )
        u = jax.lax.slice_in_dim(u, None, -1, axis=axis)
        return u
    elif method == "inner":
        u = fdx.difference(
            u, step_size=step_size, axis=axis, method="central", accuracy=accuracy
        )
        u = jax.lax.slice_in_dim(u, 1, -1, axis=axis)
        return u
    elif method == "upwind":

        # get plus-minus
        u_plus, u_minus = plusminus(a if a is not None else u)

        du_b = difference(u, step_size=step_size, axis=axis, method="left")
        du_f = difference(u, step_size=step_size, axis=axis, method="right")

        u_minus = jax.lax.slice_in_dim(u_minus, None, -1, axis=axis)
        u_plus = jax.lax.slice_in_dim(u_plus, 1, None, axis=axis)
        # u_minus = jax.lax.slice_in_dim(u_minus, -1, None, axis=axis)
        # u_plus = jax.lax.slice_in_dim(u_plus, None, -1, axis=axis)

        return du_f * u_minus + du_b * u_plus
    else:
        msg = f"Unrecognized method: {method}"
        msg += "\nNeeds to be: 'forward', 'backward', 'central'."
        raise ValueError(msg)


def x_average_1D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        + -- ⋅ -- +
        u -- u̅ -- u
        + -- ⋅ -- +

    Args:
        u (Array): the field [Nx,]

    Returns:
        ubar (Array): the field averaged [Nx-1,]

    """

    @kex.kmap(kernel_size=(2,), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def x_difference_1D(
    u: Array, step_size: Array, padding: tp.Optional[tp.Tuple] = "valid"
) -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        + -- ⋅ -- +
        u -- u̅ -- u
        + -- ⋅ -- +

    Args:
        u (Array): the field [Nx,]

    Returns:
        ubar (Array): the field averaged [Nx-1,]

    """

    @kex.kmap(kernel_size=(2,), padding=padding)
    def kernel_fn(u):
        return u[0] - u[-1]

    return kernel_fn(u) / step_size


def x_average_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- u̅ -- u
        |         |
        ⋅         ⋅
        |         |
        u -- u̅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny]

    """

    @kex.kmap(kernel_size=(2, 1), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def x_interp_linear_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- u̅ -- u
        |         |
        ⋅         ⋅
        |         |
        u -- u̅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny]

    """
    return 0.5 * (u[:-1] + u[1:])


def x_difference_2D(
    u: Array, step_size: Array, padding: tp.Optional[tp.Tuple] = "valid"
) -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- u̅ -- u
        |         |
        ⋅         ⋅
        |         |
        u -- u̅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny]

    """

    @kex.kmap(kernel_size=(2, 1), padding=padding)
    def kernel_fn(u):
        return u[1, 0] - u[0, 0]

    return kernel_fn(u) / step_size


def x_difference_2D_(
    u: Array, step_size: Array, padding: tp.Optional[tp.Tuple] = "valid"
) -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- u̅ -- u
        |         |
        ⋅         ⋅
        |         |
        u -- u̅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny]

    """
    return (u[1:] - u[:-1]) / step_size


def y_average_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        u̅         u̅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx, Ny-1]

    """

    @kex.kmap(kernel_size=(1, 2), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def y_interp_linear_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        u̅         u̅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx, Ny-1]

    """
    return 0.5 * (u[:, :-1] + u[:, 1:])


def y_difference_2D(
    u: Array, step_size: Array, padding: tp.Optional[tp.Tuple] = "valid"
) -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        u̅         u̅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx, Ny-1]

    """

    @kex.kmap(kernel_size=(1, 2), padding=padding)
    def kernel_fn(u):
        return u[0, 1] - u[0, 0]

    return kernel_fn(u) / step_size


def y_difference_2D_(
    u: Array, step_size: Array, padding: tp.Optional[tp.Tuple] = "valid"
) -> Array:
    """Returns the two-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        u̅         u̅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx, Ny-1]

    """
    return (u[:, 1:] - u[:, :-1]) / step_size


def center_average_2D(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the four-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        ⋅    u̅    ⋅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny-1]

    """

    @kex.kmap(kernel_size=(2, 2), padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def center_average_2D_(u: Array, padding: tp.Optional[tp.Tuple] = "valid") -> Array:
    """Returns the four-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        ⋅    u̅    ⋅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny-1]

    """

    return 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])


def kernel_average(
    u: Array, kernel_size: tp.Tuple, padding: tp.Optional[tp.Tuple] = "valid"
) -> Array:
    """Returns the four-point average at the centres between grid points.

    Grid:
        u -- ⋅ -- u
        |         |
        ⋅    u̅    ⋅
        |         |
        u -- ⋅ -- u

    Args:
        u (Array): the field [Nx,Ny]

    Returns:
        ubar (Array): the field averaged [Nx-1, Ny-1]

    """

    @kex.kmap(kernel_size=kernel_size, padding=padding)
    def kernel_fn(u):
        return jnp.mean(u)

    return kernel_fn(u)


def get_kernel_size(operations):
    msg = "too few/many operations. Need [1,3]"
    msg += f"\n{operations}"
    assert len(operations) <= 3 and len(operations) >= 1, msg

    if len(operations) == 1:
        if operations[0] is None:
            kernel_size = (1,)
        else:
            kernel_size = (2,)

    elif len(operations) == 2:
        x_op, y_op = operations
        do_something = ["right", "left", "inner", "outer"]
        case1 = x_op in do_something and y_op in do_something
        case2 = x_op is None and y_op in do_something
        case3 = x_op in do_something and y_op is None
        case4 = x_op is None and y_op is None

        if case1:
            kernel_size = (2, 2)
        elif case2:
            kernel_size = (1, 2)
        elif case3:
            kernel_size = (2, 1)
        elif case4:
            kernel_size = (1, 1)
        else:
            msg = "Unrecognized operations"
            msg += f"\nx - {x_op}"
            msg += f"\ny - {y_op}"
            raise ValueError(msg)

    elif len(operations) == 3:
        raise ValueError("Not Implemented")

    return kernel_size


def _field_operation_1D(u, operations, padding):
    kernel_size = get_kernel_size(operations)
    # print(kernel_size)
    # print(operations, padding, kernel_size)

    # pad values
    # print(u)
    u = jnp.pad(u, pad_width=padding, mode="edge")
    # print(u)

    # change values via the averaging
    u = kernel_average(u, kernel_size=kernel_size, padding="valid")

    return u


def _domain_operation_1D(xmin, xmax, dx, operation):
    if operation is None:
        pass
    elif operation == "right":
        xmin += dx * 0.5
        xmax += dx * 0.5
    elif operation == "left":
        xmin -= dx * 0.5
        xmax -= dx * 0.5
    elif operation == "inner":
        xmin += dx * 0.5
        xmax -= dx * 0.5
    elif operation == "outer":
        xmin -= dx * 0.5
        xmax += dx * 0.5
    else:
        msg = "Unrecognized argument for operation"
        msg += f"\noperation: {operation}"
        raise ValueError(msg)

    return xmin, xmax


def grid_operator(u: Field, operations: tp.Iterable = None):
    assert len(u.domain.dx) == len(operations)

    pads = list()

    pads = [
        DEFAULT_PADDING[iop] if iop in list(DEFAULT_PADDING.keys()) else (0, 0)
        for iop in operations
    ]

    if len(u.domain.dx) == 1:
        # operate on field values
        u_values = _field_operation_1D(u.values.squeeze(), operations, pads)
        # operate on domain
        xmin, xmax = _domain_operation_1D(
            u.domain.xmin[0], u.domain.xmax[0], u.domain.dx[0], operations[0]
        )
        # create field
        print(u_values.shape)
        return Field(u_values, Domain(xmin=(xmin,), xmax=(xmax,), dx=u.domain.dx))

    else:
        # len(u.domain.dx) == 2:
        # operate on field values
        u_values = _field_operation_1D(u.values.squeeze(), operations, pads)
        # operate on domain
        # print(u.domain.xmin, u.domain.xmax, u.domain.dx, operations)
        out = [
            _domain_operation_1D(xmin, xmax, dx, iop)
            for xmin, xmax, dx, iop in zip(
                u.domain.xmin, u.domain.xmax, u.domain.dx, operations
            )
        ]
        xmin, xmax = zip(*out)
        #         vmap_fn = jax.vmap(_domain_operation_1D, in_axes=(0,0,0,0))

        #         xmin, xmax = vmap_fn(
        #             jnp.asarray(u.domain.xmin),
        #             jnp.asarray(u.domain.xmax),
        #             jnp.asarray(u.domain.dx),
        #             jnp.asarray(operations))
        # print(xmin,xmax)
        return Field(u_values, Domain(xmin=xmin, xmax=xmax, dx=u.domain.dx))
