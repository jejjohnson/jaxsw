import typing as tp

import jax.numpy as jnp
import kernex as kex
from jaxtyping import Array

from jaxsw._src.domain.base import Domain
from jaxsw._src.fields.base import Field

DEFAULT_PADDING = dict(right=(0, 1), left=(1, 0), inner=(0, 0), outer=(1, 1))
OPERATION_MAP = dict(right=0, left=1, inner=2, outer=3)


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
