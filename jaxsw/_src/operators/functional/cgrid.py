import typing as tp
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxsw._src.domain.base import Domain
from jaxsw._src.domain import base_v2 as domain_utils
from jaxsw._src.fields.base import Field
from jaxsw._src.operators.functional import grid as F_grid
import finitediffx as fdx


DIRECTIONS = {
    "right": (1.0, 1.0),
    "left": (-1.0, -1.0),
    "inner": (1.0, -1.0),
    "outer": (-1.0, 1.0),
    None: (0.0, 0.0),
}

STAGGER_BOOLS = {"0": 1.0, "1": 0.5}

PADDING = {
    "both": (1, 1),
    "right": (0, 1),
    "left": (1, 0),
    None: (0, 0),
}


def domain_limits_transform(
    xmin: float,
    xmax: float,
    dx: float,
    direction: tp.Optional[str] = None,
    stagger: tp.Optional[bool] = None,
) -> tp.Tuple:
    # convert staggers to bools
    if stagger is None:
        stagger = "0"
    stagger = str(int(stagger))

    # TODO: check size of dx
    xmin += dx * STAGGER_BOOLS[stagger] * DIRECTIONS[direction][0]
    xmax += dx * STAGGER_BOOLS[stagger] * DIRECTIONS[direction][1]
    return xmin, xmax


def batch_domain_limits_transform(
    xmin: tp.Iterable[float],
    xmax: tp.Iterable[float],
    dx: tp.Iterable[float],
    direction: tp.Iterable[str] = None,
    stagger: tp.Iterable[bool] = None,
) -> tp.Tuple:
    if direction is None:
        direction = (None,) * len(xmin)

    if stagger is None:
        stagger = (False,) * len(xmin)

    msg = "Incorrect shapes"
    msg += f"\nxmin: {len(xmin)} | "
    msg += f"xmax: {len(xmax)} | "
    msg += f"dx: {len(dx)} | "
    msg += f"direction: {len(direction)} | "
    msg += f"stagger: {len(stagger)}"
    assert len(xmin) == len(xmax) == len(dx) == len(direction) == len(stagger), msg

    limits = [
        domain_limits_transform(imin, imax, idx, idirection, istagger)
        for imin, imax, idx, idirection, istagger in zip(
            xmin, xmax, dx, direction, stagger
        )
    ]

    xmin, xmax = zip(*limits)
    return xmin, xmax


def stagger_domain(
    domain: Domain, direction: tp.Iterable[str], stagger: tp.Iterable[bool]
):
    msg = "Incorrect shapes"
    msg += f"\nxmin: {len(domain.xmin)} | "
    msg += f"xmax: {len(domain.xmax)} | "
    msg += f"dx: {len(domain.dx)} | "
    msg += f"direction: {len(direction)} | "
    msg += f"stagger: {len(stagger)}"
    assert (
        len(domain.xmin)
        == len(domain.xmax)
        == len(domain.dx)
        == len(direction)
        == len(stagger)
    ), msg

    # change domain limits
    xmin, xmax = batch_domain_limits_transform(
        domain.xmin, domain.xmax, domain.dx, direction, stagger
    )
    domains = [
        domain_utils.init_domain_1d(float(ixmin), float(ixmax), float(idx))
        for ixmin, ixmax, idx in zip(xmin, xmax, domain.dx)
    ]
    import functools

    domain = functools.reduce(lambda a, b: a * b, domains)

    # print(domains[0], domains[1])
    # domain = sum(domains)
    # create new domain
    # domain = Domain(xmin=xmin, xmax=xmax, dx=domain.dx)

    return domain


def slice_interior_axis(u: Field, axis: int = 0):
    assert axis <= len(u.domain.Nx)

    # dynamic slice
    u_interior = jax.lax.slice_in_dim(u[:], start_index=1, limit_index=-1, axis=axis)

    xmin = [
        xmin + u.domain.dx[i] if axis == i else xmin
        for i, xmin in enumerate(u.domain.xmin)
    ]
    xmax = [
        xmax - u.domain.dx[i] if axis == i else xmax
        for i, xmax in enumerate(u.domain.xmax)
    ]
    Nx = [Nx - 2 if axis == i else Nx for i, Nx in enumerate(u.domain.Nx)]

    domain = Domain(xmin=xmin, xmax=xmax, dx=u.domain.dx)
    domain = Domain.from_numpoints(xmin=xmin, xmax=xmax, N=Nx)

    return Field(values=u_interior, domain=domain)


def slice_interior_all(u: Field):
    num_axis = len(u.domain.Nx)

    # change x limits
    xmin = [xmin + u.domain.dx[i] for i, xmin in enumerate(u.domain.xmin)]
    xmax = [xmax - u.domain.dx[i] for i, xmax in enumerate(u.domain.xmax)]
    # change number of points
    Nx = [Nx - 2 for Nx in u.domain.Nx]

    # get slices
    slices = [slice(1, -1) for i in range(num_axis)]

    u_interior = u[tuple(slices)]

    domain = Domain(xmin=xmin, xmax=xmax, dx=u.domain.dx)
    domain = Domain.from_numpoints(xmin=xmin, xmax=xmax, N=Nx)

    return Field(values=u_interior, domain=domain)


def pad_all_axis(u: Field, mode: str = "edge", **kwargs):
    num_axis = len(u.domain.Nx)

    pad_width = [PADDING["both"] for _ in range(num_axis)]

    # pad interior points
    u_interior = jnp.pad(u[:], pad_width=pad_width, mode=mode, **kwargs)

    # change x limits
    xmin = [
        xmin - pad_width[i][0] * u.domain.dx[i] for i, xmin in enumerate(u.domain.xmin)
    ]
    xmax = [
        xmax + pad_width[i][1] * u.domain.dx[i] for i, xmax in enumerate(u.domain.xmax)
    ]
    # change number of points
    Nx = [Nx - sum(pad_width[i]) for i, Nx in enumerate(u.domain.Nx)]

    domain = Domain(xmin=xmin, xmax=xmax, dx=u.domain.dx)

    return Field(values=u_interior, domain=domain)


def pad_along_axis(u: Field, pad_width: tp.Iterable[tuple], mode="edge", **kwargs):
    # assert same size
    assert len(pad_width) <= len(u.domain.Nx)

    axis = [1 if sum(PADDING[ipad]) > 0 else 0 for ipad in pad_width]

    pad_width = [PADDING[ipad] for ipad in pad_width]

    # pad interior points
    u_interior = jnp.pad(u[:], pad_width=pad_width, mode=mode, **kwargs)

    xmin = [
        xmin - pad_width[i][0] * u.domain.dx[i] if axis else xmin
        for i, xmin in enumerate(u.domain.xmin)
    ]
    xmax = [
        xmax + pad_width[i][1] * u.domain.dx[i] if axis else xmax
        for i, xmax in enumerate(u.domain.xmax)
    ]

    Nx = [
        Nx - sum(pad_width[i]) if axis == i else Nx for i, Nx in enumerate(u.domain.Nx)
    ]

    domain = Domain(xmin=xmin, xmax=xmax, dx=u.domain.dx)

    return Field(values=u_interior, domain=domain)


def diffx_midpoint(u: Array, step_size: float):
    return fdx.difference(
        u, step_size=step_size, axis=0, accuracy=1, derivative=1, method="backward"
    )[1:]


def diffy_midpoint(u: Array, step_size: float):
    return fdx.difference(
        u[:], step_size=step_size, axis=1, accuracy=1, derivative=1, method="backward"
    )[:, 1:]


def diffx2_centerpoint(u: Array, step_size: float):
    return fdx.difference(
        u, step_size=step_size, axis=0, accuracy=1, derivative=2, method="backward"
    )[1:-1]


def diffy2_centerpoint(u: Array, step_size: float):
    return fdx.difference(
        u, step_size=step_size, axis=1, accuracy=1, derivative=2, method="backward"
    )[:, 1:-1]


def difference(u: Field, axis=0, derivative=1) -> Field:
    assert derivative >= 1 and derivative <= 2
    assert axis >= 0 and axis <= 1

    # calculate 1st derivative (midpoint)
    if derivative == 1 and axis == 0:
        u_values = diffx_midpoint(u=u[:], step_size=u.domain.dx[0])
        domain = stagger_domain(
            u.domain, direction=("inner", None), stagger=(True, False)
        )

    # calculate 1st derivative (midpoint)
    elif derivative == 1 and axis == 1:
        u_values = diffy_midpoint(u=u[:], step_size=u.domain.dx[1])
        domain = stagger_domain(
            u.domain, direction=(None, "inner"), stagger=(False, True)
        )

    # calculate 2st derivative (gridpoint)
    elif derivative == 2 and axis == 0:
        u_values = diffx2_centerpoint(u=u[:], step_size=u.domain.dx[0])
        domain = stagger_domain(
            u.domain, direction=("inner", None), stagger=(False, False)
        )

    # calculate 2st derivative (gridpoint)
    elif derivative == 2 and axis == 1:
        u_values = diffy2_centerpoint(u=u[:], step_size=u.domain.dx[0])
        domain = stagger_domain(
            u.domain, direction=(False, "inner"), stagger=(False, False)
        )
    else:
        msg = f"Incorrect combo of axis and derivative:"
        msg += f"\nderivative: {derivative} | axis: {axis}"
        raise ValueError(msg)

    return Field(u_values, domain=domain)
