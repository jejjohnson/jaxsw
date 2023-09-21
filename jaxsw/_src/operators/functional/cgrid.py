import typing as tp
import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxsw._src.domain.base import Domain
from jaxsw._src.fields.base import Field
from jaxsw._src.operators.functional import grid as F_grid


DIRECTIONS = {
    "right": (1.0, 1.0),
    "left": (-1.0, -1.0),
    "inner": (1.0, -1.0),
    "outer": (-1.0, 1.0),
    None: (0.0, 0.0),
}

STAGGER_BOOLS = {"0": 1.0, "1": 0.5}


def stagger_domain(domain: Domain, direction: tp.Tuple[str], stagger: tp.Tuple[bool]):
    # check sizes
    msg = "Incorrect sizes"
    msg += f"\n {len(direction)} | {len(stagger)} | {len(domain.dx)}"
    assert len(direction) == len(stagger) == len(domain.dx)

    # convert bools to strings
    stagger = list(map(lambda x: str(int(x)), stagger))

    # do staggering
    x_limits = [
        (
            xmin + idx * STAGGER_BOOLS[istagger] * DIRECTIONS[idirection][0],
            xmax + idx * STAGGER_BOOLS[istagger] * DIRECTIONS[idirection][1],
        )
        if idirection is not None
        else (xmin, xmax)
        for xmin, xmax, idx, idirection, istagger in zip(
            domain.xmin, domain.xmax, domain.dx, direction, stagger
        )
    ]
    # parse outputs
    xmin, xmax = zip(*x_limits)

    # create new domain
    return Domain(xmin=xmin, xmax=xmax, dx=domain.dx)


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


PADDING = {
    "both": (1, 1),
    "right": (0, 1),
    "left": (1, 0),
    None: (0, 0),
}


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
