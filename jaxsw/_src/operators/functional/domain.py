import typing as tp
from jaxsw._src.domain.base import Domain


def limits_stagger(
    xmin,
    xmax,
    dx,
    operation: tp.Optional[str] = None,
    accuracy: int = 1,
):
    # add stagger
    stagger = accuracy * dx * 0.5
    print(dx)
    print(stagger)

    if operation is None:
        pass
    elif operation in ["right", "forward", "left", "backward"]:
        xmin += stagger
        xmax -= stagger
    elif operation in ["inner", "central"]:
        xmin += stagger
        xmax -= stagger
    else:
        msg = "Unrecognized argument for operation"
        msg += f"\noperation: {operation}"
        raise ValueError(msg)

    return xmin, xmax


def domain_stagger(
    domain, axis: int, operation: tp.Optional[str] = None, accuracy: int = 1
) -> Domain:
    xmin, xmax, dx = domain.xmin, domain.xmax, domain.dx
    # get new xmin, xmax
    out = [
        limits_stagger(xmin, xmax, dx, operation, accuracy)
        if i == axis
        else (xmin, xmax)
        for i, (xmin, xmax, dx) in enumerate(zip(xmin, xmax, dx))
    ]
    xmin, xmax = zip(*out)

    return Domain(xmin=xmin, xmax=xmax, dx=dx)
