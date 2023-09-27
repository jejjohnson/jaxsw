from jaxtyping import Array
import finitediffx as fdx
from jaxsw._src.fields.base import Field
from jaxsw._src.operators.functional.cgrid import stagger_domain


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
    print(u.values.shape, u.domain.Nx)
    assert derivative >= 1 and derivative <= 2
    assert axis >= 0 and axis <= 1

    # calculate 1st derivative (midpoint)
    if derivative == 1 and axis == 0:
        u_values = diffx_midpoint(u=u[:], step_size=u.domain.dx[0])
        domain = stagger_domain(
            u.domain, direction=("inner", None), stagger=(True, False)
        )

        print(u_values.shape, domain.Nx)

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
        u_values = diffy2_centerpoint(u=u[:], step_size=u.domain.dx[1])
        domain = stagger_domain(
            u.domain, direction=(False, "inner"), stagger=(False, False)
        )
    else:
        msg = f"Incorrect combo of axis and derivative:"
        msg += f"\nderivative: {derivative} | axis: {axis}"
        raise ValueError(msg)

    return Field(u_values, domain=domain)


def laplacian_centerpoint(u: Array, step_size: float):
    return fdx.laplacian(u, step_size=step_size, accuracy=1, method="backward")[
        1:-1, 1:-1
    ]


def y_average(psi):
    return 0.5 * (psi[:, :-1] + psi[:, 1:])


def x_average(psi):
    return 0.5 * (psi[:-1] + psi[1:])


def center_average(psi):
    return 0.25 * (psi[:-1, :-1] + psi[:-1, 1:] + psi[1:, :-1] + psi[1:, 1:])


# def laplacian(u: Array, step_size: int) -> Array:
#     return fdx.laplacian(u, step_size=step_size, accuracy=1)


# def divergence(u: Array, v: Array, step_size: int) -> Array:
#     return fdx.divergence(u, v, step_size=step_size, accuracy=1)


# def vorticity(u: Array, step_size: int) -> Array:
#     du_dx: Array = difference(u=u, step_size=step_size[0], axis=0, derivative=1)
#     du_dy: Array = difference(u=u, step_size=step_size[1], axis=1, derivative=1)

#     return du_dy - du_dx


# def x_average(u):
#     return u


# def y_average(u):
#     return u


# def center_average(u):
#     return u


# def u_at_h(u) -> Array:
#     return u


# def u_at_v(u) -> Array:
#     return u


# def v_at_h(u) -> Array:
#     return u


# def v_at_u(v: Array) -> Array:
#     return v

# # ==========================================
# # TESTING
# # ==========================================
# def diffx(psi, step_size):
#     return (psi[1:] - psi[:-1]) / step_size
# def diffy(psi, step_size):
#     return (psi[:, 1:] - psi[:,:-1]) / step_size
# def diffx2(psi, step_size):
#     return (psi[:-2, :] - 2*psi[1:-1, :] + psi[2:, :]) / step_size**2
# def diffy2(psi, step_size):
#     return (psi[:, :-2] - 2*psi[:, 1:-1] + psi[:, 2:]) / step_size**2
# def del2(psi, step_size):
#     return diffx2(psi, step_size[0])[:, 1:-1] + diffy2(psi, step_size[1])[1:-1, :]
