from jaxtyping import Array
import finitediffx as fdx


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
