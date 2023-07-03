import jax.numpy as jnp
from jaxtyping import Array

PAD_1D = (1, 1)
PAD_2D = ((1, 1), (1, 1))
PAD_3D = ((1, 1), (1, 1), (1, 1))


# def apply_bcs_1d(x: Array, bc: str = "dirichlet", pad: bool = True) -> Array:
#     if pad:
#         x = jnp.pad(x, pad_width=((1, 1)), mode="constant")

#     if bc == "dirichlet":
#         x = dirichlet_1d(x)
#     elif bc == "neumann":
#         x = neumann_1d(x)
#     elif bc == "dirichlet_face":
#         x = dirichlet_face_1d(x)
#     elif bc == "periodic":
#         x = periodic_1d(x)
#     else:
#         raise ValueError(f"Unrecognized bc: {bc}")

#     return x


# def apply_bcs_2d(x: Array, bc: str = "dirichlet", pad: bool = True) -> Array:
#     if pad:
#         x = jnp.pad(x, pad_width=((1, 1), (1, 1)), mode="constant")

#     if bc == "dirichlet":
#         x = dirichlet_2d(x)
#     elif bc == "neumann":
#         x = neumann_2d(x)
#     elif bc == "dirichlet_face":
#         x = dirichlet_face_2d(x)
#     elif bc == "periodic":
#         x = periodic_2d(x)
#     else:
#         raise ValueError(f"Unrecognized bc: {bc}")

#     return x


def apply_periodic_pad_ND(u, pad_width):
    return jnp.pad(u, pad_width=pad_width, mode="wrap")


def apply_periodic_pad_1D(u):
    return apply_periodic_pad_ND(u, pad_width=PAD_1D)


def apply_periodic_pad_2D(u):
    return apply_periodic_pad_ND(u, pad_width=PAD_2D)


def apply_periodic_pad_3D(u):
    return apply_periodic_pad_ND(u, pad_width=PAD_3D)


def apply_periodic_x(u: Array) -> Array:
    u = u.at[0].set(u[-2])
    u = u.at[-1].set(u[1])
    return u


def apply_periodic_y(u: Array) -> Array:
    u = u.at[:, 0].set(u[:, -2])
    u = u.at[:, -1].set(u[:, 1])
    return u


def apply_periodic_z(u: Array) -> Array:
    u = u.at[..., 0].set(u[..., -2])
    u = u.at[..., -1].set(u[..., 1])
    return u


def apply_neumann_pad_ND(u, pad_width):
    return jnp.pad(u, pad_width=pad_width, mode="symmetric")


def apply_neumann_pad_1D(u):
    return apply_neumann_pad_ND(u, pad_width=PAD_1D)


def apply_neumann_pad_2D(u):
    return apply_neumann_pad_ND(u, pad_width=PAD_2D)


def apply_neumann_pad_3D(u):
    return apply_neumann_pad_ND(u, pad_width=PAD_3D)


def apply_neumann_x(x: Array) -> Array:
    x = x.at[0].set(x[1])
    x = x.at[-1].set(x[-2])

    return x


def apply_neumann_y(u: Array) -> Array:
    u = u.at[:, 0].set(u[:, 1])
    u = u.at[:, -1].set(u[:, -2])
    return u


def apply_neumann_z(u: Array) -> Array:
    u = u.at[..., 0].set(u[..., 1])
    u = u.at[..., -1].set(u[..., -2])
    return u


def apply_neumann_corners_xy(x: Array) -> Array:
    # corners
    x = x.at[0, 0].set(x[1, 1])
    x = x.at[-1, 0].set(x[-2, 1])
    x = x.at[0, -1].set(x[1, -2])
    x = x.at[-1, -1].set(x[-2, -2])

    return x


def periodic_1d(x: Array) -> Array:
    x = x.at[0].set(x[-2])
    x = x.at[-1].set(x[1])

    return x


def periodic_2d(x: Array) -> Array:
    x = x.at[0, :].set(x[-2, :])
    x = x.at[-1, :].set(x[1, :])
    x = x.at[:, 0].set(x[:, -2])
    x = x.at[:, -1].set(x[:, 1])

    return x


def neumann_1d(x: Array) -> Array:
    x = x.at[0].set(x[1])
    x = x.at[-1].set(x[-2])

    return x


def neumann_2d(x: Array) -> Array:
    x = x.at[0, :].set(x[1, :])
    x = x.at[-1, :].set(x[-2, :])
    x = x.at[:, 0].set(x[:, 1])
    x = x.at[:, -1].set(x[:, -2])

    # corners
    x = x.at[0, 0].set(x[1, 1])
    x = x.at[-1, 0].set(x[-2, 1])
    x = x.at[0, -1].set(x[1, -2])
    x = x.at[-1, -1].set(x[-2, -2])

    return x


def apply_dirichlet_x_edge(u):
    u = u.at[0].set(-u[1])
    u = u.at[-1].set(-u[-2])
    return u


def apply_dirichlet_x_face(u):
    u = u.at[0].set(jnp.asarray(0.0, dtype=u.dtype))
    u = u.at[-1].set(jnp.asarray(0.0, dtype=u.dtype))
    return u


def apply_dirichlet_corners_edges(u):
    # corners
    u = u.at[0, 0].set(-u[0, 1] - u[1, 0] - u[1, 1])
    u = u.at[-1, 0].set(-u[-1, 1] - u[-2, 0] - u[-2, 1])
    u = u.at[0, -1].set(-u[1, -1] - u[0, -2] - u[1, -2])
    u = u.at[-1, -1].set(-u[-1, -2] - u[-2, -2] - u[-2, -1])

    return u


def apply_dirichlet_corners_faces(u: Array) -> Array:
    # corners
    u = u.at[0, 0].set(jnp.asarray(0.0, dtype=u.dtype))
    u = u.at[-1, 0].set(jnp.asarray(0.0, dtype=u.dtype))
    u = u.at[0, -1].set(jnp.asarray(0.0, dtype=u.dtype))
    u = u.at[-1, -1].set(jnp.asarray(0.0, dtype=u.dtype))
    return u


def apply_dirichlet_y_edge(u):
    u = u.at[:, 0].set(-u[:, 1])
    u = u.at[:, -1].set(-u[:, -2])
    return u


def apply_dirichlet_y_face(u):
    u = u.at[:, 0].set(jnp.asarray(0.0, dtype=u.dtype))
    u = u.at[:, -1].set(jnp.asarray(0.0, dtype=u.dtype))
    return u


def apply_dirichlet_z(u):
    u = u.at[..., 0].set(-u[..., 1])
    u = u.at[..., -1].set(-u[..., -2])
    return u


def apply_dirichlet_pad_face_1D(u):
    u = jnp.pad(u, pad_width=PAD_1D, mode="constant")
    return apply_dirichlet_x_face(u)


def apply_dirichlet_pad_edge_1D(u):
    u = jnp.pad(u, pad_width=PAD_1D, mode="constant")
    return apply_dirichlet_x_edge(u)


# def apply_dirichlet_pad_1D(u):
#     constant_values = (-u[0], -u[-1])
#     return jnp.pad(
#         u, pad_width=PAD_1D, mode="constant", constant_values=constant_values
#     )


def apply_dirichlet_pad_edge_2D(u):
    u = jnp.pad(u, pad_width=PAD_2D, mode="constant")
    return apply_dirichlet_y_edge(apply_dirichlet_x_edge(u))


def apply_dirichlet_pad_face_2D(u):
    u = jnp.pad(u, pad_width=PAD_2D, mode="constant")
    return apply_dirichlet_y_face(apply_dirichlet_x_face(u))


# def apply_dirichlet_pad_2D(u):
#     constant_values = ((-u[0], -u[-1]), (-u[:, 0], -u[:, -1]))
#     return jnp.pad(
#         u, pad_width=PAD_2D, mode="constant", constant_values=constant_values
#     )


# def apply_dirichlet_pad_3D(u):
#     u = jnp.pad(u, pad_width=PAD_3D, mode="constant")
#     return apply_dirichlet_z(apply_dirichlet_y(apply_dirichlet_x(u)))


# def apply_dirichlet_pad_3D(u):
#     constant_values = (
#         (-u[0], -u[-1]),
#         (-u[:, 0], -u[:, -1]),
#         (-u[..., 0], -u[..., -1]),
#     )
#     return jnp.pad(
#         u, pad_width=PAD_3D, mode="constant", constant_values=constant_values
#     )


def dirichlet_2d(x: Array) -> Array:
    # edges
    x = x.at[0, :].set(-x[1, :])
    x = x.at[-1, :].set(-x[-2, :])
    x = x.at[:, 0].set(-x[:, 1])
    x = x.at[:, -1].set(-x[:, -2])

    # corners
    x = x.at[0, 0].set(-x[0, 1] - x[1, 0] - x[1, 1])
    x = x.at[-1, 0].set(-x[-1, 1] - x[-2, 0] - x[-2, 1])
    x = x.at[0, -1].set(-x[1, -1] - x[0, -2] - x[1, -2])
    x = x.at[-1, -1].set(-x[-1, -2] - x[-2, -2] - x[-2, -1])

    return x


def dirichlet_face_1d(x: Array) -> Array:
    # faces
    x = x.at[0].set(0.0)
    x = x.at[-1].set(0.0)

    return x


def dirichlet_face_2d(x: Array) -> Array:
    # faces
    x = x.at[0, :].set(0.0)
    x = x.at[-1, :].set(0.0)
    x = x.at[:, 0].set(0.0)
    x = x.at[:, -1].set(0.0)

    # corners
    x = x.at[0, 0].set(0.0)
    x = x.at[-1, 0].set(0.0)
    x = x.at[0, -1].set(0.0)
    x = x.at[-1, -1].set(0.0)

    return x
