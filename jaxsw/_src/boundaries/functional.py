import jax.numpy as jnp
from jaxtyping import Array


def apply_bcs_1d(x: Array, bc: str = "dirichlet", pad: bool = True) -> Array:
    if pad:
        x = jnp.pad(x, pad_width=((1, 1)), mode="constant")

    if bc == "dirichlet":
        x = dirichlet_1d(x)
    elif bc == "neumann":
        x = neumann_1d(x)
    elif bc == "dirichlet_face":
        x = dirichlet_face_1d(x)
    elif bc == "periodic":
        x = periodic_1d(x)
    else:
        raise ValueError(f"Unrecognized bc: {bc}")

    return x


def apply_bcs_2d(x: Array, bc: str = "dirichlet", pad: bool = True) -> Array:
    if pad:
        x = jnp.pad(x, pad_width=((1, 1), (1, 1)), mode="constant")

    if bc == "dirichlet":
        x = dirichlet_2d(x)
    elif bc == "neumann":
        x = neumann_2d(x)
    elif bc == "dirichlet_face":
        x = dirichlet_face_2d(x)
    elif bc == "periodic":
        x = periodic_2d(x)
    else:
        raise ValueError(f"Unrecognized bc: {bc}")

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


def dirichlet_1d(x: Array) -> Array:
    # edges
    x = x.at[0].set(-x[1])
    x = x.at[-1].set(-x[-2])

    return x


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
