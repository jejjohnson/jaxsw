import typing as tp

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


class LayerDomain(eqx.Module):
    heights: Array = eqx.static_field()
    reduced_gravities: Array = eqx.static_field()
    Nz: float = eqx.static_field()
    A: Array = eqx.static_field()
    A_layer_2_mode: Array = eqx.static_field()
    A_mode_2_layer: Array = eqx.static_field()
    lambd: Array = eqx.static_field()

    def __init__(self, heights: tp.List[float], reduced_gravities: tp.List[float]):
        num_layers = len(heights)

        msg = "Incorrect number of heights to reduced gravities."
        msg += f"\nHeights: {heights} | {num_layers}"
        msg += f"\nReduced Gravities: {reduced_gravities} | {len(reduced_gravities)}"
        assert num_layers - 1 == len(reduced_gravities), msg

        self.heights = jnp.asarray(heights)
        self.reduced_gravities = jnp.asarray(reduced_gravities)
        self.Nz = num_layers

        # calculate matrix M
        A = create_qg_multilayer_mat(heights, reduced_gravities)
        self.A = jnp.asarray(A)

        # create layer to mode matrices
        lambd, A_layer_2_mode, A_mode_2_layer = compute_layer_to_mode_matrices(A)
        self.lambd = jnp.asarray(lambd)
        self.A_layer_2_mode = jnp.asarray(A_layer_2_mode)
        self.A_mode_2_layer = jnp.asarray(A_mode_2_layer)


def create_qg_multilayer_mat(
    heights: tp.List[float], reduced_gravities: tp.List[float]
) -> np.ndarray:
    """Computes the Matrix that is used to connected a stacked
    isopycnal Quasi-Geostrophic model.

    Args:
        heights (tp.List[float]): the height for each layer
            Size = [Nx]
        reduced_gravities (tp.List[float]): the reduced gravities
            for each layer, Size = [Nx-1]

    Returns:
        np.ndarray: The Matrix connecting the layers, Size = [Nz, Nx]
    """
    num_heights = len(heights)

    # initialize matrix
    A = np.zeros((num_heights, num_heights))

    # top rows
    A[0, 0] = 1.0 / (heights[0] * reduced_gravities[0])
    A[0, 1] = -1.0 / (heights[0] * reduced_gravities[0])

    # interior rows
    for i in range(1, num_heights - 1):
        A[i, i - 1] = -1.0 / (heights[i] * reduced_gravities[i - 1])
        A[i, i] = (
            1.0 / heights[i] * (1 / reduced_gravities[i] + 1 / reduced_gravities[i - 1])
        )
        A[i, i + 1] = -1.0 / (heights[i] * reduced_gravities[num_heights - 2])

    # bottom rows
    A[-1, -1] = 1.0 / (heights[num_heights - 1] * reduced_gravities[num_heights - 2])
    A[-1, -2] = -1.0 / (heights[num_heights - 1] * reduced_gravities[num_heights - 2])
    return A


def compute_layer_to_mode_matrices(A):
    # eigenvalue decomposition
    lambd_r, R = jnp.linalg.eig(A)
    _, L = jnp.linalg.eig(A.T)

    # extract real components
    lambd, R, L = lambd_r.real, R.real, L.real

    # create matrices
    Cl2m = np.diag(1.0 / np.diag(L.T @ R)) @ L.T
    Cm2l = R
    # create diagonal matrix
    return lambd, -Cl2m, -Cm2l
