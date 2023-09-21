import finitediffx as fdx
import numpy as np

from jaxsw._src.operators.functional import grid as F_grid


def init_tau(domain, tau0: float = 2.0e-5):
    """
    Args
    ----
        tau0 (float): wind stress magnitude m/s^2
            default=2.0e-5"""
    # initial TAU
    tau = np.zeros((2, domain.Nx[0], domain.Nx[1]))

    # create staggered coordinates (y-direction)
    y_coords = np.arange(domain.Nx[1]) + 0.5

    # create tau
    tau[0, :, :] = -tau0 * np.cos(2 * np.pi * (y_coords / domain.Nx[1]))

    return tau


def calculate_wind_forcing(tau, domain):
    # move from edges to nodes
    tau_x = F_grid.x_average_2D(tau[0], padding=((1, 0), (0, 0)))
    tau_y = F_grid.y_average_2D(tau[1], padding=((0, 0), (1, 0)))

    # compute finite difference
    dF2dX = fdx.difference(
        tau_y, axis=0, step_size=domain.dx[0], accuracy=1, method="central"
    )
    dF1dY = fdx.difference(
        tau_x, axis=1, step_size=domain.dx[1], accuracy=1, method="central"
    )
    curl_stagg = dF2dX - dF1dY

    return F_grid.center_average_2D(curl_stagg.squeeze()[1:, 1:])
