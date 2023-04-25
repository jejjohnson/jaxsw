""" model2d_nonlinear.py

2D shallow water model with:

- varying Coriolis force
- nonlinear terms
- lateral friction
- periodic boundary conditions

"""

import matplotlib.pyplot as plt
import numpy as np


# grid setup
n_x = 200
dx = 5e3
l_x = n_x * dx

n_y = 104
dy = 5e3
l_y = n_y * dy

x, y = (np.arange(n_x) * dx, np.arange(n_y) * dy)
Y, X = np.meshgrid(y, x, indexing="ij")

# physical parameters
gravity = 9.81
depth = 100.0
coriolis_f = 2e-4
coriolis_beta = 2e-11
coriolis_param = coriolis_f + Y * coriolis_beta
lateral_viscosity = 1e-3 * coriolis_f * dx**2

# other parameters
periodic_boundary_x = False
linear_momentum_equation = False

adams_bashforth_a = 1.5 + 0.1
adams_bashforth_b = -(0.5 + 0.1)

dt = 0.125 * min(dx, dy) / np.sqrt(gravity * depth)

phase_speed = np.sqrt(gravity * depth)
rossby_radius = np.sqrt(gravity * depth) / coriolis_param.mean()

# plot parameters
plot_range = 10
plot_every = 10
max_quivers = 41

# initial conditions
u0 = 10 * np.exp(-((Y - y[n_y // 2]) ** 2) / (0.02 * l_x) ** 2)
v0 = np.zeros_like(u0)
# approximate balance h_y = -(f/g)u
h_geostrophy = np.cumsum(-dy * u0 * coriolis_param / gravity, axis=0)
h0 = (
    depth
    + h_geostrophy
    # make sure h0 is centered around depth
    - h_geostrophy.mean()
    # small perturbation
    + 0.2 * np.sin(X / l_x * 10 * np.pi) * np.cos(Y / l_y * 8 * np.pi)
)


def prepare_plot():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    cs = update_plot(0, h0, u0, v0, ax)
    plt.colorbar(cs, label="$\\eta$ (m)")
    return fig, ax


def update_plot(t, h, u, v, ax):
    eta = h - depth

    quiver_stride = (slice(1, -1, n_y // max_quivers), slice(1, -1, n_x // max_quivers))

    ax.clear()
    cs = ax.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        eta[1:-1, 1:-1],
        vmin=-plot_range,
        vmax=plot_range,
        cmap="RdBu_r",
    )

    if np.any((u[quiver_stride] != 0) | (v[quiver_stride] != 0)):
        ax.quiver(
            x[quiver_stride[1]] / 1e3,
            y[quiver_stride[0]] / 1e3,
            u[quiver_stride],
            v[quiver_stride],
            clip_on=False,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("$y$ (km)")
    ax.set_title(
        "t=%5.2f days, R=%5.1f km, c=%5.1f m/s "
        % (t / 86400, rossby_radius / 1e3, phase_speed)
    )
    plt.pause(0.1)
    return cs


def enforce_boundaries(arr, grid):
    assert grid in ("h", "u", "v")
    if periodic_boundary_x:
        arr[:, 0] = arr[:, -2]
        arr[:, -1] = arr[:, 1]
    elif grid == "u":
        arr[:, -2] = 0.0
    if grid == "v":
        arr[-2, :] = 0.0
    return arr


def iterate_shallow_water():
    # allocate arrays
    u, v, h = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
    du, dv, dh = np.empty((n_y, n_x)), np.empty((n_y, n_x)), np.empty((n_y, n_x))
    du_new, dv_new, dh_new = (
        np.empty((n_y, n_x)),
        np.empty((n_y, n_x)),
        np.empty((n_y, n_x)),
    )
    fe, fn = np.empty((n_y, n_x)), np.empty((n_y, n_x))
    q, ke = np.empty((n_y, n_x)), np.empty((n_y, n_x))

    # initial conditions
    h[...] = h0
    u[...] = u0
    v[...] = v0

    # boundary values of h must not be used
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = np.nan

    h = enforce_boundaries(h, "h")
    u = enforce_boundaries(u, "u")
    v = enforce_boundaries(v, "v")

    first_step = True

    # time step equations
    while True:
        hc = np.pad(h[1:-1, 1:-1], 1, "edge")
        hc = enforce_boundaries(hc, "h")

        fe[1:-1, 1:-1] = 0.5 * (hc[1:-1, 1:-1] + hc[1:-1, 2:]) * u[1:-1, 1:-1]
        fn[1:-1, 1:-1] = 0.5 * (hc[1:-1, 1:-1] + hc[2:, 1:-1]) * v[1:-1, 1:-1]
        fe = enforce_boundaries(fe, "u")
        fn = enforce_boundaries(fn, "v")

        dh_new[1:-1, 1:-1] = -(
            (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
            + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
        )

        if linear_momentum_equation:
            v_avg = 0.25 * (v[1:-1, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[:-2, 2:])
            du_new[1:-1, 1:-1] = (
                coriolis_param[1:-1, 1:-1] * v_avg
                - gravity * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx
            )
            u_avg = 0.25 * (u[1:-1, 1:-1] + u[1:-1, :-2] + u[2:, 1:-1] + u[2:, :-2])
            dv_new[1:-1, 1:-1] = (
                -coriolis_param[1:-1, 1:-1] * u_avg
                - gravity * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy
            )

        else:  # nonlinear momentum equation
            # planetary and relative vorticity
            q[1:-1, 1:-1] = coriolis_param[1:-1, 1:-1] + (
                (v[1:-1, 2:] - v[1:-1, 1:-1]) / dx - (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
            )
            # potential vorticity
            q[1:-1, 1:-1] *= 1.0 / (
                0.25 * (hc[1:-1, 1:-1] + hc[1:-1, 2:] + hc[2:, 1:-1] + hc[2:, 2:])
            )
            q = enforce_boundaries(q, "h")

            du_new[1:-1, 1:-1] = -gravity * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx + 0.5 * (
                q[1:-1, 1:-1] * 0.5 * (fn[1:-1, 1:-1] + fn[1:-1, 2:])
                + q[:-2, 1:-1] * 0.5 * (fn[:-2, 1:-1] + fn[:-2, 2:])
            )
            dv_new[1:-1, 1:-1] = -gravity * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy - 0.5 * (
                q[1:-1, 1:-1] * 0.5 * (fe[1:-1, 1:-1] + fe[2:, 1:-1])
                + q[1:-1, :-2] * 0.5 * (fe[1:-1, :-2] + fe[2:, :-2])
            )
            ke[1:-1, 1:-1] = 0.5 * (
                0.5 * (u[1:-1, 1:-1] ** 2 + u[1:-1, :-2] ** 2)
                + 0.5 * (v[1:-1, 1:-1] ** 2 + v[:-2, 1:-1] ** 2)
            )
            ke = enforce_boundaries(ke, "h")

            du_new[1:-1, 1:-1] += -(ke[1:-1, 2:] - ke[1:-1, 1:-1]) / dx
            dv_new[1:-1, 1:-1] += -(ke[2:, 1:-1] - ke[1:-1, 1:-1]) / dy

        if first_step:
            u[1:-1, 1:-1] += dt * du_new[1:-1, 1:-1]
            v[1:-1, 1:-1] += dt * dv_new[1:-1, 1:-1]
            h[1:-1, 1:-1] += dt * dh_new[1:-1, 1:-1]
            first_step = False
        else:
            u[1:-1, 1:-1] += dt * (
                adams_bashforth_a * du_new[1:-1, 1:-1]
                + adams_bashforth_b * du[1:-1, 1:-1]
            )
            v[1:-1, 1:-1] += dt * (
                adams_bashforth_a * dv_new[1:-1, 1:-1]
                + adams_bashforth_b * dv[1:-1, 1:-1]
            )
            h[1:-1, 1:-1] += dt * (
                adams_bashforth_a * dh_new[1:-1, 1:-1]
                + adams_bashforth_b * dh[1:-1, 1:-1]
            )

        h = enforce_boundaries(h, "h")
        u = enforce_boundaries(u, "u")
        v = enforce_boundaries(v, "v")

        if lateral_viscosity > 0:
            # lateral friction
            fe[1:-1, 1:-1] = lateral_viscosity * (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx
            fn[1:-1, 1:-1] = lateral_viscosity * (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
            fe = enforce_boundaries(fe, "u")
            fn = enforce_boundaries(fn, "v")

            u[1:-1, 1:-1] += dt * (
                (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
            )

            fe[1:-1, 1:-1] = lateral_viscosity * (v[1:-1, 2:] - u[1:-1, 1:-1]) / dx
            fn[1:-1, 1:-1] = lateral_viscosity * (v[2:, 1:-1] - u[1:-1, 1:-1]) / dy
            fe = enforce_boundaries(fe, "u")
            fn = enforce_boundaries(fn, "v")

            v[1:-1, 1:-1] += dt * (
                (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
                + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
            )

        # rotate quantities
        du[...] = du_new
        dv[...] = dv_new
        dh[...] = dh_new

        yield h, u, v


if __name__ == "__main__":
    fig, ax = prepare_plot()

    model = iterate_shallow_water()
    for iteration, (h, u, v) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt
            update_plot(t, h, u, v, ax)

        # stop if user closes plot window
        if not plt.fignum_exists(fig.number):
            break
