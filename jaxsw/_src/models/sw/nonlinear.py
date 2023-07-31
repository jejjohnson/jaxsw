import jax.numpy as jnp
from jaxsw._src.models.pde import DynamicalSystem
from jaxsw._src.operators.functional import grid as F_grid
from jaxtyping import Array
import equinox as eqx
from jaxsw._src.models.sw import Params, State


class ShallowWater2D(DynamicalSystem):
    @staticmethod
    def boundary_f(state, component: str = "h"):
        if component == "h":
            return state
        elif component == "u":
            u = state.u.at[-2, :].set(jnp.asarray(0.0))
            return eqx.tree_at(lambda x: x.u, state, u)
        elif component == "v":
            v = state.v.at[:, -2].set(jnp.asarray(0.0))
            return eqx.tree_at(lambda x: x.v, state, v)
        else:
            msg = f"Unrecognized component: {component}"
            msg += "\nNeeds to be h, u, or v"
            raise ValueError(msg)

    @staticmethod
    def equation_of_motion(t: float, state: State, args):
        """2D Linear Shallow Water Equations

        Equation:
            ∂h/∂t + H (∂u/∂x + ∂v/∂y) = 0
            ∂u/∂t - fv = - g ∂h/∂x - ku
            ∂v/∂t + fu = - g ∂h/∂y - kv
        """

        # apply boundary conditions
        h = enforce_boundaries(state.h, "h")
        u = enforce_boundaries(state.u, "u")
        v = enforce_boundaries(state.v, "v")
        # update state
        state = eqx.tree_at(lambda x: x.u, state, u)
        state = eqx.tree_at(lambda x: x.v, state, v)
        state = eqx.tree_at(lambda x: x.h, state, h)

        # apply RHS
        h_rhs, u_rhs, v_rhs = equation_of_motion(state, args)

        # update state
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)

        return state

    @staticmethod
    def equation_of_motion_fast(t: float, state: State, args):
        """2D Linear Shallow Water Equations

        Equation:
            ∂h/∂t + H (∂u/∂x + ∂v/∂y) = 0
            ∂u/∂t - fv = - g ∂h/∂x - ku
            ∂v/∂t + fu = - g ∂h/∂y - kv
        """

        # apply boundary conditions
        h: Array = enforce_boundaries(state.h, "h")
        u = enforce_boundaries(state.u, "u")
        v = enforce_boundaries(state.v, "v")
        # update state
        state = eqx.tree_at(lambda x: x.u, state, u)
        state = eqx.tree_at(lambda x: x.v, state, v)
        state = eqx.tree_at(lambda x: x.h, state, h)

        # apply RHS
        h_rhs, u_rhs, v_rhs = equation_of_motion_original(state, args)

        # update state
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)

        return state


def enforce_boundaries(u, grid: str, periodic_x: bool = False):
    assert grid in ["h", "u", "v"]

    if periodic_x:
        u = u.at[0, :].set(u[-2, :])
        u = u.at[-1, :].set(u[1, :])

    elif grid == "u":
        u = u.at[-2, :].set(0.0)

    if grid == "v":
        u = u.at[:, -2].set(0.0)

    return u


def equation_of_motion(state: State, params: Params):
    h, u, v = state.h, state.u, state.v

    domain = params.domain

    Nx, Ny = h.shape

    # ================================
    # HEIGHT
    # ================================

    # pad
    h_ghost = jnp.pad(h[1:-1, 1:-1], pad_width=((1, 1), (1, 1)), mode="edge")
    h_ghost = enforce_boundaries(h_ghost, "h")

    # assert (Nx, Ny) == h_ghost.shape

    # move h onto u, v
    # grid interpolation
    h_on_u = F_grid.x_average_2D(h_ghost, padding="valid")
    h_on_v = F_grid.y_average_2D(h_ghost, padding="valid")

    # assert (Nx-1, Ny) == h_on_u.shape
    # assert (Nx, Ny-1) == h_on_v.shape

    # remove
    # x-axis: first grid point
    # y-axis: ghost points
    h_on_u = h_on_u[1:, 1:-1]
    h_on_v = h_on_v[1:-1, 1:]

    # assert (Nx-2, Ny-2) == h_on_u.shape
    # assert (Nx-2, Ny-2) == h_on_v.shape

    # hu, hv (interior only)
    hu_on_u = jnp.zeros_like(h_ghost)
    hv_on_v = jnp.zeros_like(h_ghost)
    hu_on_u = hu_on_u.at[1:-1, 1:-1].set(h_on_u * u[1:-1, 1:-1])
    hv_on_v = hv_on_v.at[1:-1, 1:-1].set(h_on_v * v[1:-1, 1:-1])

    # assert (Nx, Ny) == hu_on_u.shape
    # assert (Nx, Ny) == hv_on_v.shape

    hu_on_u = enforce_boundaries(hu_on_u, "h")
    hv_on_v = enforce_boundaries(hv_on_v, "h")

    # grid difference
    dhu_dx = F_grid.x_difference_2D(hu_on_u, step_size=domain.dx[0])
    dhv_dy = F_grid.y_difference_2D(hv_on_v, step_size=domain.dx[1])

    # assert (Nx-1, Ny) == dhu_dx.shape
    # assert (Nx, Ny-1) == dhv_dy.shape

    # remove
    # x-axis: last grid point
    # y-axis: ghost points
    dhu_dx = dhu_dx[:-1, 1:-1]
    dhv_dy = dhv_dy[1:-1, :-1]

    # assert (Nx-2, Ny-2) == dhu_dx.shape
    # assert (Nx-2, Ny-2) == dhv_dy.shape

    h_rhs = jnp.zeros_like(h)
    h_rhs = h_rhs.at[1:-1, 1:-1].set(-(dhu_dx + dhv_dy))

    # ================================
    # U-VELOCITY
    # ================================
    # PLANETARY VORTICITY
    planetary_vort = params.coriolis_param(domain)[1:-1, 1:-1]
    # assert (Nx-2, Ny-2) == planetary_vort.shape

    dv_dx = F_grid.x_difference_2D(v, step_size=domain.dx[0])
    du_dy = F_grid.y_difference_2D(u, step_size=domain.dx[1])

    # assert (Nx-1, Ny) == dv_dx.shape
    # assert (Nx, Ny-1) == du_dy.shape

    # remove:
    # x-axis: last grid point
    # y-axis: ghost points
    dv_dx = dv_dx[1:, 1:-1]
    du_dy = du_dy[1:-1, 1:]

    # assert (Nx-2, Ny-2) == dv_dx.shape
    # assert (Nx-2, Ny-2) == du_dy.shape

    relative_vort = dv_dx - du_dy

    h_on_vort = F_grid.center_average_2D(h_ghost)
    # assert (Nx-1, Ny-1) == h_on_vort.shape

    # remove:
    # x-axis: last grid point
    h_on_vort = h_on_vort[1:, 1:]
    # assert (Nx-2, Ny-2) == h_on_vort.shape

    pv = jnp.zeros_like(h)
    pv = pv.at[1:-1, 1:-1].set((planetary_vort + relative_vort) / h_on_vort)
    # assert (Nx, Ny) == pv.shape

    # dhv ---> q domain
    hv_on_q = F_grid.x_average_2D(hv_on_v)
    # assert (Nx-1, Ny) == hv_on_q.shape

    hv_on_q = hv_on_q[1:, 1:-1]
    # assert (Nx-2, Ny-2) == hv_on_q.shape

    # ADVECTION TERM
    adv_rhs = jnp.zeros_like(h)
    adv_rhs = adv_rhs.at[1:-1, 1:-1].set(hv_on_q * pv[1:-1, 1:-1])

    # advection --> v domain
    adv_rhs_on_u = F_grid.y_average_2D(adv_rhs)
    # assert (Nx, Ny-1) == adv_rhs_on_u.shape

    adv_rhs_on_u = adv_rhs_on_u[1:-1, :-1]

    # assert (Nx-2, Ny-2) == adv_rhs_on_u.shape

    # KINETIC ENERGY
    ke_on_h = jnp.zeros_like(h)

    u2_on_h = F_grid.x_average_2D(u**2)
    v2_on_h = F_grid.y_average_2D(v**2)

    # assert (Nx-1, Ny) == u2_on_h.shape
    # assert (Nx, Ny-1) == v2_on_h.shape

    u2_on_h = u2_on_h[:-1, 1:-1]
    v2_on_h = v2_on_h[1:-1, :-1]

    # assert (Nx-2, Ny-2) == u2_on_h.shape
    # assert (Nx-2, Ny-2) == v2_on_h.shape

    ke_on_h = ke_on_h.at[1:-1, 1:-1].set(0.5 * (u2_on_h + v2_on_h))

    ke_on_h = enforce_boundaries(ke_on_h, "h")

    dke_on_u = -F_grid.x_difference_2D(ke_on_h, step_size=domain.dx[0])
    # assert (Nx-1, Ny) ==  dke_on_u.shape

    dke_on_u = dke_on_u[1:, 1:-1]
    # assert (Nx-2, Ny-2) == dke_on_u.shape

    dh_dx_on_u = F_grid.x_difference_2D(h, step_size=domain.dx[0])
    # assert (Nx-1, Ny) == dh_dx_on_u.shape

    dh_dx_on_u = dh_dx_on_u[1:, 1:-1]
    p_work_on_u = -params.gravity * dh_dx_on_u
    # assert (Nx-2, Ny-2) == p_work_on_u.shape

    u_rhs = jnp.zeros_like(h)
    u_rhs = u_rhs.at[1:-1, 1:-1].set(adv_rhs_on_u + dke_on_u + p_work_on_u)

    # ================================
    # V-VELOCITY
    # ================================

    # dhv ---> q domain
    hu_on_q = F_grid.y_average_2D(hu_on_u)
    # assert (Nx, Ny-1) == hu_on_q.shape

    hu_on_q = hu_on_q[1:-1, 1:]

    # ADVECTION TERM
    adv_rhs = jnp.zeros_like(h)
    adv_rhs = -adv_rhs.at[1:-1, 1:-1].set(hu_on_q * pv[1:-1, 1:-1])

    # advection --> v domain
    adv_rhs_on_v = F_grid.x_average_2D(adv_rhs)
    # assert (Nx-1, Ny) == adv_rhs_on_v.shape

    adv_rhs_on_v = adv_rhs_on_v[:-1, 1:-1]
    # assert (Nx-2, Ny-2) == adv_rhs_on_v.shape

    dke_on_v = -F_grid.y_difference_2D(ke_on_h, step_size=domain.dx[1])
    # assert (Nx, Ny-1) ==  dke_on_v.shape

    dke_on_v = dke_on_v[1:-1, 1:]
    # assert (Nx-2, Ny-2) == dke_on_u.shape

    dh_dx_on_v = F_grid.y_difference_2D(h, step_size=domain.dx[1])
    # assert (Nx, Ny-1) == dh_dx_on_v.shape

    dh_dx_on_v = dh_dx_on_v[1:-1, 1:]
    p_work_on_v = -params.gravity * dh_dx_on_v

    # assert (Nx-2, Ny-2) == p_work_on_v.shape

    v_rhs = jnp.zeros_like(h)
    v_rhs = v_rhs.at[1:-1, 1:-1].set(adv_rhs_on_v + dke_on_v + p_work_on_v)

    return h_rhs, u_rhs, v_rhs


def equation_of_motion_original(state, params):
    h, u, v = state.h, state.u, state.v

    domain = params.domain

    h = enforce_boundaries(h, "h", False)
    v = enforce_boundaries(v, "v", False)
    u = enforce_boundaries(u, "u", False)

    # pad
    # print("Ghost")
    h_node = jnp.pad(h[1:-1, 1:-1], 1, "edge")
    h_node = enforce_boundaries(h_node, "h", False)

    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    h_on_u = F_grid.x_interp_linear_2D(h_node[1:, 1:-1])

    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    h_on_v = F_grid.y_interp_linear_2D(h_node[1:-1, 1:])

    # hu, hv (interior only)
    # print("HU --> U,V")
    flux_on_u = jnp.zeros_like(h)
    flux_on_v = jnp.zeros_like(h)
    flux_on_u = flux_on_u.at[1:-1, 1:-1].set(h_on_u * u[1:-1, 1:-1])
    flux_on_v = flux_on_v.at[1:-1, 1:-1].set(h_on_v * v[1:-1, 1:-1])

    flux_on_u = enforce_boundaries(flux_on_u, "h", False)
    flux_on_v = enforce_boundaries(flux_on_v, "h", False)

    # finite difference
    # u --> h | top edge --> cell node | right edge --> cell center
    dh_dx = F_grid.x_difference_2D_(flux_on_u[:-1, 1:-1], step_size=domain.dx[0])
    # v --> h | right edge --> cell node | top edge --> cell center
    dh_dy = F_grid.y_difference_2D_(flux_on_v[1:-1, :-1], step_size=domain.dx[1])

    # print("H_RHS")
    h_rhs = jnp.zeros_like(h)
    h_rhs = h_rhs.at[1:-1, 1:-1].set(-(dh_dx + dh_dy))

    # planetary and relative vorticity
    planetary_vort = params.coriolis_param(domain)[1:-1, 1:-1]

    # relative vorticity
    # v --> q | right edge --> cell face | top edge --> cell node
    dv_dx = F_grid.x_difference_2D_(v[1:, 1:-1], step_size=domain.dx[0])
    # u --> q | top edge --> cell face | right edge --> cell node
    du_dy = F_grid.y_difference_2D_(u[1:-1, 1:], step_size=domain.dx[1])

    relative_vort = dv_dx - du_dy

    # calculate potential vorticity
    # h --> q | cell node --> cell face | cell face --> cell node
    # move h (cell node) to vort (cell center)
    h_on_vort = F_grid.center_average_2D_(h_node[1:, 1:])

    # print("Potential VORTICITY")
    potential_vort = jnp.zeros_like(h)
    potential_vort = potential_vort.at[1:-1, 1:-1].set(
        (planetary_vort + relative_vort) / h_on_vort
    )

    # enforce boundaries
    potential_vort = enforce_boundaries(potential_vort, "h", False)

    # flux on v (top edge) ---> vort (cell center)
    flux_on_q = F_grid.x_interp_linear_2D(flux_on_v)
    adv_rhs = F_grid.y_interp_linear_2D(potential_vort[1:-1, :-1] * flux_on_q[1:, :-1])

    # kinetic energy
    ke_on_h = jnp.zeros_like(h)

    # u --> h | top edge --> cell node | right edge --> cell center
    u2_on_h = F_grid.x_interp_linear_2D(u[:-1, 1:-1] ** 2)
    # v --> h | right edge --> cell node | top edge --> cell center
    v2_on_h = F_grid.y_interp_linear_2D(v[1:-1, :-1] ** 2)

    ke_on_h = ke_on_h.at[1:-1, 1:-1].set(0.5 * (u2_on_h + v2_on_h))

    # enforce boundary conditions
    ke_on_h = enforce_boundaries(ke_on_h, "h", False)

    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    dke_on_u = -F_grid.x_difference_2D_(ke_on_h[1:, 1:-1], step_size=domain.dx[0])

    # pressure work
    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    p_work = -params.gravity * F_grid.x_difference_2D_(
        h[1:, 1:-1], step_size=domain.dx[0]
    )

    u_rhs = jnp.zeros_like(h)
    u_rhs = u_rhs.at[1:-1, 1:-1].set(adv_rhs + p_work + dke_on_u)

    # u --> q | top edge --> cell face | right edge --> cell node
    flux_on_q = F_grid.y_interp_linear_2D(flux_on_u)
    adv_rhs = -F_grid.x_interp_linear_2D(potential_vort[:-1, 1:-1] * flux_on_q[:-1, 1:])

    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    dke_on_v = -F_grid.y_difference_2D_(ke_on_h[1:-1, 1:], step_size=domain.dx[1])

    # pressure work
    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    p_work = -params.gravity * F_grid.y_difference_2D_(
        h[1:-1, 1:], step_size=domain.dx[1]
    )

    v_rhs = jnp.zeros_like(h)
    v_rhs = v_rhs.at[1:-1, 1:-1].set(adv_rhs + p_work + dke_on_v)

    return h_rhs, u_rhs, v_rhs
