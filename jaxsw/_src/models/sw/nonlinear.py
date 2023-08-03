import jax.numpy as jnp
from jaxsw._src.models.pde import DynamicalSystem
from jaxsw._src.operators.functional import grid as F_grid
from jaxtyping import Array
import equinox as eqx
from jaxsw._src.models.sw import Params, State


def enforce_boundaries(u: Array, component: str = "h", periodic: bool = False):
    if periodic:
        u = u.at[0, :].set(u[-2, :])
        u = u.at[-1, :].set(u[1, :])
    if component == "h":
        return u
    elif component == "u":
        return u.at[-2, :].set(jnp.asarray(0.0))
    elif component == "v":
        return u.at[:, -2].set(jnp.asarray(0.0))
    else:
        msg = f"Unrecognized component: {component}"
        msg += "\nNeeds to be h, u, or v"
        raise ValueError(msg)


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
    def equation_of_motion(t: float, state: State, args: Params):
        """2D Shallow Water Equations

        Equation:
            ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
            ∂u/∂t + u ∂u/∂x + v ∂u/∂x - fv = - g ∂h/∂x - ku
            ∂v/∂t + u ∂v/∂x + v ∂v/∂x + fu = - g ∂h/∂y - kv
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
        h_rhs, u_rhs, v_rhs = equation_of_motion(state, args)

        # update state
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)

        return state

    @staticmethod
    def equation_of_motion_advection(t: float, state: State, args: Params):
        """2D Shallow Water Equations

        Equation:
            ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
            ∂u/∂t + u ∂u/∂x + v ∂u/∂x - fv = - g ∂h/∂x - ku
            ∂v/∂t + u ∂v/∂x + v ∂v/∂x + fu = - g ∂h/∂y - kv
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
        h_rhs, u_rhs, v_rhs = equation_of_motion_advection(state, args)

        # update state
        state = eqx.tree_at(lambda x: x.u, state, u_rhs)
        state = eqx.tree_at(lambda x: x.v, state, v_rhs)
        state = eqx.tree_at(lambda x: x.h, state, h_rhs)

        return state


def equation_of_motion(state: State, params: Params):
    interp_method = "upwind"
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
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    h_on_u = F_grid.interp(h_node, axis=0, method=interp_method)[1:, 1:-1]

    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    h_on_v = F_grid.interp(h_node, axis=1, method=interp_method)[1:-1, 1:]

    # hu, hv (interior only)
    # print("HU --> U,V")
    uh_on_u = jnp.zeros_like(h)
    vh_on_v = jnp.zeros_like(h)
    uh_on_u = uh_on_u.at[1:-1, 1:-1].set(h_on_u * u[1:-1, 1:-1])
    vh_on_v = vh_on_v.at[1:-1, 1:-1].set(h_on_v * v[1:-1, 1:-1])

    uh_on_u = enforce_boundaries(uh_on_u, "h", False)
    vh_on_v = enforce_boundaries(vh_on_v, "h", False)

    # ########################################
    # HEIGHT EQUATION
    # ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
    # ########################################

    # finite difference
    # u --> h | top edge --> cell node | right edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    duh_dx = F_grid.difference(
        uh_on_u, step_size=domain.dx[0], axis=0, accuracy=1, method="right"
    )[:-1, 1:-1]

    # v --> h | right edge --> cell node | top edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    dvh_dy = F_grid.difference(
        vh_on_v, step_size=domain.dx[1], axis=1, accuracy=1, method="right"
    )[1:-1, :-1]

    # print("H_RHS")
    h_rhs = jnp.zeros_like(h)
    h_rhs = h_rhs.at[1:-1, 1:-1].set(-(duh_dx + dvh_dy))

    # ####################################################
    # NONLINEAR TERMS
    # ∂u/∂t + u ∂u/∂x + v ∂u/∂x - fv = - g ∂h/∂x - ku
    # ###################################################
    # planetary and relative vorticity
    planetary_vort = params.coriolis_param(domain)[1:-1, 1:-1]

    # relative vorticity
    # v --> q | right edge --> cell face | top edge --> cell node
    # [Nx+2,Nx+2] --> [Nx+1,Nx+2] --> [Nx,Ny]
    dv_dx = F_grid.difference(
        v, axis=0, step_size=domain.dx[0], accuracy=1, method="right"
    )[1:, 1:-1]
    # u --> q | top edge --> cell face | right edge --> cell node
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    du_dy = F_grid.difference(
        u, axis=1, step_size=domain.dx[1], accuracy=1, method="right"
    )[1:-1, 1:]

    relative_vort = dv_dx - du_dy

    # calculate potential vorticity
    # h --> q | cell node --> cell face | cell face --> cell node
    # [Nx+2,Ny+2] --> [Nx+1,Ny+1] --> [Nx,Ny]
    h_on_vort = F_grid.interp_center(h_node, method="linear")[1:, 1:]

    # print("Potential VORTICITY")
    # [Nx+2,Ny+2]
    potential_vort = jnp.zeros_like(h)
    potential_vort = potential_vort.at[1:-1, 1:-1].set(
        (planetary_vort + relative_vort) / h_on_vort
    )

    # enforce boundaries
    potential_vort = enforce_boundaries(potential_vort, "h", False)

    # u --> h | top edge --> cell node | right edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    u2_on_h = F_grid.interp(u**2, axis=0, method=interp_method)[:-1, 1:-1]
    # v --> h | right edge --> cell node | top edge --> cell center
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    v2_on_h = F_grid.interp(v**2, axis=1, method=interp_method)[1:-1, :-1]

    # ####################################################
    # U-VELOCITY EQUATION
    # ∂u/∂t + u ∂u/∂x + v ∂u/∂x - fv = - g ∂h/∂x - ku
    # ###################################################

    u_rhs = jnp.zeros_like(h)

    # WORK RHS
    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny]
    p_work = (
        -params.gravity * F_grid.difference(h, step_size=domain.dx[0], axis=0)[1:, 1:-1]
    )

    u_rhs += u_rhs.at[1:-1, 1:-1].set(p_work)

    # ADVECTION RHS
    # flux on v (top edge) ---> vort (cell center)
    # [Nx+2,Ny+2] --> [Nx+1,Ny+2] --> [Nx,Ny+2]
    vh_on_q = F_grid.interp(vh_on_v, axis=0, method=interp_method)[1:]
    # PV: [Nx+2,Ny+2] --> [Nx,Ny+1], [Nx,Ny+1] --> [Nx,Ny]
    adv_rhs = F_grid.interp(
        potential_vort[1:-1] * vh_on_q, axis=1, method=interp_method
    )

    # PV: [Nx,Ny+1] --> [Nx,Ny]
    adv_rhs = adv_rhs[:, :-1]

    u_rhs += u_rhs.at[1:-1, 1:-1].set(adv_rhs)

    # KINETIC ENERGY RHS
    ke_on_h = jnp.zeros_like(h)
    ke_on_h = ke_on_h.at[1:-1, 1:-1].set(0.5 * (u2_on_h + v2_on_h))

    # enforce boundary conditions
    ke_on_h = enforce_boundaries(ke_on_h, "h", False)

    # move h to u | (cell node) -> (top edge) | (cell center) --> (right edge)
    # [Nx+2,Ny+2] --> [Nx+1,Ny+1] --> [Nx,Ny]
    dke_on_u = -F_grid.difference(ke_on_h, step_size=domain.dx[0], axis=0)[1:, 1:-1]

    u_rhs += u_rhs.at[1:-1, 1:-1].set(dke_on_u)

    # ####################################################
    # V-VELOCITY EQUATION
    # ∂u/∂t + u ∂v/∂x + v ∂v/∂x + fu = - g ∂h/∂y - kv
    # ###################################################

    v_rhs = jnp.zeros_like(h)

    # WORK TERM: - g ∂h/∂y
    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny]
    p_work = (
        -params.gravity * F_grid.difference(h, axis=1, step_size=domain.dx[1])[1:-1, 1:]
    )

    v_rhs += v_rhs.at[1:-1, 1:-1].set(p_work)

    # flux on u (top edge) ---> vort (cell center)
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx+2,Ny]
    uh_on_q = F_grid.interp(uh_on_u, axis=1, method=interp_method)[:, 1:]
    # PV: [Nx+2,Ny+2] --> [Nx+1,Ny+2], [Nx+1,Ny+2] --> [Nx,Ny]
    adv_rhs = -F_grid.interp(
        potential_vort[:, 1:-1] * uh_on_q, axis=0, method=interp_method
    )

    # PV: [Nx+1,Ny] --> [Nx,Ny]
    adv_rhs = adv_rhs[:-1]

    v_rhs += v_rhs.at[1:-1, 1:-1].set(adv_rhs)

    # move h to v | (cell node) --> (right edge) | (cell center) --> (top edge)
    dke_on_v = -F_grid.y_difference_2D_(ke_on_h[1:-1, 1:], step_size=domain.dx[1])

    v_rhs += v_rhs.at[1:-1, 1:-1].set(dke_on_v)

    return h_rhs, u_rhs, v_rhs


def equation_of_motion_advection(state: State, params: Params):
    interp_method = "linear"
    adv_method = "linear"

    h, u, v = state.h, state.u, state.v

    domain = params.domain

    h = enforce_boundaries(h, "h", False)
    v = enforce_boundaries(v, "v", False)
    u = enforce_boundaries(u, "u", False)

    # pad
    # print("Ghost")
    h_pad = jnp.pad(h[1:-1, 1:-1], 1, "edge")
    h_pad = enforce_boundaries(h_pad, "h", False)

    u_pad = jnp.pad(u[1:-1, 1:-1], pad_width=((1, 1), (1, 1)), mode="constant")
    v_pad = jnp.pad(v[1:-1, 1:-1], pad_width=((1, 1), (1, 1)), mode="constant")
    v_pad = enforce_boundaries(v_pad, "v", False)
    u_pad = enforce_boundaries(u_pad, "u", False)
    u_pad = jnp.pad(u_pad, pad_width=((1, 0), (0, 0)), mode="constant")
    v_pad = jnp.pad(v_pad, pad_width=((0, 0), (1, 0)), mode="constant")

    # ####################################
    # Nonlinear Dynamcis
    #
    # ####################################
    # [Nx+3,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny+1]
    u_at_v = F_grid.interp_center(u_pad)[1:]
    # [Nx+2,Ny+3] --> [Nx+1,Ny+2] --> [Nx+1,Ny]
    v_at_u = F_grid.interp_center(v_pad)[:, 1:]

    # [Nx+3,Ny+2] --> [Nx+2,Ny+2] --> [Nx+2,Ny]
    ubar_x = F_grid.interp(u_pad, axis=0, method=interp_method)[:, 1:-1]
    # [Nx+3,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny+1]
    ubar_y = F_grid.interp(u_pad, axis=1, method=interp_method)[1:-1, :]

    # [Nx+2,Ny+3] --> [Nx+1,Ny+3] --> [Nx+1,Ny+1]
    vbar_x = F_grid.interp(v_pad, axis=0, method=interp_method)[:, 1:-1]
    # [Nx+2,Ny+3] --> [Nx+2,Ny+2] --> [Nx+2,Ny]
    vbar_y = F_grid.interp(v_pad, axis=1, method=interp_method)[1:-1, :]

    # ########################################
    # HEIGHT EQUATION
    # ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
    # ########################################

    h_rhs = jnp.zeros_like(h)

    # H --> U,V
    # [Nx+2,Ny+2] --> [Nx+1, Ny+2] --> [Nx+1, Ny]
    h_on_u = F_grid.interp(h_pad, axis=0, method=interp_method)[:, 1:-1]
    # [Nx+2,Ny+2] --> [Nx+2,Ny+1] --> [Nx,Ny+1]
    h_on_v = F_grid.interp(h_pad, axis=1, method=interp_method)[1:-1, :]

    # Advection Terms:  ∂/∂x((H+h)u) + ∂/∂y((H+h)v)
    # duh_dx, dvh_dy
    duh_dx = F_grid.difference(
        h_on_u * u_pad[1:-1, 1:-1], axis=0, step_size=domain.dx[0], method="right"
    )
    dvh_dy = F_grid.difference(
        h_on_v * v_pad[1:-1, 1:-1], axis=1, step_size=domain.dx[1], method="right"
    )

    h_rhs += h_rhs.at[1:-1, 1:-1].set(-(duh_dx + dvh_dy))

    # ####################################################
    # U-VELOCITY EQUATION
    # ∂u/∂t + u ∂u/∂x + v ∂u/∂x - fv = - g ∂h/∂x - ku
    # ###################################################

    u_rhs = jnp.zeros_like(u_pad)

    # work term: - g ∂h/∂x
    # [Nx+2,Ny+2] --> [Nx+1, Ny+2] --> [Nx+1,Ny]
    dhdx_on_u = F_grid.difference(
        h_pad, axis=0, step_size=domain.dx[0], method="right"
    )[:, 1:-1]
    u_rhs += u_rhs.at[1:-1, 1:-1].set(-params.gravity * dhdx_on_u)

    # nonlinear advection terms: u ∂u/∂x + v ∂u/∂x
    if adv_method == "upwind":
        # **UPWIND SCHEME**
        udu_dx = F_grid.difference(
            ubar_x, axis=0, step_size=domain.dx[0], method="upwind", a=ubar_x
        )
        vdu_dy = F_grid.difference(
            ubar_y, axis=1, step_size=domain.dx[1], method="upwind", a=v_at_u
        )
    else:
        # **LINEAR SCHEME**
        udu_dx = 0.5 * F_grid.difference(
            ubar_x**2, axis=0, step_size=domain.dx[0], method="right"
        )
        vdu_dy = v_at_u[:, 1:] * F_grid.difference(
            ubar_y, axis=1, step_size=domain.dx[1], method="right"
        )

    u_rhs += u_rhs.at[1:-1, 1:-1].set(-udu_dx - vdu_dy)

    # planetary forces: + fv
    fv = params.coriolis_f0 * v_at_u[:, 1:]

    u_rhs += u_rhs.at[1:-1, 1:-1].set(fv)

    # ####################################################
    # V-VELOCITY EQUATION
    # ∂v/∂t + u ∂v/∂x + v ∂v/∂x + fu = - g ∂h/∂y - kv
    # ###################################################
    v_rhs = jnp.zeros_like(v_pad)

    # work term: - g ∂h/∂x
    # [Nx+2,Ny+2] --> [Nx+1, Ny+2] --> [Nx+1,Ny]
    dhdy_on_v = F_grid.difference(
        h_pad, axis=1, step_size=domain.dx[1], method="right"
    )[1:-1]
    v_rhs += v_rhs.at[1:-1, 1:-1].set(-params.gravity * dhdy_on_v)

    # nonlinear advection terms: u ∂v/∂x + v ∂v/∂x
    if adv_method == "upwind":
        # **UPWIND SCHEME**
        vdv_dx = F_grid.difference(
            vbar_x, axis=0, step_size=domain.dx[0], method="upwind", a=u_at_v
        )
        vdv_dy = F_grid.difference(
            vbar_y, axis=1, step_size=domain.dx[1], method="upwind", a=vbar_y
        )
    else:
        # **STANDARD SCHEME**
        vdv_dx = u_at_v[1:] * F_grid.difference(
            vbar_x, axis=0, step_size=domain.dx[0], method="right"
        )
        vdv_dy = 0.5 * F_grid.difference(
            vbar_y**2, axis=1, step_size=domain.dx[1], method="right"
        )

    v_rhs += v_rhs.at[1:-1, 1:-1].set(-vdv_dx - vdv_dy)

    # planetary forces: - fu
    fu = params.coriolis_f0 * u_at_v[1:]

    v_rhs += v_rhs.at[1:-1, 1:-1].set(-fu)

    return h_rhs, u_rhs[1:], v_rhs[:, 1:]
