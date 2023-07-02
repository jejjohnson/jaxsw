from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import lineax as lx
from jaxopt import linear_solve as jopt_linear_solver
from jaxtyping import Array


class SDResults(NamedTuple):
    u: Array
    residual: Array
    b: Array
    iteration: int
    loss: Array


def l1_norm(u, u_ref):
    l1_diff = jnp.sum(jnp.abs(u - u_ref)) / jnp.sum(jnp.abs(u_ref))
    return l1_diff


def l2_norm(u: Array, u_ref: Array) -> Array:
    l2_diff = jnp.linalg.norm(u - u_ref, "fro") / jnp.linalg.norm(u_ref, "fro")
    return l2_diff


def steepest_descent(
    b: Array,
    matvec_fn: Callable[Array, Array],
    u_init: Optional[Array] = None,
    target_criterion: float = 1e-4,
    max_iterations: int = 1,
    criterion: str = "l2",
) -> Array:
    residual = jnp.zeros_like(b)
    if u_init is None:
        u_init = jnp.zeros_like(b)

    # initialize state
    state = SDResults(
        u=u_init,
        residual=residual,
        b=b,
        iteration=0,
        loss=1 + target_criterion,
    )

    if criterion == "l1":
        criterion = l1_norm
    elif criterion == "l2":
        criterion = l2_norm
    else:
        raise ValueError("Unrecognized criterion")

    def condition_fn(state) -> bool:
        return (state.iteration < max_iterations) & (state.loss > target_criterion)

    def body_fn(state):
        Ax = matvec_fn(state.u)

        # compute residual
        residual = state.b - Ax

        # compute eom of residual
        Aresidual = matvec_fn(residual)

        # compute stepsize
        alpha = jnp.sum(residual * residual) / jnp.sum(residual * Aresidual)

        # update solution
        u = state.u.copy() + alpha * residual

        # compute loss
        loss = criterion(u, state.u)

        state = state._replace(
            u=u, residual=residual, iteration=state.iteration + 1, loss=loss
        )

        return state

    state = jax.lax.while_loop(condition_fn, body_fn, state)

    return state


class CGResults(NamedTuple):
    u: Array
    residual: Array
    direction: Array
    b: Array
    iteration: int
    loss: Array


def conjugate_gradient(
    b: Array,
    matvec_fn: Callable[Array, Array],
    u_init: Optional[Array] = None,
    target_criterion: float = 1e-4,
    max_iterations: int = 1,
    criterion: str = "l2",
) -> Array:
    residual = jnp.zeros_like(b)
    if u_init is None:
        u_init = jnp.zeros_like(b)

    # compute initial residual
    residual = b - matvec_fn(u_init)

    # compute initial direction
    direction = residual.copy()

    # initialize state
    state = CGResults(
        u=u_init,
        residual=residual,
        direction=direction,
        b=b,
        iteration=0,
        loss=1 + target_criterion,
    )

    if criterion == "l1":
        criterion = l1_norm
    elif criterion == "l2":
        criterion = l2_norm
    else:
        raise ValueError("Unrecognized criterion")

    def condition_fn(state) -> bool:
        return (state.iteration < max_iterations) & (state.loss > target_criterion)

    def body_fn(state):
        # compute search direction
        A_direction = matvec_fn(state.direction)

        # compute stepsize
        alpha = jnp.sum(state.residual * state.residual) / jnp.sum(
            state.direction * A_direction
        )

        # update solution
        u = state.u + alpha * state.direction

        # update residual
        residual = state.residual - alpha * A_direction

        # update search direction
        beta = jnp.sum(residual * residual) / jnp.sum(state.residual * state.residual)

        # update direction
        direction = residual + beta * state.direction

        # compute loss
        loss = criterion(u, state.u)

        state = state._replace(
            u=u,
            residual=residual,
            direction=direction,
            iteration=state.iteration + 1,
            loss=loss,
        )

        return state

    state = jax.lax.while_loop(condition_fn, body_fn, state)

    return state


def lx_linear_solver(
    matvec_fn: Callable,
    b: Array,
    solver: lx.AbstractLinearSolver = lx.GMRES,
    x0: Optional[Array] = None,
    verbose: bool = False,
):
    # get the size of structure
    in_structure = jax.eval_shape(lambda: b)

    # define operator based on vector
    operator = lx.FunctionLinearOperator(matvec_fn, in_structure)

    # get solution
    if x0 is None:
        x0 = matvec_fn(b)
    solution = lx.linear_solve(operator, b, solver, options=dict(y0=x0))

    if verbose:
        print(solution.stats)

    return solution.value


def jaxopt_linear_solver(
    matvec_fn: Callable,
    b: Array,
    x0: Optional[Array] = None,
    solver: Callable = jopt_linear_solver.solve_cg,
    **kwargs
):
    if x0 is None:
        x0 = matvec_fn(b)
    return solver(matvec=matvec_fn, b=b, init=x0, **kwargs)
