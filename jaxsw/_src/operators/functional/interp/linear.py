from jaxtyping import Array


def linear_2pts(qm: Array, qp: Array) -> Array:
    """
    2-points linear reconstruction:

    qm--x--qp

    """
    return 0.5 * (qm + qp)


def linear_3pts_left(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    """
    return -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp


def linear_3pts_right(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points linear left-biased stencil reconstruction:

    qp--x--q0-----qm

    """
    return linear_3pts_left(qm=qp, q0=q0, qp=qm)


def linear_4pts(qmm: Array, qm: Array, qp: Array, qpp: Array) -> Array:
    """
    4-points linear reconstruction:

    qmm-----qm--x--qp-----qpp

    """
    return -1.0 / 12.0 * qmm + 7.0 / 12.0 * qm + 7.0 / 12.0 * qp - 1.0 / 12.0 * qpp


def linear_5pts_left(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """
    5-points linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    """
    return (
        1.0 / 30.0 * qmm
        - 13.0 / 60.0 * qm
        + 47.0 / 60.0 * q0
        + 9.0 / 20.0 * qp
        - 1. / 20. * qpp
    )


def linear_5pts_right(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """
    5-points linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    """
    return linear_5pts_left(qmm=qpp, qm=qp, q0=q0, qp=qm, qpp=qmm)
