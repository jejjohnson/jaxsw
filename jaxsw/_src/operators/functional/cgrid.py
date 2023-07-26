from jaxtyping import Array

from jaxsw._src.operators.functional import grid as F_grid


def edge_to_node_1D(u: Array) -> Array:
    """1D Arakawa C-Grid. Moves the array from
    the top-down edge to a node. Typically used
    for converting the velocity to the quantity
    of interest.
        Input:  u -- + -- u
        Output: . -- u̅ -- .

    Args:
        u (Array): Array on a edge
            Size = [Nx]

    Returns:
        Array: Array on the node
            Size = [Nx - 1]
    """
    u = F_grid.x_average_1D(u, padding="valid")
    return u


def node_to_edge_1D(u: Array) -> Array:
    """1D Arakawa C-Grid. Moves the array from
    the node to the edge. Typically used
    for converting the quantity of interest to
    the velocity.
        Input:  u -- . -- u
        Output: x -- u̅ -- x

    Args:
        u (Array): Array on the node
            Size = [Nx]

    Returns:
        Array: Array on the edge
            Size = [Nx - 1]
    """
    u = edge_to_node_1D(u)
    return u


def node_to_face_2D(u: Array) -> Array:
    """Transforms the variable from cell node
    to the cell face.

    Example:
        Transform from Psi --> Q
    """
    return F_grid.center_average_2D(u, padding="valid")


def node_to_edge_tb_2D(u: Array) -> Array:
    """Transforms the variable from Cell Node
    to the Top-Down Edge.

    Example:
        Transform from PSI --> U-VEL
    """
    return F_grid.x_average_2D(u, padding="valid")


def node_to_edge_lr_2D(u: Array) -> Array:
    """Transforms the variable from cell node
    to the Left-Right Edge.

    Example:
        Transform from PSI --> V-VEL
    """
    return F_grid.y_average_2D(u, padding="valid")


def face_to_node_2D(u: Array) -> Array:
    """Transforms the variable from Cell Face
    edge to the cell node.

    Example:
        Transform from Q --> Psi
    """
    return F_grid.center_average_2D(u, padding="valid")


def face_to_edge_tb_2D(u: Array) -> Array:
    """Transforms the variable from Cell Face
    to the Top-Down Edge.

    Example:
        Transform from Q --> U-VEL
    """
    return F_grid.y_average_2D(u, padding="valid")


def face_to_edge_lr_2D(u: Array) -> Array:
    """Transforms the variable from cell face
    to the Left-Right Edge.

    Example:
        Transform from Q --> V-VEL
    """
    return F_grid.x_average_2D(u, padding="valid")


def edge_tb_to_edge_lr(u: Array) -> Array:
    """Transforms the variable from Top-Down
    edge to the Left-Right Edge.

    Example:
        Transform from U-Vel --> V-VEL
    """
    return F_grid.center_average_2D(u, padding="valid")


def edge_tb_to_node_2D(u: Array) -> Array:
    """Transforms the variable from Top-Down
    edge to the cell node.

    Example:
        Transform from U-Vel --> Psi
    """
    return F_grid.x_average_2D(u, padding="valid")


def edge_tb_to_face_2D(u: Array) -> Array:
    """Transforms the variable from Top-Down
    edge to the cell face.

    Example:
        Transform from U-Vel --> Q
    """
    return F_grid.y_average_2D(u, padding="valid")


def edge_lr_to_node_2D(u: Array) -> Array:
    """Transforms the variable from Left-Right
    edge to the cell node.

    Example:
        Transform from V-Vel --> Psi
    """
    return F_grid.y_average_2D(u, padding="valid")


def edge_lr_to_face_2D(u: Array) -> Array:
    """Transforms the variable from Left-Right
    edge to the cell face.

    Example:
        Transform from V-Vel --> Q
    """
    return F_grid.x_average_2D(u, padding="valid")


def edge_lr_to_edge_tb_2D(u: Array) -> Array:
    """Transforms the variable from Left-Right
    edge to the Top-Down Edge.

    Example:
        Transform from V-VEL --> U-VEL
    """
    return F_grid.center_average_2D(u, padding="valid")
