"""
Testing the full Arakawa-C grid.
        P -- U -- P
        |         |
        V    Q    V
        |         |
        P -- U -- P

"""
import numpy as np
import pytest

from jaxsw._src.domain.base import Domain
from jaxsw._src.operators.functional import grid as F_grid


@pytest.fixture
def PSIDomain_1D():
    # STREAM FUNCTION - CELL CORNER
    nrows = 5
    return Domain(xmin=(0,), xmax=(nrows + 1,), dx=(1,))


@pytest.fixture
def UDomain_1D():
    # ZONAL VELOCITY - EAST-WEST EDGE
    nrows = 5
    return Domain(xmin=(0.5,), xmax=(nrows + 0.5,), dx=(1,))


@pytest.fixture
def PSIDomain_2D():
    # STREAM FUNCTION - CELL CORNER
    nrows, ncols = 3, 5
    return Domain(xmin=(0, 0), xmax=(nrows + 1, ncols + 1), dx=1)


@pytest.fixture
def QDomain_2D():
    # POTENTIAL VORTICITY - CELL FACE
    nrows, ncols = 3, 5
    return Domain(xmin=(0.5, 0.5), xmax=(nrows + 0.5, ncols + 0.5), dx=1)


@pytest.fixture
def UDomain_2D():
    # ZONAL VELOCITY - EAST-WEST EDGE
    nrows, ncols = 3, 5
    return Domain(xmin=(0.5, 0), xmax=(nrows + 0.5, ncols + 1), dx=1)


@pytest.fixture
def VDomain_2D():
    # MERIDIONAL VELOCITY - NORTH-SOUTH EDGE
    nrows, ncols = 3, 5
    return Domain(xmin=(0, 0.5), xmax=(nrows + 1, ncols + 0.5), dx=1)


####################################
# 1D PSI TRANSFORMATIONS
####################################


def test_x_average_1D_node_to_edge(PSIDomain_1D, UDomain_1D):
    # =======================================================
    # PSI Domain (Cell-Node) ----> U Domain (Top-Bottom Edge)
    # =======================================================
    # X-Direction
    psi_on_u_x = F_grid.x_average_1D(PSIDomain_1D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(UDomain_1D.grid[..., 0], psi_on_u_x)


####################################
# 1D U-VELOCITY TRANSFORMATIONS
####################################


def test_x_average_1D_edge_to_node(UDomain_1D, PSIDomain_1D):
    # =======================================================
    # PSI Domain (Cell-Node) ----> U Domain (Top-Bottom Edge)
    # =======================================================
    # X-Direction
    u_on_psi_x = F_grid.x_average_1D(UDomain_1D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(PSIDomain_1D.grid[1:-1, 0], u_on_psi_x)


####################################
# 2D U-VELOCITY TRANSFORMATIONS
####################################


def test_y_average_2D_tb_edge_to_face(UDomain_2D, QDomain_2D):
    # =======================================================
    # U Domain (Top-Down Edges) ----> Q Domain (Cell-Face)
    # =======================================================
    # X-Direction
    u_on_q_x = F_grid.y_average_2D(UDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(QDomain_2D.grid[..., 0], u_on_q_x)

    # Y-Direction
    u_on_q_y = F_grid.y_average_2D(UDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(QDomain_2D.grid[..., 1], u_on_q_y)


def test_x_average_2D_yb_edge_to_node(UDomain_2D, PSIDomain_2D):
    # =======================================================
    # U Domain (Top-Down Edges) ----> PSI Domain (Cell-Node)
    # =======================================================
    # X-Direction
    u_on_psi_x = F_grid.x_average_2D(UDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(PSIDomain_2D.grid[1:-1, :, 0], u_on_psi_x)

    # Y-Direction
    u_on_psi_y = F_grid.x_average_2D(UDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(PSIDomain_2D.grid[1:-1, :, 1], u_on_psi_y)


def test_center_average_2D_tb_edge_to_lr_edge(UDomain_2D, VDomain_2D):
    # =======================================================
    # U Domain (Top-Down Edges) ----> V Domain (Left-Right Edges)
    # =======================================================
    # X-Direction
    u_on_v_x = F_grid.center_average_2D(UDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(VDomain_2D.grid[1:-1, :, 0], u_on_v_x)

    # Y-Direction
    u_on_v_y = F_grid.center_average_2D(UDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(VDomain_2D.grid[1:-1, :, 1], u_on_v_y)


####################################
# 2D V-VELOCITY TRANSFORMATIONS
####################################


def test_x_average_2D_lr_edge_to_face(VDomain_2D, QDomain_2D):
    # =======================================================
    # U Domain (Face) ----> V Domain (Top-Down Edge)
    # =======================================================
    # X-Direction
    v_on_q_x = F_grid.x_average_2D(VDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(QDomain_2D.grid[..., 0], v_on_q_x)

    # Y-Direction
    v_on_q_y = F_grid.x_average_2D(VDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(QDomain_2D.grid[..., 1], v_on_q_y)


def test_y_average_2D_lr_edge_to_node(VDomain_2D, PSIDomain_2D):
    # =======================================================
    # U Domain (Face) ----> PSI Domain (Cell-Node)
    # =======================================================
    # X-Direction
    v_on_psi_x = F_grid.y_average_2D(VDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(PSIDomain_2D.grid[:, 1:-1, 0], v_on_psi_x)

    # Y-Direction
    v_on_psi_y = F_grid.y_average_2D(VDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(PSIDomain_2D.grid[:, 1:-1, 1], v_on_psi_y)


def test_center_average_2D_lr_edge_to_ew_edge(VDomain_2D, UDomain_2D):
    # =======================================================
    # V Domain (Top-Down Edge) ----> U Domain (Left-Right Edge)
    # =======================================================
    # X-Direction
    v_on_u_x = F_grid.center_average_2D(VDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(UDomain_2D.grid[:, 1:-1, 0], v_on_u_x)

    # Y-Direction
    v_on_u_y = F_grid.center_average_2D(VDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(UDomain_2D.grid[:, 1:-1, 1], v_on_u_y)


####################################
# 2D Q TRANSFORMATIONS
####################################


def test_y_average_2D_face_to_tb_edge(QDomain_2D, UDomain_2D):
    # =======================================================
    # Q Domain (Face) ----> U Domain (Top-Bottom Edge)
    # =======================================================
    # X-Direction
    q_on_u_x = F_grid.y_average_2D(QDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(UDomain_2D.grid[:, 1:-1, 0], q_on_u_x)

    # Y-Direction
    q_on_u_y = F_grid.y_average_2D(QDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(UDomain_2D.grid[:, 1:-1, 1], q_on_u_y)


def test_x_average_2D_face_to_lr_edge(QDomain_2D, VDomain_2D):
    # =======================================================
    # Q Domain (Face) ----> V Domain (Left-Right Edge)
    # =======================================================
    # X-Direction
    q_on_v_x = F_grid.x_average_2D(QDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(VDomain_2D.grid[1:-1, :, 0], q_on_v_x)

    # Y-Direction
    q_on_v_y = F_grid.x_average_2D(QDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(VDomain_2D.grid[1:-1, :, 1], q_on_v_y)


def test_center_average_2D_face_to_node(QDomain_2D, PSIDomain_2D):
    # =======================================================
    # Q Domain (Face) ----> PSI Domain (Cell-Node)
    # =======================================================
    # X-Direction
    q_on_psi_x = F_grid.center_average_2D(QDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(PSIDomain_2D.grid[1:-1, 1:-1, 0], q_on_psi_x)

    # Y-Direction
    q_on_psi_y = F_grid.center_average_2D(QDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(PSIDomain_2D.grid[1:-1, 1:-1, 1], q_on_psi_y)


####################################
# 2D PSI TRANSFORMATIONS
####################################


def test_x_average_2D_node_to_tb_edge(PSIDomain_2D, UDomain_2D):
    # =======================================================
    # PSI Domain (Cell-Node) ----> U Domain (Top-Bottom Edge)
    # =======================================================
    # X-Direction
    psi_on_u_x = F_grid.x_average_2D(PSIDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(UDomain_2D.grid[..., 0], psi_on_u_x)

    # Y-Direction
    psi_on_u_y = F_grid.x_average_2D(PSIDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(UDomain_2D.grid[..., 1], psi_on_u_y)


def test_y_average_2D_node_to_lr_edge(PSIDomain_2D, VDomain_2D):
    # =======================================================
    # Q Domain (Face) ----> V Domain (Left-Right Edge)
    # =======================================================
    # X-Direction
    psi_on_v_x = F_grid.y_average_2D(PSIDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(VDomain_2D.grid[..., 0], psi_on_v_x)

    # Y-Direction
    psi_on_v_y = F_grid.y_average_2D(PSIDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(VDomain_2D.grid[..., 1], psi_on_v_y)


def test_center_average_2D_node_to_face(PSIDomain_2D, QDomain_2D):
    # =======================================================
    # Q Domain (Face) ----> PSI Domain (Cell-Node)
    # =======================================================
    # X-Direction
    psi_on_q_x = F_grid.center_average_2D(PSIDomain_2D.grid[..., 0], padding="valid")

    np.testing.assert_array_equal(QDomain_2D.grid[..., 0], psi_on_q_x)

    # Y-Direction
    psi_on_q_y = F_grid.center_average_2D(PSIDomain_2D.grid[..., 1], padding="valid")

    np.testing.assert_array_equal(QDomain_2D.grid[..., 1], psi_on_q_y)
