import numpy as np
from square_hole_grid import square_grid_with_square_hole_points


# ------------------------------------------------------------
# Index utilities
# ------------------------------------------------------------

def GetEzDofNum(i, j, nx, ny, mask):
    """Return Ez DoF index or -1 if outside domain."""
    if i < 0 or j < 0 or i >= nx or j >= ny:
        return -1
    if ~mask[j, i]:
        return -1
    return j * nx + i


def GetHxDofNum(i, j, nx, ny, mask):
    """Return Hx DoF index or -1 if outside domain."""
    if i < 0 or j < 0 or i >= nx or j >= ny - 1:
        return -1
    if ~mask[j, i]:
        return -1
    return j * nx + i


def GetHyDofNum(i, j, nx, ny, mask):
    """Return Hy DoF index or -1 if outside domain."""
    if i < 0 or j < 0 or i >= nx - 1 or j >= ny:
        return -1
    if ~mask[j, i]:
        return -1
    return j * (nx - 1) + i + nx * (ny - 1)


# ------------------------------------------------------------
# Matrix builders
# ------------------------------------------------------------

def build_AE_matrix(nx, ny, ezMask, hxMask, hyMask, E_coef):
    """Construct the A_E matrix."""
    n_e = nx * ny
    n_hx = nx * (ny - 1)
    n_hy = (nx - 1) * ny
    AE = np.zeros((n_e, n_hx + n_hy))

    for j in range(ny):
        for i in range(nx):

            if ~ezMask[j, i]:
                ez_ix = GetEzDofNum(i, j, nx, ny, ezMask)
                AE[ez_ix][:] = 0
                continue

            ez_ix = GetEzDofNum(i, j, nx, ny, ezMask)

            # dHy/dx
            hy_plus = GetHyDofNum(i, j, nx, ny, hyMask)
            hy_minus = GetHyDofNum(i - 1, j, nx, ny, hyMask)

            if hy_plus == -1:
                AE[ez_ix][hy_minus] = -2.0 * E_coef
            elif hy_minus == -1:
                AE[ez_ix][hy_plus] = 2.0 * E_coef
            else:
                AE[ez_ix][hy_plus] = E_coef
                AE[ez_ix][hy_minus] = -E_coef

            # -dHx/dy
            hx_plus = GetHxDofNum(i, j, nx, ny, hxMask)
            hx_minus = GetHxDofNum(i, j - 1, nx, ny, hxMask)

            if hx_plus == -1:
                AE[ez_ix][hx_minus] = 2.0 * E_coef
            elif hx_minus == -1:
                AE[ez_ix][hx_plus] = -2.0 * E_coef
            else:
                AE[ez_ix][hx_plus] = -E_coef
                AE[ez_ix][hx_minus] = E_coef

    return AE


def build_AH_matrix(nx, ny, ezMask, hxMask, hyMask, H_coef):
    """Construct the A_H matrix."""
    n_e = nx * ny
    n_hx = nx * (ny - 1)
    n_hy = (nx - 1) * ny
    AH = np.zeros((n_hx + n_hy, n_e))

    # Hx part
    for j in range(ny - 1):
        for i in range(nx):
            if ~hxMask[j, i]:
                hx_ix = GetHxDofNum(i, j, nx, ny, hxMask)
                AH[hx_ix][:] = 0
                continue

            hx_ix = GetHxDofNum(i, j, nx, ny, hxMask)

            ez_plus = GetEzDofNum(i, j + 1, nx, ny, ezMask)
            ez_minus = GetEzDofNum(i, j, nx, ny, ezMask)

            AH[hx_ix][ez_plus] = -H_coef
            AH[hx_ix][ez_minus] = H_coef

    # Hy part
    for j in range(ny):
        for i in range(nx - 1):
            if ~hyMask[j, i]:
                hy_ix = GetHyDofNum(i, j, nx, ny, hyMask)
                AH[hy_ix][:] = 0
                continue

            hy_ix = GetHyDofNum(i, j, nx, ny, hyMask)

            ez_plus = GetEzDofNum(i + 1, j, nx, ny, ezMask)
            ez_minus = GetEzDofNum(i, j, nx, ny, ezMask)

            AH[hy_ix][ez_plus] = H_coef
            AH[hy_ix][ez_minus] = -H_coef

    return AH


# ------------------------------------------------------------
# High-level function for notebooks
# ------------------------------------------------------------

def generate_fdtd_matrices(
        nx,
        ny,
        L,
        hole_center=(0, 0),
        hole_size=0.0,
        eps_r=1.0
    ):
    """
    Generate A_E and A_H FDTD matrices.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    L : float
        Side length of the domain.
    hole_center : tuple(float, float)
        (x, y) center of square hole.
    hole_size : float
        Side length of hole. Use 0 for no hole.
    eps_r : float
        Relative permittivity.

    Returns
    -------
    AE, AH : ndarray
        The A_E and A_H matrices.
    """

    X, Y, keep_mask, XY_kept, hole_bounds, inner_nodes, outer_nodes = \
        square_grid_with_square_hole_points(
            N_outer=nx,
            L=L,
            hole_center=hole_center,
            hole_size=hole_size,
            align="snap",
            include_outer_boundary=True
        )

    h = X[0][1] - X[0][0]

    ezMask = keep_mask.copy()
    hxMask = ezMask[:-1, :] * ezMask[1:, :]
    hyMask = ezMask[:, :-1] * ezMask[:, 1:]

    eps = eps_r
    mu0 = 1.0

    E_coef = 1.0 / (h * eps)
    H_coef = 1.0 / (h * mu0)

    AE = build_AE_matrix(nx, ny, ezMask, hxMask, hyMask, E_coef)
    AH = build_AH_matrix(nx, ny, ezMask, hxMask, hyMask, H_coef)

    return AE, AH

def AE_AH_to_A(AE, AH):
    zeros_e = np.zeros((AE.shape[0], AE.shape[0]))  # top-left block
    zeros_h = np.zeros((AH.shape[0], AH.shape[0]))  # bottom-right block

    A = np.block([
        [zeros_e, AE],
        [AH,      zeros_h]
    ])
    return A

def forward_diff_rect(N, dx=1):
    # Create a matrix of size (N-1) x N
    D = np.zeros((N, N))
    for i in range(N-1):
        D[i, i] = -1.0
        D[i, i+1] = 1.0
    return D / dx

def backward_diff_rect(N, dx=1):
    # Create a matrix of size (N-1) x N
    D = np.zeros((N, N-1))
    for i in range(1, N-1):
        D[i, i] = 1.0
        D[i, i-1] = -1.0
    D[0,0] = 2
    D[N-1, N-2] = -2
    return D / dx 

def backward_diff_body(nx, ny, dx=1):
    # Create a matrix of size (N-1) x N
    D = np.zeros((nx, ny-1))
    for i in range(1, N-1):
        D[i, i] = 1.0
        D[i, i-1] = -1.0
    D[0,0] = 2
    D[N-1, N-2] = -2
    return D / dx 
