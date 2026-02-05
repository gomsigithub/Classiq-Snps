#!/usr/bin/env python3
"""
square_hole_grid.py

Build and visualize a uniform square mesh with a square hole removed.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Function Definition
# ---------------------------------------------------------------------------

def square_grid_with_square_hole_points(
    N_outer: int,
    L: float,
    hole_center: tuple[float, float],
    hole_size: float,
    *,
    align: str = "strict",          # "strict" or "snap"
    include_outer_boundary: bool = True,
    include_corners: bool = True,
    atol: float = 1e-12,
):
    """
    Create a uniform meshgrid on [0,L]x[0,L], remove points inside a square hole,
    and return holed arrays plus boundary index sets.

    Parameters
    ----------
    N_outer : int
        Number of grid points per side on the outer domain (including boundaries).
    L : float
        Side length of the outer square.
    hole_center : (float, float)
        (cx, cy) coordinates of the hole center.
    hole_size : float
        Side length of the (axis-aligned) square hole.
    align : {"strict","snap"}
        "strict" - require hole edges to fall exactly on grid lines,
        "snap"   - adjust each edge to nearest grid line.
    include_outer_boundary : bool
        Whether to return outer boundary masks/indices.
    include_corners : bool
        If False, exclude corner points from each edge mask.
    atol : float
        Tolerance for floating-point equality.

    Returns
    -------
    X_holed, Y_holed : 2D arrays (with NaN inside hole)
    keep_mask : 2D bool array (True outside/on hole boundary)
    XY_kept : (K, 2) array of kept node coordinates
    hole_bounds : dict {xL,xR,yB,yT} actual hole edges (after snapping)
    inner_boundary_nodes : dict of boundary info (nodes forming hole edges)
    outer_boundary_nodes : dict or None (nodes forming outer edges if include_outer_boundary=True)
   	"""

    if N_outer < 2:
        raise ValueError("N_outer must be >= 2")
    if hole_size <= 0.0:
        raise ValueError("hole_size must be positive")

    # --- Build uniform grid ---
    x = np.linspace(0, L, N_outer)
    y = np.linspace(0, L, N_outer)
    X, Y = np.meshgrid(x, y, indexing="xy")
    h = L / (N_outer-1)

    # --- Requested hole edges ---
    cx, cy = hole_center
    a = hole_size
    xL_req, xR_req = cx - a/2, cx + a/2
    yB_req, yT_req = cy - a/2, cy + a/2

    # --- Handle alignment ---
    def is_on_grid(val):
        return np.isclose((val / h) - np.round(val / h), 0.0, atol=atol)

    if align == "strict":
        if not (is_on_grid(xL_req) and is_on_grid(xR_req)
                and is_on_grid(yB_req) and is_on_grid(yT_req)):
            raise ValueError(
                "Hole edges not aligned with grid lines. "
                "Use align='snap' or choose compatible parameters."
            )
        xL, xR, yB, yT = xL_req, xR_req, yB_req, yT_req
    elif align == "snap":
        xL = np.round(xL_req / h) * h
        xR = np.round(xR_req / h) * h
        yB = np.round(yB_req / h) * h
        yT = np.round(yT_req / h) * h
        xL, xR = np.clip(sorted([xL, xR]), 0.0, L)
        yB, yT = np.clip(sorted([yB, yT]), 0.0, L)
    else:
        raise ValueError("align must be 'strict' or 'snap'")

    # --- Create mask and holed-out arrays ---
    inside = (X > xL) & (X < xR) & (Y > yB) & (Y < yT)
    keep_mask = ~inside
    X_holed = X.copy(); Y_holed = Y.copy()
    X_holed[~keep_mask] = np.nan
    Y_holed[~keep_mask] = np.nan
    XY_kept = np.c_[X[keep_mask], Y[keep_mask]]
    hole_bounds = dict(xL=float(xL), xR=float(xR), yB=float(yB), yT=float(yT))

    # --- Helper for edge info ---
    def edge_info(mask2d):
        ij = np.argwhere(mask2d)
        flat = np.flatnonzero(mask2d.ravel(order="C"))
        return {"mask": mask2d, "ij": ij, "flat": flat}

    # --- Inner (hole) boundary edges ---
    on_left   = np.isclose(X, xL, atol=atol) & (Y >= yB - atol) & (Y <= yT + atol)
    on_right  = np.isclose(X, xR, atol=atol) & (Y >= yB - atol) & (Y <= yT + atol)
    on_bottom = np.isclose(Y, yB, atol=atol) & (X >= xL - atol) & (X <= xR + atol)
    on_top    = np.isclose(Y, yT, atol=atol) & (X >= xL - atol) & (X <= xR + atol)

    if not include_corners:
        at_corners = (
            (np.isclose(X, xL, atol=atol) | np.isclose(X, xR, atol=atol)) &
            (np.isclose(Y, yB, atol=atol) | np.isclose(Y, yT, atol=atol))
        )
        on_left &= ~at_corners; on_right &= ~at_corners
        on_bottom &= ~at_corners; on_top &= ~at_corners

    inner_nodes = {
        "left":   edge_info(on_left),
        "right":  edge_info(on_right),
        "bottom": edge_info(on_bottom),
        "top":    edge_info(on_top),
    }

    # --- Outer boundary (optional) ---
    outer_nodes = None
    if include_outer_boundary:
        left_o   = np.isclose(X, 0.0, atol=atol)
        right_o  = np.isclose(X, L,   atol=atol)
        bottom_o = np.isclose(Y, 0.0, atol=atol)
        top_o    = np.isclose(Y, L,   atol=atol)
        if not include_corners:
            at_corners = (
                (np.isclose(X, 0.0, atol=atol) | np.isclose(X, L, atol=atol)) &
                (np.isclose(Y, 0.0, atol=atol) | np.isclose(Y, L, atol=atol))
            )
            left_o &= ~at_corners; right_o &= ~at_corners
            bottom_o &= ~at_corners; top_o &= ~at_corners
        outer_nodes = {
            "left":   edge_info(left_o),
            "right":  edge_info(right_o),
            "bottom": edge_info(bottom_o),
            "top":    edge_info(top_o),
        }

    return X_holed, Y_holed, keep_mask, XY_kept, hole_bounds, inner_nodes, outer_nodes


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_holed, Y_holed, keep_mask, XY_kept, hole_bounds, inner_nodes, outer_nodes = (
        square_grid_with_square_hole_points(
            N_outer=8,
            L=1.0,
            hole_center=(0.5, 0.5),
            hole_size=0.4,
            align="snap",
            include_outer_boundary=True,
        )
    )

    print("Hole bounds:", hole_bounds)
    print("Kept points:", XY_kept.shape[0], "/", X_holed.size)
    print("Inner edge node counts:",
          {k: v["ij"].shape[0] for k, v in inner_nodes.items()})

    # --- Plot holed-out grid ---
    plt.figure(figsize=(6,6))
	
	# Draw gridlines: loop over rows & cols, skip nan segments automatically
    for i in range(X_holed.shape[0]):  # horizontal lines
        plt.plot(X_holed[i, :], Y_holed[i, :], linewidth=0.5, color='k')
    for j in range(X_holed.shape[1]):  # vertical lines
        plt.plot(X_holed[:, j], Y_holed[:, j], linewidth=0.5, color='k')

    plt.scatter(X_holed, Y_holed, s=8)
    plt.gca().set_aspect('equal')
    plt.title("Uniform grid with square hole (nodes in hole replaced with NANs)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.show()
