"""
Build IO balancing constraints with an embedded footprint target, in the
b_ij parametrisation used in the GBM / footprint-extended write-up:

    flows are  f_ij = b_ij * x_j

    row margin   i :   sum_j  x_j      * b_ij = u_i
    col margin   j :   sum_i  x_j      * b_ij = v_j
    footprint      :   sum    S_j x_j  * b_ij = E

The footprint constraint is just another row of G with coefficients S_j x_j,
solved by KRAS like any other (non-unity, possibly relaxable) constraint.

Vectorisation is row-major (C order): position of b_ij is  i * n_cols + j.
"""

import numpy as np


def build_io_footprint_constraints(n_rows, n_cols, x, u, v,
                                   S=None, E=None,
                                   footprint_mode="per_row",
                                   include_row=True,
                                   include_col=True):
    """
    Returns
    -------
    G : (n_C, n_rows*n_cols) ndarray
    c : (n_C,) ndarray
    kind : list[str]   one of {"row", "col", "footprint"} per constraint,
                       so you can assign sigma by role.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n_var = n_rows * n_cols

    def pos(i, j):
        return i * n_cols + j

    rows_G, c_list, kind = [], [], []

    if include_row:
        for i in range(n_rows):
            g = np.zeros(n_var)
            for j in range(n_cols):
                g[pos(i, j)] = x[j]            # coefficient x_j
            rows_G.append(g); c_list.append(u[i]); kind.append("row")

    if include_col:
        for j in range(n_cols):
            g = np.zeros(n_var)
            for i in range(n_rows):
                g[pos(i, j)] = x[j]            # coefficient x_j
            rows_G.append(g); c_list.append(v[j]); kind.append("col")

    if S is not None and E is not None:
        S = np.asarray(S, dtype=float)
        w = S * x                              # footprint coefficient S_j x_j
        if footprint_mode == "global":
            g = np.zeros(n_var)
            for i in range(n_rows):
                for j in range(n_cols):
                    g[pos(i, j)] = w[j]
            rows_G.append(g); c_list.append(float(E)); kind.append("footprint")
        elif footprint_mode == "per_row":
            E = np.asarray(E, dtype=float)     # one target per row
            for i in range(n_rows):
                g = np.zeros(n_var)
                for j in range(n_cols):
                    g[pos(i, j)] = w[j]
                rows_G.append(g); c_list.append(float(E[i])); kind.append("footprint")
        else:
            raise ValueError("footprint_mode must be 'global' or 'per_row'")

    return np.array(rows_G), np.array(c_list), kind


def sigma_by_kind(kind, sigma_margin=0.0, sigma_footprint=1.0):
    """Assign a standard deviation to each constraint by its role."""
    return np.array([sigma_footprint if k == "footprint" else sigma_margin
                     for k in kind], dtype=float)


def footprint_of(B, x, S):
    """Total embodied footprint of matrix B given output x and intensities S."""
    B = np.asarray(B, dtype=float)
    return float(np.sum(B * (S * x)[None, :]))


def row_footprints(B, x, S):
    """Per-row embodied footprint vector."""
    B = np.asarray(B, dtype=float)
    return (B * (S * x)[None, :]).sum(axis=1)
