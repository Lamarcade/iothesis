"""
KRAS -- Konfliktfreies RAS (conflict-free RAS).

Reference implementation of the generalised iterative scaling method of

    Lenzen, M., Gallego, B. & Wood, R. (2009)
    "Matrix Balancing under Conflicting Information",
    Economic Systems Research 21(1), 23-44.
    doi:10.1080/09535310802688661

KRAS extends GRAS (Junius & Oosterhaven, 2003) so that it can:
  (a) impose constraints on arbitrary subsets of matrix elements (G a = c),
  (b) use non-unity real constraint coefficients g_ij in R,
  (c) balance negative elements while preserving their sign,
  (d) reconcile *conflicting* / inconsistent constraints by relaxing each
      constraint value c_i towards its realisation (G a)_i, bounded by the
      constraint's standard deviation sigma_i.

The minimisation underlying the scaling is the GRAS information-gain function
    f(a, a0) = sum_k |a_k| * ln( a_k / (e * a0_k) )           (Eq. 7)
subject to  G a = c.

The per-constraint multiplicative update (Eqs. 8, 17, 19, 23) is:
    a_k <- a_k * r_i ** ( sign(a_k) * g_ik )
where r_i > 0 solves, for the constraint currently being processed,
    P_i(r) = sum_k  g_ik * a_k * r ** ( sign(a_k) * g_ik )  -  c_i  =  0.
P_i is monotonically increasing in r on (0, inf), so the positive root is
unique and is found here by a safeguarded Newton / bisection solver
(numpy only -- no SciPy dependency).
"""

from __future__ import annotations

import numpy as np

try:                                   # optional: enables large sparse systems
    from scipy import sparse as _sp
    _HAVE_SCIPY = True
except ImportError:                    # pragma: no cover
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Per-constraint scaling factor r_i  (root of the GRAS polynomial, Eq. 19)
# --------------------------------------------------------------------------- #
def _solve_scaling(coeffs: np.ndarray,
                   vals: np.ndarray,
                   target: float,
                   tol: float = 1e-13,
                   max_iter: int = 200) -> float:
    """
    Solve  sum_k coeffs_k * vals_k * r ** (sign(vals_k) * coeffs_k) = target
    for r > 0.  Only entries with vals_k != 0 and coeffs_k != 0 participate
    (a zero element cannot be moved by multiplicative scaling -- the known
    "unadjustable zeros" property of RAS/GRAS).
    """
    mask = (coeffs != 0.0) & (vals != 0.0)
    if not np.any(mask):
        # Nothing to scale: constraint cannot be enforced multiplicatively.
        return 1.0

    g = coeffs[mask].astype(float)
    a = vals[mask].astype(float)
    e = np.sign(a) * g                 # effective exponent on r
    base = g * a                       # signed contribution at r = 1

    def P(r: float) -> float:
        return float(np.sum(base * r ** e) - target)

    def dP(r: float) -> float:
        return float(np.sum(base * e * r ** (e - 1.0)))

    # P is monotonically increasing in r > 0  ->  unique positive root.
    p1 = P(1.0)
    if abs(p1) <= tol:
        return 1.0

    # Bracket the root by expanding outward from r = 1.
    if p1 > 0.0:                       # need to *decrease* P -> r < 1
        hi, lo = 1.0, 0.5
        while P(lo) > 0.0:
            hi, lo = lo, lo * 0.5
            if lo < 1e-300:
                return lo              # no sign change: clamp
    else:                              # need to *increase* P -> r > 1
        lo, hi = 1.0, 2.0
        while P(hi) < 0.0:
            lo, hi = hi, hi * 2.0
            if hi > 1e300:
                return hi              # no sign change: clamp

    # Safeguarded Newton inside [lo, hi] (bisection fallback if Newton strays).
    r = 0.5 * (lo + hi)
    for _ in range(max_iter):
        pr = P(r)
        if abs(pr) <= tol:
            return r
        if pr > 0.0:
            hi = r
        else:
            lo = r
        d = dP(r)
        step_ok = False
        if d != 0.0:
            r_newton = r - pr / d
            if lo < r_newton < hi:
                r, step_ok = r_newton, True
        if not step_ok:
            r = 0.5 * (lo + hi)
        if hi - lo <= tol * max(1.0, r):
            break
    return r

# --------------------------------------------------------------------------- #
# Main KRAS routine
# --------------------------------------------------------------------------- #
def kras(a0,
         G,
         c,
         sigma=None,
         alpha: float = 0.1,
         max_sweeps: int = 5000,
         tol: float = 1e-8,
         stall_tol: float = 1e-6,
         min_gras_sweeps: int = 2,
         verbose: bool = False):
    """
    Balance / reconcile a (vectorised) matrix under linear constraints.

    Parameters
    ----------
    a0 : (n_var,) array_like
        Initial estimate of the unknown vectorised matrix.  Signs are taken
        from a0 and preserved throughout (GRAS property).  Exact zeros stay
        zero.
    G : (n_C, n_var) array_like
        Constraint matrix.  Row i encodes constraint  sum_k G[i,k]*a_k = c[i].
        Row/column sums, single cells, aggregates and non-unity coefficients
        are all just rows of G.
    c : (n_C,) array_like
        Target values for each constraint.
    sigma : (n_C,) array_like, optional
        Standard deviation (reliability) of each constraint value.
        sigma_i = 0  ->  the constraint is treated as exact (never relaxed).
        Larger sigma_i  ->  the constraint is relaxed more readily when the
        system is over-determined / conflicting.  Defaults to all zeros
        (pure GRAS: no constraint relaxation).
    alpha : float, default 0.1
        Constraint-relaxation step, as a fraction of one sigma per touch
        (Eq. 23).  Larger = faster but coarser relaxation.
    max_sweeps : int
        Maximum number of full passes over all constraints.
    tol : float
        Convergence tolerance on the max constraint residual
        max_i |(G a)_i - c_i^cur| (against the *current*, possibly relaxed, c).
    stall_tol : float
        Relative-improvement threshold used to detect that the pure-GRAS
        phase has stalled (which only happens under conflicting constraints);
        once stalled, constraint relaxation (the "K" in KRAS) switches on.
    min_gras_sweeps : int
        Minimum GRAS sweeps before relaxation may switch on.

    Returns
    -------
    a : (n_var,) ndarray
        Balanced vector.
    info : dict
        Diagnostics: 'sweeps', 'phase_switch', 'c_final', 'residual_vs_c0',
        'residual_vs_cfinal', 'converged'.
    """
    a = np.asarray(a0, dtype=float).copy()
    c0 = np.asarray(c, dtype=float).copy()

    # Accept either a dense array or a scipy.sparse matrix for G.  The dense
    # path is unchanged; the sparse path stores G as CSR so that G @ a and the
    # per-row supports are cheap even for large systems (EXIOBASE scale).
    is_sparse = _HAVE_SCIPY and _sp.issparse(G)
    if is_sparse:
        G = G.tocsr()
        n_C, n_var = G.shape
        # (idx, g_i) for each constraint row, read straight from CSR buffers.
        supports = [(G.indices[G.indptr[i]:G.indptr[i + 1]],
                     G.data[G.indptr[i]:G.indptr[i + 1]].astype(float))
                    for i in range(n_C)]
    else:
        G = np.asarray(G, dtype=float)
        n_C, n_var = G.shape
        supports = []
        for i in range(n_C):
            idx = np.nonzero(G[i])[0]
            supports.append((idx, G[i, idx].astype(float)))

    if a.shape[0] != n_var:
        raise ValueError("a0 length must equal number of columns of G")
    if c0.shape[0] != n_C:
        raise ValueError("c length must equal number of rows of G")

    sigma = (np.zeros(n_C) if sigma is None
             else np.asarray(sigma, dtype=float).copy())

    c_cur = c0.copy()
    phase = "gras"                     # "gras" -> "kras"
    phase_switch = None
    prev_dist = np.inf
    converged = False

    for sweep in range(1, max_sweeps + 1):
        for i in range(n_C):
            idx, g_i = supports[i]
            if idx.size == 0:
                continue
            Ga_i = float(np.dot(g_i, a[idx]))

            # --- KRAS: relax the constraint value towards its realisation ---
            if phase == "kras" and sigma[i] > 0.0:
                resid = c_cur[i] - Ga_i
                step = min(abs(resid), alpha * sigma[i])
                c_cur[i] -= np.sign(resid) * step          # Eq. 23

            # --- solve for the scaling factor and apply the update ----------
            r = _solve_scaling(g_i, a[idx], c_cur[i])
            if r > 0.0 and np.isfinite(r):
                a[idx] = a[idx] * r ** (np.sign(a[idx]) * g_i)

        # --- distances at end of sweep ---------------------------------------
        Ga = G @ a
        dist_cur = float(np.max(np.abs(Ga - c_cur))) if n_C else 0.0
        dist_c0 = float(np.max(np.abs(Ga - c0))) if n_C else 0.0

        if verbose:
            print(f"sweep {sweep:5d}  phase={phase:4s}  "
                  f"max|Ga-c_cur|={dist_cur:.3e}  max|Ga-c0|={dist_c0:.3e}")

        if phase == "gras":
            improvement = (prev_dist - dist_c0) / prev_dist if prev_dist > 0 else 0.0
            if dist_c0 <= tol:
                converged = True
                break
            if sweep >= min_gras_sweeps and improvement < stall_tol:
                # GRAS has stalled on conflicting constraints -> turn on "K".
                if np.any(sigma > 0.0):
                    phase = "kras"
                    phase_switch = sweep
                else:
                    # No reliability info: best feasible GRAS solution reached.
                    break
            prev_dist = dist_c0
        else:  # kras phase: converged once Ga matches the (relaxed) c_cur
            if dist_cur <= tol:
                converged = True
                break

    Ga = G @ a
    info = {
        "sweeps": sweep,
        "phase_switch": phase_switch,
        "c_final": c_cur,
        "residual_vs_c0": Ga - c0,
        "residual_vs_cfinal": Ga - c_cur,
        "converged": converged,
    }
    return a, info


# --------------------------------------------------------------------------- #
# Convenience builders for the classic row/column-sum balancing problem
# --------------------------------------------------------------------------- #
def build_margin_constraints(n_rows: int, n_cols: int, u, v):
    """
    Build (G, c) for the standard problem of fixing row sums u and column
    sums v of an (n_rows x n_cols) matrix vectorised in row-major (C) order.

    Returns G (shape (n_rows+n_cols, n_rows*n_cols)) and c (row sums then
    column sums).  Note: row and column totals must be consistent
    (sum(u) == sum(v)) for an exact RAS solution to exist; if not, supply
    sigma > 0 to let KRAS find a compromise.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n_var = n_rows * n_cols
    G = np.zeros((n_rows + n_cols, n_var))
    for i in range(n_rows):
        for j in range(n_cols):
            G[i, i * n_cols + j] = 1.0           # row-sum constraint i
    for j in range(n_cols):
        for i in range(n_rows):
            G[n_rows + j, i * n_cols + j] = 1.0   # col-sum constraint j
    c = np.concatenate([u, v])
    return G, c
