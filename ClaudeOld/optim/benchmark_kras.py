"""
Scalability / stability benchmark for the KRAS implementation.

For a range of matrix dimensions n it builds an n x n IO balancing problem in
the b_ij parametrisation (flows = b_ij * x_j) with sparse constraints, and
measures, for two scenarios:

  A) Consistent RAS  -- recover exact margins from a perturbed start (sigma=0).
  B) KRAS footprint  -- soft margins + a global footprint target (conflicting),
                        reconciled by constraint relaxation.

Reported per n: variables, constraints, wall-clock time, sweeps, max constraint
residual (stability), and whether it converged.  A rough time-vs-n scaling
exponent is fitted at the end.
"""

import time
import numpy as np
from scipy import sparse

from kras import kras

np.set_printoptions(precision=4, suppress=True)


# --------------------------------------------------------------------------- #
# Sparse constraint builder (margins with coeff x_j, optional global footprint)
# --------------------------------------------------------------------------- #
def build_sparse_io(n, x, S=None, u=None, v=None, E=None):
    """Return (G_csr, c) for an n x n problem; footprint row appended if E given."""
    n_var = n * n
    rows, cols, data = [], [], []
    cval = []
    r = 0
    # row margins:  sum_j x_j b_ij = u_i
    for i in range(n):
        for j in range(n):
            rows.append(r); cols.append(i * n + j); data.append(x[j])
        cval.append(u[i]); r += 1
    # col margins:  sum_i x_j b_ij = v_j
    for j in range(n):
        for i in range(n):
            rows.append(r); cols.append(i * n + j); data.append(x[j])
        cval.append(v[j]); r += 1
    # global footprint:  sum_ij S_j x_j b_ij = E
    if E is not None:
        w = S * x
        for i in range(n):
            for j in range(n):
                rows.append(r); cols.append(i * n + j); data.append(w[j])
        cval.append(float(E)); r += 1
    G = sparse.csr_matrix((data, (rows, cols)), shape=(r, n_var))
    return G, np.array(cval)


def make_problem(n, seed=0):
    rng = np.random.default_rng(seed)
    B0 = rng.uniform(0.01, 0.3, size=(n, n))      # positive technical coeffs
    x = rng.uniform(50.0, 200.0, size=n)          # outputs
    S = rng.uniform(0.05, 2.0, size=n)            # footprint intensities
    flows = B0 * x[None, :]
    u = flows.sum(axis=1)                          # row margins
    v = flows.sum(axis=0)                          # col margins
    return B0, x, S, u, v


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def run(n_list, time_budget_s=60.0):
    print(f"{'n':>5} {'n_var':>8} {'n_C':>6}  {'scenario':<12} "
          f"{'time_s':>9} {'sweeps':>7} {'max_resid':>11} {'conv':>5}")
    print("-" * 74)

    results = {"A": [], "B": []}

    for n in n_list:
        B0, x, S, u, v = make_problem(n)

        # ---- Scenario A: consistent RAS (recover margins from perturbed start)
        rng = np.random.default_rng(1)
        A_init = (B0 * np.exp(rng.normal(0, 0.4, size=B0.shape))).flatten()
        G, c = build_sparse_io(n, x, u=u, v=v)
        t0 = time.perf_counter()
        a, info = kras(A_init, G, c, sigma=None, tol=1e-6, max_sweeps=20000)
        dt = time.perf_counter() - t0
        resid = float(np.max(np.abs(G @ a - c)))
        print(f"{n:>5} {n*n:>8} {G.shape[0]:>6}  {'A consistent':<12} "
              f"{dt:>9.3f} {info['sweeps']:>7} {resid:>11.2e} "
              f"{str(info['converged']):>5}")
        results["A"].append((n, dt, info["sweeps"], resid, info["converged"]))

        # ---- Scenario B: KRAS with a conflicting global footprint target ------
        E_consistent = float(np.sum(S * v))
        E_target = 0.85 * E_consistent
        G2, c2 = build_sparse_io(n, x, S=S, u=u, v=v, E=E_target)
        n_C = G2.shape[0]
        # margins soft-ish, footprint more reliable so it steers the solution
        sig = np.empty(n_C)
        sig[:2 * n] = 0.02 * v.mean()        # margins
        sig[-1] = 0.20 * v.mean()            # footprint target
        t0 = time.perf_counter()
        a2, info2 = kras(B0.flatten(), G2, c2, sigma=sig, alpha=0.1,
                         tol=1e-5, max_sweeps=20000)
        dt2 = time.perf_counter() - t0
        resid2 = float(np.max(np.abs(G2 @ a2 - info2["c_final"])))
        fp_ach = float(np.sum((a2.reshape(n, n)) * (S * x)[None, :]))
        print(f"{n:>5} {n*n:>8} {n_C:>6}  {'B footprint':<12} "
              f"{dt2:>9.3f} {info2['sweeps']:>7} {resid2:>11.2e} "
              f"{str(info2['converged']):>5}"
              f"   (footprint {fp_ach:.1f} -> target {E_target:.1f}, "
              f"consistent {E_consistent:.1f})")
        results["B"].append((n, dt2, info2["sweeps"], resid2, info2["converged"]))

        if dt + dt2 > time_budget_s:
            print(f"\n[stopping: last n took {dt + dt2:.1f}s > budget {time_budget_s}s]")
            break

    # ---- rough scaling exponent for the consistent scenario ------------------
    A = np.array([(n, dt) for (n, dt, *_ ) in results["A"] if dt > 0])
    if len(A) >= 3:
        # fit log(time) = p*log(n) + b over the larger sizes
        big = A[A[:, 0] >= A[:, 0].max() / 8]
        p, b = np.polyfit(np.log(big[:, 0]), np.log(big[:, 1]), 1)
        print("\n" + "-" * 74)
        print(f"Consistent-RAS wall-clock scales roughly as  n^{p:.2f}  "
              f"(over n in [{int(big[:,0].min())}, {int(big[:,0].max())}]).")
        print("Per-sweep cost is O(n_C * support) = O(n^2); total = that x #sweeps.")
    return results


if __name__ == "__main__":
    n_list = [5, 10, 20, 40, 80, 160, 320]
    run(n_list, time_budget_s=90.0)
