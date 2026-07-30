import numpy as np
from kras import kras
from io_footprint import (build_io_footprint_constraints, sigma_by_kind,
                          footprint_of, row_footprints)

np.set_printoptions(precision=4, suppress=True)

# --- a small synthetic IO problem in the b_ij parametrisation -------------- #
n = 4
B0 = np.array([[0.20, 0.10, 0.05, 0.02],
               [0.08, 0.25, 0.06, 0.03],
               [0.04, 0.07, 0.30, 0.05],
               [0.02, 0.03, 0.08, 0.22]])
x = np.array([100., 120., 90., 150.])          # sector outputs
S = np.array([2.0, 0.3, 1.2, 0.1])             # footprint intensities (sector j)

# margins implied by B0 (so the baseline is self-consistent)
u = (B0 * x[None, :]).sum(axis=1)              # row margins  sum_j x_j b_ij
v = (B0 * x[None, :]).sum(axis=0)              # col margins  sum_i x_j b_ij

print("baseline row footprints :", row_footprints(B0, x, S))
print("baseline total footprint: %.2f" % footprint_of(B0, x, S))
print("sum_j S_j v_j (= total with exact col margins): %.2f"
      % float(np.sum(S * v)))
print()

# =========================================================================== #
print("=" * 70)
print("MODE 1 -- per-row footprint target (recompose within fixed margins)")
print("=" * 70)
# With col margins fixed, the TOTAL footprint is locked at sum_j S_j v_j, so
# per-row targets can only REDISTRIBUTE footprint across rows (sum preserved).
# Here: partially equalise row footprints (blend baseline with the row mean).
base_fp = row_footprints(B0, x, S)
E_target = 0.5 * base_fp + 0.5 * base_fp.mean()   # sums to the locked total
print("baseline -> target row footprints:")
for i in range(n):
    print("  row %d: %6.2f -> %6.2f" % (i, base_fp[i], E_target[i]))
# feasibility: E_i/u_i must stay within [min S_j, max S_j]
ratio = E_target / u
print("target footprint-intensity per row:", ratio,
      " (must lie in [%.1f, %.1f])" % (S.min(), S.max()))

G, c, kind = build_io_footprint_constraints(n, n, x, u, v,
                                            S=S, E=E_target,
                                            footprint_mode="per_row")
# margins exact (sigma=0), footprint soft (sigma>0) so KRAS steers toward it
sigma = sigma_by_kind(kind, sigma_margin=0.0, sigma_footprint=u.mean() * 0.02)
a, info = kras(B0.flatten(), G, c, sigma=sigma, alpha=0.1, tol=1e-9,
               max_sweeps=20000)
B1 = a.reshape(n, n)

print("\nbalanced B (row-normalised view):\n", B1)
print("row margins  :", (B1 * x[None, :]).sum(1), " target", u)
print("col margins  :", (B1 * x[None, :]).sum(0), " target", v)
print("row footprint:", row_footprints(B1, x, S), " target", E_target)
print("margins exact:", np.allclose((B1 * x).sum(1), u, atol=1e-6)
      and np.allclose((B1 * x).sum(0), v, atol=1e-6))
print("total footprint %.2f  (baseline %.2f)"
      % (footprint_of(B1, x, S), footprint_of(B0, x, S)))

# =========================================================================== #
print("\n" + "=" * 70)
print("MODE 2 -- global footprint target vs margins: the sigma trade-off")
print("=" * 70)
# Ask for a 30% lower TOTAL footprint than sum_j S_j v_j. With exact col
# margins this is over-determined -> who yields depends on sigma ratios.
E_glob = 0.70 * float(np.sum(S * v))
print("global target E = %.2f  (consistent value = %.2f)\n"
      % (E_glob, float(np.sum(S * v))))

for sig_fp_mult in [0.2, 1.0, 5.0]:
    G, c, kind = build_io_footprint_constraints(n, n, x, u, v,
                                                S=S, E=E_glob,
                                                footprint_mode="global")
    # margins and footprint BOTH soft; vary their relative reliability
    sig = np.array([
        (v.mean() * 0.05) if k in ("row", "col")
        else (v.mean() * 0.05 * sig_fp_mult)
        for k in kind])
    a, info = kras(B0.flatten(), G, c, sigma=sig, alpha=0.1, tol=1e-7,
                   max_sweeps=20000)
    B2 = a.reshape(n, n)
    fp = footprint_of(B2, x, S)
    col_err = np.max(np.abs((B2 * x).sum(0) - v))
    print("sigma_fp/sigma_margin = %4.1f  ->  total footprint = %7.2f "
          "(target %.2f) | max col-margin error = %.3f"
          % (sig_fp_mult, fp, E_glob, col_err))

print("\n(higher footprint reliability -> footprint hits target, margins give;"
      "\n lower -> margins hold, footprint stays near its consistent value.)")
