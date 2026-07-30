import numpy as np
from kras import kras, build_margin_constraints

np.set_printoptions(precision=4, suppress=True)


def test_consistent_ras():
    """Sanity check: a consistent row/col-sum problem must reproduce plain RAS."""
    print("=" * 64)
    print("0) Consistent RAS (sanity check)")
    A0 = np.array([[10., 20., 30.],
                   [40., 50., 60.],
                   [70., 80., 90.]])
    u = np.array([100., 200., 300.])   # target row sums
    v = np.array([150., 250., 200.])   # target col sums  (sum u == sum v == 600)
    G, c = build_margin_constraints(3, 3, u, v)
    a, info = kras(A0.flatten(), G, c, sigma=None, tol=1e-12)
    A = a.reshape(3, 3)
    print("balanced matrix:\n", A)
    print("row sums:", A.sum(1), " (target", u, ")")
    print("col sums:", A.sum(0), " (target", v, ")")
    assert np.allclose(A.sum(1), u, atol=1e-6)
    assert np.allclose(A.sum(0), v, atol=1e-6)
    print("PASS\n")


def test_cole():
    """Cole (1992) RAS-infeasible problem, equal sigma -> a ~ (0, 4/3.. ) etc.
    Paper (equal sigma): a = (0, 1.333, 1.333, 1.333),
    Ga - c = (1/3, -1/3, 1/3, -1/3, 1/3)."""
    print("=" * 64)
    print("1) Cole (1992) RAS-infeasible problem")
    G = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]], dtype=float)
    c = np.array([1, 3, 1, 3, 1], dtype=float)
    a0 = np.array([1, 1, 1, 1], dtype=float)
    sigma = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    a, info = kras(a0, G, c, sigma=sigma, alpha=0.1, tol=1e-9)
    print("a            =", a)
    print("Ga           =", G @ a)
    print("Ga - c       =", info["residual_vs_c0"])
    print("phase switch at sweep", info["phase_switch"], "| sweeps", info["sweeps"])
    # Expected (paper): residual constant +/- 1/3 across constraints.
    assert np.allclose(np.abs(info["residual_vs_c0"]), 1/3, atol=2e-2), info["residual_vs_c0"]
    assert abs(a[0]) < 2e-2
    print("PASS (matches paper: residual ~ +/-1/3 across constraints)\n")


def test_mohr():
    """Moehr et al. (1987) RAS-infeasible problem (ill-set initial estimate)."""
    print("=" * 64)
    print("2) Moehr et al. (1987) RAS-infeasible problem")
    # 4x4 matrix, constraints: 4 row sums then 4 column sums
    u = np.array([301., 104., 105., 10.])
    v = np.array([100., 220., 100., 100.])
    G, c = build_margin_constraints(4, 4, u, v)
    A0 = np.array([[90., 0., 95., 95.],
                   [5., 101., 2., 2.],
                   [5., 101., 2., 2.],
                   [0., 18., 1., 1.]])
    sigma = np.full(8, 0.1)
    a, info = kras(A0.flatten(), G, c, sigma=sigma, alpha=0.1, tol=1e-9)
    A = a.reshape(4, 4)
    print("balanced matrix:\n", A)
    print("row sums:", A.sum(1), " (target", u, ")")
    print("col sums:", A.sum(0), " (target", v, ")")
    print("residual vs c0:", info["residual_vs_c0"])
    print("phase switch at sweep", info["phase_switch"], "| sweeps", info["sweeps"])
    # Paper: all realisations are 0.33 away (3.3 sigma) from imposed constraints.
    res = info["residual_vs_c0"]
    print("|residual| range: [%.3f, %.3f]" % (np.abs(res).min(), np.abs(res).max()))
    assert np.allclose(np.abs(res), np.abs(res)[0], atol=5e-2)
    print("PASS (compromise solution, ~constant residual as in paper)\n")


def test_nonunity_and_negative():
    """Exercise non-unity coefficients and a negative element with sign preservation."""
    print("=" * 64)
    print("3) Non-unity coefficient + negative element")
    # vector a = [a1, a2, a3], a3 starts negative and must stay negative.
    a0 = np.array([3., 5., -1.])
    # constraint: a1 + a2 - a3 = 2  (the worked example from the paper, Sec 3.2)
    G = np.array([[1., 1., -1.]])
    c = np.array([2.])
    a, info = kras(a0, G, c, sigma=None, tol=1e-12)
    print("a       =", a, " (a3 stays negative:", a[2] < 0, ")")
    print("Ga      =", G @ a, " target", c)
    assert np.allclose(G @ a, c, atol=1e-8)
    assert a[2] < 0
    print("PASS\n")


if __name__ == "__main__":
    test_consistent_ras()
    test_cole()
    test_mohr()
    test_nonunity_and_negative()
    print("All tests passed.")
