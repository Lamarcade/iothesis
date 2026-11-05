import numpy as np
from kras import kras, build_margin_constraints
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from read_Z import read_Z, read_x, read_Y
from convert_sectors import aggregate_to_sector_matrix, aggregate_to_sector_vector, load_concordance

np.set_printoptions(precision=4, suppress=True)

ZIP = os.path.join(ROOT, "EXIOBASE_data.zip")
Z = read_Z(year=2019, zip_path=ZIP)
x_raw = read_x(year=2019, zip_path=ZIP)
Y_raw = read_Y(year=2019, zip_path=ZIP)

Z_agg = aggregate_to_sector_matrix(Z)
x = aggregate_to_sector_vector(x_raw).reindex(Z_agg.columns).values   # total output
y = aggregate_to_sector_vector(Y_raw).reindex(Z_agg.columns).values   # final demand


u = (Z_agg * x[None, :]).sum(axis=1)              # row margins  sum_j x_j b_ij
v = (Z_agg * x[None, :]).sum(axis=0)              # Use, col margins  sum_i x_j b_ij

print(u)
print()
print(v)

#%% Projection KRAS : ancien Z (2015) -> information 2019, étude de convergence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def history_benchmark():
    OLD_YEAR = 2015

    # Matrice ancienne agrégée, alignée sur l'ordre des secteurs GICS de 2019
    Z_old = aggregate_to_sector_matrix(read_Z(year=OLD_YEAR, zip_path=ZIP))
    Z_old = Z_old.reindex(index=Z_agg.index, columns=Z_agg.columns)

    A_true = Z_agg.values                 # vrai 2019 (vérité terrain)
    A_old = Z_old.values
    n_rows, n_cols = A_true.shape
    a0 = A_old.flatten()                   # row-major : cohérent avec build_margin_constraints

    if (A_old == 0).any():
        print(f"Attention : {(A_old == 0).sum()} cellule(s) nulle(s) dans Z {OLD_YEAR} "
            "ne pourront pas être ajustées par GRAS (plancher d'erreur).")

    # Contraintes de base : marges (sommes lignes/colonnes) du vrai 2019
    u_true = A_true.sum(axis=1)
    v_true = A_true.sum(axis=0)
    G_m, c_m = build_margin_constraints(n_rows, n_cols, u_true, v_true)

    # Cellules classées par magnitude décroissante du vrai 2019
    a_true = A_true.flatten()
    order = np.argsort(-np.abs(a_true))


    def project_with_n_cells(n):
        """KRAS avec marges 2019 + n cellules exactes connues ; renvoie (proj, err rel, info)."""
        if n > 0:
            idx = order[:n]
            G_cells = np.zeros((n, a0.size))
            G_cells[np.arange(n), idx] = 1.0
            G = np.vstack([G_m, G_cells])
            c = np.concatenate([c_m, a_true[idx]])
        else:
            G, c = G_m, c_m
        a, info = kras(a0, G, c)
        A_proj = a.reshape(n_rows, n_cols)
        rel_err = np.linalg.norm(A_proj - A_true) / np.linalg.norm(A_true)
        return A_proj, rel_err, info


    # Balayage de n : convergence de la matrice projetée vers le vrai 2019
    n_values = [0, 5, 10, 20, 40, 60, 80, 100, n_rows * n_cols]
    errors = []
    for n in n_values:
        A_proj, rel_err, info = project_with_n_cells(n)
        errors.append(rel_err)
        print(f"n={n:4d}  rel.err={rel_err:.4e}  converged={info['converged']} sweeps={info['sweeps']}")

    # Référence : ancienne matrice brute vs vrai 2019 (sans projection)
    baseline = np.linalg.norm(A_old - A_true) / np.linalg.norm(A_true)
    print(f"\nBaseline (Z {OLD_YEAR} brut vs 2019) : {baseline:.4e}")

    # Courbe de convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, errors, "o-", label="Projection KRAS")
    ax.axhline(baseline, ls="--", color="gray", label=f"Z {OLD_YEAR} brut (sans projection)")
    ax.set_xlabel("Nombre de cellules 2019 connues ajoutées comme contraintes (n)")
    ax.set_ylabel("Erreur relative de Frobenius vs vrai 2019")
    ax.set_title(f"Projection KRAS du Z {OLD_YEAR} vers 2019 — convergence")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "kras_convergence.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

#%% Décarbonation des secteurs GICS vs empreinte CO2 initiale 2019 (style footprint_demo)
from io_pipeline import load_extension
from io_footprint import build_io_footprint_constraints, footprint_of, row_footprints

# Intensité carbone réelle par secteur GICS : émissions CO2 combustion / production
F = load_extension(2019, "air_emissions", zip_path=ZIP)["F"]
emis = F.loc["CO2 - combustion - air"]                   # Series (region, sector), kg CO2
emis_gics = aggregate_to_sector_vector(emis).reindex(Z_agg.columns).values

# Normalisation en grandeurs O(1) : dans la paramétrisation b_ij, les coefficients de
# contrainte (x_j pour les marges, S_j*x_j pour l'empreinte) servent d'EXPOSANTS dans la
# mise à jour r**g de KRAS. Avec les unités brutes (x~1e7 M€, S*x~1e12 kg) r**g déborde.
# Seules les réductions relatives importent -> on ramène tout à l'ordre 1.
n = len(x)
X0 = x.mean()
x_s = x / X0                                              # sorties O(1) (coeff. de marge)
w_target = emis_gics / emis_gics.mean()                  # coeff. empreinte O(1) (= S*x_s)
S = w_target / x_s                                        # intensité normalisée par secteur

B0 = Z_agg.values / x[None, :]                            # coefficients techniques A = Z/x
u_io = (B0 * x_s[None, :]).sum(axis=1)                    # marges lignes (échelle O(1))
v_io = (B0 * x_s[None, :]).sum(axis=0)                    # marges colonnes

# Empreinte initiale par secteur + cibles de décarbonation (échelle normalisée)
base_fp = row_footprints(B0, x_s, S)
base_total = footprint_of(B0, x_s, S)
print("\nEmpreinte CO2 par secteur (initiale, normalisée) :")
for lab, e in zip(Z_agg.columns, base_fp):
    print(f"  {lab:24s} {e:.3e}")
print(f"Total initial : {base_total:.3e}\n")

r = 0.30                                                  # objectif : -30% par secteur
E_target = (1 - r) * base_fp

# Contraintes : marges + une cible d'empreinte par ligne (mode per_row)
G, c, kind = build_io_footprint_constraints(n, n, x_s, u_io, v_io,
                                            S=S, E=E_target, footprint_mode="per_row")

# Arbitrage : on fait varier la fiabilité de l'empreinte relativement aux marges
sweep = [0.2, 1.0, 5.0, 20.0]                             # sigma_fp / sigma_marge
fp_achieved, margin_dist = [], []
print("Arbitrage distorsion des marges <-> respect des cibles d'empreinte :")
for mult in sweep:
    sigma = np.array([(v_io.mean() * 0.05) if k in ("row", "col")
                      else (v_io.mean() * 0.05 * mult) for k in kind])
    a, info = kras(B0.flatten(), G, c, sigma=sigma, alpha=0.1, tol=1e-7, max_sweeps=20000)
    B1 = a.reshape(n, n)
    fp = footprint_of(B1, x_s, S)
    row_ach = (B1 * x_s[None, :]).sum(1)
    col_ach = (B1 * x_s[None, :]).sum(0)
    dist = max(np.max(np.abs(row_ach - u_io)), np.max(np.abs(col_ach - v_io))) / v_io.mean()
    fp_achieved.append(fp)
    margin_dist.append(dist)
    print(f"  sigma_fp/sigma_marge={mult:5.1f}  empreinte={fp:.3e} "
          f"(cible {E_target.sum():.3e}, initiale {base_total:.3e}) "
          f"| distorsion marges={dist:.3f} | converged={info['converged']}")

# Graphe de l'arbitrage (double axe y)
fig2, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xscale("log")
l1, = ax1.plot(sweep, fp_achieved, "o-", color="tab:red", label="Empreinte atteinte")
ax1.axhline(base_total, ls=":", color="gray", label="Empreinte initiale")
ax1.axhline(E_target.sum(), ls="--", color="tab:green", label="Cible (-30%)")
ax1.set_xlabel("sigma_empreinte / sigma_marges  (Si augmente: empreinte plus relâchée / moins prioritaire)")
ax1.set_ylabel("Empreinte CO2 totale", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
l2, = ax2.plot(sweep, margin_dist, "s-", color="tab:blue", label="Distorsion marges")
ax2.set_ylabel("Distorsion relative des marges", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax1.set_title("Décarbonation GICS : arbitrage empreinte vs marges (KRAS, 2019)")
ax1.legend(loc="center right")
fig2.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), "decarbonation_gics.png")
fig2.savefig(out2, dpi=150)
print(f"Saved: {out2}")

#%% Heatmaps des flux : matrice initiale vs après KRAS, balayage de la cible d'empreinte
SIGMA_RATIO = 0.2                          # un seul point de l'arbitrage (empreinte prioritaire)
r_targets = [0.3, 0.6, 0.9]

sigma_hm = np.array([(v_io.mean() * 0.05) if k in ("row", "col")
                     else (v_io.mean() * 0.05 * SIGMA_RATIO) for k in kind])

labels = list(Z_agg.columns)
flow0 = Z_agg.values                        # flux initiaux (M€)
flows, titles, B_list = [flow0], ["Initiale (2019)"], []
print("\nBalayage de la cible d'empreinte (sigma_fp/sigma_marge=%.1f) :" % SIGMA_RATIO)
for rt in r_targets:
    E_t = (1 - rt) * base_fp
    G_r, c_r, _ = build_io_footprint_constraints(n, n, x_s, u_io, v_io,
                                                 S=S, E=E_t, footprint_mode="per_row")
    a_r, info_r = kras(B0.flatten(), G_r, c_r, sigma=sigma_hm, alpha=0.1,
                       tol=1e-7, max_sweeps=20000)
    B_r = a_r.reshape(n, n)
    B_list.append(B_r)
    flows.append(B_r * x[None, :])          # flux reconstruits en M€ (B * x)
    titles.append(f"Cible à -{int(rt * 100)} %")
    print(f"  r={rt:.0%}  empreinte={footprint_of(B_r, x_s, S):.3e} "
          f"converged={info_r['converged']}")

# Figure combinée, échelle de couleur log partagée. On plancher vmin au plus petit flux
# réel de 2019 : les flux poussés vers ~0 par KRAS sont clippés (au lieu d'étirer l'échelle
# jusqu'à 1e-130 et d'écraser le contraste sur le gros des flux).
vmax = np.log10(np.concatenate([m[m > 0].ravel() for m in flows]).max())
vmin = np.log10(flow0[flow0 > 0].min())   # plancher : plus petit flux réel 2019

ncols = len(flows)
fig3, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.6))
for k, (ax, M, title) in enumerate(zip(axes, flows, titles)):
    disp = np.log10(np.where(M > 0, M, np.nan))
    im = ax.imshow(disp, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels if k == 0 else [], fontsize=6)
    ax.set_xlabel("Secteur destination", fontsize=8)
axes[0].set_ylabel("Secteur origine", fontsize=8)
cbar = fig3.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
cbar.set_label("log10(flux M€)")
fig3.suptitle("Flux inter-sectoriels GICS selon objectif de décarbonation"
              f"(sigma_fp/sigma_marge={SIGMA_RATIO})", fontsize=12)
out3 = os.path.join(os.path.dirname(__file__), "decarbonation_heatmaps.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")

#%% Compromis par secteur : empreinte réelle atteinte vs production rebouclée (Leontief)
scenarios = ["Initiale"] + [f"-{int(rt * 100)} %" for rt in r_targets]
B_all = [B0] + B_list

fp_by_sec = np.array([row_footprints(B, x_s, S) for B in B_all])          # (n_scen, n) normalisé

# Production rebouclée : à demande finale y FIXE, on laisse l'output réagir au changement
# de structure A après KRAS via le modèle de Leontief  x = (I - A)^-1 · y.
# (Avec A = B0 et y réel, on retrouve la production réelle 2019 -> contrôle.)
prod_by_sec = np.array([np.linalg.solve(np.eye(n) - B, y) for B in B_all])  # (n_scen, n) M€
if (prod_by_sec < 0).any():
    print("Attention : production négative recalculée (structure A non admissible).")
print(f"\nContrôle production initiale recalculée vs réelle : "
      f"max écart relatif = {np.max(np.abs(prod_by_sec[0] - x) / x):.2e}")

# Empreinte EX-POST : même métrique mais évaluée avec la production rebouclée x_new.
# Capte à la fois le changement de structure (B) ET la réaction de la production.
fp_expost_by_sec = np.array([row_footprints(B_all[s], prod_by_sec[s] / X0, S)
                             for s in range(len(scenarios))])              # (n_scen, n) normalisé

# Réductions d'empreinte ex-post réellement obtenues par secteur (vs initiale)
print("\nRéduction d'empreinte EX-POST par secteur (%) :")
for j, lab in enumerate(labels):
    reals = [100 * (1 - fp_expost_by_sec[s + 1, j] / base_fp[j]) for s in range(len(r_targets))]
    print(f"  {lab:24s} " + "  ".join(
        f"-{int(rt*100)}%->{rl:5.1f}%" for rt, rl in zip(r_targets, reals)))

# Empreinte totale : cible (production fixe) vs ex-post (production rebouclée)
print("\nEmpreinte totale : ex-ante (cible, prod. fixe) vs ex-post (prod. rebouclée) :")
for s, name in enumerate(scenarios):
    fa = footprint_of(B_all[s], x_s, S)
    fe = fp_expost_by_sec[s].sum()
    print(f"  {name:10s} ex-ante={fa:.3e} ({100*(fa/base_total-1):+5.1f}%)  "
          f"ex-post={fe:.3e} ({100*(fe/base_total-1):+5.1f}%)")

# Variation de production totale rebouclée par scénario
print("\nProduction totale rebouclée (vs initiale) :")
for s, name in enumerate(scenarios):
    tot = prod_by_sec[s].sum()
    print(f"  {name:10s} {tot:.3e} M€  ({100*(tot/prod_by_sec[0].sum()-1):+.1f} %)")

# Barres groupées : empreinte ex-post (gauche) et production rebouclée (droite), par secteur
xpos = np.arange(n)
width = 0.8 / len(scenarios)
fig4, (axA, axB) = plt.subplots(1, 2, figsize=(17, 6))
for s, name in enumerate(scenarios):
    off = (s - (len(scenarios) - 1) / 2) * width
    axA.bar(xpos + off, fp_expost_by_sec[s], width, label=name)
    axB.bar(xpos + off, prod_by_sec[s], width, label=name)

axA.set_title("Empreinte CO2 ex-post par secteur (production rebouclée)")
axA.set_ylabel("Empreinte (unité normalisée)")
axB.set_title("Production rebouclée par secteur  x = (I − A)⁻¹·y  (y fixe)")
axB.set_ylabel("Production (M€)")
for ax in (axA, axB):
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.legend(fontsize=8, title="Cible")
    ax.grid(True, axis="y", alpha=0.3)
fig4.suptitle("Compromis décarbonation par secteur : empreinte ex-post vs production rebouclée "
              f"(sigma_fp/sigma_marge={SIGMA_RATIO})", fontsize=13)
fig4.tight_layout()
out4 = os.path.join(os.path.dirname(__file__), "decarbonation_compromis.png")
fig4.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")
