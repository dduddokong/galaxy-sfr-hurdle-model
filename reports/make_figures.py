"""Hero figure for the README: reports/figures/hero.png.

All numbers are read from results JSON files in this repo (nothing typed by hand):
  results/official/reference_key_numbers.json        organizer baseline RMSE, squared-error shares
  results/rstress/final_hmass/hmass_key_numbers.json ENV_OD, V2_third and H_mass official RMSE
Usage (from repo root): python reports/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Inlined copy of the shared portfolio style
COLORS = {"primary": "#1F4E79", "accent": "#E07A1F", "muted": "#9AA5B1",
          "light": "#D6E2EE", "text": "#1F2933", "note": "#6B7785"}
plt.rcParams.update({
    "figure.figsize": (10, 4.2), "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": COLORS["text"], "axes.edgecolor": "#C9D1D9",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#EDF0F3", "grid.linewidth": 0.8, "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"], "legend.frameon": False, "axes.axisbelow": True,
})

ref = json.loads((ROOT / "results/official/reference_key_numbers.json").read_text(encoding="utf-8"))
hm = json.loads((ROOT / "results/rstress/final_hmass/hmass_key_numbers.json").read_text(encoding="utf-8"))

base = ref["hgb_baseline"]["rmse_all"]
models = [
    ("Organizer HGB baseline", base, COLORS["muted"]),
    ("ENV_OD (mid-term hurdle)", hm["tab_ver1_rmse"], COLORS["light"]),
    ("V2_third (boosted hurdle)", hm["official_rmse_v2"], COLORS["primary"]),
    ("H_mass (final)", hm["official_rmse_hmass"], COLORS["accent"]),
]
share = ref["hgb_sse_share_by_true_sfr_rank"]
shares = [("Top 0.1%", share["top_0.1pct"]), ("Top 1%", share["top_1pct"]), ("Top 10%", share["top_10pct"])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})

# (a) RMSE ladder
labels = [m[0] for m in models][::-1]
vals = [m[1] for m in models][::-1]
cols = [m[2] for m in models][::-1]
ax1.barh(labels, vals, color=cols, height=0.6, edgecolor="white")
for i, v in enumerate(vals):
    ax1.text(v + 0.12, i, f"{v:.3f}", va="center", fontsize=10.5, color=COLORS["text"],
             fontweight="bold" if i == 0 else "normal")
ax1.set_xlim(0, base * 1.18)
ax1.set_xlabel("Official validation RMSE (linear SFR, lower is better)")
ax1.grid(axis="y", visible=False)
ax1.set_title("(a) Validation RMSE by model stage", fontsize=11, loc="left")

# (b) Error concentration in the baseline
x = [s[0] for s in shares]
y = [100 * s[1] for s in shares]
ax2.bar(x, y, color=[COLORS["accent"], COLORS["muted"], COLORS["muted"]], width=0.6)
for i, v in enumerate(y):
    ax2.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10.5, color=COLORS["text"],
             fontweight="bold" if i == 0 else "normal")
ax2.set_ylim(0, 110)
ax2.set_ylabel("Share of baseline squared error (%)")
ax2.set_xlabel("Galaxies ranked by true SFR")
ax2.grid(axis="x", visible=False)
ax2.set_title("(b) Where the baseline error comes from", fontsize=11, loc="left")

pct = 100 * (hm["official_rmse_hmass"] / base - 1)
fig.suptitle(f"Two-part hurdle model cuts validation RMSE from {base:.3f} to "
             f"{hm['official_rmse_hmass']:.3f} ({pct:.1f}%)",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=COLORS["text"])
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.text(0.01, -0.02, "Source: results/official/reference_key_numbers.json, "
         "results/rstress/final_hmass/hmass_key_numbers.json. Official validation rows were inspected "
         "several times, so RMSE values are exploratory.", fontsize=7.5, color=COLORS["note"],
         ha="left", va="top")
fig.savefig(FIG / "hero.png")
print("->", FIG / "hero.png")
