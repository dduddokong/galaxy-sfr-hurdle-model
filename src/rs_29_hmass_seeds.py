"""RS R26 — I56 H_mass 하이브리드(RS-I<11 + V2_third) vs V2_third: 무작위 분할 seed 0–9 (사전 등록 docs/27)

seed 마다 학습 셋 전체로 V2_third(rs_25.bb)·RS-I 를 적합해 검증 셋에서 비교한다. V2_third 재계산 값은 hl_04b 와 일치해야 한다.
출력: results/rstress/rs_29_hmass_seeds.json
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import rs_00_common as C
from rs_09_final_model import predict_rs
from rs_15_seed_splits import fit_rs
from rs_25_cross_stacking import bb

HL = pd.read_csv(C.ROOT / "results" / "hsloop" / "hl_04b_confirm_bases_units.csv").pivot(index="unit", columns="arm", values="rmse_all")
t0 = C.tick()
rows = []
for s in range(10):
    tr, te = train_test_split(C.IDX, test_size=0.2, random_state=s, stratify=C.POS.astype(int))
    te = np.sort(te)
    v2 = bb(tr, te)["v2"]
    rs_i = predict_rs(fit_rs(tr), C.X.iloc[te].reset_index(drop=True))["pred"]
    lo = C.LMS[te] < 11.0
    hm = np.where(lo, rs_i, v2)
    y = C.y[te]
    rm = lambda p, m=slice(None): float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
    r = {"seed": s, "V2_third": rm(v2), "V2_third_hl": float(HL.loc[f"seed{s}", "T_V1B"]), "H_mass": rm(hm),
         "mae_V2_third": float(np.mean(np.abs(v2 - y))), "mae_H_mass": float(np.mean(np.abs(hm - y))),
         "lo_V2_third": rm(v2, lo), "lo_H_mass": rm(hm, lo), "frac_interp": float(lo.mean())}
    rows.append(r)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, f"{C.tick() - t0:.0f}s", flush=True)
R = pd.DataFrame(rows)
assert np.allclose(R.V2_third, R.V2_third_hl, atol=1e-6), "V2_third 재계산이 hl_04b 와 다르다"
d = (R.H_mass - R.V2_third).to_numpy()
dm = (R.mae_H_mass - R.mae_V2_third).to_numpy()
dl = (R.lo_H_mass - R.lo_V2_third).to_numpy()
out = {"script": "src/rs_29_hmass_seeds.py", "prereg": "docs/27_i56_prereg_hmass_seed_splits.md", "rows": rows,
       "mean": {k: float(R[k].mean()) for k in ["V2_third", "H_mass", "mae_V2_third", "mae_H_mass", "lo_V2_third", "lo_H_mass", "frac_interp"]},
       "delta_rmse": {"mean": float(d.mean()), "median": float(np.median(d)), "n_H_mass_better": int((d < 0).sum()),
                      "min": float(d.min()), "max": float(d.max())},
       "delta_mae": {"mean": float(dm.mean()), "n_H_mass_better": int((dm < 0).sum())},
       "delta_rmse_lo11": {"mean": float(dl.mean()), "n_H_mass_better": int((dl < 0).sum())}}
out["verdict"] = ("H_mass 동등 이상" if d.mean() <= 0 and (d < 0).sum() >= 7 else
                  "H_mass 허용 범위" if d.mean() <= 0.124 else "V2_third 우위")
out["elapsed_s"] = C.tick() - t0
C.save_json(out, C.OUT / "rs_29_hmass_seeds.json")
print({k: v for k, v in out.items() if k != "rows"})
