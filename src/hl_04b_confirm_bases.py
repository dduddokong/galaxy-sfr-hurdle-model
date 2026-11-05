"""루프 L8 확인: ver1 기준의 분산 감소 방식(random_state 평균 / 행 부트스트랩 / 둘 다) × 영역 1/3 결합.

분할마다 ver1 random_state 42–46 (→ V1, V1B), 부트스트랩 5개 (→ V1BB5), V1MIX10 = ½(V1B + V1BB5),
영역 후보 P_iso, K_mix3 학습. 레시피 T_* = 영역 안 1/3씩 (기준, P_iso, K_mix3).
판정: seed ≥ 7/10 & 공간 ≥ 4/5 에서 T_V1B 보다 낮으면 기준 교체 채택.
출력: results/hsloop/hl_04b_confirm_bases.json, hl_04b_confirm_bases_units.csv
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

import hl_00_common as C
import hl_02_candidates as H

y = C.y
t0 = time.time()


def v1(tr, te, rs):
    return np.maximum(np.prod(C.ver1_parts(tr, te, rs), axis=0), 0.0)


units = []
for s in range(10):
    tr, te = train_test_split(C.IDX, test_size=0.20, random_state=s, stratify=(y > 0).astype(int))
    units.append((f"seed{s}", tr, np.sort(te)))
P = C.df[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
ijk = np.floor(np.mod(P, C.BOX) / 40.0).astype(int)
groups = 25 * ijk[:, 0] + 5 * ijk[:, 1] + ijk[:, 2]
for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(C.IDX, groups=groups)):
    units.append((f"spatial{f}", tr, te))

rows, oof = [], {}
for name, tr, te in units:
    seeds = [v1(tr, te, rs) for rs in C.V1_SEEDS]
    boots = [v1(np.random.default_rng(k).choice(tr, len(tr), replace=True), te, 42) for k in range(5)]
    B = {"V1": seeds[0], "V1B": np.mean(seeds, 0), "V1BB5": np.mean(boots, 0)}
    B["V1MIX10"] = 0.5 * (B["V1B"] + B["V1BB5"])
    dm = C.DOM[te]
    trd = tr[C.DOM[tr]]
    pi, k3 = H.mix(trd, te[dm], p_cal=True), H.mix3(trd, te[dm])
    R = dict(B)
    for k in ("V1B", "V1BB5", "V1MIX10"):
        o = B[k].copy()
        o[dm] = (B[k][dm] + pi + k3) / 3
        R[f"T_{k}"] = o
    for arm, p in R.items():
        rows.append(dict(unit=name, arm=arm, **C.metrics(y[te], p)))
        if name.startswith("spatial"):
            oof.setdefault(arm, np.full(C.N, np.nan))[te] = p
    print(f"{name}: {time.time() - t0:.0f}s  " + "  ".join(f"{r['arm']} {r['rmse_all']:.3f}" for r in rows if r["unit"] == name), flush=True)
for arm, p in oof.items():
    rows.append(dict(unit="spatial_pooled", arm=arm, **C.metrics(y, p)))
tab = pd.DataFrame(rows)
tab.to_csv(C.OUT / "hl_04b_confirm_bases_units.csv", index=False, encoding="utf-8")
W = tab.pivot(index="unit", columns="arm", values="rmse_all")
S, Sp = W.loc[[f"seed{s}" for s in range(10)]], W.loc[[f"spatial{f}" for f in range(5)]]
summ = {arm: {"seeds_mean": float(S[arm].mean()), "seeds_median": float(S[arm].median()),
              "spatial_pooled": float(W.loc["spatial_pooled", arm]),
              "seeds_better_than_V1": int((S[arm] < S.V1).sum()), "spatial_better_than_V1": int((Sp[arm] < Sp.V1).sum()),
              "seeds_better_than_T_V1B": int((S[arm] < S.T_V1B).sum()), "spatial_better_than_T_V1B": int((Sp[arm] < Sp.T_V1B).sum()),
              "seeds_mean_ratio_med_ge100": float(tab[(tab.arm == arm) & tab.unit.str.startswith("seed")].ratio_med_ge100.mean()),
              "seeds_mean_ratio_med_ge400": float(tab[(tab.arm == arm) & tab.unit.str.startswith("seed")].ratio_med_ge400.mean())}
        for arm in W.columns}
out = {"script": "src/hl_04b_confirm_bases.py", "summary": summ, "elapsed_seconds": time.time() - t0}
(C.OUT / "hl_04b_confirm_bases.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in summ.items():
    print(f"{k:10s} seeds {v['seeds_mean']:.3f} (med {v['seeds_median']:.3f})  spatial {v['spatial_pooled']:.3f}  "
          f">V1 {v['seeds_better_than_V1']}/10 {v['spatial_better_than_V1']}/5  >T_V1B {v['seeds_better_than_T_V1B']}/10 {v['spatial_better_than_T_V1B']}/5  "
          f"r100 {v['seeds_mean_ratio_med_ge100']:.3f} r400 {v['seeds_mean_ratio_med_ge400']:.3f}")
