"""RS R24 판정 — docs/24 Part A 선택·채택 규칙을 rs_27_b3_fix_{42,43,44,spatial,seeds}.json 에서 계산한다.
출력: results/rstress/rs_27_b3_fix_summary.json (key 'adopted_variant' 를 rs_28 이 읽는다)
"""
import json

import numpy as np

import rs_00_common as C

REPS = ["42", "43", "44"]
R = {u: json.loads((C.OUT / f"rs_27_b3_fix_{u}.json").read_text(encoding="utf-8")) for u in REPS + ["spatial", "seeds"]}
V = ["base", "cap", "wins", "ridge", "all"]
fr = {v: np.concatenate([R[u]["fold_rmse_band"][v] for u in REPS]) for v in V}
out = {"script": "src/rs_27b_fix_summary.py", "prereg": "docs/24_i52_i53_prereg_fix_and_router.md",
       "inner_mean_band_rmse": {v: float(fr[v].mean()) for v in V},
       "spatial_band_rmse": {v: R["spatial"]["metrics_band"][v]["rmse_all"] for v in V},
       "max_pred_band": {u: R[u]["max_pred_band"] for u in REPS + ["spatial"]}}
rows = R["seeds"]["rows"]
out["seeds_mean"] = {v: float(np.mean([r[f"FI_{v}"] for r in rows])) for v in V}
out["seeds_mean"]["V2_third"] = float(np.mean([r["V2_third_hl"] for r in rows]))
out["seeds_mean"]["V1"] = float(np.mean([r["V1_hl"] for r in rows]))
out["seeds_max_pred"] = {v: float(max(r[f"FI_{v}_max"] for r in rows)) for v in V}
cmp = {}
for v in V[1:]:
    d = fr[v] - fr["base"]
    ds = np.array([r[f"FI_{v}"] - r["FI_base"] for r in rows])
    dv2 = np.array([r[f"FI_{v}"] - r["V2_third_hl"] for r in rows])
    sp = np.array(R["spatial"]["fold_rmse_band"][v]) - np.array(R["spatial"]["fold_rmse_band"]["base"])
    cmp[v] = {"inner_n_better": int((d < 0).sum()), "inner_mean_diff": float(d.mean()),
              "seeds_mean_diff_vs_FI": float(ds.mean()), "seeds_n_better_vs_FI": int((ds < 0).sum()),
              "seeds_mean_diff_vs_V2": float(dv2.mean()), "seeds_n_better_vs_V2": int((dv2 < 0).sum()),
              "spatial_n_better": int((sp < 0).sum())}
out["vs_base"] = cmp
sel = min(V, key=lambda v: out["inner_mean_band_rmse"][v])
out["selected"] = sel
if sel == "base":
    out["adopted_variant"], out["adopted"] = "base", False
else:
    c = cmp[sel]
    ok = c["inner_mean_diff"] < 0 and c["inner_n_better"] >= 8 and c["seeds_mean_diff_vs_FI"] < 0
    out["adopted"] = bool(ok)
    out["adopted_variant"] = sel if ok else "base"
C.save_json(out, C.OUT / "rs_27_b3_fix_summary.json")
print("inner", {v: round(x, 3) for v, x in out["inner_mean_band_rmse"].items()})
print("spatial", {v: round(x, 3) for v, x in out["spatial_band_rmse"].items()})
print("seeds", {v: round(x, 3) for v, x in out["seeds_mean"].items()})
print("seeds max", {v: round(x, 1) for v, x in out["seeds_max_pred"].items()})
for v, c in cmp.items():
    print(v, c)
print("selected", sel, "adopted", out["adopted"], "->", out["adopted_variant"])
