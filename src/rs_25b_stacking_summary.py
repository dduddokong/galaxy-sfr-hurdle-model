"""RS R22 판정 — docs/23 규칙을 rs_25_cross_stacking_{42,43,44,spatial}.json 에서 계산한다.
출력: results/rstress/rs_25_cross_stacking_summary.json
"""
import json
from math import comb

import numpy as np

import rs_00_common as C

UNITS = ["42", "43", "44", "spatial"]
REPS = UNITS[:3]
R = {u: json.loads((C.OUT / f"rs_25_cross_stacking_{u}.json").read_text(encoding="utf-8")) for u in UNITS}
ARMS = ["BB", "V1", "V1B", "FI", "M1_cal", "M1", "M1_off", "M2", "M2B"]
PAIRS = {"test1": [("M1", "BB"), ("M1_off", "BB"), ("M1", "FI"), ("M1", "M1_cal")],
         "test2": [("M2B", "BB"), ("M2", "V1"), ("M2B", "V1B")]}


def fold_vals(u, arm, key="rmse"):
    return np.array([f["metrics"][arm][key] for f in R[u]["folds"]])


def sign_p(k, n):
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


out = {"script": "src/rs_25b_stacking_summary.py", "prereg": "docs/23_i51_prereg_cross_stacking.md",
       "n_folds": {u: len(R[u]["folds"]) for u in UNITS}}
out["mean_rmse"] = {u: {a: float(fold_vals(u, a).mean()) for a in ARMS} for u in UNITS}
out["mean_rmse_inner3"] = {a: float(np.mean([fold_vals(u, a).mean() for u in REPS])) for a in ARMS}
out["mean_rmse_hi_inner3"] = {a: float(np.mean([fold_vals(u, a, "rmse_hi").mean() for u in REPS])) for a in ARMS}
out["mean_mae_inner3"] = {a: float(np.mean([fold_vals(u, a, "mae").mean() for u in REPS])) for a in ARMS}
for test, ps in PAIRS.items():
    rows = []
    for arm, ref in ps:
        d = np.concatenate([fold_vals(u, arm) - fold_vals(u, ref) for u in REPS])
        ds = fold_vals("spatial", arm) - fold_vals("spatial", ref)
        k = int((d < 0).sum())
        sig = bool(k >= 12 and d.mean() < 0)
        rows.append({"arm": arm, "ref": ref, "n_better": k, "mean_diff": float(d.mean()), "sign_p": sign_p(k, len(d)),
                     "significant": sig, "spatial_better": int((ds < 0).sum()), "spatial_mean_diff": float(ds.mean()),
                     "confirmed": bool(sig and (ds < 0).sum() >= 3)})
    out[test] = rows
    main = [r for r in rows if r["ref"] == "BB"]
    out[f"{test}_conclusion"] = "스태킹이 개선" if any(r["confirmed"] for r in main) else "스태킹 이득 없음"
info = [f["interp"] for u in REPS for f in R[u]["folds"]]
out["M1_slope"] = {"mean": float(np.mean([i["M1_slope_p10_p90"] for i in info])),
                   "range": [float(min(i["M1_slope_p10_p90"] for i in info)), float(max(i["M1_slope_p10_p90"] for i in info))]}
out["M1_cal_slope"] = {"mean": float(np.mean([i["M1_cal_slope_p10_p90"] for i in info]))}
terms = list(info[0]["M1_phys_range"])
out["M1_phys_range_mean"] = {t: float(np.mean([i["M1_phys_range"][t] for i in info])) for t in terms}
out["M1_off_phys_range_mean"] = {t: float(np.mean([i["M1_off_phys_range"][t] for i in info])) for t in info[0]["M1_off_phys_range"]}
out["max_pred"] = {u: [f["metrics"]["_max_pred"] for f in R[u]["folds"]] for u in UNITS}
C.save_json(out, C.OUT / "rs_25_cross_stacking_summary.json")
print("inner3 mean RMSE", {a: round(v, 3) for a, v in out["mean_rmse_inner3"].items()})
print("spatial mean RMSE", {a: round(v, 3) for a, v in out["mean_rmse"]["spatial"].items()})
print("hi", {a: round(v, 2) for a, v in out["mean_rmse_hi_inner3"].items()})
print("mae", {a: round(v, 3) for a, v in out["mean_mae_inner3"].items()})
for t in PAIRS:
    for r in out[t]:
        print(t, f"{r['arm']:7s} vs {r['ref']:6s} better {r['n_better']:2d}/15 mean {r['mean_diff']:+.3f} p {r['sign_p']:.3f} "
                 f"spatial {r['spatial_better']}/5 ({r['spatial_mean_diff']:+.3f}) confirmed {r['confirmed']}")
    print(t, out[f"{t}_conclusion"])
print("M1 slope", out["M1_slope"], "cal", out["M1_cal_slope"])
print("M1 phys", {t: round(v, 3) for t, v in out["M1_phys_range_mean"].items()})
print("M1_off phys", {t: round(v, 3) for t, v in out["M1_off_phys_range_mean"].items()})
