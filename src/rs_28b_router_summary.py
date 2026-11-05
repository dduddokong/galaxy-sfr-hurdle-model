"""RS R25 판정 — docs/24 Part B 규칙을 rs_28_hybrid_router_{42,43,44,spatial}.json 에서 계산한다.
출력: results/rstress/rs_28_hybrid_router_summary.json
"""
import json
from collections import Counter
from math import comb

import numpy as np

import rs_00_common as C

UNITS = ["42", "43", "44", "spatial"]
REPS = UNITS[:3]
R = {u: json.loads((C.OUT / f"rs_28_hybrid_router_{u}.json").read_text(encoding="utf-8")) for u in UNITS}
ARMS = ["BB", "FI", "H_mass", "H_sfr", "H_learn_hard", "H_learn_soft"]


def fv(u, arm, key="rmse"):
    return np.array([f["metrics"][arm][key] for f in R[u]["folds"]])


def sign_p(k, n):
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


out = {"script": "src/rs_28b_router_summary.py", "prereg": "docs/24_i52_i53_prereg_fix_and_router.md",
       "fi_variant": R["42"]["fi_variant"]}
out["mean_rmse_inner3"] = {a: float(np.mean([fv(u, a).mean() for u in REPS])) for a in ARMS}
out["mean_rmse_spatial"] = {a: float(fv("spatial", a).mean()) for a in ARMS}
out["mean_mae_inner3"] = {a: float(np.mean([fv(u, a, "mae").mean() for u in REPS])) for a in ARMS}
out["mean_rmse_hi_inner3"] = {a: float(np.mean([fv(u, a, "rmse_hi").mean() for u in REPS])) for a in ARMS}
cov_keys = list(R["42"]["folds"][0]["metrics"]["H_mass"]["coverage_fi"])
out["coverage_fi_inner3"] = {a: {k: float(np.nanmean([f["metrics"][a]["coverage_fi"][k] if f["metrics"][a]["coverage_fi"][k] is not None else np.nan
                                                      for u in REPS for f in R[u]["folds"]])) for k in cov_keys} for a in ARMS}
cmp = {}
for a in ARMS[1:]:
    d = np.concatenate([fv(u, a) - fv(u, "BB") for u in REPS])
    ds = fv("spatial", a) - fv("spatial", "BB")
    k = int((d < 0).sum())
    sig = bool(k >= 12 and d.mean() < 0)
    cmp[a] = {"n_better": k, "mean_diff": float(d.mean()), "sign_p": sign_p(k, len(d)), "significant": sig,
              "spatial_better": int((ds < 0).sum()), "spatial_mean_diff": float(ds.mean()),
              "confirmed": bool(sig and (ds < 0).sum() >= 3), "equivalent": bool(d.mean() <= 0.124)}
out["vs_BB"] = cmp
out["H_sfr_choices"] = dict(Counter(f"{f['info']['H_sfr_choice']['t']}|{f['info']['H_sfr_choice']['tau']}" for u in UNITS for f in R[u]["folds"]))
routers = ["H_mass", "H_sfr", "H_learn_hard", "H_learn_soft"]
conf = [a for a in routers if cmp[a]["confirmed"]]
eq = [a for a in routers if cmp[a]["equivalent"]]
if conf:
    out["conclusion"] = "하이브리드가 블랙박스보다 낫다: " + ", ".join(conf)
elif eq:
    best = min(eq, key=lambda a: cmp[a]["mean_diff"])
    out["conclusion"] = f"하이브리드는 블랙박스와 동등하며, 해석 모델 담당 비율 {out['coverage_fi_inner3'][best]['all']:.3f} ({best})"
else:
    out["conclusion"] = "하이브리드 이득 없음"
C.save_json(out, C.OUT / "rs_28_hybrid_router_summary.json")
print("inner3", {a: round(v, 3) for a, v in out["mean_rmse_inner3"].items()})
print("spatial", {a: round(v, 3) for a, v in out["mean_rmse_spatial"].items()})
print("mae", {a: round(v, 3) for a, v in out["mean_mae_inner3"].items()})
print("hi", {a: round(v, 2) for a, v in out["mean_rmse_hi_inner3"].items()})
for a, c in cmp.items():
    print(f"{a:13s} better {c['n_better']:2d}/15 mean {c['mean_diff']:+.3f} p {c['sign_p']:.3f} spatial {c['spatial_better']}/5 "
          f"({c['spatial_mean_diff']:+.3f}) conf {c['confirmed']} eq {c['equivalent']}")
for a in routers:
    print("coverage", a, {k: (round(v, 3) if v == v else None) for k, v in out["coverage_fi_inner3"][a].items()})
print("H_sfr choices", out["H_sfr_choices"])
print(out["conclusion"])
