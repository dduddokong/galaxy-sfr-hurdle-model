"""RS R25 — I53 하이브리드: 고SFR = 블랙박스(V2_third), 그 아래 = 해석 모델(FI*), 경계는 학습된 분류 라우터 (사전 등록 docs/24 Part B)

FI* = RS-I<11 + G_add × B3 변형(rs_27_b3_fix_summary.json 'adopted_variant').
라우터: H_mass(고정 log M* < 11) / H_sfr(P(SFR ≥ t) ≥ τ → BB, (t, τ) 는 T 내부 OOF 로 선택) / H_learn_hard·soft(어느 쪽이 맞는지 직접 분류)
중첩 교차 적합(rs_25 와 같은 구조). 단위: 42 43 44 | spatial
출력: results/rstress/rs_28_hybrid_router_{unit}.json
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

import rs_00_common as C
from rs_09_final_model import predict_rs
from rs_15_seed_splits import fit_rs
from rs_20_highmass_booster import CLS, LO, XA
from rs_25_cross_stacking import EPS, bb
from rs_27_b3_fix import VARIANTS, b3_variant

ADOPT = json.loads((C.OUT / "rs_27_b3_fix_summary.json").read_text(encoding="utf-8"))["adopted_variant"]
TS, TAUS = [10.0, 30.0, 100.0], [0.2, 0.35, 0.5]
NB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=42)
SEGS = [("zero", 0, 0), ("0-10", 0, 10), ("10-100", 10, 100), ("ge100", 100, np.inf)]


def fi_star(tr, te):
    P = predict_rs(fit_rs(tr), C.X.iloc[te].reset_index(drop=True))
    out = P["pred"].copy()
    hi = C.LMS[te] >= LO
    trb = tr[C.LMS[tr] >= LO]
    out[hi] = P["p_on"][hi] * b3_variant(XA.iloc[trb].reset_index(drop=True), C.y[trb], CLS[trb],
                                         XA.iloc[te[hi]].reset_index(drop=True), **VARIANTS[ADOPT])
    return out


def feats(rows, p_fi, p_bb):
    F = C.F.iloc[rows].reset_index(drop=True).copy()
    F["lfi"], F["lbb"] = np.log10(np.maximum(p_fi, 0) + EPS), np.log10(np.maximum(p_bb, 0) + EPS)
    return F


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def coverage(use_fi, y, lms):
    c = {"all": float(use_fi.mean()), "lo11": float(use_fi[lms < LO].mean()), "hi11": float(use_fi[lms >= LO].mean())}
    for name, lo_, hi_ in SEGS:
        m = (y == 0) if name == "zero" else (y > lo_) & (y <= hi_) if lo_ == 0 else (y > lo_) & (y <= hi_)
        c[f"sfr_{name}"] = float(use_fi[m].mean()) if m.any() else None
    return c


def outer_fold(T, V):
    inner = list(KFold(5, shuffle=True, random_state=0).split(T))
    oof_bb, oof_fi = np.empty(len(T)), np.empty(len(T))
    for a, b in inner:
        oof_bb[b] = bb(T[a], T[b])["v2"]
        oof_fi[b] = fi_star(T[a], T[b])
    bbV, fiV = bb(T, V)["v2"], fi_star(T, V)
    yT, yV, lmsV = C.y[T], C.y[V], C.LMS[V]
    FT, FV = feats(T, oof_fi, oof_bb), feats(V, fiV, bbV)
    preds, use_fi, info = {"BB": bbV, "FI": fiV}, {"BB": np.zeros(len(V), bool), "FI": np.ones(len(V), bool)}, {}
    m = lmsV < LO
    preds["H_mass"], use_fi["H_mass"] = np.where(m, fiV, bbV), m
    # H_sfr: (t, τ) 를 T 내부 OOF 로 선택
    grid = {}
    for t in TS:
        lab = (yT >= t).astype(int)
        pO = np.empty(len(T))
        for a, b in inner:
            pO[b] = HistGradientBoostingClassifier(**NB).fit(FT.iloc[a], lab[a]).predict_proba(FT.iloc[b])[:, 1]
        for tau in TAUS:
            grid[f"{t}|{tau}"] = rmse(np.where(pO >= tau, oof_bb, oof_fi), yT)
    best = min(grid, key=grid.get)
    t, tau = (float(x) for x in best.split("|"))
    pV = HistGradientBoostingClassifier(**NB).fit(FT, (yT >= t).astype(int)).predict_proba(FV)[:, 1]
    preds["H_sfr"], use_fi["H_sfr"] = np.where(pV >= tau, bbV, fiV), pV < tau
    info["H_sfr_grid"], info["H_sfr_choice"] = grid, {"t": t, "tau": tau}
    info["oof_rmse_T"] = {"BB": rmse(oof_bb, yT), "FI": rmse(oof_fi, yT)}
    # H_learn: 어느 쪽이 맞는지 직접 분류 (비용 가중)
    eb, ef = (oof_bb - yT) ** 2, (oof_fi - yT) ** 2
    z, w = (eb < ef).astype(int), np.abs(eb - ef)
    w = w / w.mean()
    pB = HistGradientBoostingClassifier(**NB).fit(FT, z, sample_weight=w).predict_proba(FV)[:, 1]
    preds["H_learn_hard"], use_fi["H_learn_hard"] = np.where(pB > 0.5, bbV, fiV), pB <= 0.5
    preds["H_learn_soft"] = pB * bbV + (1 - pB) * fiV
    use_fi["H_learn_soft"] = pB <= 0.5                     # 기록용(가중 0.5 이하 = 해석 우세)
    info["H_learn_frac_z1_T"] = float(z.mean())
    hi = lmsV >= LO
    res = {k: {"rmse": rmse(p, yV), "mae": float(np.mean(np.abs(p - yV))), "rmse_hi": rmse(p[hi], yV[hi]),
               "rmse_lo": rmse(p[~hi], yV[~hi]), "coverage_fi": coverage(use_fi[k], yV, lmsV)} for k, p in preds.items()}
    return res, info


if __name__ == "__main__":
    for unit in (sys.argv[1:] or ["42"]):
        t0 = C.tick()
        folds = ([(np.asarray(a), np.asarray(b)) for a, b in C.SPATIAL] if unit == "spatial"
                 else [(C.TR[a], C.TR[b]) for a, b in C.INNER[int(unit)]])
        out = {"script": "src/rs_28_hybrid_router.py", "unit": unit, "prereg": "docs/24_i52_i53_prereg_fix_and_router.md",
               "fi_variant": ADOPT, "folds": []}
        for f, (T, V) in enumerate(folds):
            res, info = outer_fold(T, V)
            out["folds"].append({"fold": f, "metrics": res, "info": info})
            print(unit, f, {k: round(v["rmse"], 3) for k, v in res.items()}, info["H_sfr_choice"],
                  {k: round(v["coverage_fi"]["all"], 3) for k, v in res.items()}, f"{C.tick() - t0:.0f}s", flush=True)
            C.save_json(out, C.OUT / f"rs_28_hybrid_router_{unit}.json")
        out["elapsed_s"] = C.tick() - t0
        C.save_json(out, C.OUT / f"rs_28_hybrid_router_{unit}.json")
