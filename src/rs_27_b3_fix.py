"""RS R24 — I52 B3 폭주 수정안 (사전 등록 docs/24 Part A)

팔: base / cap(클래스 크기 ≤ 클래스 학습 최대) / wins(입력 0.5–99.5 백분위 자르기) / ridge(크기 모델 릿지 1e-2) / all
단위: 42 43 44 (TR 내부 KFold, 고질량 행, p_on = G_add OOF) | spatial | seeds (FI 변형 전체 RMSE, rs_24 와 같은 분할)
출력: results/rstress/rs_27_b3_fix_{unit}.json
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import rs_00_common as C
from rs_20_highmass_booster import BOOST, CLS, GATE_TERMS, HI, LO, TERMS7, XA, AddMultinom, rmse

VARIANTS = {"base": {}, "cap": {"cap": True}, "wins": {"wins": True}, "ridge": {"ridge": 1e-2},
            "all": {"cap": True, "wins": True, "ridge": 1e-2}}
WCOLS = TERMS7 + ["lbh_hh"]


def b3_variant(Xtr, ytr, cls_tr, Xte, cap=False, wins=False, ridge=1e-4):
    """B3 조건부 크기 E[SFR | SFR>0, x] (고질량 행)."""
    if wins:
        Xtr, Xte = Xtr.copy(), Xte.copy()
        for c in WCOLS:
            lo_, hi_ = np.quantile(Xtr[c], [0.005, 0.995])
            Xtr[c], Xte[c] = Xtr[c].clip(lo_, hi_), Xte[c].clip(lo_, hi_)
    pos = cls_tr >= 0
    P = AddMultinom().fit(Xtr[pos], cls_tr[pos], GATE_TERMS).p(Xte)
    sel = {c: cls_tr == c for c in (0, 1, 2)}
    zero = lambda m: np.zeros(int(m.sum()))
    mQ = C.AdditiveMean({"lms": "free", "lbh": "free"}, n_knots=6, ridge=ridge).fit_mean(Xtr[sel[0]], zero(sel[0]), ytr[sel[0]])
    mS = C.AdditiveMean({t: "free" for t in TERMS7}, n_knots=8, ridge=ridge).fit_mean(Xtr[sel[1]], zero(sel[1]), ytr[sel[1]])
    off = np.full(int(sel[2].sum()), np.log10(HI))
    mH = C.AdditiveMean({t: "inc" for t in BOOST}, n_knots=4, ridge=ridge).fit_mean(Xtr[sel[2]], off, ytr[sel[2]])
    E = np.stack([10 ** mQ.predict(Xte), 10 ** mS.predict(Xte), 10 ** (np.log10(HI) + mH.predict(Xte))], axis=1)
    if cap:
        E = np.minimum(E, np.array([ytr[sel[c]].max() for c in (0, 1, 2)]))
    return (P * E).sum(1)


def band_unit(rows, folds, p_on):
    y, lms, cls = C.y[rows], C.LMS[rows], CLS[rows]
    band = lms >= LO
    Xr = XA.iloc[rows].reset_index(drop=True)
    pred = {v: np.full(len(rows), np.nan) for v in VARIANTS}
    fold_of = np.full(len(rows), -1)
    for f, (a, b) in enumerate(folds):
        ab, bb = a[band[a]], b[band[b]]
        Xa, Xb = Xr.iloc[ab].reset_index(drop=True), Xr.iloc[bb].reset_index(drop=True)
        for v, kw in VARIANTS.items():
            pred[v][bb] = p_on[bb] * b3_variant(Xa, y[ab], cls[ab], Xb, **kw)
        fold_of[bb] = f
    yb, fb = y[band], fold_of[band]
    return {"band_n": int(band.sum()),
            "fold_rmse_band": {v: [rmse(p[band][fb == f], yb[fb == f]) for f in range(len(folds))] for v, p in pred.items()},
            "metrics_band": {v: C.metrics(yb, p[band]) for v, p in pred.items()},
            "max_pred_band": {v: float(np.nanmax(p[band])) for v, p in pred.items()}, "max_y_band": float(yb.max())}


def seeds_unit():
    from rs_09_final_model import predict_rs
    from rs_15_seed_splits import fit_rs
    hl = pd.read_csv(C.ROOT / "results" / "hsloop" / "hl_04b_confirm_bases_units.csv").pivot(index="unit", columns="arm", values="rmse_all")
    rows = []
    for s in range(10):
        tr, te = train_test_split(C.IDX, test_size=0.2, random_state=s, stratify=C.POS.astype(int))
        te = np.sort(te)
        P = predict_rs(fit_rs(tr), C.X.iloc[te].reset_index(drop=True))
        hi = C.LMS[te] >= LO
        trb = tr[C.LMS[tr] >= LO]
        Xa, Xb = XA.iloc[trb].reset_index(drop=True), XA.iloc[te[hi]].reset_index(drop=True)
        y = C.y[te]
        r = {"seed": s, "V2_third_hl": float(hl.loc[f"seed{s}", "T_V1B"]), "V1_hl": float(hl.loc[f"seed{s}", "V1"])}
        for v, kw in VARIANTS.items():
            fi = P["pred"].copy()
            fi[hi] = P["p_on"][hi] * b3_variant(Xa, C.y[trb], CLS[trb], Xb, **kw)
            r[f"FI_{v}"] = rmse(fi, y)
            r[f"FI_{v}_hi"] = rmse(fi[hi], y[hi])
            r[f"FI_{v}_max"] = float(fi.max())
        rows.append(r)
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items() if not k.endswith(("_hi", "_max"))}, flush=True)
    return {"rows": rows}


if __name__ == "__main__":
    for unit in (sys.argv[1:] or ["42"]):
        t0 = C.tick()
        if unit == "seeds":
            res = seeds_unit()
        else:
            d = np.load(C.CACHE / f"rs06_oof_{unit}.npz")
            folds = C.SPATIAL if unit == "spatial" else C.INNER[int(unit)]
            res = band_unit(d["rows"], folds, d["p_G_add"])
            print(unit, {v: round(m["rmse_all"], 3) for v, m in res["metrics_band"].items()}, res["max_pred_band"], flush=True)
        out = {"script": "src/rs_27_b3_fix.py", "unit": unit, "prereg": "docs/24_i52_i53_prereg_fix_and_router.md",
               **res, "elapsed_s": C.tick() - t0}
        C.save_json(out, C.OUT / f"rs_27_b3_fix_{unit}.json")
