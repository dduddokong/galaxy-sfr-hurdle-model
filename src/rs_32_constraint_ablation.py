"""RS R30 — 루프 3: RS-I 크기 모델의 단조 감소 제약 검증 (사전 등록 docs/30 §3)

팔(한 요소만 바꿈; 차단기·천장·나머지 항 공유): C(현행) / BH_free / BH_drop / OD_free / OD_drop.
단위: 내부 15 fold(r42–r44, rs_28 바깥 fold) + official(기록용). log M* ≥ 11 행은 rs_31 이 저장한 V2_third 예측을 쓴다.
항 곡선은 공식 학습 셋 SFR>0 은하의 지지 구간(1–99 백분위)에 놓은 공통 격자에서 계산한다.
출력: results/rstress/rs_32_constraint_ablation.json, _cache/rs_32_arms/<unit>.npz
"""
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import rs_00_common as C
from rs_09_final_model import GATE_TERMS
from rs_31_hmass_repro import PRED_DIR, units

ARM_DIR = C.CACHE / "rs_32_arms"
ARM_DIR.mkdir(parents=True, exist_ok=True)
BASE = dict(C.MAG_ARMS["C_band"])
BH = ["lbh_b1", "lbh_b2", "lbh_b3"]
ARMS = {"C": BASE,
        "BH_free": {k: ("free" if k in BH else v) for k, v in BASE.items()},
        "BH_drop": {k: v for k, v in BASE.items() if k not in BH},
        "OD_free": {k: ("free" if k == "od" else v) for k, v in BASE.items()},
        "OD_drop": {k: v for k, v in BASE.items() if k != "od"}}
CURVE_TERMS = ["od", "lbh_b2", "lbh_b3"]
BAND = {"lbh_b2": (10.3, 11.0), "lbh_b3": (11.0, 13.5)}


def grid_for(name, rows):
    Xt = C.X.iloc[rows]
    pos = C.POS[rows]
    if name == "od":
        v = Xt.od.to_numpy()[pos]
    else:
        lo, hi = BAND[name]
        m = pos & (Xt.lms.to_numpy() >= lo) & (Xt.lms.to_numpy() < hi) & (Xt.lbh.to_numpy() > C.X.lbh.min() + 1e-6)
        v = Xt.lbh.to_numpy()[m]
    a, b = np.quantile(v, [0.01, 0.99])
    return np.linspace(a, b, 41)


GRIDS = {n: grid_for(n, C.TR) for n in CURVE_TERMS}


def max_rise(c):
    c = np.asarray(c, float)
    return float(np.max(c - np.minimum.accumulate(c)))


def fit_unit(tr, te):
    Xt = C.X.iloc[tr].reset_index(drop=True)
    yv, pos, ly, lms = C.y[tr], C.POS[tr], C.LOGY[tr], C.LMS[tr]
    knots = {nm: C.ramp_knots(Xt[nm].to_numpy(float), 10) for nm in GATE_TERMS}
    A = np.hstack([C.ramp_basis(Xt[nm].to_numpy(float), knots[nm]) for nm in GATE_TERMS])
    sc = StandardScaler().fit(A)
    gate = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(A), pos.astype(int))
    p = np.where(pos)[0]
    curve = C.fit_ceiling(lms[p], ly[p], q=0.95)
    Xv = C.X.iloc[te].reset_index(drop=True)
    Av = np.hstack([C.ramp_basis(Xv[nm].to_numpy(float), knots[nm]) for nm in GATE_TERMS])
    p_on = gate.predict_proba(sc.transform(Av))[:, 1]
    cv = C.eval_ceiling(curve, Xv.lms.to_numpy())
    preds, curves = {}, {}
    for arm, terms in ARMS.items():
        mag = C.AdditiveMean(terms).fit_mean(Xt.iloc[p], C.eval_ceiling(curve, lms[p]), yv[p])
        preds[arm] = p_on * 10 ** (cv + mag.predict(Xv))
        curves[arm] = {n: (mag.curve(n, GRIDS[n]) if n in terms else np.zeros(len(GRIDS[n]))) for n in CURVE_TERMS}
    return preds, curves


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else float("nan")


if __name__ == "__main__":
    t0 = C.tick()
    U = [(n, k, tr, te) for n, k, tr, te in units() if k in ("inner", "official")]
    res = {}
    for name, kind, tr, te in U:
        f = ARM_DIR / f"{name}.npz"
        if not f.exists():
            preds, curves = fit_unit(tr, te)
            np.savez_compressed(f, te=te, **{f"pred_{a}": v for a, v in preds.items()},
                                **{f"curve_{a}_{n}": c for a, cs in curves.items() for n, c in cs.items()})
            print(name, "fitted", f"{C.tick() - t0:.0f}s", flush=True)
    # rs_31 의 V2_third 예측을 기다렸다가 결합
    for name, kind, tr, te in U:
        q = PRED_DIR / f"{name}.npz"
        while not q.exists():
            time.sleep(30)
        time.sleep(2)
        z, qq = np.load(ARM_DIR / f"{name}.npz"), np.load(q)
        assert np.array_equal(z["te"], te) and np.array_equal(qq["te"], te)
        y, lms = C.y[te], C.LMS[te]
        lo = lms < 11.0
        assert np.allclose(z["pred_C"], qq["rsi"], rtol=0, atol=1e-9), f"{name}: C 팔이 rs_31 RS-I 와 다름"
        u = {}
        for a in ARMS:
            pa = z[f"pred_{a}"]
            hm = np.where(lo, pa, qq["v2"])
            m0, mp = lo & (y == 0), lo & (y > 0)
            u[a] = {"rmse_all": rmse(hm, y), "rmse_lo11": rmse(pa[lo], y[lo]), "rmse_lo11_zero": rmse(pa[m0], y[m0]),
                    "logrmse_lo11_pos": rmse(np.log10(pa[mp]), np.log10(y[mp])),
                    "curves": {n: z[f"curve_{a}_{n}"].tolist() for n in CURVE_TERMS},
                    "max_rise": {n: max_rise(z[f"curve_{a}_{n}"]) for n in CURVE_TERMS}}
        res[name] = u
    inner = [n for n, k, _, _ in U if k == "inner"]
    summ = {}
    for a in ARMS:
        if a == "C":
            continue
        s = {}
        for met in ["rmse_all", "rmse_lo11", "rmse_lo11_zero", "logrmse_lo11_pos"]:
            d = np.array([res[n][a][met] - res[n]["C"][met] for n in inner])
            s[met] = {"mean": float(d.mean()), "n_arm_lower": int((d < 0).sum()), "min": float(d.min()), "max": float(d.max()),
                      "official": float(res["official"][a][met] - res["official"]["C"][met])}
        summ[a] = s
    curve_stats = {}
    for a in ["C", "BH_free", "OD_free"]:
        cs = {}
        for n in CURVE_TERMS:
            M = np.array([np.asarray(res[u][a]["curves"][n]) - res[u][a]["curves"][n][0] for u in inner])
            med = np.median(M, axis=0)
            cs[n] = {"grid": GRIDS[n].tolist(), "median_curve": med.tolist(), "fold_sd_mean": float(M.std(0).mean()),
                     "fold_sd_max": float(M.std(0).max()), "max_rise_median_curve": max_rise(med),
                     "max_rise_official": res["official"][a]["max_rise"][n],
                     "n_folds_rise_ge_0.05": int(sum(res[u][a]["max_rise"][n] >= 0.05 for u in inner)),
                     "total_change_official": float(res["official"][a]["curves"][n][-1] - res["official"][a]["curves"][n][0])}
        curve_stats[a] = cs
    rule = {}
    for n in ["lbh_b2", "od"]:
        free = "BH_free" if n.startswith("lbh") else "OD_free"
        o, m = curve_stats[free][n]["max_rise_official"], curve_stats[free][n]["max_rise_median_curve"]
        rule[n] = {"max_rise_official": o, "max_rise_inner_median": m,
                   "verdict_official": "비단조(제약이 만든 모양)" if o >= 0.05 else "같은 방향(제약과 데이터 일치)",
                   "verdict_inner": "비단조(제약이 만든 모양)" if m >= 0.05 else "같은 방향(제약과 데이터 일치)"}
    out = {"script": "src/rs_32_constraint_ablation.py", "prereg": "docs/30_loop_prereg_hmass_repro_boundary_constraint.md",
           "arms": {a: t for a, t in ARMS.items()}, "units": res, "paired_vs_C_inner15": summ, "curve_stats": curve_stats,
           "rule_docs28_3_3": rule, "elapsed_s": C.tick() - t0}
    C.save_json(out, C.OUT / "rs_32_constraint_ablation.json")
    print(json.dumps(summ, ensure_ascii=False), rule)
