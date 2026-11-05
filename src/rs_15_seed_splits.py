"""RS R12 — 결합 모델 RS-C 의 무작위 분할 seed 0–9 확인 (기존 확인 관례: I37·hs_04 의 seed 10개)

각 seed s: train_test_split(test_size=0.2, random_state=s, stratify=SFR>0) → 학습 셋으로 V1(ver1, random_state 42)과 RS-I(G_add × C_band_m)를 적합,
검증 셋에서 RS-C = where(log M* < 11, RS-I, V1) 와 V1, RS-I 단독을 비교.
판정(결과 전 고정): 'RS-C 유지' = seed 10개 평균 Δ(RS-C − V1) ≤ +0.124 (docs/14 §2-1). 부호 일치 seed 수와 log M* < 11 행 RMSE 도 보고.
출력: results/rstress/rs_15_seed_splits.json
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

import rs_00_common as C
from rs_09_final_model import GATE_TERMS  # noqa: F401  (같은 게이트 정의)
from rs_09_final_model import predict_rs

NB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=42)


def fit_rs(tr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xt = C.X.iloc[tr].reset_index(drop=True)
    y, pos, ly, lms = C.y[tr], C.POS[tr], C.LOGY[tr], C.LMS[tr]
    knots = {nm: C.ramp_knots(Xt[nm].to_numpy(float), 10) for nm in GATE_TERMS}
    A = np.hstack([C.ramp_basis(Xt[nm].to_numpy(float), knots[nm]) for nm in GATE_TERMS])
    sc = StandardScaler().fit(A)
    gate = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(A), pos.astype(int))
    p = np.where(pos)[0]
    curve = C.fit_ceiling(lms[p], ly[p], q=0.95)
    mag = C.AdditiveMean(C.MAG_ARMS["C_band"]).fit_mean(Xt.iloc[p], C.eval_ceiling(curve, lms[p]), y[p])
    return {"gate_knots": knots, "gate_scaler": sc, "gate": gate, "ceiling_curve": curve, "magnitude": mag}


if __name__ == "__main__":
    t0 = C.tick()
    rows = []
    for s in range(10):
        tr, te = train_test_split(C.IDX, test_size=0.2, random_state=s, stratify=C.POS.astype(int))
        te = np.sort(te)
        Ft, yt = C.F.iloc[tr], C.y[tr]
        sw = HistGradientBoostingClassifier(**NB).fit(Ft[C.MASS12 + [C.OD]], (yt > 0).astype(int))
        mg = HistGradientBoostingRegressor(**NB).fit(Ft.loc[yt > 0, C.MASS12], yt[yt > 0])
        v1 = np.maximum(sw.predict_proba(C.F.iloc[te][C.MASS12 + [C.OD]])[:, 1] * mg.predict(C.F.iloc[te][C.MASS12]), 0.0)
        rs_i = predict_rs(fit_rs(tr), C.X.iloc[te].reset_index(drop=True))["pred"]
        y, lms = C.y[te], C.LMS[te]
        rsc = np.where(lms < 11.0, rs_i, v1)
        rm = lambda p, m=slice(None): float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
        lo = lms < 11.0
        rows.append({"seed": s, "V1": rm(v1), "RS_C": rm(rsc), "RS_I": rm(rs_i), "below11_V1": rm(v1, lo), "below11_RS_I": rm(rs_i, lo),
                     "mae_V1": float(np.mean(np.abs(v1 - y))), "mae_RS_C": float(np.mean(np.abs(rsc - y)))})
        print(rows[-1], f"{C.tick() - t0:.0f}s", flush=True)
    d = np.array([r["RS_C"] - r["V1"] for r in rows])
    dl = np.array([r["below11_RS_I"] - r["below11_V1"] for r in rows])
    out = {"script": "src/rs_15_seed_splits.py", "rows": rows,
           "mean_delta_RS_C_minus_V1": float(d.mean()), "seeds_RS_C_better": int((d < 0).sum()),
           "mean_delta_below11": float(dl.mean()), "seeds_below11_RS_I_better": int((dl < 0).sum()),
           "RS_C_maintained": bool(d.mean() <= 0.124), "elapsed_s": C.tick() - t0}
    C.save_json(out, C.OUT / "rs_15_seed_splits.json")
    print({k: v for k, v in out.items() if k != "rows"})
