"""RS R22 — I51 교차 스태킹 (사전 등록 docs/23)

시험 1: 블랙박스(BB = V2_third) 예측 → 해석 메타(평균 척도 가법 포아송): M1_cal, M1, M1_off
시험 2: 해석(FI = RS-I<11 + G_add·B3) 예측 → 블랙박스 메타(ver1 구조 HGB + lfi): M2, M2B
중첩 교차 적합: 바깥 fold 의 학습 T 안 KFold 5 로 BB·FI OOF(메타 학습 입력), T 전체 적합으로 V 입력. V 정답 미사용.
단위: 42 43 44 (TR 내부 KFold 바깥) | spatial (공간 블록 5-fold)
출력: results/rstress/rs_25_cross_stacking_{unit}.json
"""
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

import hl_00_common as H0
import hl_02_candidates as H2
import rs_00_common as C
from rs_09_final_model import predict_rs
from rs_15_seed_splits import fit_rs
from rs_20_highmass_booster import CLS, LO, XA
from rs_20_highmass_booster import fit_predict as b3_fit

LN10 = np.log(10.0)
EPS = 0.01
PHYS = dict(C.MAG_ARMS["B_band"])                       # 무제약 물리 항 (질량대별 BH·헤일로 포함)


# ---------------------------------------------------------------- 기반 모델
def bb(tr, te):
    parts = [H0.ver1_parts(tr, te, rs) for rs in H0.V1_SEEDS]
    prods = [np.maximum(p * m, 0.0) for p, m in parts]
    v1, v1b = prods[0], np.mean(prods, axis=0)
    v2 = v1b.copy()
    dm = H0.DOM[te]
    if dm.any():
        trd = tr[H0.DOM[tr]]
        v2[dm] = (v1b[dm] + H2.mix(trd, te[dm], p_cal=True) + H2.mix3(trd, te[dm])) / 3
    return {"v1": v1, "v1b": v1b, "v2": v2}


def fi(tr, te):
    P = predict_rs(fit_rs(tr), C.X.iloc[te].reset_index(drop=True))
    out = P["pred"].copy()
    hi = C.LMS[te] >= LO
    trb = tr[C.LMS[tr] >= LO]
    b3 = b3_fit(XA.iloc[trb].reset_index(drop=True), C.y[trb], CLS[trb], XA.iloc[te[hi]].reset_index(drop=True))["B3"]
    out[hi] = P["p_on"][hi] * b3
    return out


# ---------------------------------------------------------------- 메타 1: 0 포함 전체 행 포아송 가법 모델
class MeanAll(C.AdditiveMean):
    """AdditiveMean 과 같지만 초기값을 log10(max(y, EPS)) 로 잡아 SFR = 0 행을 포함해 적합한다."""

    def fit_mean(self, Xd, off, yv):
        C.AdditiveLog.fit(self, Xd, np.log10(np.maximum(yv, EPS)) - off)
        B, lo_, hi_, _ = self._design(Xd)
        A = np.hstack([np.ones((len(B), 1)), B])
        sd = A.std(0); sd[0] = 1.0; sd[sd == 0] = 1.0
        b0 = self.coef_.copy()
        b0[0] += np.log10(yv.mean() / np.mean(10 ** (off + A @ b0)))
        n, lam = len(yv), self.ridge_m

        def fg(b):
            eta = LN10 * (off + A @ b)
            m = np.exp(np.clip(eta, -50, 50))
            f = (m - yv * eta).sum() / n + lam * np.sum((sd[1:] * b[1:]) ** 2)
            g = A.T @ (LN10 * (m - yv)) / n
            g[1:] += 2 * lam * sd[1:] ** 2 * b[1:]
            return f, g
        res = minimize(fg, b0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "maxfun": 6000})
        self.coef_, self.a_ = res.x, float(res.x[0])
        self.opt_ = {"success": bool(res.success), "nit": int(res.nit)}
        return self


def meta1(T, V, bb_T, bb_V):
    Xt, Xv = C.X.iloc[T].reset_index(drop=True).copy(), C.X.iloc[V].reset_index(drop=True).copy()
    Xt["lbb"], Xv["lbb"] = np.log10(bb_T + EPS), np.log10(bb_V + EPS)
    yt = C.y[T]
    z_t, z_v = np.zeros(len(T)), np.zeros(len(V))
    out, info = {}, {}
    m = MeanAll({"lbb": "free"}).fit_mean(Xt, z_t, yt)
    out["M1_cal"] = 10 ** m.predict(Xv)
    q = np.quantile(Xt.lbb, [0.1, 0.9])
    g = m.curve("lbb", q)
    info["M1_cal_slope_p10_p90"] = float((g[1] - g[0]) / (q[1] - q[0]))
    m = MeanAll({"lbb": "free", **PHYS}).fit_mean(Xt, z_t, yt)
    out["M1"] = 10 ** m.predict(Xv)
    g = m.curve("lbb", q)
    info["M1_slope_p10_p90"] = float((g[1] - g[0]) / (q[1] - q[0]))
    info["M1_phys_range"] = {t: float(np.subtract(*np.quantile(m.term(Xt, t), [0.995, 0.005]))) for t in m.slices if t != "lbb"}
    m = MeanAll(PHYS).fit_mean(Xt, Xt.lbb.to_numpy(), yt)
    out["M1_off"] = 10 ** (Xv.lbb.to_numpy() + m.predict(Xv))
    info["M1_off_phys_range"] = {t: float(np.subtract(*np.quantile(m.term(Xt, t), [0.995, 0.005]))) for t in m.slices}
    info["M1_off_intercept"] = m.a_
    return out, info


# ---------------------------------------------------------------- 메타 2: ver1 구조 HGB + lfi
def meta2(T, V, fi_T, fi_V):
    Ft, Fv = C.F.iloc[T].copy(), C.F.iloc[V].copy()
    Ft["lfi"], Fv["lfi"] = np.log10(fi_T + EPS), np.log10(fi_V + EPS)
    yt = C.y[T]
    preds = []
    for rs in H0.V1_SEEDS:
        nb = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=rs)
        sw = HistGradientBoostingClassifier(**nb).fit(Ft[C.MASS12 + [C.OD, "lfi"]], (yt > 0).astype(int))
        mg = HistGradientBoostingRegressor(**nb).fit(Ft.loc[yt > 0, C.MASS12 + ["lfi"]], yt[yt > 0])
        preds.append(np.maximum(sw.predict_proba(Fv[C.MASS12 + [C.OD, "lfi"]])[:, 1] * mg.predict(Fv[C.MASS12 + ["lfi"]]), 0.0))
    return {"M2": preds[0], "M2B": np.mean(preds, axis=0)}


def outer_fold(T, V):
    oof_bb, oof_fi = np.empty(len(T)), np.empty(len(T))
    for a, b in KFold(5, shuffle=True, random_state=0).split(T):
        oof_bb[b] = bb(T[a], T[b])["v2"]
        oof_fi[b] = fi(T[a], T[b])
    base = bb(T, V)
    fv = fi(T, V)
    preds = {"BB": base["v2"], "V1": base["v1"], "V1B": base["v1b"], "FI": fv}
    m1, info = meta1(T, V, oof_bb, base["v2"])
    preds.update(m1)
    preds.update(meta2(T, V, oof_fi, fv))
    y, hi = C.y[V], C.LMS[V] >= LO
    res = {k: {"rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(np.mean(np.abs(p - y))),
               "rmse_hi": float(np.sqrt(np.mean((p[hi] - y[hi]) ** 2))), "rmse_lo": float(np.sqrt(np.mean((p[~hi] - y[~hi]) ** 2)))}
           for k, p in preds.items()}
    res["_oof_rmse_inner"] = {"BB": float(np.sqrt(np.mean((oof_bb - C.y[T]) ** 2))), "FI": float(np.sqrt(np.mean((oof_fi - C.y[T]) ** 2)))}
    res["_max_pred"] = {"oof_BB": float(oof_bb.max()), "oof_FI": float(oof_fi.max()), "V_FI": float(fv.max()), "V_BB": float(base["v2"].max()),
                        "y_V": float(y.max())}
    return res, info


if __name__ == "__main__":
    for unit in (sys.argv[1:] or ["42"]):
        t0 = C.tick()
        if unit == "spatial":
            folds = [(np.asarray(tr), np.asarray(te)) for tr, te in C.SPATIAL]
        else:
            folds = [(C.TR[a], C.TR[b]) for a, b in C.INNER[int(unit)]]
        out = {"script": "src/rs_25_cross_stacking.py", "unit": unit, "prereg": "docs/23_i51_prereg_cross_stacking.md", "folds": []}
        for f, (T, V) in enumerate(folds):
            res, info = outer_fold(T, V)
            out["folds"].append({"fold": f, "metrics": res, "interp": info})
            print(unit, f, {k: round(v["rmse"], 3) for k, v in res.items() if not k.startswith("_")}, f"{C.tick() - t0:.0f}s", flush=True)
            C.save_json(out, C.OUT / f"rs_25_cross_stacking_{unit}.json")
        out["elapsed_s"] = C.tick() - t0
        C.save_json(out, C.OUT / f"rs_25_cross_stacking_{unit}.json")
