"""RS R17 — I43 고질량(log M* ≥ 11) 3모드 + 100 기준 부스터 해석 모델 (사전 등록 docs/15)

E[SFR|x] = p_on · [P_Q·E_Q + P_S·E_S + P_H·E_H],  Q = sSFR<1e-11, S = SF & SFR<100, H = SFR≥100 (SFR>0 행)
  P_c : 가법 다항 로지스틱 (램프 기저, lms od lbh lmh dgs dgh lmr + lbh_hh)
  E_H = 100 · 10^{a + Σ b_k},  b_k 단조 증가·≥0 (lmh, dgs, lbh_hh, lms)
팔: B3(p_on = V1 스위치), B3_free(E_H 부스터 무제약), B3_addgate(p_on = G_add), B2_ref(Q / SF 2모드).
단위: 42 43 44 (TR 내부 KFold) | spatial (B3_addgate·B3_free_addgate만) | official (TR 학습 → VA, 기록 전용)
출력: results/rstress/rs_20_highmass_booster_{unit}.json, 그림 reports/figures/rs_20_highmass_booster.png (반복 42)
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import rs_00_common as C

LO, HH, HI = 11.0, 13.5, 100.0
TERMS7 = ["lms", "od", "lbh", "lmh", "dgs", "dgh", "lmr"]
GATE_TERMS = TERMS7 + ["lbh_hh"]
BOOST = ["lmh", "dgs", "lbh_hh", "lms"]
XA = C.X.copy()
XA["lbh_hh"] = np.where(XA.lmh >= HH, XA.lbh, 5.0)          # 무거운 헤일로 층 안의 BH (층 밖 = 기준점 5.0 → 기여 0)
SSFR = np.where(C.POS, C.LOGY - C.LMS, -np.inf)
CLS = np.where(~C.POS, -1, np.where(SSFR < -11.0, 0, np.where(C.y < HI, 1, 2)))   # -1 SFR=0, 0 Q, 1 S, 2 H


class AddMultinom:
    """가법 다항 로지스틱. curve(nm, grid, a, b) = 클래스 a 대 b 의 logit 기여 (grid[0] 기준)."""

    def fit(self, Xa, z, names):
        self.names = names
        self.k = {nm: C.ramp_knots(Xa[nm].to_numpy(float), 8) for nm in names}
        A = self._B(Xa)
        self.sc = StandardScaler().fit(A)
        self.sc.scale_[self.sc.scale_ == 0] = 1.0
        self.m = LogisticRegression(C=1.0, max_iter=5000).fit(self.sc.transform(A), z)
        sizes = [len(self.k[nm]) - 1 for nm in names]
        self.sl = dict(zip(names, np.split(np.arange(sum(sizes)), np.cumsum(sizes)[:-1])))
        return self

    def _B(self, Xd):
        return np.hstack([C.ramp_basis(Xd[nm].to_numpy(float), self.k[nm]) for nm in self.names])

    def p(self, Xd):
        return self.m.predict_proba(self.sc.transform(self._B(Xd)))

    def curve(self, nm, grid, a, b):
        B = C.ramp_basis(np.asarray(grid, float), self.k[nm])
        sl = self.sl[nm]
        z = (B - self.sc.mean_[sl]) / self.sc.scale_[sl]
        c = z @ (self.m.coef_[a][sl] - self.m.coef_[b][sl])
        return c - c[0]


def fit_predict(Xtr, ytr, cls_tr, Xte, collect=None):
    """고질량대 학습 행(Xtr) → 시험 행(Xte)의 조건부 크기 [SFR>0 조건]. 반환 dict(팔 → E[SFR|SFR>0, x])."""
    pos = cls_tr >= 0
    gm = AddMultinom().fit(Xtr[pos], cls_tr[pos], GATE_TERMS)
    P = gm.p(Xte)
    g2 = AddMultinom().fit(Xtr[pos], (cls_tr[pos] >= 1).astype(int), GATE_TERMS)
    psf = g2.p(Xte)[:, 1]
    sel = {c: cls_tr == c for c in (0, 1, 2)}
    zero = lambda m: np.zeros(int(m.sum()))
    mQ = C.AdditiveMean({"lms": "free", "lbh": "free"}, n_knots=6).fit_mean(Xtr[sel[0]], zero(sel[0]), ytr[sel[0]])
    mS = C.AdditiveMean({t: "free" for t in TERMS7}, n_knots=8).fit_mean(Xtr[sel[1]], zero(sel[1]), ytr[sel[1]])
    off_h = np.full(int(sel[2].sum()), np.log10(HI))
    mH = C.AdditiveMean({t: "inc" for t in BOOST}, n_knots=4).fit_mean(Xtr[sel[2]], off_h, ytr[sel[2]])
    mHf = C.AdditiveMean({t: "free" for t in BOOST}, n_knots=4).fit_mean(Xtr[sel[2]], off_h, ytr[sel[2]])
    sf = sel[1] | sel[2]
    mSF = C.AdditiveMean({t: "free" for t in TERMS7}, n_knots=8).fit_mean(Xtr[sf], zero(sf), ytr[sf])
    EQ, ES = 10 ** mQ.predict(Xte), 10 ** mS.predict(Xte)
    EH, EHf = 10 ** (np.log10(HI) + mH.predict(Xte)), 10 ** (np.log10(HI) + mHf.predict(Xte))
    out = {"B3": P[:, 0] * EQ + P[:, 1] * ES + P[:, 2] * EH,
           "B3_free": P[:, 0] * EQ + P[:, 1] * ES + P[:, 2] * EHf,
           "B2_ref": (1 - psf) * EQ + psf * 10 ** mSF.predict(Xte)}
    if collect is not None:
        h = Xtr[sel[2]]
        pz = Xtr[pos]
        cur = {"base_H": float(HI * 10 ** mH.a_), "opt_H": mH.opt_}
        for t in BOOST:
            lo_, hi_ = np.quantile(h[t], [0.005, 0.995])
            g = np.linspace(lo_, hi_, 30)
            c = mH.curve(t, g)
            cur[f"boost_{t}"] = {"grid": g, "curve": c - c[0], "rise": float(c[-1] - c[0])}
        for t in ("od", "lbh", "lbh_hh", "lmh"):
            lo_, hi_ = np.quantile(pz[t], [0.005, 0.995])
            if hi_ <= lo_:
                hi_ = pz[t].max()
            g = np.linspace(lo_, hi_, 30)
            cur[f"logit_QvsS_{t}"] = {"grid": g, "curve": gm.curve(t, g, 0, 1), "total": float(gm.curve(t, g, 0, 1)[-1])}
            cur[f"logit_HvsS_{t}"] = {"grid": g, "curve": gm.curve(t, g, 2, 1), "total": float(gm.curve(t, g, 2, 1)[-1])}
        collect.append(cur)
    return out


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def run_folds(rows, folds, gates):
    """rows: 전체 행 인덱스, folds: rows 위치 인덱스 쌍. gates: 이름 → p_on(rows 정렬). 고질량 행 예측과 fold 표지 반환."""
    y, lms, cls = C.y[rows], C.LMS[rows], CLS[rows]
    band = lms >= LO
    Xr = XA.iloc[rows].reset_index(drop=True)
    mag = {}
    fold_of = np.full(len(rows), -1)
    coll = []
    for f, (a, b) in enumerate(folds):
        ab, bb = a[band[a]], b[band[b]]
        res = fit_predict(Xr.iloc[ab].reset_index(drop=True), y[ab], cls[ab], Xr.iloc[bb].reset_index(drop=True), coll)
        for k, v in res.items():
            mag.setdefault(k, np.full(len(rows), np.nan))[bb] = v
        fold_of[bb] = f
        print(f"  fold {f} done", flush=True)
    preds = {}
    for gname, p in gates.items():
        for k, v in mag.items():
            preds[k if gname == "V1sw" else f"{k}_{gname}"] = p * v
    return preds, fold_of, coll


def summarize_curves(coll):
    s = {"base_H": [c["base_H"] for c in coll]}
    for t in BOOST:
        r = np.array([c[f"boost_{t}"]["rise"] for c in coll])
        act = r > 0.05
        s[f"boost_{t}"] = {"rise": r, "active_n": int(act.sum()), "n": len(r),
                          "cv_active": float(r[act].std(ddof=1) / r[act].mean()) if act.sum() > 1 else None}
    for t in ("od", "lbh", "lbh_hh", "lmh"):
        for pair in ("QvsS", "HvsS"):
            v = np.array([c[f"logit_{pair}_{t}"]["total"] for c in coll])
            s[f"logit_{pair}_{t}"] = {"total": v, "n_pos": int((v > 0).sum())}
    return s


if __name__ == "__main__":
    v1_in, v1_off = C.v1_cache()
    for unit in (sys.argv[1:] or ["42"]):
        t0 = C.tick()
        out = {"script": "src/rs_20_highmass_booster.py", "unit": unit, "prereg": "docs/15_i43_prereg_highmass_booster.md"}
        if unit in ("42", "43", "44"):
            rep = int(unit)
            d = np.load(C.CACHE / f"rs06_oof_{unit}.npz")
            rows = d["rows"]
            assert (rows == C.TR).all()
            gates = {"V1sw": v1_in[f"r{unit}_p0"], "addgate": d["p_G_add"]}
            preds, fold_of, coll = run_folds(rows, C.INNER[rep], gates)
            y, lms = C.y[rows], C.LMS[rows]
            band = lms >= LO
            r12 = json.loads((C.OUT / f"rs_12_highmass_modes_{unit}.json").read_text(encoding="utf-8"))
            kmix = np.load(C.ROOT / "results" / "hsloop" / "runs" / "K_mix3_preds.npz")[f"inner_r{unit}"]
            v2 = np.load(C.ROOT / "results" / "hsloop" / "runs" / "BL_v1b_Piso_Kmix3_third_preds.npz")[f"inner_r{unit}"]
            rs_i = d["p_G_add"] * d["E_C_band_m"]
            preds.update({"V1": d["v1"], "RS-I": rs_i, "K_mix3": kmix, "V2_third": v2})
            nb = np.array([int((band & (fold_of == f)).sum()) for f in range(5)])
            fold_rmse = {k: [rmse(p[band & (fold_of == f)], y[band & (fold_of == f)]) for f in range(5)] for k, p in preds.items()}
            fold_rmse["twomode_free(rs12)"] = list(np.sqrt(np.array(r12["fold_sse_band"]["twomode_free"]) / nb))
            out["band_n"] = int(band.sum())
            out["metrics_band"] = {k: C.metrics(y[band], p[band]) for k, p in preds.items()}
            out["fold_rmse_band"] = fold_rmse
            out["twomode_free_rs12_rmse_band"] = r12["metrics_band"]["twomode_free"]["rmse_all"]
            lo11 = ~band
            out["whole_sample_rmse"] = {
                "V1": rmse(d["v1"], y), "RS-C (RS-I<11 + V1)": rmse(np.where(lo11, rs_i, d["v1"]), y),
                "RS-I<11 + B3": rmse(np.where(lo11, rs_i, preds["B3"]), y),
                "RS-I<11 + B3_addgate (완전 해석)": rmse(np.where(lo11, rs_i, preds["B3_addgate"]), y),
                "RS-I<11 + B3_free": rmse(np.where(lo11, rs_i, preds["B3_free"]), y),
                "V2_third": rmse(v2, y), "RS-I<11 + V2_third": rmse(np.where(lo11, rs_i, v2), y)}
            out["curves"] = summarize_curves(coll)
            if unit == "42":
                out["curves_raw_fold"] = coll
        elif unit == "spatial":
            d = np.load(C.CACHE / "rs06_oof_spatial.npz")
            rows = d["rows"]
            preds, fold_of, coll = run_folds(rows, C.SPATIAL, {"addgate": d["p_G_add"]})
            y, lms = C.y[rows], C.LMS[rows]
            band = lms >= LO
            rs_i = d["p_G_add"] * d["E_C_band_m"]
            preds.update({"V1": d["v1"], "RS-I": rs_i})
            out["band_n"] = int(band.sum())
            out["metrics_band"] = {k: C.metrics(y[band], p[band]) for k, p in preds.items()}
            out["fold_rmse_band"] = {k: [rmse(p[band & (fold_of == f)], y[band & (fold_of == f)]) for f in range(5)] for k, p in preds.items()}
            out["whole_sample_rmse"] = {"V1": rmse(d["v1"], y), "RS-C (RS-I<11 + V1)": rmse(np.where(~band, rs_i, d["v1"]), y),
                                        "RS-I<11 + B3_addgate (완전 해석)": rmse(np.where(~band, rs_i, preds["B3_addgate"]), y)}
            out["curves"] = summarize_curves(coll)
        elif unit == "official":
            d = np.load(C.CACHE / "rs06_oof_official.npz")
            va = d["rows"]
            assert (va == C.VA).all()
            tr = C.TR[C.LMS[C.TR] >= LO]
            vb = C.LMS[va] >= LO
            res = fit_predict(XA.iloc[tr].reset_index(drop=True), C.y[tr], CLS[tr], XA.iloc[va[vb]].reset_index(drop=True))
            y = C.y[va]
            preds = {"V1": v1_off["v1"], "RS-I": d["p_G_add"] * d["E_C_band_m"]}
            for k, v in res.items():
                for gname, p in (("", v1_off["p0"]), ("_addgate", d["p_G_add"])):
                    full = preds["V1"].copy() if gname == "" else preds["RS-I"].copy()
                    full[vb] = p[vb] * v
                    preds[k + gname] = full
            rs_i = preds["RS-I"]
            out["band_n"] = int(vb.sum())
            out["metrics_band"] = {k: C.metrics(y[vb], p[vb]) for k, p in preds.items()}
            out["whole_sample_rmse"] = {"V1": rmse(preds["V1"], y), "RS-C (RS-I<11 + V1)": rmse(np.where(~vb, rs_i, preds["V1"]), y),
                                        "RS-I<11 + B3": rmse(np.where(~vb, rs_i, preds["B3"]), y),
                                        "RS-I<11 + B3_addgate (완전 해석)": rmse(preds["B3_addgate"], y)}
        out["elapsed_s"] = C.tick() - t0
        C.save_json(out, C.OUT / f"rs_20_highmass_booster_{unit}.json")
        print(unit, "band RMSE", {k: round(v["rmse_all"], 3) for k, v in out["metrics_band"].items()})
        print(unit, "whole", {k: round(v, 3) for k, v in out["whole_sample_rmse"].items()}, f"{out['elapsed_s']:.0f}s", flush=True)
        if unit == "42":
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
            for t, col in zip(BOOST, ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]):
                for i, c in enumerate(coll):
                    ax[0].plot(c[f"boost_{t}"]["grid"] - c[f"boost_{t}"]["grid"][0], c[f"boost_{t}"]["curve"], color=col, alpha=0.6,
                               label=t if i == 0 else None)
            ax[0].set_xlabel("distance from reference (term units)"); ax[0].set_ylabel("booster [dex above base]")
            ax[0].set_title("SFR≥100 mode: boosters (5 folds)", fontsize=9); ax[0].legend(fontsize=7)
            for j, (t, pair) in enumerate([("od", "QvsS"), ("lbh", "QvsS")]):
                for c in coll:
                    ax[1 + j].plot(c[f"logit_{pair}_{t}"]["grid"], c[f"logit_{pair}_{t}"]["curve"], color="#8172b2", alpha=0.6)
                    ax[1 + j].plot(c[f"logit_HvsS_{t}"]["grid"], c[f"logit_HvsS_{t}"]["curve"], color="#dd8452", alpha=0.6)
                ax[1 + j].axhline(0, color="grey", lw=0.6)
                ax[1 + j].set_xlabel(t); ax[1 + j].set_ylabel("logit vs SF<100 (purple: quenched, orange: SFR≥100)")
                ax[1 + j].set_title(f"mode probability: {t} term (5 folds)", fontsize=9)
            fig.tight_layout()
            fig.savefig(C.FIG / "rs_20_highmass_booster.png", dpi=120)
