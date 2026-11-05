"""I38 — "유무는 환경, 크기는 질량" 분업의 직접 검정 (사전 등록: docs/12_i38_prereg_routing_separability.md).

A. 라우팅 2×2 (+ OD 전용 스위치): 스위치 {MASS12, MASS12+OD, OD} × 크기 {MASS12, MASS12+OD}, 공식 분할 + 공간 5-fold, 짝 ΔMSE·구간 기여·스위치 AP.
B. 분리 가능성 격자: log M* 0.25 dex × 과밀도 10분위 칸의 정지 비율(sSFR<1e-11 주, SFR=0 보조), 환경 고정 질량 효과 vs 질량 고정 환경 효과,
   가법 로짓 모델의 설명 몫, 저밀도에서의 질량 소광 서명.
출력: results/i38/i38_routing_separability.json (수치의 단일 출처), i38_arms.csv, i38_grid.csv
사용: python src/i38_routing_separability.py
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "galaxy_sfr.csv"
OUT = ROOT / "results" / "i38"; OUT.mkdir(parents=True, exist_ok=True)
RANDOM_SEED, BOX = 42, 200.0
MASS_RATIO_FLOOR, BH_SEED, BH_SEED_TOL = -6.0, 5.0, 6e-7
MASS6 = ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass", "log10_main_prog_mass",
         "log10_main_prog_stellar_mass", "log10_mass_ratio"]
MASS12 = MASS6 + ["log_mstar_over_mhalo", "log_mbh_over_mstar", "dlog_mhalo_growth", "dlog_mstar_growth",
                  "merger_flag", "bh_floor_flag"]
OD = "log10_overdensity_plus_1"
NB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)
SW = {"S_M": MASS12, "S_MO": MASS12 + [OD], "S_O": [OD]}
MG = {"M_M": MASS12, "M_MO": MASS12 + [OD]}
ARMS = {"ENV0": ("S_M", "M_M"), "ENV_OD": ("S_MO", "M_M"), "SM_MMO": ("S_M", "M_MO"), "SMO_MMO": ("S_MO", "M_MO"), "SO_MM": ("S_O", "M_M")}
CONTRASTS = {"E_sw|M": ("ENV_OD", "ENV0"), "E_mag|M": ("SM_MMO", "ENV0"), "E_mag|MO": ("SMO_MMO", "ENV_OD"),
             "E_sw|MO": ("SMO_MMO", "SM_MMO"), "E_swO_vs_swM": ("SO_MM", "ENV0"), "E_swO_vs_swMO": ("SO_MM", "ENV_OD")}


def mass12_features(Xin):
    F = Xin[MASS6].copy()
    F["log_mstar_over_mhalo"] = Xin.log10_stellar_mass - Xin.log10_halo_mass
    F["log_mbh_over_mstar"] = Xin.log10_BH_mass - Xin.log10_stellar_mass
    F["dlog_mhalo_growth"] = Xin.log10_halo_mass - Xin.log10_main_prog_mass
    F["dlog_mstar_growth"] = Xin.log10_stellar_mass - Xin.log10_main_prog_stellar_mass
    F["merger_flag"] = (Xin.log10_mass_ratio > MASS_RATIO_FLOOR + 1e-9).astype(int)
    F["bh_floor_flag"] = (Xin.log10_BH_mass <= BH_SEED + BH_SEED_TOL).astype(int)
    return F[MASS12]


def metrics(yv, pr, p):
    q90 = float(np.quantile(yv, 0.90)); se = (pr - yv) ** 2; z = (yv == 0).astype(int)
    return {"rmse_all": float(se.mean() ** 0.5), "mae_all": float(np.abs(pr - yv).mean()),
            "rmse_zero": float(se[yv == 0].mean() ** 0.5), "rmse_top10": float(se[yv >= q90].mean() ** 0.5),
            "msepart_zero": float(se[yv == 0].sum() / len(yv)), "msepart_bulk": float(se[(yv > 0) & (yv < q90)].sum() / len(yv)),
            "msepart_top10": float(se[yv >= q90].sum() / len(yv)),
            "ap_zero": float(average_precision_score(z, 1 - p)), "logloss": float(log_loss(1 - z, np.clip(p, 1e-15, 1 - 1e-15)))}


def paired(yv, pa, pb, ids):
    d = (pa - yv) ** 2 - (pb - yv) ** 2; n = len(yv); q90 = float(np.quantile(yv, 0.90))
    ia = np.lexsort((ids, -np.abs(d)))[0]; k = int(math.floor(0.001 * n)); top = np.lexsort((ids, -yv))[:k]
    keep = np.ones(n, bool); keep[top] = False
    return {"delta_mse": float(d.mean()), "delta_mae": float((np.abs(pa - yv) - np.abs(pb - yv)).mean()),
            "delta_drop_max_paired_row": float(np.delete(d, ia).mean()), "delta_drop_top0p1_y": float(d[keep].mean()),
            "segment_contrib": {"zero": float(d[yv == 0].sum() / n), "bulk": float(d[(yv > 0) & (yv < q90)].sum() / n), "top10": float(d[yv >= q90].sum() / n)}}


t_start = time.time()
df = pd.read_csv(RAW)
FEATURES = [c for c in df.columns if c not in ("subhalo_id", "sfr_linear")]
X, y = df[FEATURES].copy(), df["sfr_linear"].copy()
FULL = pd.concat([mass12_features(X), X[[OD]]], axis=1); Y = y.to_numpy(float); N = len(df)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=(y > 0).astype(int))
TR, VA = X_train.index.to_numpy(), X_val.index.to_numpy()
P = np.mod(X[["pos_x", "pos_y", "pos_z"]].to_numpy(float), BOX)
ijk = np.floor(P / 40.0).astype(int); groups = 25 * ijk[:, 0] + 5 * ijk[:, 1] + ijk[:, 2]
FOLDS = list(GroupKFold(n_splits=5).split(np.arange(N), groups=groups))


def run_unit(tr, va):
    Fa, Fb, ya = FULL.iloc[tr], FULL.iloc[va], Y[tr]; pos = ya > 0
    p = {k: HistGradientBoostingClassifier(**NB).fit(Fa[f], pos.astype(int)).predict_proba(Fb[f])[:, 1] for k, f in SW.items()}
    m = {k: HistGradientBoostingRegressor(**NB).fit(Fa.loc[pos, f], ya[pos]).predict(Fb[f]) for k, f in MG.items()}
    return {a: (np.maximum(p[s] * m[g], 0.0), p[s]) for a, (s, g) in ARMS.items()}


off = run_unit(TR, VA); yv = Y[VA]
A = {"official": {a: metrics(yv, pr, p) for a, (pr, p) in off.items()},
     "paired_official": {c: paired(yv, off[a][0], off[b][0], VA) for c, (a, b) in CONTRASTS.items()}}
print(f"official {time.time() - t_start:.0f}s: " + " ".join(f"{a} {m['rmse_all']:.4f}" for a, m in A["official"].items()))
OOF = {a: np.full(N, np.nan) for a in ARMS}; OOF_P = {a: np.full(N, np.nan) for a in ARMS}; fold_d = {c: [] for c in CONTRASTS}
for f, (tr, va) in enumerate(FOLDS):
    o = run_unit(tr, va)
    for a in ARMS:
        OOF[a][va], OOF_P[a][va] = o[a]
    for c, (a, b) in CONTRASTS.items():
        fold_d[c].append(float(np.mean((o[a][0] - Y[va]) ** 2) - np.mean((o[b][0] - Y[va]) ** 2)))
    print(f"spatial{f} {time.time() - t_start:.0f}s")
A["spatial_pooled"] = {a: metrics(Y, OOF[a], OOF_P[a]) for a in ARMS}
A["paired_spatial_pooled"] = {c: dict(paired(Y, OOF[a], OOF[b], np.arange(N)), folds_delta_mse=fold_d[c],
                                      folds_with_delta_lt_0=int(sum(x < 0 for x in fold_d[c])), folds_with_delta_gt_0=int(sum(x > 0 for x in fold_d[c])))
                              for c, (a, b) in CONTRASTS.items()}
A["interaction_spatial_pooled"] = A["paired_spatial_pooled"]["E_mag|MO"]["delta_mse"] - A["paired_spatial_pooled"]["E_mag|M"]["delta_mse"]
A["interaction_official"] = A["paired_official"]["E_mag|MO"]["delta_mse"] - A["paired_official"]["E_mag|M"]["delta_mse"]
em, es = A["paired_spatial_pooled"]["E_mag|MO"], A["paired_spatial_pooled"]["E_sw|M"]
consistent = max(em["folds_with_delta_lt_0"], em["folds_with_delta_gt_0"]) >= 4
A["verdict"] = ("MAG_SENSITIVE_TO_ENV" if (em["delta_mse"] < 0 and em["folds_with_delta_lt_0"] >= 4 and abs(em["delta_mse"]) >= 0.1 * abs(es["delta_mse"]))
                else "MAG_INSENSITIVE_TO_ENV")
A["ratio_mag_over_switch_effect_spatial"] = abs(em["delta_mse"]) / abs(es["delta_mse"])
A["switch_mass_share"] = {u: {"ap_S_M": A[u]["ENV0"]["ap_zero"], "ap_S_O": A[u]["SO_MM"]["ap_zero"], "ap_S_MO": A[u]["ENV_OD"]["ap_zero"],
                              "complement_gain": A[u]["ENV_OD"]["ap_zero"] - max(A[u]["ENV0"]["ap_zero"], A[u]["SO_MM"]["ap_zero"]),
                              "od_only_ge_mass_only": bool(A[u]["SO_MM"]["ap_zero"] >= A[u]["ENV0"]["ap_zero"])} for u in ("official", "spatial_pooled")}
print(f"A verdict {A['verdict']}: E_sw|M {es['delta_mse']:+.3f}, E_mag|MO {em['delta_mse']:+.3f} ({em['folds_with_delta_lt_0']}/5), ratio {A['ratio_mag_over_switch_effect_spatial']:.3f}; "
      f"AP S_M {A['spatial_pooled']['ENV0']['ap_zero']:.3f} S_O {A['spatial_pooled']['SO_MM']['ap_zero']:.3f} S_MO {A['spatial_pooled']['ENV_OD']['ap_zero']:.3f}")
pd.DataFrame([{"unit": u, "arm": a, **A[u][a]} for u in ("official", "spatial_pooled") for a in ARMS]).to_csv(OUT / "i38_arms.csv", index=False)

# ================================================================ B. 분리 가능성 격자
ms = X.log10_stellar_mass.to_numpy(); od = X[OD].to_numpy()
ssfr = Y / 10 ** ms
Q = {"ssfr": ssfr < 1e-11, "sfr0": Y == 0}
MEDGES = np.arange(9.3, 12.31, 0.25); mb = np.clip(np.digitize(ms, MEDGES) - 1, 0, len(MEDGES) - 2)
dq = pd.qcut(od, 10, labels=False)
B = {"mass_edges": [float(v) for v in MEDGES], "od_decile_edges": [float(v) for v in np.quantile(od, np.linspace(0, 1, 11))], "min_cell_n": 50}
grid_rows = []
for qname, qmask in Q.items():
    g = pd.DataFrame({"mb": mb, "dq": dq, "q": qmask}).groupby(["mb", "dq"]).agg(n=("q", "size"), fq=("q", "mean")).reset_index()
    g = g[g.n >= 50]
    for r in g.itertuples():
        grid_rows.append({"definition": qname, "mass_bin": int(r.mb), "mass_lo": float(MEDGES[r.mb]), "od_decile": int(r.dq), "n": int(r.n), "f_q": float(r.fq)})
    # 질량 고정 시 환경 효과 (칸별 상위 − 하위 10분위, 유효 칸 기준)
    env_eff = []
    for b, sub in g.groupby("mb"):
        lo, hi = sub.loc[sub.dq.idxmin()], sub.loc[sub.dq.idxmax()]
        if hi.dq - lo.dq >= 5:
            env_eff.append({"mass_bin": int(b), "mass_lo": float(MEDGES[int(b)]), "dq_lo": int(lo.dq), "dq_hi": int(hi.dq), "f_lo": float(lo.fq), "f_hi": float(hi.fq), "effect": float(hi.fq - lo.fq), "n": int(sub.n.sum())})
    # 환경 고정 시 질량 효과 (10분위별 가장 무거운 유효 칸 − 가장 가벼운 유효 칸)
    mass_eff = []
    for d, sub in g.groupby("dq"):
        lo, hi = sub.loc[sub.mb.idxmin()], sub.loc[sub.mb.idxmax()]
        if hi.mb - lo.mb >= 4:
            mass_eff.append({"od_decile": int(d), "mb_lo": int(lo.mb), "mb_hi": int(hi.mb), "mass_lo": float(MEDGES[int(lo.mb)]), "mass_hi": float(MEDGES[int(hi.mb)]),
                             "f_lo": float(lo.fq), "f_hi": float(hi.fq), "effect": float(hi.fq - lo.fq), "n": int(sub.n.sum())})
    w_env = sum(e["n"] for e in env_eff); w_mass = sum(e["n"] for e in mass_eff)
    env_w = sum(e["effect"] * e["n"] for e in env_eff) / w_env; mass_w = sum(e["effect"] * e["n"] for e in mass_eff) / w_mass
    # 가법 로짓 모델 vs 포화(칸 평균) vs 영(전체 평균)
    yq = qmask.astype(int)
    Xa = np.hstack([np.eye(len(MEDGES) - 1)[mb], np.eye(10)[dq]])
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xa, yq)
    ll_add = log_loss(yq, lr.predict_proba(Xa)[:, 1])
    cell_mean = pd.Series(yq).groupby([mb, dq]).transform("mean").to_numpy()
    ll_sat = log_loss(yq, np.clip(cell_mean, 1e-6, 1 - 1e-6)); ll_null = log_loss(yq, np.full(N, yq.mean()))
    # 저밀도(하위 3개 10분위)에서 질량 소광 서명: 질량에 따른 f_q
    lowd = dq <= 2
    low_curve = pd.DataFrame({"mb": mb[lowd], "q": qmask[lowd]}).groupby("mb").agg(n=("q", "size"), fq=("q", "mean"))
    low_curve = low_curve[low_curve.n >= 50]
    B[qname] = {"n_valid_cells": int(len(g)), "env_effect_at_fixed_mass": env_eff, "mass_effect_at_fixed_env": mass_eff,
                "env_effect_weighted": float(env_w), "mass_effect_weighted": float(mass_w), "ratio_env_over_mass": float(env_w / mass_w) if mass_w != 0 else None,
                "additive_logit": {"logloss_null": float(ll_null), "logloss_additive": float(ll_add), "logloss_saturated": float(ll_sat),
                                   "share_of_cell_deviance_explained_by_additive": float(1 - (ll_add - ll_sat) / (ll_null - ll_sat))},
                "low_density_mass_curve": {str(float(MEDGES[int(b)])): {"n": int(r.n), "f_q": float(r.fq)} for b, r in low_curve.iterrows()},
                "low_density_mass_quenching_signature": float(low_curve.fq.iloc[-1] - low_curve.fq.iloc[0]) if len(low_curve) >= 2 else None}
    print(f"B[{qname}]: env effect {env_w:.4f} vs mass effect {mass_w:.4f} (ratio {B[qname]['ratio_env_over_mass']:.1f}); additive share {B[qname]['additive_logit']['share_of_cell_deviance_explained_by_additive']:.3f}; "
          f"low-density mass signature {B[qname]['low_density_mass_quenching_signature']}")
# 사전 등록 규칙 그대로: N 가중 환경 효과 >= 5 x N 가중 질량 효과 (질량 효과가 0 이하이면 자동 충족). 첫 실행은 비율(env/mass)의 부호 때문에 잘못 판정했음(QC 기록).
B["verdict"] = "ENV_DOMINANT_ONOFF" if B["ssfr"]["env_effect_weighted"] >= 5 * B["ssfr"]["mass_effect_weighted"] else "MASS_MATTERS_FOR_ONOFF"
for qn in ("ssfr", "sfr0"):
    B[qn]["ratio_env_over_abs_mass"] = float(B[qn]["env_effect_weighted"] / abs(B[qn]["mass_effect_weighted"])) if B[qn]["mass_effect_weighted"] != 0 else None
pd.DataFrame(grid_rows).to_csv(OUT / "i38_grid.csv", index=False)

res = {"script": "src/i38_routing_separability.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "prereg": "docs/12_i38_prereg_routing_separability.md", "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
       "A_routing": dict(A, arms={a: {"switch": SW[s], "magnitude": MG[g]} for a, (s, g) in ARMS.items()}, contrasts={c: f"{a} - {b}" for c, (a, b) in CONTRASTS.items()}),
       "B_separability": B, "elapsed_seconds": time.time() - t_start}
(OUT / "i38_routing_separability.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"B verdict {B['verdict']}; wrote ({time.time() - t_start:.0f}s)")
