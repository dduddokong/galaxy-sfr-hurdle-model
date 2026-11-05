"""중간보고서 Part I 가 노트북 안에서 다시 계산하는 공간 검증·보조 수치의 대조값 (2026-10-31 지적사항 반영).

노트북(`make_report_notebook.py` → Part I 5–7절)과 **같은 정의·같은 코드**로 아래를 계산해 저장한다. 생성기는 본문 수치를
이 json 에서 채우고, `qc_report_notebook.py`(R2h) 가 노트북 실행본의 요약 출력과 1e-6 안에서 같은지 대조한다.

(A) 공통 HGB 베이스라인의 좌표 제거 실험 — 6.1 "원본 좌표를 쓰면 공간 분할에서 성능이 떨어진다"는 주장을 같은 검증 행의 짝 차이로 검정.
    X13(원본 13열) / X10(위치 제외) / X7(위치·속도 제외), 공식 분할 + 공간 블록 5-fold pooled. Δ = MSE(X13) − MSE(X10 또는 X7).
(B) 2단계 팔 4개(ENV0·ENV_OD·ENV_TRAIN10·ENV_BOTH, 크기 모델 공유)의 공간 블록 5-fold pooled 와 짝 대비 — 연구 기록 R05 값과 같아야 한다.
(C) 완충 구역(buffer) 변형: 검증 은하로부터 BUFFER 단위 안의 학습 은하를 학습에서 제외한 공간 5-fold. 블록 경계 이웃 누수의 민감도.
(C') 완충 폭 민감도 스윕 0/2.5/5/7.5/10 단위 — Part II 10.0c 가 재계산(폭 선택이 아니라 대비 부호의 안정성 확인; 사용자 결정 2026-10-31).
(D) Figure 2(b) 와 같은 0.25 dex 별 질량 구간(N≥50)의 정지 비율과 log M* < 11.3 구간의 최소·최대 — 본문·그림 일치용.
(E) 예비 모델(ENV_OD) 스위치의 순열 중요도(공식 검증 행, log loss, 5회 반복, seed 42).
(F) 카탈로그 10번째 이웃 거리 분포(완충 폭 근거), 공간 fold 0 의 블록 목록(동결 평가 블록 후보).

출력: results/official/part1_spatial.json
사용: python src/official_03_part1_spatial.py
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "galaxy_sfr.csv"
OUT = ROOT / "results" / "official" / "part1_spatial.json"
RANDOM_SEED, BOX, BUFFER = 42, 200.0, 5.0

# ---------------------------------------------------------------- 특성 (노트북 4절과 같은 정의)
MASS_RATIO_FLOOR, BH_SEED, BH_SEED_TOL = -6.0, 5.0, 6e-7
MASS6 = ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass", "log10_main_prog_mass",
         "log10_main_prog_stellar_mass", "log10_mass_ratio"]
MASS12 = MASS6 + ["log_mstar_over_mhalo", "log_mbh_over_mstar", "dlog_mhalo_growth", "dlog_mstar_growth",
                  "merger_flag", "bh_floor_flag"]
OD, TRAIN10 = "log10_overdensity_plus_1", "log10_knn10_train_relative_density"
POS, VEL = ["pos_x", "pos_y", "pos_z"], ["vel_x", "vel_y", "vel_z"]
HGB_KW = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)
ARMS = {"ENV0": MASS12, "ENV_OD": MASS12 + [OD], "ENV_TRAIN10": MASS12 + [TRAIN10], "ENV_BOTH": MASS12 + [OD, TRAIN10]}
CONTRASTS = [("C-PROVIDER", "ENV_OD", "ENV0"), ("C-TRAIN10", "ENV_TRAIN10", "ENV0"),
             ("C-PROVIDER-GIVEN-TRAIN10", "ENV_BOTH", "ENV_TRAIN10"), ("C-TRAIN10-GIVEN-PROVIDER", "ENV_BOTH", "ENV_OD"),
             ("C-BOTH", "ENV_BOTH", "ENV0")]


def mass12_features(Xin):
    F = Xin[MASS6].copy()
    F["log_mstar_over_mhalo"] = Xin.log10_stellar_mass - Xin.log10_halo_mass
    F["log_mbh_over_mstar"] = Xin.log10_BH_mass - Xin.log10_stellar_mass
    F["dlog_mhalo_growth"] = Xin.log10_halo_mass - Xin.log10_main_prog_mass
    F["dlog_mstar_growth"] = Xin.log10_stellar_mass - Xin.log10_main_prog_stellar_mass
    F["merger_flag"] = (Xin.log10_mass_ratio > MASS_RATIO_FLOOR + 1e-9).astype(int)
    F["bh_floor_flag"] = (Xin.log10_BH_mass <= BH_SEED + BH_SEED_TOL).astype(int)
    return F[MASS12]


def knn10_density(P, ref_rows, query_rows, query_in_ref, k=10):
    ref_rows = np.sort(np.asarray(ref_rows)); query_rows = np.asarray(query_rows)
    n_eff = len(ref_rows) - (1 if query_in_ref else 0)
    d, i = cKDTree(P[ref_rows], boxsize=BOX).query(P[query_rows], k=k + 1 if query_in_ref else k)
    if query_in_ref:
        d = np.sort(np.where(ref_rows[i] == query_rows[:, None], np.inf, d), axis=1)
    r10 = np.maximum(d[:, k - 1], 1e-8)
    return np.log10((k * BOX ** 3) / ((4.0 * np.pi / 3.0) * n_eff * r10 ** 3))


# ---------------------------------------------------------------- 지표 (템플릿 evaluate_sfr 와 같은 구간 정의)
def evaluate(y, pred, p=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    q90 = float(np.quantile(y, 0.90))
    masks = {"all": np.ones(len(y), bool), "zero": y == 0, "positive": y > 0, "top10": y >= q90, "bulk": (y > 0) & (y < q90)}
    se = (pred - y) ** 2
    out = {"n": int(len(y))}
    for k, m in masks.items():
        out[f"rmse_{k}"] = float(mean_squared_error(y[m], pred[m]) ** 0.5)
        out[f"mae_{k}"] = float(mean_absolute_error(y[m], pred[m]))
        out[f"r2_{k}"] = float(r2_score(y[m], pred[m])) if k != "zero" else None
        out[f"msepart_{k}"] = float(se[m].sum() / len(y))
    out["mse_all"] = float(se.mean())
    if p is not None:
        z = (y == 0).astype(int)
        out["ap_zero"] = float(average_precision_score(z, 1 - p))
        out["brier"] = float(brier_score_loss(1 - z, p))
        out["logloss"] = float(log_loss(1 - z, np.clip(p, 1e-15, 1 - 1e-15)))
    return out


def dmse(y, pa, pb):
    y = np.asarray(y, float)
    return {"delta_mse": float(np.mean((pa - y) ** 2) - np.mean((pb - y) ** 2)),
            "delta_mae": float(np.mean(np.abs(pa - y)) - np.mean(np.abs(pb - y)))}


# ---------------------------------------------------------------- 데이터·분할 (고정 셀과 같은 분할)
t0 = time.time()
df = pd.read_csv(RAW)
FEATURES = [c for c in df.columns if c not in ("subhalo_id", "sfr_linear")]
X, y = df[FEATURES].copy(), df["sfr_linear"].copy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=(y > 0).astype(int))
TR, VA = X_train.index.to_numpy(), X_val.index.to_numpy()
FULL = pd.concat([mass12_features(X), X[[OD]]], axis=1)
P_all = np.mod(X[POS].to_numpy(float), BOX)
ijk = np.floor(P_all / 40.0).astype(int)
groups = 25 * ijk[:, 0] + 5 * ijk[:, 1] + ijk[:, 2]
FOLDS = list(GroupKFold(n_splits=5).split(np.arange(len(P_all)), groups=groups))
N = len(df)
Y = y.to_numpy()

# (F) 10번째 이웃 거리, fold 0 블록
d_all, _ = cKDTree(P_all, boxsize=BOX).query(P_all, k=11)
r10_all = d_all[:, 10]
r10_info = {"median": float(np.median(r10_all)), "mean": float(r10_all.mean()),
            "p90": float(np.quantile(r10_all, 0.90)), "p95": float(np.quantile(r10_all, 0.95)),
            "frac_lt_buffer": float((r10_all < BUFFER).mean())}
frozen_blocks = sorted(int(g) for g in np.unique(groups[FOLDS[0][1]]))

# ---------------------------------------------------------------- (A) 베이스라인 좌표 제거 실험
ABL = {"X13": FEATURES, "X10": [c for c in FEATURES if c not in POS], "X7": [c for c in FEATURES if c not in POS + VEL]}
abl_off, abl_oof, abl_fold_mse = {}, {}, []
for k, cols in ABL.items():
    m = HistGradientBoostingRegressor(**HGB_KW).fit(X_train[cols], y_train)
    abl_off[k] = m.predict(X_val[cols])
    abl_oof[k] = np.full(N, np.nan)
for f, (tr, va) in enumerate(FOLDS):
    fm = {}
    for k, cols in ABL.items():
        m = HistGradientBoostingRegressor(**HGB_KW).fit(X.iloc[tr][cols], y.iloc[tr])
        abl_oof[k][va] = m.predict(X.iloc[va][cols])
        fm[k] = float(np.mean((abl_oof[k][va] - Y[va]) ** 2))
    abl_fold_mse.append(fm)
ABL_PAIRS = [("X13-X10", "X13", "X10"), ("X13-X7", "X13", "X7"), ("X10-X7", "X10", "X7")]
baseline_ablation = {
    "variants": ABL,
    "official": {k: evaluate(y_val, abl_off[k]) for k in ABL},
    "spatial_pooled": {k: evaluate(Y, abl_oof[k]) for k in ABL},
    "spatial_folds_mse": abl_fold_mse,
    "paired_official": {cid: dmse(y_val, abl_off[a_], abl_off[b_]) for cid, a_, b_ in ABL_PAIRS},
    "paired_spatial_pooled": {cid: dict(dmse(Y, abl_oof[a_], abl_oof[b_]),
                                        folds_delta_mse=[fm[a_] - fm[b_] for fm in abl_fold_mse],
                                        folds_with_delta_lt_0=int(sum(fm[a_] < fm[b_] for fm in abl_fold_mse)))
                              for cid, a_, b_ in ABL_PAIRS},
}
print(f"(A) baseline ablation {time.time() - t0:.0f}s")
for k in ABL:
    print(f"  {k:4s} official RMSE {baseline_ablation['official'][k]['rmse_all']:.4f}  spatial pooled RMSE {baseline_ablation['spatial_pooled'][k]['rmse_all']:.4f}")
for cid, v in baseline_ablation["paired_official"].items():
    print(f"  {cid}: official dMSE {v['delta_mse']:+.4f}  spatial dMSE {baseline_ablation['paired_spatial_pooled'][cid]['delta_mse']:+.4f} "
          f"(folds<0 {baseline_ablation['paired_spatial_pooled'][cid]['folds_with_delta_lt_0']}/5)")


# ---------------------------------------------------------------- (B)(C) 2단계 팔 4개
def split_frames(tr_rows, va_rows):
    Ftr, Fva = FULL.iloc[tr_rows].copy(), FULL.iloc[va_rows].copy()
    Ftr[TRAIN10] = knn10_density(P_all, tr_rows, tr_rows, query_in_ref=True)
    Fva[TRAIN10] = knn10_density(P_all, tr_rows, va_rows, query_in_ref=False)
    return Ftr, Fva


def run_arms(tr_rows, va_rows):
    Ftr, Fva = split_frames(tr_rows, va_rows)
    ytr = y.iloc[tr_rows].to_numpy()
    shared = HistGradientBoostingRegressor(**HGB_KW).fit(Ftr.loc[ytr > 0, MASS12], ytr[ytr > 0])
    out, models = {}, {}
    for name, feats in ARMS.items():
        sw = HistGradientBoostingClassifier(**HGB_KW).fit(Ftr[feats], (ytr > 0).astype(int))
        p, m = sw.predict_proba(Fva[feats])[:, 1], shared.predict(Fva[MASS12])
        out[name] = (np.maximum(p * m, 0.0), p); models[name] = sw
    return out, models, Ftr, Fva


def run_cv(folds, label):
    oof = {k: np.full(N, np.nan) for k in ARMS}; oof_p = {k: np.full(N, np.nan) for k in ARMS}; fold_mse = []; n_tr = []
    for f, (tr, va) in enumerate(folds):
        out_f, _, _, _ = run_arms(tr, va)
        for k, (pr, p) in out_f.items():
            oof[k][va], oof_p[k][va] = pr, p
        fold_mse.append({k: float(np.mean((out_f[k][0] - Y[va]) ** 2)) for k in ARMS}); n_tr.append(int(len(tr)))
        print(f"  {label} fold {f}: n_train {len(tr):,}  " + "  ".join(f"{k} {np.sqrt(fold_mse[-1][k]):.3f}" for k in ARMS))
    pooled = {k: evaluate(Y, oof[k], oof_p[k]) for k in ARMS}
    contrasts = {cid: dict(dmse(Y, oof[a_], oof[b_]), folds_delta_mse=[fm[a_] - fm[b_] for fm in fold_mse],
                           folds_with_delta_lt_0=int(sum(fm[a_] < fm[b_] for fm in fold_mse))) for cid, a_, b_ in CONTRASTS}
    return {"pooled": pooled, "folds_mse": fold_mse, "n_train_per_fold": n_tr, "contrasts_pooled": contrasts}


t1 = time.time()
OFF_OUT, OFF_MODELS, F_train, F_val = run_arms(TR, VA)
arms_official = {k: evaluate(y_val, OFF_OUT[k][0], OFF_OUT[k][1]) for k in ARMS}
contrasts_official = {cid: dmse(y_val, OFF_OUT[a_][0], OFF_OUT[b_][0]) for cid, a_, b_ in CONTRASTS}
spatial = run_cv(FOLDS, "spatial")
print(f"(B) arms official + spatial {time.time() - t1:.0f}s")

# 완충 구역: 검증 은하로부터 BUFFER 단위(주기 상자 최소상 거리) 안의 학습 은하 제외
def buffered_folds(width):
    out = []
    for tr, va in FOLDS:
        d_tr, _ = cKDTree(P_all[va], boxsize=BOX).query(P_all[tr], k=1)
        out.append((tr[d_tr > width], va))
    return out


def run_buffered(width):
    res = run_cv(buffered_folds(width), f"buffered{width:g}")
    res["buffer"] = width
    res["n_train_unbuffered_per_fold"] = [int(len(tr)) for tr, _ in FOLDS]
    res["frac_train_removed_per_fold"] = [1 - nb / nu for nb, nu in zip(res["n_train_per_fold"], res["n_train_unbuffered_per_fold"])]
    return res


t1 = time.time()
buffered = run_buffered(BUFFER)
print(f"(C) buffered spatial {time.time() - t1:.0f}s")
# (C') 완충 폭 민감도 스윕 (Part II 10.0c 가 재계산; 선택이 아니라 부호 안정성 확인용). 0 = 완충 없음(= spatial), 5 = 본 설정.
t1 = time.time()
SWEEP_WIDTHS = [2.5, 7.5, 10.0]
buffer_sweep = {"widths": [0.0, 2.5, 5.0, 7.5, 10.0], "0": {"pooled": spatial["pooled"], "contrasts_pooled": spatial["contrasts_pooled"],
                                                            "n_train_per_fold": spatial["n_train_per_fold"], "frac_train_removed_per_fold": [0.0] * 5},
                "5": {k: buffered[k] for k in ("pooled", "contrasts_pooled", "n_train_per_fold", "frac_train_removed_per_fold")}}
for w in SWEEP_WIDTHS:
    r = run_buffered(w)
    buffer_sweep[f"{w:g}"] = {k: r[k] for k in ("pooled", "contrasts_pooled", "n_train_per_fold", "frac_train_removed_per_fold")}
print(f"(C') buffer sweep {time.time() - t1:.0f}s")
for w in buffer_sweep["widths"]:
    b = buffer_sweep[f"{w:g}"]
    print(f"  width {w:4.1f}: removed {np.mean(b['frac_train_removed_per_fold']):.1%}  " + "  ".join(f"{k} {b['pooled'][k]['rmse_all']:.4f}" for k in ARMS) +
          f"  C-PROVIDER {b['contrasts_pooled']['C-PROVIDER']['delta_mse']:+.3f} ({b['contrasts_pooled']['C-PROVIDER']['folds_with_delta_lt_0']}/5)"
          f"  C-TRAIN10 {b['contrasts_pooled']['C-TRAIN10']['delta_mse']:+.3f} ({b['contrasts_pooled']['C-TRAIN10']['folds_with_delta_lt_0']}/5)")
for k in ARMS:
    print(f"  {k:12s} spatial RMSE {spatial['pooled'][k]['rmse_all']:.4f}  buffered RMSE {buffered['pooled'][k]['rmse_all']:.4f}")
for cid in spatial["contrasts_pooled"]:
    s, b = spatial["contrasts_pooled"][cid], buffered["contrasts_pooled"][cid]
    print(f"  {cid}: spatial dMSE {s['delta_mse']:+.4f} ({s['folds_with_delta_lt_0']}/5)  buffered dMSE {b['delta_mse']:+.4f} ({b['folds_with_delta_lt_0']}/5)")

# ---------------------------------------------------------------- (D) Figure 2(b) 질량 구간
ms = X.log10_stellar_mass.to_numpy()
quench = (Y / 10 ** ms < 1e-11)
mbins = np.arange(9.3, 12.31, 0.25)
mid = 0.5 * (mbins[1:] + mbins[:-1]); idx = np.digitize(ms, mbins) - 1
fq_m = [float(quench[idx == i].mean()) if (idx == i).sum() >= 50 else None for i in range(len(mid))]
lt = [v for i, v in enumerate(fq_m) if mbins[i] < 11.3 and v is not None]
fig2_mass_bins = {"edges": [float(v) for v in mbins], "min_n": 50,
                  "frac_q_by_bin": {f"{mbins[i]:.2f}-{mbins[i + 1]:.2f}": v for i, v in enumerate(fq_m)},
                  "min_lower_edge_lt_11.3": float(min(lt)), "max_lower_edge_lt_11.3": float(max(lt))}
print(f"(D) Figure 2(b) bins (lower edge < 11.3): {fig2_mass_bins['min_lower_edge_lt_11.3']:.4f} – {fig2_mass_bins['max_lower_edge_lt_11.3']:.4f}")

# ---------------------------------------------------------------- (E) ENV_OD 스위치 순열 중요도
t1 = time.time()
pi = permutation_importance(OFF_MODELS["ENV_OD"], F_val[ARMS["ENV_OD"]], (y_val > 0).astype(int), scoring="neg_log_loss",
                            n_repeats=5, random_state=RANDOM_SEED, n_jobs=1)
perm = {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(ARMS["ENV_OD"], pi.importances_mean, pi.importances_std)}
order = sorted(perm, key=lambda c: -perm[c]["mean"])
print(f"(E) permutation importance {time.time() - t1:.0f}s: " + ", ".join(f"{c} {perm[c]['mean']:.4f}" for c in order[:5]))

out = {
    "script": "src/official_03_part1_spatial.py",
    "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
    "split": {"n_train": int(len(TR)), "n_val": int(len(VA)), "test_size": 0.2, "random_state": RANDOM_SEED, "stratify": "y>0"},
    "spatial_cv": {"block_width": 40.0, "n_blocks": int(len(np.unique(groups))), "n_splits": 5, "fold_sizes": [int(len(va)) for _, va in FOLDS],
                   "fold0_blocks": frozen_blocks, "fold0_n_galaxies": int(len(FOLDS[0][1]))},
    "r10_catalogue": r10_info,
    "baseline_ablation": baseline_ablation,
    "arms_official": arms_official, "contrasts_official": contrasts_official,
    "arms_spatial": spatial, "arms_buffered": buffered, "buffer_sweep": buffer_sweep,
    "fig2_mass_bins": fig2_mass_bins,
    "perm_importance_switch_ENV_OD": {"features": ARMS["ENV_OD"], "importances": perm, "order": order,
                                      "scoring": "neg_log_loss", "n_repeats": 5, "random_state": RANDOM_SEED},
    "elapsed_seconds": time.time() - t0,
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
