"""가설 1 — 크기 head 설정의 공식 학습 셋 내부 CV 재선택 (사전 등록: docs/08_h1_prereg_magnitude_cv.md).

선택에는 공식 학습 행(166,749)만 쓴다. StratifiedKFold(5, seed 42) pooled OOF 로 후보 109개(BASE_nb + R09 격자 108)의
MSE·MAE 를 계산하고, 사전 등록 규칙(MAE ≤ 1.01×BASE, MSE 최소)으로 하나를 고른 뒤 공식 검증을 한 번만 한다.
재개 가능: results/h1/inner_oof/<candidate>.npy 가 있으면 그 후보는 건너뛴다.

출력: results/h1/h1_inner_cv_grid.csv, h1_inner_cv.json (수치의 단일 출처), h1_progress.log
사용: python src/h1_inner_cv_select.py
"""
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "galaxy_sfr.csv"
OUT = ROOT / "results" / "h1"
OOF_DIR = OUT / "inner_oof"
OUT.mkdir(parents=True, exist_ok=True); OOF_DIR.mkdir(exist_ok=True)
LOG = OUT / "h1_progress.log"
RANDOM_SEED, BOX, MAE_TOL = 42, 200.0, 1.01

MASS_RATIO_FLOOR, BH_SEED, BH_SEED_TOL = -6.0, 5.0, 6e-7
MASS6 = ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass", "log10_main_prog_mass",
         "log10_main_prog_stellar_mass", "log10_mass_ratio"]
MASS12 = MASS6 + ["log_mstar_over_mhalo", "log_mbh_over_mstar", "dlog_mhalo_growth", "dlog_mstar_growth",
                  "merger_flag", "bh_floor_flag"]
OD = "log10_overdensity_plus_1"
SWITCH_F = MASS12 + [OD]
NB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def mass12_features(Xin):
    F = Xin[MASS6].copy()
    F["log_mstar_over_mhalo"] = Xin.log10_stellar_mass - Xin.log10_halo_mass
    F["log_mbh_over_mstar"] = Xin.log10_BH_mass - Xin.log10_stellar_mass
    F["dlog_mhalo_growth"] = Xin.log10_halo_mass - Xin.log10_main_prog_mass
    F["dlog_mstar_growth"] = Xin.log10_stellar_mass - Xin.log10_main_prog_stellar_mass
    F["merger_flag"] = (Xin.log10_mass_ratio > MASS_RATIO_FLOOR + 1e-9).astype(int)
    F["bh_floor_flag"] = (Xin.log10_BH_mass <= BH_SEED + BH_SEED_TOL).astype(int)
    return F[MASS12]


# ---------------------------------------------------------------- 후보 (사전 등록 격자, 변경 금지)
CANDS = {"BASE_nb": dict(NB)}
for k, (lr, it, leaves, msl, l2) in enumerate(itertools.product((0.04, 0.08, 0.12), (100, 150, 300), (15, 31, 63), (20, 50), (0.0, 1.0)), 1):
    CANDS[f"HGB-{k:03d}"] = dict(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                 l2_regularization=l2, early_stopping=False, max_bins=255, random_state=RANDOM_SEED)
assert len(CANDS) == 109


def evaluate(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    q90 = float(np.quantile(y, 0.90)); se = (pred - y) ** 2
    out = {"n": int(len(y)), "mse_all": float(se.mean()), "rmse_all": float(se.mean() ** 0.5),
           "mae_all": float(np.abs(pred - y).mean()), "r2_all": float(r2_score(y, pred))}
    for k, m in {"zero": y == 0, "positive": y > 0, "top10": y >= q90, "bulk": (y > 0) & (y < q90)}.items():
        out[f"rmse_{k}"] = float(mean_squared_error(y[m], pred[m]) ** 0.5)
        out[f"mae_{k}"] = float(mean_absolute_error(y[m], pred[m]))
        out[f"msepart_{k}"] = float(se[m].sum() / len(y))
    out["rmse_all_without_worst"] = float(math.sqrt(np.sort(se)[:-1].mean()))
    return out


def paired(y, pa, pb, ids):
    y = np.asarray(y, float); d = (pa - y) ** 2 - (pb - y) ** 2; n = len(y)
    ia = np.lexsort((ids, -np.abs(d)))[0]
    k = int(math.floor(0.001 * n)); top = np.lexsort((ids, -y))[:k]; keep = np.ones(n, bool); keep[top] = False
    return {"delta_mse": float(d.mean()), "delta_mae": float((np.abs(pa - y) - np.abs(pb - y)).mean()),
            "delta_drop_max_paired_row": float(np.delete(d, ia).mean()), "dropped_row_raw_id": int(ids[ia]),
            "delta_drop_top0p1_y": float(d[keep].mean()), "top0p1_k": k}


# ---------------------------------------------------------------- 데이터
t_start = time.time()
df = pd.read_csv(RAW)
FEATURES = [c for c in df.columns if c not in ("subhalo_id", "sfr_linear")]
X, y = df[FEATURES].copy(), df["sfr_linear"].copy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=(y > 0).astype(int))
TR, VA = X_train.index.to_numpy(), X_val.index.to_numpy()
FULL = pd.concat([mass12_features(X), X[[OD]]], axis=1)
Ftr, ytr = FULL.iloc[TR].reset_index(drop=True), y_train.to_numpy()
inner = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED).split(np.zeros(len(TR)), (ytr > 0).astype(int)))
log(f"start: train {len(TR)} val {len(VA)}; inner folds {[len(v) for _, v in inner]}; candidates {len(CANDS)}")

# ---------------------------------------------------------------- 내부 CV: 스위치(고정) OOF p, 후보별 OOF m
p_path = OOF_DIR / "_switch_p.npy"
if p_path.exists():
    P_OOF = np.load(p_path)
else:
    P_OOF = np.zeros(len(TR))
    for f, (tr, va) in enumerate(inner):
        clf = HistGradientBoostingClassifier(**NB).fit(Ftr.iloc[tr][SWITCH_F], (ytr[tr] > 0).astype(int))
        P_OOF[va] = clf.predict_proba(Ftr.iloc[va][SWITCH_F])[:, 1]
    np.save(p_path, P_OOF)
log("switch OOF ready")

rows = []
for i, (name, params) in enumerate(CANDS.items(), 1):
    fp = OOF_DIR / f"{name}.npy"
    if fp.exists():
        M = np.load(fp)
    else:
        t0 = time.time(); M = np.zeros(len(TR), dtype=np.float64)
        for f, (tr, va) in enumerate(inner):
            pos = tr[ytr[tr] > 0]
            reg = HistGradientBoostingRegressor(**params).fit(Ftr.iloc[pos][MASS12], ytr[pos])
            M[va] = reg.predict(Ftr.iloc[va][MASS12])
        np.save(fp, M)
        log(f"[{i}/{len(CANDS)}] {name} fitted in {time.time() - t0:.0f}s")
    pred = np.maximum(P_OOF * M, 0.0)
    e = evaluate(ytr, pred)
    rows.append({"candidate": name, **{k: params.get(k) for k in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")},
                 "early_stopping": params.get("early_stopping", "auto"), **{f"inner_{k}": v for k, v in e.items()}})
grid = pd.DataFrame(rows)
grid.to_csv(OUT / "h1_inner_cv_grid.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- 사전 등록 선택 규칙
base = grid.loc[grid.candidate == "BASE_nb"].iloc[0]
mae_cap = MAE_TOL * base.inner_mae_all
feas = grid[grid.inner_mae_all <= mae_cap].copy()
feas["gid"] = feas.candidate.map(lambda s: -1 if s == "BASE_nb" else int(s.split("-")[1]))
feas = feas.sort_values(["inner_mse_all", "inner_mae_all", "gid"])
best = feas.iloc[0]
status = "NO_CANDIDATE" if (best.candidate == "BASE_nb" or best.inner_mse_all >= base.inner_mse_all) else "SELECTED"
uncon = grid.sort_values(["inner_mse_all", "inner_mae_all"]).iloc[0]   # 진단 전용
selection = {"rule": "min inner MSE subject to inner MAE <= 1.01 x MAE(BASE_nb); ties (MSE, MAE, grid id)",
             "base_inner_mse": float(base.inner_mse_all), "base_inner_mae": float(base.inner_mae_all), "mae_cap": float(mae_cap),
             "n_feasible": int(len(feas)), "status": status, "selected": None if status == "NO_CANDIDATE" else best.candidate,
             "selected_inner_mse": float(best.inner_mse_all), "selected_inner_mae": float(best.inner_mae_all),
             "unconstrained_min_mse_candidate_DIAGNOSTIC_ONLY": uncon.candidate,
             "unconstrained_min_mse": float(uncon.inner_mse_all), "unconstrained_min_mse_mae": float(uncon.inner_mae_all)}
log(f"selection: {status} {selection['selected']} inner MSE {best.inner_mse_all:.4f} (base {base.inner_mse_all:.4f}) "
    f"MAE {best.inner_mae_all:.4f} (cap {mae_cap:.4f}); feasible {len(feas)}/{len(grid)}")

# ---------------------------------------------------------------- 공식 검증 1회 (+ 공간 5-fold 보조)
arms = {"BASE_nb": CANDS["BASE_nb"]}
if status == "SELECTED":
    arms[best.candidate] = CANDS[best.candidate]


def run_unit(tr, va):
    Fa, ya, Fb, yb = FULL.iloc[tr], y.iloc[tr].to_numpy(), FULL.iloc[va], y.iloc[va].to_numpy()
    p = HistGradientBoostingClassifier(**NB).fit(Fa[SWITCH_F], (ya > 0).astype(int)).predict_proba(Fb[SWITCH_F])[:, 1]
    out = {}
    for name, params in arms.items():
        t0 = time.time()
        m = HistGradientBoostingRegressor(**params).fit(Fa.loc[ya > 0, MASS12], ya[ya > 0]).predict(Fb[MASS12])
        out[name] = (np.maximum(p * m, 0.0), time.time() - t0)
    return out


off = run_unit(TR, VA)
official = {n: dict(evaluate(y_val, pr), train_seconds=s) for n, (pr, s) in off.items()}
official_paired = ({f"{best.candidate}-BASE_nb": paired(y_val.to_numpy(), off[best.candidate][0], off["BASE_nb"][0], VA)}
                   if status == "SELECTED" else {})
log("official evaluation done: " + ", ".join(f"{n} RMSE {v['rmse_all']:.4f} MAE {v['mae_all']:.4f}" for n, v in official.items()))

P = X[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
ijk = np.floor(np.mod(P, BOX) / 40.0).astype(int); groups = 25 * ijk[:, 0] + 5 * ijk[:, 1] + ijk[:, 2]
folds = list(GroupKFold(n_splits=5).split(np.arange(len(P)), groups=groups))
OOF = {n: np.full(len(df), np.nan) for n in arms}; fold_delta = []
for f, (tr, va) in enumerate(folds):
    o = run_unit(tr, va)
    for n in arms:
        OOF[n][va] = o[n][0]
    if status == "SELECTED":
        yv = y.iloc[va].to_numpy()
        fold_delta.append(float(np.mean((o[best.candidate][0] - yv) ** 2) - np.mean((o["BASE_nb"][0] - yv) ** 2)))
spatial = {n: evaluate(y, OOF[n]) for n in arms}
spatial_paired = ({f"{best.candidate}-BASE_nb": dict(paired(y.to_numpy(), OOF[best.candidate], OOF["BASE_nb"], np.arange(len(df))),
                                                      folds_delta_mse=fold_delta, folds_with_delta_lt_0=int(sum(d < 0 for d in fold_delta)))}
                  if status == "SELECTED" else {})
log("spatial done: " + ", ".join(f"{n} RMSE {v['rmse_all']:.4f}" for n, v in spatial.items()))

verdict = "NOT_TESTED_NO_CANDIDATE"
if status == "SELECTED":
    pr = official_paired[f"{best.candidate}-BASE_nb"]
    if pr["delta_mse"] >= 0:
        verdict = "REJECTED_OFFICIAL_RMSE_NOT_LOWER"
    elif pr["delta_drop_max_paired_row"] >= 0 or pr["delta_drop_top0p1_y"] >= 0:
        verdict = "TAIL_FRAGILE"
    else:
        verdict = "SUPPORTED_OFFICIAL_RMSE_LOWER"

res = {"script": "src/h1_inner_cv_select.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "prereg": "docs/08_h1_prereg_magnitude_cv.md", "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
       "split": {"n_train": int(len(TR)), "n_val": int(len(VA))}, "inner_folds": [int(len(v)) for _, v in inner],
       "n_candidates": len(CANDS), "mae_tolerance_relative": MAE_TOL, "selection": selection,
       "official": official, "paired_official": official_paired, "spatial_pooled": spatial, "paired_spatial_pooled": spatial_paired,
       "verdict": verdict, "elapsed_seconds": time.time() - t_start}
(OUT / "h1_inner_cv.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"verdict {verdict}; wrote h1_inner_cv.json ({time.time() - t_start:.0f}s)")
