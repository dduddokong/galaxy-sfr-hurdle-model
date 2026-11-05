"""I37 — 크기 head에 BH 잔차를 더한 단일 모델의 안정성 검정 (사전 등록: docs/11_i37_prereg_bh_resid_single.md).

후보: ENV_OD(기준) / M12r(1차) / 2차 변형 M12r_lin, M0r, M12r_sw.
단위: 무작위 seed 0–9 (80/20 층화) / 공간 원점 0–7 (40 단위 블록 5-fold, 원점 이동) / 공식 분할(보고만).
판정: seed ΔMSE<0 ≥ 8/10 & 원점 ≥ 6/8 & seed 평균 ΔMAE ≤ +0.010 → ADOPT_CANDIDATE, 아니면 NOT_ADOPTED.
출력: results/i37/i37_bh_resid_single.json (수치의 단일 출처), i37_units.csv, i37_progress.log
사용: python src/i37_bh_resid_single.py
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "galaxy_sfr.csv"
OUT = ROOT / "results" / "i37"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "i37_progress.log"
RANDOM_SEED, BOX, SEEDS, ORIGINS = 42, 200.0, list(range(10)), list(range(8))
MAE_TOL_ABS, MIN_SEEDS, MIN_ORIGINS = 0.010, 8, 6

MASS_RATIO_FLOOR, BH_SEED, BH_SEED_TOL = -6.0, 5.0, 6e-7
MASS6 = ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass", "log10_main_prog_mass",
         "log10_main_prog_stellar_mass", "log10_mass_ratio"]
MASS12 = MASS6 + ["log_mstar_over_mhalo", "log_mbh_over_mstar", "dlog_mhalo_growth", "dlog_mstar_growth",
                  "merger_flag", "bh_floor_flag"]
M0 = [c for c in MASS12 if c not in ("log10_BH_mass", "log_mbh_over_mstar", "bh_floor_flag")]
OD, R, RL = "log10_overdensity_plus_1", "bh_resid", "bh_resid_lin"
SWITCH_F = MASS12 + [OD]
NB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)
# 팔 → (스위치 입력, 크기 입력)
ARMS = {"ENV_OD": (SWITCH_F, MASS12), "M12r": (SWITCH_F, MASS12 + [R]), "M12r_lin": (SWITCH_F, MASS12 + [RL]),
        "M0r": (SWITCH_F, M0 + [R]), "M12r_sw": (SWITCH_F + [R], MASS12 + [R])}
PRIMARY = "M12r"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True)
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


def evaluate(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    q90 = float(np.quantile(y, 0.90)); se = (pred - y) ** 2
    out = {"n": int(len(y)), "mse_all": float(se.mean()), "rmse_all": float(se.mean() ** 0.5),
           "mae_all": float(np.abs(pred - y).mean()), "r2_all": float(r2_score(y, pred))}
    for k, m in {"zero": y == 0, "positive": y > 0, "top10": y >= q90, "bulk": (y > 0) & (y < q90)}.items():
        out[f"rmse_{k}"] = float(mean_squared_error(y[m], pred[m]) ** 0.5)
        out[f"mae_{k}"] = float(mean_absolute_error(y[m], pred[m]))
        out[f"msepart_{k}"] = float(se[m].sum() / len(y))
    return out


def paired(y, pa, pb, ids):
    y = np.asarray(y, float); d = (pa - y) ** 2 - (pb - y) ** 2; n = len(y)
    ia = np.lexsort((ids, -np.abs(d)))[0]; k = int(math.floor(0.001 * n)); top = np.lexsort((ids, -y))[:k]
    keep = np.ones(n, bool); keep[top] = False
    q90 = float(np.quantile(y, 0.90))
    return {"delta_mse": float(d.mean()), "delta_mae": float((np.abs(pa - y) - np.abs(pb - y)).mean()),
            "delta_drop_max_paired_row": float(np.delete(d, ia).mean()), "dropped_row_raw_id": int(ids[ia]),
            "delta_drop_top0p1_y": float(d[keep].mean()), "top0p1_k": k,
            "segment_contrib": {"zero": float(d[y == 0].sum() / n), "bulk": float(d[(y > 0) & (y < q90)].sum() / n), "top10": float(d[y >= q90].sum() / n)}}


def run_unit(tr, va):
    Fa, Fb = FULL.iloc[tr].copy(), FULL.iloc[va].copy(); ya = Y[tr]
    cols = ["log10_stellar_mass", "log10_halo_mass"]
    bh = HistGradientBoostingRegressor(**NB).fit(Fa[cols], Fa.log10_BH_mass)
    lin = LinearRegression().fit(Fa[cols], Fa.log10_BH_mass)
    for F in (Fa, Fb):
        F[R] = F.log10_BH_mass - bh.predict(F[cols]); F[RL] = F.log10_BH_mass - lin.predict(F[cols])
    pos = ya > 0; cache = {}; out = {}
    for name, (sf, mf) in ARMS.items():
        ks, km = ("sw", tuple(sf)), ("mg", tuple(mf))
        if ks not in cache:
            cache[ks] = HistGradientBoostingClassifier(**NB).fit(Fa[sf], pos.astype(int)).predict_proba(Fb[sf])[:, 1]
        if km not in cache:
            cache[km] = HistGradientBoostingRegressor(**NB).fit(Fa.loc[pos, mf], ya[pos]).predict(Fb[mf])
        out[name] = np.maximum(cache[ks] * cache[km], 0.0)
    return out


def summarize(label, va, preds):
    yv = Y[va]; ids = np.asarray(va)
    return {"unit": label, "n_val": int(len(va)), "metrics": {n: evaluate(yv, p) for n, p in preds.items()},
            "paired_vs_ENV_OD": {n: paired(yv, preds[n], preds["ENV_OD"], ids) for n in ARMS if n != "ENV_OD"}}


t_start = time.time()
df = pd.read_csv(RAW)
FEATURES = [c for c in df.columns if c not in ("subhalo_id", "sfr_linear")]
X, y = df[FEATURES].copy(), df["sfr_linear"].copy()
FULL = pd.concat([mass12_features(X), X[[OD]]], axis=1); Y = y.to_numpy(float); N = len(df)
P = np.mod(X[["pos_x", "pos_y", "pos_z"]].to_numpy(float), BOX)
units = []

# 공식 분할 (보고만)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=(y > 0).astype(int))
TR, VA = X_train.index.to_numpy(), X_val.index.to_numpy()
units.append(summarize("official", VA, run_unit(TR, VA)))
log("official: " + ", ".join(f"{n} {m['rmse_all']:.4f}" for n, m in units[-1]["metrics"].items()))

# 무작위 seed 0–9
for s in SEEDS:
    tr_i, va_i = train_test_split(np.arange(N), test_size=0.20, random_state=s, stratify=(Y > 0).astype(int))
    units.append(summarize(f"seed{s}", va_i, run_unit(tr_i, va_i)))
    pv = units[-1]["paired_vs_ENV_OD"][PRIMARY]
    log(f"seed{s}: {PRIMARY}-ENV_OD dMSE {pv['delta_mse']:+.3f} dMAE {pv['delta_mae']:+.4f} ({time.time() - t_start:.0f}s)")

# 공간 원점 0–7: 블록 격자 원점을 이동한 5-fold, pooled
for o in ORIGINS:
    rng = np.random.default_rng(o); shift = rng.uniform(0, 40.0, size=3)
    ijk = np.floor(np.mod(P + shift, BOX) / 40.0).astype(int); groups = 25 * ijk[:, 0] + 5 * ijk[:, 1] + ijk[:, 2]
    oof = {n: np.full(N, np.nan) for n in ARMS}
    for tr_i, va_i in GroupKFold(n_splits=5).split(np.arange(N), groups=groups):
        pr = run_unit(tr_i, va_i)
        for n in ARMS:
            oof[n][va_i] = pr[n]
    u = summarize(f"origin{o}", np.arange(N), oof); u["shift"] = [float(v) for v in shift]; units.append(u)
    pv = u["paired_vs_ENV_OD"][PRIMARY]
    log(f"origin{o}: {PRIMARY}-ENV_OD pooled dMSE {pv['delta_mse']:+.3f} dMAE {pv['delta_mae']:+.4f} ({time.time() - t_start:.0f}s)")

# 판정
def tally(prefix, arm):
    us = [u for u in units if u["unit"].startswith(prefix)]
    d = [u["paired_vs_ENV_OD"][arm]["delta_mse"] for u in us]; a = [u["paired_vs_ENV_OD"][arm]["delta_mae"] for u in us]
    flips = sum((u["paired_vs_ENV_OD"][arm]["delta_drop_max_paired_row"] >= 0) != (u["paired_vs_ENV_OD"][arm]["delta_mse"] >= 0)
                or (u["paired_vs_ENV_OD"][arm]["delta_drop_top0p1_y"] >= 0) != (u["paired_vs_ENV_OD"][arm]["delta_mse"] >= 0) for u in us)
    return {"n_units": len(us), "n_delta_mse_lt_0": int(sum(x < 0 for x in d)), "mean_delta_mse": float(np.mean(d)), "sd_delta_mse": float(np.std(d, ddof=1)),
            "mean_delta_mae": float(np.mean(a)), "n_delta_mae_lt_0": int(sum(x < 0 for x in a)), "n_units_with_sensitivity_sign_flip": int(flips),
            "delta_mse_list": d}
tallies = {arm: {"seed": tally("seed", arm), "origin": tally("origin", arm)} for arm in ARMS if arm != "ENV_OD"}
t = tallies[PRIMARY]
conds = {"seeds_ge_8_of_10": t["seed"]["n_delta_mse_lt_0"] >= MIN_SEEDS, "origins_ge_6_of_8": t["origin"]["n_delta_mse_lt_0"] >= MIN_ORIGINS,
         "seed_mean_delta_mae_le_0p010": t["seed"]["mean_delta_mae"] <= MAE_TOL_ABS}
verdict = "ADOPT_CANDIDATE" if all(conds.values()) else "NOT_ADOPTED"
log(f"verdict {verdict}: seeds {t['seed']['n_delta_mse_lt_0']}/10 mean {t['seed']['mean_delta_mse']:+.3f}; origins {t['origin']['n_delta_mse_lt_0']}/8 mean {t['origin']['mean_delta_mse']:+.3f}; seed mean dMAE {t['seed']['mean_delta_mae']:+.4f}")

rows = [{"unit": u["unit"], "arm": n, **u["metrics"][n], **({f"d_{k}": v for k, v in u["paired_vs_ENV_OD"][n].items() if not isinstance(v, dict)} if n != "ENV_OD" else {})}
        for u in units for n in ARMS]
pd.DataFrame(rows).to_csv(OUT / "i37_units.csv", index=False, encoding="utf-8")
res = {"script": "src/i37_bh_resid_single.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "prereg": "docs/11_i37_prereg_bh_resid_single.md", "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
       "arms": {k: {"switch": v[0], "magnitude": v[1]} for k, v in ARMS.items()}, "primary": PRIMARY,
       "rule": {"min_seeds_lt_0": MIN_SEEDS, "min_origins_lt_0": MIN_ORIGINS, "seed_mean_delta_mae_max": MAE_TOL_ABS},
       "official": units[0], "units": {u["unit"]: u for u in units[1:]}, "tallies": tallies, "conditions": conds, "verdict": verdict,
       "elapsed_seconds": time.time() - t_start}
(OUT / "i37_bh_resid_single.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"wrote i37_bh_resid_single.json ({time.time() - t_start:.0f}s)")
