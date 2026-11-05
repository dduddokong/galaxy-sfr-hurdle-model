"""공식 템플릿(설명회 2026-10-16) 기준 분할·지표 재현 + 지표 민감도 진단.

템플릿 고정 셀을 그대로 옮김:
  - train_test_split(test_size=0.20, random_state=42, stratify=(y>0))
  - 지표: 선형 sfr_linear의 MAE/RMSE/R2, 구간 All / SFR=0 / SFR>0 / Top 10% SFR
  - FINAL_SCORE = 검증 RMSE(All)
설명회 화면에 나온 값(HGB baseline RMSE 9.525463, Median baseline 12.345089)과 일치하는지 확인한다.

산출: results/official/reference_key_numbers.json, results/official/reference_metrics.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "official"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
TARGET, ID_COLUMN = "sfr_linear", "subhalo_id"

df = pd.read_csv(ROOT / "data" / "raw" / "galaxy_sfr.csv")
FEATURES = [c for c in df.columns if c not in (ID_COLUMN, TARGET)]
X, y = df[FEATURES], df[TARGET]
regime = (y > 0).astype(int)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=regime)


def regression_metrics(t, p):
    return {"MAE": mean_absolute_error(t, p), "RMSE": mean_squared_error(t, p) ** 0.5, "R2": r2_score(t, p)}


def evaluate_sfr(t, p, label):
    t, p = np.asarray(t, float), np.asarray(p, float)
    top = t >= np.quantile(t, 0.90)
    masks = {"All": np.ones_like(t, bool), "SFR = 0": t == 0, "SFR > 0": t > 0, "Top 10% SFR": top}
    return pd.DataFrame([{"Model": label, "Regime": k, "N": int(m.sum()), **regression_metrics(t[m], p[m])}
                         for k, m in masks.items()])


rows = []
dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
rows.append(evaluate_sfr(y_val, dummy.predict(X_val), "Median baseline"))

hgb = HistGradientBoostingRegressor(learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
                                    random_state=RANDOM_SEED)
t0 = time.time()
hgb.fit(X_train, y_train)
hgb_sec = time.time() - t0
p_hgb = hgb.predict(X_val)
rows.append(evaluate_sfr(y_val, p_hgb, "HGB baseline"))

# 진단: 같은 HGB를 log1p(SFR)로 학습 후 역변환 — 타깃 변환이 선형 RMSE에 주는 영향
hgb_log = HistGradientBoostingRegressor(learning_rate=0.08, max_iter=150, max_leaf_nodes=31,
                                        random_state=RANDOM_SEED).fit(X_train, np.log1p(y_train))
p_log = np.expm1(hgb_log.predict(X_val))
rows.append(evaluate_sfr(y_val, p_log, "HGB log1p-target (diag)"))
metrics = pd.concat(rows, ignore_index=True)
metrics.to_csv(OUT / "reference_metrics.csv", index=False)

# 진단: 선형 RMSE의 오차 집중도 (HGB baseline 잔차 제곱합 중 상위 SFR 은하 비중)
yv = y_val.to_numpy()
se = (p_hgb - yv) ** 2
order = np.argsort(-yv)
share = {f"top_{q}pct": float(se[order[: int(len(yv) * q / 100)]].sum() / se.sum()) for q in (0.1, 1, 10)}
# 진단: 최대 SSE 기여 은하 1개를 뺐을 때 RMSE
worst = np.argmax(se)


def m(label, regime_name, col):
    return float(metrics.loc[(metrics.Model == label) & (metrics.Regime == regime_name), col].iloc[0])


key = {
    "split": {"n_train": len(y_train), "n_val": len(y_val), "val_zero_frac": float((y_val == 0).mean()),
              "val_top10_threshold_sfr": float(np.quantile(yv, 0.90))},
    "target_linear": {"mean": float(y.mean()), "std": float(y.std()), "median": float(y.median()),
                      "q99": float(y.quantile(0.99)), "max": float(y.max()),
                      "val_max": float(yv.max())},
    "median_baseline_rmse_all": m("Median baseline", "All", "RMSE"),
    "hgb_baseline": {"rmse_all": m("HGB baseline", "All", "RMSE"), "mae_all": m("HGB baseline", "All", "MAE"),
                     "r2_all": m("HGB baseline", "All", "R2"), "rmse_top10": m("HGB baseline", "Top 10% SFR", "RMSE"),
                     "train_seconds": hgb_sec},
    "hgb_log1p_diag": {"rmse_all": m("HGB log1p-target (diag)", "All", "RMSE"),
                       "mae_all": m("HGB log1p-target (diag)", "All", "MAE")},
    "hgb_sse_share_by_true_sfr_rank": share,
    "hgb_worst_single_galaxy": {"true_sfr": float(yv[worst]), "pred_sfr": float(p_hgb[worst]),
                                "sse_share": float(se[worst] / se.sum()),
                                "rmse_all_without_it": float(np.sqrt(np.delete(se, worst).mean()))},
    # 전 데이터에서 SFR 상위 0.1% 은하의 물리적 성격 (선형 RMSE를 지배하는 집단)
    "top0p1pct_profile": (lambda top: {
        "n": int(len(top)),
        "sfr_threshold": float(y.quantile(0.999)),
        "median_log10_stellar_mass": float(top.log10_stellar_mass.median()),
        "median_log10_halo_mass": float(top.log10_halo_mass.median()),
        "median_log10_overdensity_plus_1": float(top.log10_overdensity_plus_1.median()),
        "frac_mass_ratio_above_floor": float((top.log10_mass_ratio > df.log10_mass_ratio.min()).mean()),
        "frac_mass_ratio_above_floor_all": float((df.log10_mass_ratio > df.log10_mass_ratio.min()).mean()),
    })(df[y >= y.quantile(0.999)]),
    "matches_briefing_screen": {"hgb_rmse_9.525463": abs(m("HGB baseline", "All", "RMSE") - 9.525463) < 1e-3,
                                "median_rmse_12.345089": abs(m("Median baseline", "All", "RMSE") - 12.345089) < 1e-3},
}
(OUT / "reference_key_numbers.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
print(metrics.to_string())
print(json.dumps(key, indent=2, ensure_ascii=False))
