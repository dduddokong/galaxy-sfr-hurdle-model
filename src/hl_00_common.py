"""고 SFR 개선 루프(hl_*) 공용 하네스 — 2026-11-03 자율 실험 루프.

평가 규약 (results/hsloop/JOURNAL.md §0 에 기록)
  - 공식 분할: 템플릿 고정 셀과 동일 (80/20, seed 42, SFR>0 층화). 검증 41,688행 = 보고 전용(튜닝 금지).
  - 주 선택 지표: 공식 학습 셋(166,749행) 내부 KFold 5 × 3 반복(random_state 42,43,44) OOF 의 전체 선형 RMSE.
    반복 3개의 평균과 fold 15개의 짝 비교(ΔMSE < 0 개수)를 함께 본다.
  - 기준 모델: V1 = ver1 ENV_OD_nb (random_state 42), V1B = ver1 random_state 42–46 평균.
  - 영역 모델: log M* ≥ 11.5 행만 교체(밖은 기준 모델 그대로). 후보 함수는 영역 행만 예측한다.
  - 확인 단계(hl_confirm): 무작위 분할 seed 0–9, 공간 블록 5-fold.

캐시: results/hsloop/_cache/base_inner.npz, base_official.npz (V1·V1B 와 구성 요소 p0=P(SFR>0), m=E[SFR|SFR>0])
사용: python src/hl_00_common.py   (캐시 생성)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "hsloop"
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)
SEED, BOX = 42, 200.0
DOMAIN_LOGMS = 11.5
V1_SEEDS = [42, 43, 44, 45, 46]
REPS = [42, 43, 44]

MASS6 = ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass", "log10_main_prog_mass",
         "log10_main_prog_stellar_mass", "log10_mass_ratio"]
MASS12 = MASS6 + ["log_mstar_over_mhalo", "log_mbh_over_mstar", "dlog_mhalo_growth", "dlog_mstar_growth",
                  "merger_flag", "bh_floor_flag"]
OD = "log10_overdensity_plus_1"
HSF = MASS12 + [OD, "log_dMs_lin", "speed"]          # hs_02 의 영역 모델 입력

df = pd.read_csv(ROOT / "data" / "raw" / "galaxy_sfr.csv")
y = df.sfr_linear.to_numpy()
N = len(df)
F = df[MASS6 + [OD]].copy()
F["log_mstar_over_mhalo"] = df.log10_stellar_mass - df.log10_halo_mass
F["log_mbh_over_mstar"] = df.log10_BH_mass - df.log10_stellar_mass
F["dlog_mhalo_growth"] = df.log10_halo_mass - df.log10_main_prog_mass
F["dlog_mstar_growth"] = df.log10_stellar_mass - df.log10_main_prog_stellar_mass
F["merger_flag"] = (df.log10_mass_ratio > -6 + 1e-9).astype(int)
F["bh_floor_flag"] = (df.log10_BH_mass <= 5 + 6e-7).astype(int)
F["log_dMs_lin"] = np.log10(np.clip(10 ** df.log10_stellar_mass - 10 ** df.log10_main_prog_stellar_mass, 1e6, None))
F["speed"] = np.sqrt(df.vel_x ** 2 + df.vel_y ** 2 + df.vel_z ** 2)
LOGMS = df.log10_stellar_mass.to_numpy()
DOM = LOGMS >= DOMAIN_LOGMS
LSSFR = np.where(y > 0, np.log10(np.clip(y, 1e-12, None)) - LOGMS, -np.inf)
SFM = LSSFR >= -11.0                                   # 별생성 모드 (docs/04 정지 주 정의의 여집합)

IDX = np.arange(N)
TR, VA = train_test_split(IDX, test_size=0.20, random_state=SEED, stratify=(y > 0).astype(int))
VA = np.sort(VA)          # TR 은 train_test_split 순서 유지: ver1 HGB 조기종료 내부 분할이 행 순서에 의존(정렬 시 공식 7.560 ≠ 7.434)
INNER = {r: list(KFold(5, shuffle=True, random_state=r).split(TR)) for r in REPS}   # TR 위치 인덱스


# ---------------------------------------------------------------- ver1
def ver1_parts(tr, te, rs):
    NB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=rs)
    Xtr, ytr = F.iloc[tr], y[tr]
    sw = HistGradientBoostingClassifier(**NB).fit(Xtr[MASS12 + [OD]], (ytr > 0).astype(int))
    mg = HistGradientBoostingRegressor(**NB).fit(Xtr.loc[ytr > 0, MASS12], ytr[ytr > 0])
    return sw.predict_proba(F.iloc[te][MASS12 + [OD]])[:, 1], mg.predict(F.iloc[te][MASS12])


def base_parts(tr, te):
    p0s, ms = zip(*[ver1_parts(tr, te, rs) for rs in V1_SEEDS])
    v1 = np.maximum(p0s[0] * ms[0], 0.0)
    v1b = np.mean([np.maximum(a * b, 0.0) for a, b in zip(p0s, ms)], axis=0)
    return {"v1": v1, "v1b": v1b, "p0": p0s[0], "m": ms[0]}


def build_cache():
    t0 = time.time()
    inner = {}
    for r in REPS:
        arr = {k: np.full(len(TR), np.nan) for k in ("v1", "v1b", "p0", "m")}
        for a, b in INNER[r]:
            parts = base_parts(TR[a], TR[b])
            for k in arr:
                arr[k][b] = parts[k]
        for k in arr:
            inner[f"r{r}_{k}"] = arr[k]
        print(f"inner rep {r}: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(CACHE / "base_inner.npz", **inner)
    off = base_parts(TR, VA)
    np.savez_compressed(CACHE / "base_official.npz", **off)
    print(f"cache done {time.time() - t0:.0f}s")


def load_cache():
    """기준 예측 캐시. 추가 기준(hl_08 의 v1b10, v1bb5 등)이 있으면 합친다."""
    inner, off = dict(np.load(CACHE / "base_inner.npz")), dict(np.load(CACHE / "base_official.npz"))
    for tag in ("v1var",):
        if (CACHE / f"{tag}_inner.npz").exists():
            inner.update(np.load(CACHE / f"{tag}_inner.npz")); off.update(np.load(CACHE / f"{tag}_official.npz"))
    return inner, off


# ---------------------------------------------------------------- 지표
SEG = [("zero", 0, 0), ("0-10", 0, 10), ("10-30", 10, 30), ("30-100", 30, 100), ("100-400", 100, 400), ("ge400", 400, np.inf)]


def metrics(yt, p):
    yt, p = np.asarray(yt, float), np.asarray(p, float)
    se = (p - yt) ** 2
    out = {"n": int(len(yt)), "rmse_all": float(np.sqrt(se.mean())), "mae_all": float(np.abs(p - yt).mean())}
    q90 = np.quantile(yt, 0.9)
    out["rmse_top10"] = float(np.sqrt(se[yt >= q90].mean()))
    for name, lo, hi in SEG:
        m = (yt == 0) if name == "zero" else ((yt > lo) & (yt < hi)) if lo == 0 else ((yt >= lo) & (yt < hi))
        out[f"n_{name}"] = int(m.sum())
        out[f"rmse_{name}"] = float(np.sqrt(se[m].mean())) if m.any() else None
        out[f"sse_share_{name}"] = float(se[m].sum() / se.sum())
    for t in (100, 400):
        m = yt >= t
        out[f"ratio_med_ge{t}"] = float(np.median(p[m] / yt[m])) if m.any() else None
        out[f"ratio_sum_ge{t}"] = float(p[m].sum() / yt[m].sum()) if m.any() else None
    # 과대예측 쪽 손해: 실제 < 100 인데 예측 ≥ 50 인 행의 SSE 몫
    fp = (yt < 100) & (p >= 50)
    out["n_pred_ge50_true_lt100"] = int(fp.sum())
    out["sse_share_pred_ge50_true_lt100"] = float(se[fp].sum() / se.sum())
    return out


def combine(base, dom_pred, dom_mask):
    out = base.copy()
    out[dom_mask] = dom_pred
    return out


# ---------------------------------------------------------------- 후보 평가
def evaluate(name, fn, base_key="v1", reps=REPS, official=True, note=""):
    """fn(tr_rows, te_rows, base_te) → te_rows 예측 (te_rows 는 영역 행만, base_te 는 그 행의 기준 예측).
    tr_rows 는 전체 학습 행(영역 밖 포함) — 영역만 쓸지는 fn 이 결정한다.
    반환: dict (inner 반복별 지표, fold 짝 비교, official 지표, 시간)."""
    inner, off = load_cache()
    t0 = time.time()
    res = {"name": name, "base": base_key, "note": note, "reps": {}}
    fold_dmse = []
    for r in reps:
        base = inner[f"r{r}_{base_key}"]
        pred = base.copy()
        for a, b in INNER[r]:
            tr, te = TR[a], TR[b]
            dm = DOM[te]
            pred[b[dm]] = fn(tr, te[dm], base[b[dm]])
            yt = y[te]
            fold_dmse.append(float(((pred[b] - yt) ** 2).mean() - ((base[b] - yt) ** 2).mean()))
        res.setdefault("_inner_pred", {})[r] = pred
        res["reps"][r] = metrics(y[TR], pred)
        res["reps"][r]["base_rmse_all"] = float(np.sqrt(((base - y[TR]) ** 2).mean()))
    res["inner_seconds"] = time.time() - t0
    res["inner_rmse_mean"] = float(np.mean([v["rmse_all"] for v in res["reps"].values()]))
    res["inner_base_rmse_mean"] = float(np.mean([v["base_rmse_all"] for v in res["reps"].values()]))
    res["fold_dmse"] = fold_dmse
    res["folds_better"] = int(sum(d < 0 for d in fold_dmse))
    res["n_folds"] = len(fold_dmse)
    for k in ("ratio_med_ge100", "ratio_med_ge400", "rmse_100-400", "rmse_ge400", "rmse_0-10", "rmse_zero",
              "sse_share_pred_ge50_true_lt100", "mae_all"):
        res[f"inner_{k}"] = float(np.mean([v[k] for v in res["reps"].values()]))
    if official:
        t1 = time.time()
        base = off[base_key]
        dm = DOM[VA]
        p = combine(base, fn(TR, VA[dm], base[dm]), dm)
        res["official"] = metrics(y[VA], p)
        res["official_base"] = metrics(y[VA], base)
        res["official_seconds"] = time.time() - t1
        res["_official_pred"] = p
    return res


def log_result(res, registry="registry.csv"):
    """한 줄 요약을 results/hsloop/registry.csv 에 추가하고 전체를 json 으로 저장."""
    row = {"name": res["name"], "base": res["base"], "inner_rmse_mean": res["inner_rmse_mean"],
           "inner_base_rmse_mean": res["inner_base_rmse_mean"], "folds_better": res["folds_better"],
           "n_folds": res["n_folds"], "inner_ratio_med_ge100": res["inner_ratio_med_ge100"],
           "inner_ratio_med_ge400": res["inner_ratio_med_ge400"], "inner_rmse_ge400": res["inner_rmse_ge400"],
           "inner_rmse_100-400": res["inner_rmse_100-400"], "inner_mae_all": res["inner_mae_all"],
           "official_rmse_all": res.get("official", {}).get("rmse_all"),
           "official_base_rmse_all": res.get("official_base", {}).get("rmse_all"),
           "inner_seconds": res["inner_seconds"], "note": res["note"]}
    path = OUT / registry
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")
    js = {k: v for k, v in res.items() if not k.startswith("_")}
    (OUT / "runs").mkdir(exist_ok=True)
    (OUT / "runs" / f"{res['name']}.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "runs" / f"{res['name']}_preds.npz", official=res.get("_official_pred", np.array([])),
                        **{f"inner_r{r}": v for r, v in res["_inner_pred"].items()})
    print(f"[{res['name']}] inner {res['inner_rmse_mean']:.4f} (base {res['inner_base_rmse_mean']:.4f}) "
          f"folds {res['folds_better']}/{res['n_folds']}  ratio100 {res['inner_ratio_med_ge100']:.3f} "
          f"ratio400 {res['inner_ratio_med_ge400']:.3f}  official {row['official_rmse_all']}  ({res['inner_seconds']:.0f}s)",
          flush=True)
    return row


if __name__ == "__main__":
    build_cache()


def load_run(name):
    z = np.load(OUT / "runs" / f"{name}_preds.npz")
    return {k: z[k] for k in z.files}


def blend_runs(new_name, parts, note=""):
    """저장된 실험 예측의 가중 결합 (재학습 없음). parts = [(run 이름 또는 'v1'/'v1b', 가중치), ...]."""
    inner, off = load_cache()
    res = {"name": new_name, "base": "v1", "note": note, "reps": {}, "_inner_pred": {}}
    fold_dmse = []
    get = lambda nm, key, r: (inner[f"r{r}_{nm}"] if key == "inner" else off[nm]) if nm in ("v1", "v1b") else \
        load_run(nm)[f"inner_r{r}" if key == "inner" else "official"]
    for r in REPS:
        pred = sum(w * get(nm, "inner", r) for nm, w in parts)
        base = inner[f"r{r}_v1"]
        res["_inner_pred"][r] = pred
        res["reps"][r] = metrics(y[TR], pred)
        res["reps"][r]["base_rmse_all"] = float(np.sqrt(((base - y[TR]) ** 2).mean()))
        for a, b in INNER[r]:
            fold_dmse.append(float(((pred[b] - y[TR][b]) ** 2).mean() - ((base[b] - y[TR][b]) ** 2).mean()))
    res["inner_seconds"] = 0.0
    res["inner_rmse_mean"] = float(np.mean([v["rmse_all"] for v in res["reps"].values()]))
    res["inner_base_rmse_mean"] = float(np.mean([v["base_rmse_all"] for v in res["reps"].values()]))
    res["fold_dmse"], res["folds_better"], res["n_folds"] = fold_dmse, int(sum(d < 0 for d in fold_dmse)), len(fold_dmse)
    for k in ("ratio_med_ge100", "ratio_med_ge400", "rmse_100-400", "rmse_ge400", "rmse_0-10", "rmse_zero",
              "sse_share_pred_ge50_true_lt100", "mae_all"):
        res[f"inner_{k}"] = float(np.mean([v[k] for v in res["reps"].values()]))
    p = sum(w * get(nm, "official", None) for nm, w in parts)
    res["official"], res["official_base"] = metrics(y[VA], p), metrics(y[VA], off["v1"])
    res["_official_pred"] = p
    return res
