"""저장된 실험 예측의 영역 결합 (재학습 없음).

결합 = 영역 밖은 기준(V1 또는 V1B), 영역 안은 w_base·기준 + Σ w_k·후보_k.
후보 run 의 저장 예측은 영역 밖이 V1 이므로 영역 행만 쓴다.
사용: python hl_03_blends.py            (L2 표준 비교 실행)
"""
import numpy as np

import hl_00_common as C


def blend_dom(name, base, parts, note=""):
    inner, off = C.load_cache()
    w_base = 1.0 - sum(w for _, w in parts)
    dtr, dva = C.DOM[C.TR], C.DOM[C.VA]
    runs = {nm: C.load_run(nm) for nm, _ in parts}
    res = {"name": name, "base": base, "note": note, "reps": {}, "_inner_pred": {}}
    fold_dmse = []
    for r in C.REPS:
        b = inner[f"r{r}_{base}"]
        pred = b.copy()
        pred[dtr] = w_base * b[dtr] + sum(w * runs[nm][f"inner_r{r}"][dtr] for nm, w in parts)
        res["_inner_pred"][r] = pred
        v1 = inner[f"r{r}_v1"]
        res["reps"][r] = C.metrics(C.y[C.TR], pred)
        res["reps"][r]["base_rmse_all"] = float(np.sqrt(((v1 - C.y[C.TR]) ** 2).mean()))
        for a, bb in C.INNER[r]:
            yt = C.y[C.TR][bb]
            fold_dmse.append(float(((pred[bb] - yt) ** 2).mean() - ((v1[bb] - yt) ** 2).mean()))
    res["inner_seconds"] = 0.0
    res["inner_rmse_mean"] = float(np.mean([v["rmse_all"] for v in res["reps"].values()]))
    res["inner_base_rmse_mean"] = float(np.mean([v["base_rmse_all"] for v in res["reps"].values()]))
    res["fold_dmse"], res["folds_better"], res["n_folds"] = fold_dmse, int(sum(d < 0 for d in fold_dmse)), len(fold_dmse)
    for k in ("ratio_med_ge100", "ratio_med_ge400", "rmse_100-400", "rmse_ge400", "rmse_0-10", "rmse_zero",
              "sse_share_pred_ge50_true_lt100", "mae_all"):
        res[f"inner_{k}"] = float(np.mean([v[k] for v in res["reps"].values()]))
    b = off[base]
    p = b.copy()
    p[dva] = w_base * b[dva] + sum(w * runs[nm]["official"][dva] for nm, w in parts)
    res["official"], res["official_base"], res["_official_pred"] = C.metrics(C.y[C.VA], p), C.metrics(C.y[C.VA], off["v1"]), p
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "L2":
        # L2: 기준 × 후보 1개, 가중 0.5 (한 요소씩)
        for base in ("v1", "v1b"):
            for nm in ("B_HSmix", "P_iso", "M_pois", "M_lin", "X_insitu", "X_prog", "X_nbr", "X_all"):
                C.log_result(blend_dom(f"BL_{base}_{nm}_0.5", base, [(nm, 0.5)], note=f"영역 0.5·{base} + 0.5·{nm}"))
