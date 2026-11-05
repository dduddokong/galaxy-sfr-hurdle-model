"""RS R27 — 최종 예측 구조 H_mass (2026-11-03 사용자 채택, key_idea I56) 의 공식 검증 예측과 핵심 수치

H_mass = log M* < 11 → RS-I (rs_09 최종 적합, 공식 학습 셋 전체), 그 위 → V2_third (hl_10 최종 적합, 공식 학습 셋 전체).
두 모델 모두 이미 공식 학습 셋 전체로 적합·저장된 검증 예측을 결합한다(재학습 없음). 공식 검증 점수는 보고 전용이다(선택에 미사용).
입력: results/rstress/models/rs_final_va_predictions.csv, results/hsloop/final/val_predictions.csv
      검증 수치: rs_28_hybrid_router_summary.json (중첩 교차검증), rs_29_hmass_seeds.json (seed 0–9)
출력: results/rstress/final_hmass/val_predictions.csv, metrics_template.csv, hmass_key_numbers.json
"""
import json

import numpy as np
import pandas as pd

import rs_00_common as C

OUT = C.OUT / "final_hmass"
OUT.mkdir(parents=True, exist_ok=True)
BOUNDARY = 11.0

rs = pd.read_csv(C.OUT / "models" / "rs_final_va_predictions.csv", encoding="utf-8-sig")
hl = pd.read_csv(C.ROOT / "results" / "hsloop" / "final" / "val_predictions.csv")
m = rs.merge(hl, left_on="row_index", right_on="row", how="inner", validate="one_to_one")
assert len(m) == len(C.VA) and np.allclose(m.sfr_true, m.y_true), "두 예측 파일의 검증 행·정답이 다르다"
assert (np.sort(m.row_index.to_numpy()) == C.VA).all()
lms = C.LMS[m.row_index.to_numpy()]
interp = lms < BOUNDARY
m["pred_h_mass"] = np.where(interp, m.pred_RS_I, m.pred_v2_third)
m["part"] = np.where(interp, "interpretable_RS-I", "blackbox_V2_third")
m["log10_stellar_mass"] = lms
m[["row_index", "subhalo_id_x", "sfr_true", "log10_stellar_mass", "part", "pred_h_mass", "pred_v2_third", "pred_RS_I", "pred_ver1"]].rename(
    columns={"subhalo_id_x": "subhalo_id"}).to_csv(OUT / "val_predictions.csv", index=False, encoding="utf-8")

y = m.sfr_true.to_numpy()


def evaluate_sfr(t, p, label):
    """official_01_reference.evaluate_sfr 와 같은 정의 (All / SFR = 0 / SFR > 0 / Top 10% SFR)."""
    top = t >= np.quantile(t, 0.90)
    rows = []
    for k, msk in {"All": np.ones_like(t, bool), "SFR = 0": t == 0, "SFR > 0": t > 0, "Top 10% SFR": top}.items():
        e = p[msk] - t[msk]
        ss = np.sum((t[msk] - t[msk].mean()) ** 2)
        rows.append({"Model": label, "Regime": k, "N": int(msk.sum()), "MAE": float(np.abs(e).mean()),
                     "RMSE": float(np.sqrt(np.mean(e ** 2))), "R2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else 0.0})
    return rows


tab = pd.DataFrame(evaluate_sfr(y, m.pred_h_mass.to_numpy(), "H_mass (RS-I<11 + V2_third)")
                   + evaluate_sfr(y, m.pred_v2_third.to_numpy(), "V2_third")
                   + evaluate_sfr(y, m.pred_ver1.to_numpy(), "ver1 (ENV_OD_nb)"))
tab.to_csv(OUT / "metrics_template.csv", index=False, encoding="utf-8")

router = json.loads((C.OUT / "rs_28_hybrid_router_summary.json").read_text(encoding="utf-8"))
seeds = json.loads((C.OUT / "rs_29_hmass_seeds.json").read_text(encoding="utf-8"))
g = lambda lab, reg, col: float(tab[(tab.Model == lab) & (tab.Regime == reg)][col].iloc[0])
H, V2 = "H_mass (RS-I<11 + V2_third)", "V2_third"
key = {
    "script": "src/rs_30_final_hmass.py",
    "boundary_log_mstar": BOUNDARY,
    "official_frac_interp": float(interp.mean()),
    "official_rmse_hmass": g(H, "All", "RMSE"), "official_rmse_v2": g(V2, "All", "RMSE"),
    "official_mae_hmass": g(H, "All", "MAE"), "official_mae_v2": g(V2, "All", "MAE"),
    "official_r2_hmass": g(H, "All", "R2"),
    "official_rmse_zero_hmass": g(H, "SFR = 0", "RMSE"), "official_rmse_pos_hmass": g(H, "SFR > 0", "RMSE"),
    "official_rmse_top10_hmass": g(H, "Top 10% SFR", "RMSE"), "official_mae_top10_hmass": g(H, "Top 10% SFR", "MAE"),
    "nested_inner_mean_rmse_hmass": router["mean_rmse_inner3"]["H_mass"], "nested_inner_mean_rmse_bb": router["mean_rmse_inner3"]["BB"],
    "nested_n_better_of15": router["vs_BB"]["H_mass"]["n_better"], "nested_spatial_better_of5": router["vs_BB"]["H_mass"]["spatial_better"],
    "nested_mean_diff": router["vs_BB"]["H_mass"]["mean_diff"],
    "nested_mae_hmass": router["mean_mae_inner3"]["H_mass"], "nested_mae_bb": router["mean_mae_inner3"]["BB"],
    "nested_frac_interp": router["coverage_fi_inner3"]["H_mass"]["all"],
    "seeds_mean_rmse_hmass": seeds["mean"]["H_mass"], "seeds_mean_rmse_v2": seeds["mean"]["V2_third"],
    "seeds_n_better_of10": seeds["delta_rmse"]["n_H_mass_better"], "seeds_mean_diff": seeds["delta_rmse"]["mean"],
    "seeds_mean_mae_hmass": seeds["mean"]["mae_H_mass"], "seeds_mean_mae_v2": seeds["mean"]["mae_V2_third"],
    "seeds_mean_rmse_lo11_hmass": seeds["mean"]["lo_H_mass"], "seeds_mean_rmse_lo11_v2": seeds["mean"]["lo_V2_third"],
}
# 노트북 Part II 10.1 표·본문용 추가 수치 (같은 검증 행, 같은 지표)
ver1, v1b, rsi = m.pred_ver1.to_numpy(), m.pred_v1b.to_numpy(), m.pred_RS_I.to_numpy()
for name, p in (("ver1", ver1), ("v1b", v1b), ("rsi", rsi), ("v2", m.pred_v2_third.to_numpy()), ("hmass", m.pred_h_mass.to_numpy())):
    r = evaluate_sfr(y, p, name)
    key[f"tab_{name}_mae"], key[f"tab_{name}_rmse"], key[f"tab_{name}_r2"] = r[0]["MAE"], r[0]["RMSE"], r[0]["R2"]
    key[f"tab_{name}_rmse_top10"] = r[3]["RMSE"]
    mt = C.metrics(y, p)
    for seg in ("zero", "0-10", "10-30", "30-100", "100-400", "ge400"):
        key[f"seg_{name}_rmse_{seg.replace('-', '_')}"] = mt[f"rmse_{seg}"]
    key[f"seg_{name}_ratio_med_ge100"] = mt["ratio_med_ge100"]
lo = interp
key["lo11_rmse_hmass"] = float(np.sqrt(np.mean((m.pred_h_mass.to_numpy()[lo] - y[lo]) ** 2)))
key["lo11_rmse_v2"] = float(np.sqrt(np.mean((m.pred_v2_third.to_numpy()[lo] - y[lo]) ** 2)))
key["hi11_sse_share_v2"] = float(np.sum((m.pred_v2_third.to_numpy()[~lo] - y[~lo]) ** 2) / np.sum((m.pred_v2_third.to_numpy() - y) ** 2))
C.save_json(key, OUT / "hmass_key_numbers.json")
print(tab.round(4).to_string(index=False))
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in key.items()})
