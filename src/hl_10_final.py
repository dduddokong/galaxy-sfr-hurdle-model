"""루프 최종 후보 재현: 공식 학습 셋 전체로 학습 → 공식 검증 예측, 템플릿 evaluate_sfr 형식 표.

최종 후보 "V2_third" (JOURNAL §결론):
  기준  = ver1 ENV_OD_nb 를 random_state BASE_SEEDS 로 학습해 평균 (V1B: 42–46)
  영역  = log10_stellar_mass ≥ 11.5
  영역 안 예측 = 1/3·기준 + 1/3·P_iso + 1/3·K_mix3
    P_iso  : HS_mix (P(별생성 | x) × smearing·10^μ(x) + (1−P)·E[정지]) 의 P 를 학습 행 내부 3-fold OOF isotonic 으로 보정
    K_mix3 : 3클래스(정지 / 별생성·SFR<100 / SFR≥100) 확률 × 클래스별 log 크기(전역 smearing)
  영역 밖 = 기준 그대로
입력: 원본 13열(subhalo_id, sfr_linear 제외) 중 질량 6 + overdensity, 파생(hl_00_common.F). 위치는 쓰지 않는다(속도는 영역 모델 speed 로만).
출력: results/hsloop/final/  (val_predictions.csv, metrics_template.csv, final.json)
사용: python src/hl_10_final.py [--seeds 10] [--thr 11.25]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import hl_00_common as C
import hl_02_candidates as H

BASE_SEEDS = list(range(42, 52)) if "--seeds" in sys.argv and sys.argv[sys.argv.index("--seeds") + 1] == "10" else C.V1_SEEDS
THR = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else C.DOMAIN_LOGMS
DOMF = C.LOGMS >= THR
OUTD = C.OUT / ("final" if THR == C.DOMAIN_LOGMS else f"final_thr{THR}")
OUTD.mkdir(exist_ok=True)


def evaluate_sfr(y_true, y_pred, label):
    """템플릿 고정 셀(DO NOT MODIFY — standardized metrics)과 같은 정의."""
    rows = []
    masks = {"All": np.ones(len(y_true), bool), "SFR = 0": y_true == 0, "SFR > 0": y_true > 0,
             "Top 10% SFR": y_true >= np.quantile(y_true, 0.90)}
    for k, m in masks.items():
        rows.append({"Model": label, "Regime": k, "N": int(m.sum()), "MAE": mean_absolute_error(y_true[m], y_pred[m]),
                     "RMSE": mean_squared_error(y_true[m], y_pred[m]) ** 0.5, "R2": r2_score(y_true[m], y_pred[m])})
    return pd.DataFrame(rows)


def predict(tr, te):
    t0 = time.time()
    base = np.mean([np.maximum(np.prod(C.ver1_parts(tr, te, rs), axis=0), 0.0) for rs in BASE_SEEDS], axis=0)
    t_base = time.time() - t0
    dm = DOMF[te]
    trd = tr[DOMF[tr]]
    p_iso = H.mix(trd, te[dm], p_cal=True)
    k3 = H.mix3(trd, te[dm])
    out = base.copy()
    out[dm] = (base[dm] + p_iso + k3) / 3
    return out, base, {"train_predict_seconds_base": t_base, "train_predict_seconds_total": time.time() - t0}


if __name__ == "__main__":
    y = C.y
    pred, base, secs = predict(C.TR, C.VA)
    yv = y[C.VA]
    v1 = np.maximum(np.prod(C.ver1_parts(C.TR, C.VA, 42), axis=0), 0.0)
    tab = pd.concat([evaluate_sfr(yv, v1, "ver1 (ENV_OD_nb)"), evaluate_sfr(yv, base, f"V1B ({len(BASE_SEEDS)} seeds)"),
                     evaluate_sfr(yv, pred, "V2_third")])
    tab.to_csv(OUTD / "metrics_template.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row": C.VA, "subhalo_id": C.df.subhalo_id.to_numpy()[C.VA], "y_true": yv, "pred_v2_third": pred,
                  "pred_v1b": base, "pred_ver1": v1}).to_csv(OUTD / "val_predictions.csv", index=False)
    js = {"script": "src/hl_10_final.py", "base_seeds": BASE_SEEDS, "domain_logMs": THR,
          "FINAL_SCORE_v2_third": float(tab[(tab.Model == "V2_third") & (tab.Regime == "All")].RMSE.iloc[0]),
          "metrics_v2_third": C.metrics(yv, pred), "metrics_v1b": C.metrics(yv, base), "metrics_ver1": C.metrics(yv, v1), **secs}
    (OUTD / "final.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    print(tab.to_string(index=False))
    print("seconds:", secs)
