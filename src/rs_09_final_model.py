"""RS 최종 산출물 — 해석 모델과 결합 모델을 공식 학습 셋(TR)으로 적합해 보존 (기존 최고 예측 모델 ver1 은 versions/ver1 에 그대로)

모델 RS-I (해석 모델, 단독):  E[SFR|X] = p_on(X) × C95(M*) × 10^{a + Σ f_k}
   p_on: 가법 로지스틱(G_add, 7항 램프), 크기: 평균 척도 가법 감쇠 C_band_m (OD·BH 감쇠 단조 제약, 질량대별 BH·헤일로)
모델 RS-C (결합):  log M* < 11.0 → RS-I,  log M* ≥ 11.0 → V1(ver1, 공식 캐시 예측)
산출: results/rstress/models/rs_final.joblib (적합 객체), rs_final_va_predictions.csv (VA 행 예측; 공식 지표는 기록용),
      rs_final_manifest.json (설정·데이터 해시·지표)
사용: python src/rs_09_final_model.py
새 데이터 예측: joblib.load → predict_rs(bundle, Xnew_df) (아래 함수; Xnew_df 는 원본 15열 형식)
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import rs_00_common as C

BOUNDARY = 11.0
GATE_TERMS = ["od", "lms", "lmh", "lbh", "dgs", "dgh", "lmr"]


def features_from_raw(raw):
    """원본 15열 DataFrame → 해석 모델 특징 (rs_00_common.X 와 같은 정의)."""
    Xd = pd.DataFrame({
        "lms": raw.log10_stellar_mass, "lmh": raw.log10_halo_mass, "lbh": raw.log10_BH_mass, "od": raw.log10_overdensity_plus_1,
        "dgs": raw.log10_stellar_mass - raw.log10_main_prog_stellar_mass,
        "dgh": raw.log10_halo_mass - raw.log10_main_prog_mass, "lmr": raw.log10_mass_ratio}).reset_index(drop=True)
    return C.add_band_cols(Xd)


def fit():
    TR = C.TR
    Xtr = C.X.iloc[TR].reset_index(drop=True)
    ytr, pos, ly, lms = C.y[TR], C.POS[TR], C.LOGY[TR], C.LMS[TR]
    knots = {nm: C.ramp_knots(Xtr[nm].to_numpy(float), 10) for nm in GATE_TERMS}
    A = np.hstack([C.ramp_basis(Xtr[nm].to_numpy(float), knots[nm]) for nm in GATE_TERMS])
    sc = StandardScaler().fit(A)
    gate = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(A), pos.astype(int))
    pidx = np.where(pos)[0]
    curve = C.fit_ceiling(lms[pidx], ly[pidx], q=0.95)
    ctr = C.eval_ceiling(curve, lms[pidx])
    mag = C.AdditiveMean(C.MAG_ARMS["C_band"]).fit_mean(Xtr.iloc[pidx], ctr, ytr[pidx])
    return {"gate_knots": knots, "gate_scaler": sc, "gate": gate, "ceiling_curve": curve, "magnitude": mag,
            "boundary_logMs": BOUNDARY, "band_fill": C.BAND_FILL, "bands": C.BANDS}


def predict_rs(bundle, Xd):
    """해석 모델 RS-I 예측과 분해. Xd = features_from_raw(...) 결과."""
    A = np.hstack([C.ramp_basis(Xd[nm].to_numpy(float), bundle["gate_knots"][nm]) for nm in GATE_TERMS])
    p_on = bundle["gate"].predict_proba(bundle["gate_scaler"].transform(A))[:, 1]
    c = C.eval_ceiling(bundle["ceiling_curve"], Xd.lms.to_numpy())
    mag = bundle["magnitude"]
    terms = {nm: mag.term(Xd, nm) for nm in mag.slices}
    log_mean_on = c + mag.a_ + sum(terms.values())
    return {"p_on": p_on, "log_ceiling": c, "intercept": mag.a_, "terms": terms, "log_mean_on": log_mean_on,
            "pred": p_on * 10 ** log_mean_on}


if __name__ == "__main__":
    t0 = C.tick()
    bundle = fit()
    out_dir = C.OUT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / "rs_final.joblib")
    VA = C.VA
    Xva = features_from_raw(C.df.iloc[VA])
    P = predict_rs(bundle, Xva)
    _, v1_off = C.v1_cache()
    lms_va = C.LMS[VA]
    comb = np.where(lms_va < BOUNDARY, P["pred"], v1_off["v1"])
    yva = C.y[VA]
    rows = pd.DataFrame({"row_index": VA, "subhalo_id": C.df.subhalo_id.iloc[VA].to_numpy(), "sfr_true": yva,
                         "pred_RS_I": P["pred"], "pred_V1": v1_off["v1"], "pred_RS_C": comb, "p_on": P["p_on"],
                         "log_ceiling": P["log_ceiling"], "log_mean_on": P["log_mean_on"],
                         **{f"term_{k}": v for k, v in P["terms"].items()}})
    rows.to_csv(out_dir / "rs_final_va_predictions.csv", index=False, encoding="utf-8-sig")
    raw = (C.ROOT / "data" / "raw" / "galaxy_sfr.csv").read_bytes()
    man = {"script": "src/rs_09_final_model.py", "raw_sha256": hashlib.sha256(raw).hexdigest(),
           "split": "train_test_split(test_size=0.2, random_state=42, stratify=SFR>0); TR 순서 유지",
           "models": {"RS_I": "G_add × C95 × 10^(a+Σf), 크기 = C_band_m (OD·BH 감쇠 단조, 질량대별 BH·헤일로, 포아송 손실)",
                      "RS_C": f"log M* < {BOUNDARY}: RS_I, 그 위: V1(ver1)", "V1": "versions/ver1 (수정 없음)"},
           "official_va_record_only": {"RS_I": C.metrics(yva, P["pred"]), "RS_C": C.metrics(yva, comb), "V1": C.metrics(yva, v1_off["v1"])},
           "frac_va_below_boundary": float((lms_va < BOUNDARY).mean()),
           "intercept_a": bundle["magnitude"].a_, "optimizer": bundle["magnitude"].opt_,
           "how_to_run": "python src/rs_09_final_model.py",
           "elapsed_s": C.tick() - t0}
    C.save_json(man, out_dir / "rs_final_manifest.json")
    print({k: round(v["rmse_all"], 3) for k, v in man["official_va_record_only"].items()}, f"{C.tick() - t0:.0f}s")
