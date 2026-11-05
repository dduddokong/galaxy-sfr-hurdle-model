"""RS R17 판정 — docs/15 H1–H4 를 rs_20_highmass_booster_{42,43,44,spatial,official}.json 에서 계산한다.
출력: results/rstress/rs_20_highmass_booster_summary.json
"""
import json

import numpy as np

import rs_00_common as C

R = {u: json.loads((C.OUT / f"rs_20_highmass_booster_{u}.json").read_text(encoding="utf-8"))
     for u in ["42", "43", "44", "spatial", "official"]}
REPS = ["42", "43", "44"]
out = {"script": "src/rs_20b_booster_summary.py", "prereg": "docs/15_i43_prereg_highmass_booster.md"}


def fold_diff(arm, ref="V1"):
    return np.concatenate([np.array(R[u]["fold_rmse_band"][arm]) - np.array(R[u]["fold_rmse_band"][ref]) for u in REPS])


# H1 성능 / H2 안정성 (15 fold)
h = {}
for arm in ["B3", "B3_addgate", "B3_free", "B2_ref", "twomode_free(rs12)", "K_mix3", "V2_third", "RS-I"]:
    d = fold_diff(arm)
    h[arm] = {"n_worse_gt1": int((d > 1.0).sum()), "n_better": int((d < 0).sum()), "mean_diff": float(d.mean()),
              "sd_diff": float(d.std(ddof=1)), "band_rmse_by_rep": [R[u]["metrics_band"][arm]["rmse_all"] for u in REPS]
              if arm in R["42"]["metrics_band"] else None}
out["fold_vs_V1"] = h
out["band_rmse_V1_by_rep"] = [R[u]["metrics_band"]["V1"]["rmse_all"] for u in REPS]
out["H1_rejected"] = bool(h["B3"]["n_worse_gt1"] >= 8)
out["H2_more_stable_than_twomode"] = bool(h["B3"]["sd_diff"] < h["twomode_free(rs12)"]["sd_diff"])

# H3 부스터 해석 안정성
h3 = {}
for t in ["lmh", "dgs", "lbh_hh", "lms"]:
    per = []
    for u in REPS:
        c = R[u]["curves"][f"boost_{t}"]
        n_act = c["active_n"]
        agree = max(n_act, c["n"] - n_act)
        per.append({"rep": u, "active_n": n_act, "agree": agree, "cv_active": c["cv_active"],
                    "rise_mean": float(np.mean(c["rise"])), "rise": c["rise"]})
    agree_ok = all(p["agree"] >= 4 for p in per)
    active_major = [p["active_n"] >= 4 for p in per]
    cv_ok = all((p["cv_active"] is not None and p["cv_active"] < 0.5) for p, a in zip(per, active_major) if a)
    h3[t] = {"per_rep": per, "agree_ok": agree_ok, "cv_ok": cv_ok, "stable": bool(agree_ok and cv_ok)}
out["H3_boosters"] = h3
out["H3_all_stable"] = bool(all(v["stable"] for v in h3.values()))
out["base_H_by_rep"] = {u: R[u]["curves"]["base_H"] for u in REPS}

# H4 클래스 확률 부호 (Q 대 S, 과밀도)
out["H4_od_QvsS_n_pos"] = {u: R[u]["curves"]["logit_QvsS_od"]["n_pos"] for u in REPS}
out["H4_supported"] = bool(all(v >= 4 for v in out["H4_od_QvsS_n_pos"].values()))
out["record_logit_totals"] = {u: {k: R[u]["curves"][k]["total"] for k in R[u]["curves"] if k.startswith("logit_")} for u in REPS}

# H5 기록
out["whole_sample"] = {u: R[u]["whole_sample_rmse"] for u in R}
out["whole_sample_mean_inner"] = {k: float(np.mean([R[u]["whole_sample_rmse"][k] for u in REPS])) for k in R["42"]["whole_sample_rmse"]}
out["tail_ratio_band"] = {u: {a: {"ge100_med": m["ratio_med_ge100"], "ge400_med": m["ratio_med_ge400"],
                                  "rmse_100-400": m["rmse_100-400"], "rmse_ge400": m["rmse_ge400"], "rmse_0-10": m["rmse_0-10"],
                                  "rmse_zero": m["rmse_zero"]}
                              for a, m in R[u]["metrics_band"].items()} for u in R}
out["spatial_band"] = {a: m["rmse_all"] for a, m in R["spatial"]["metrics_band"].items()}
sf = R["spatial"]["fold_rmse_band"]
out["spatial_folds_B3_addgate_better"] = int(sum(a < b for a, b in zip(sf["B3_addgate"], sf["V1"])))
out["official_band_record"] = {a: m["rmse_all"] for a, m in R["official"]["metrics_band"].items()}
if out["H1_rejected"]:
    out["conclusion"] = "고질량에서 부스터 구조로도 V1 수준에 못 미침"
elif out["H2_more_stable_than_twomode"] and out["H3_all_stable"]:
    out["conclusion"] = "해석 가능한 고질량 모델 후보"
else:
    out["conclusion"] = "성능은 허용, 해석은 제한적"
C.save_json(out, C.OUT / "rs_20_highmass_booster_summary.json")
print(json.dumps({k: out[k] for k in ["H1_rejected", "H2_more_stable_than_twomode", "H3_all_stable", "H4_supported", "conclusion",
                                      "H4_od_QvsS_n_pos", "spatial_folds_B3_addgate_better", "whole_sample_mean_inner"]},
                 ensure_ascii=False, indent=1))
for arm, v in h.items():
    print(arm, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "band_rmse_by_rep"})
for t, v in h3.items():
    print(t, v["stable"], [(p["active_n"], p["cv_active"] and round(p["cv_active"], 2), round(p["rise_mean"], 3)) for p in v["per_rep"]])
print("base_H", {u: [round(b, 1) for b in v] for u, v in out["base_H_by_rep"].items()})
for u in R:
    print(u, {a: {k: (round(x, 3) if x is not None else None) for k, x in m.items()} for a, m in out["tail_ratio_band"][u].items() if a in ("V1", "B3", "B3_addgate", "B2_ref", "V2_third", "K_mix3")})
