"""RS R28 — 루프 1: H_mass vs V2_third 반복 검증 재현 + 분할별 예측 저장 (사전 등록 docs/30 §1)

단위(분할 정의는 rs_28·rs_29 와 같다):
  seed0..seed9  train_test_split(test_size=0.2, random_state=s, stratify=SFR>0), 검증 행 정렬 (rs_29)
  sp0..sp4      40 단위 블록 GroupKFold 5 (C.SPATIAL; Part I FOLDS 와 같은 정의)
  r42_0..r44_4  공식 학습 셋 내부 KFold 5 (shuffle, random_state 42/43/44) — rs_28 바깥 fold (C.TR[a], C.TR[b])
  official      공식 분할(기록·중단 조건 확인용: H_mass 7.217941, V2_third 7.224293 과 1e-4 안)
분할마다 RS-I·V2_third(V1B 포함)를 학습 셋 전체로 적합해 검증 행 예측을 저장한다(루프 2·3 은 이 예측을 재결합한다).
출력: results/rstress/rs_31_hmass_repro.json, rs_31_hmass_repro_units.csv, _cache/rs_31_preds/<unit>.npz
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import rs_00_common as C
from rs_09_final_model import predict_rs
from rs_15_seed_splits import fit_rs
from rs_25_cross_stacking import bb

PRED_DIR = C.CACHE / "rs_31_preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)
BOUNDARY = 11.0
BOUNDS = [10.5, 11.0, 11.5, 12.0]


def units():
    out = [("official", "official", C.TR, C.VA)]
    for s in range(10):
        tr, te = train_test_split(C.IDX, test_size=0.2, random_state=s, stratify=C.POS.astype(int))
        out.append((f"seed{s}", "seed", tr, np.sort(te)))
    for f, (a, b) in enumerate(C.SPATIAL):
        out.append((f"sp{f}", "spatial", np.asarray(a), np.asarray(b)))
    for r in C.REPS:
        for f, (a, b) in enumerate(C.INNER[r]):
            out.append((f"r{r}_{f}", "inner", C.TR[a], C.TR[b]))
    return out


def rmse(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else float("nan")


def unit_metrics(name, kind, tr, te, rsi, v2, p_on):
    y, lms = C.y[te], C.LMS[te]
    lo = lms < BOUNDARY
    hm = np.where(lo, rsi, v2)
    keep = y < 400
    d_sse_hi = float(((hm[~lo] - y[~lo]) ** 2).sum() - ((v2[~lo] - y[~lo]) ** 2).sum())
    r = {"unit": name, "kind": kind, "n_train": int(len(tr)), "n_val": int(len(te)),
         "n_val_ge100": int((y >= 100).sum()), "n_val_ge400": int((y >= 400).sum()),
         "n_val_ge100_lo11": int(((y >= 100) & lo).sum()), "max_sfr_lo11": float(y[lo].max()),
         "rmse_hmass": rmse(hm, y), "rmse_v2": rmse(v2, y), "mae_hmass": float(np.mean(np.abs(hm - y))),
         "mae_v2": float(np.mean(np.abs(v2 - y))), "rmse_rsi": rmse(rsi, y),
         "rmse_lo11_hmass": rmse(hm, y, lo), "rmse_lo11_v2": rmse(v2, y, lo),
         "rmse_ex400_hmass": rmse(hm, y, keep), "rmse_ex400_v2": rmse(v2, y, keep),
         "frac_interp": float(lo.mean()), "hi11_max_abs_pred_diff": float(np.abs(hm[~lo] - v2[~lo]).max()),
         "d_sse_hi11": d_sse_hi, "d_sse_lo11": float(((hm[lo] - y[lo]) ** 2).sum() - ((v2[lo] - y[lo]) ** 2).sum()),
         "rmse_zero_hmass": rmse(hm, y, y == 0), "rmse_zero_v2": rmse(v2, y, y == 0)}
    r["delta_rmse"] = r["rmse_hmass"] - r["rmse_v2"]
    r["delta_mae"] = r["mae_hmass"] - r["mae_v2"]
    r["delta_rmse_lo11"] = r["rmse_lo11_hmass"] - r["rmse_lo11_v2"]
    r["delta_rmse_ex400"] = r["rmse_ex400_hmass"] - r["rmse_ex400_v2"]
    for b in BOUNDS:
        r[f"rmse_bound_{b}"] = rmse(np.where(lms < b, rsi, v2), y)
    return r


def summarize(R):
    out = {}
    for kind, g in R[R.kind != "official"].groupby("kind"):
        d = g.delta_rmse.to_numpy()
        out[kind] = {"n_units": int(len(g)), "n_hmass_better": int((d < 0).sum()),
                     "delta_rmse": {"mean": float(d.mean()), "median": float(np.median(d)), "min": float(d.min()), "max": float(d.max())},
                     "delta_mae": {"mean": float(g.delta_mae.mean()), "n_better": int((g.delta_mae < 0).sum())},
                     "delta_rmse_lo11": {"mean": float(g.delta_rmse_lo11.mean()), "n_better": int((g.delta_rmse_lo11 < 0).sum())},
                     "delta_rmse_ex400": {"mean": float(g.delta_rmse_ex400.mean()), "n_better": int((g.delta_rmse_ex400 < 0).sum())},
                     "mean_rmse_hmass": float(g.rmse_hmass.mean()), "mean_rmse_v2": float(g.rmse_v2.mean()),
                     "mean_mae_hmass": float(g.mae_hmass.mean()), "mean_mae_v2": float(g.mae_v2.mean()),
                     "mean_frac_interp": float(g.frac_interp.mean()),
                     "max_hi11_abs_pred_diff": float(g.hi11_max_abs_pred_diff.max()),
                     "max_abs_d_sse_hi11": float(g.d_sse_hi11.abs().max()),
                     "total_n_val_ge100_lo11": int(g.n_val_ge100_lo11.sum())}
    return out


if __name__ == "__main__":
    only = set(sys.argv[1:])
    t0 = C.tick()
    rows = []
    for name, kind, tr, te in units():
        if only and name not in only and kind not in only:
            continue
        f = PRED_DIR / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            assert np.array_equal(z["te"], te)
            rsi, v2, p_on = z["rsi"], z["v2"], z["p_on"]
        else:
            B = bb(tr, te)
            P = predict_rs(fit_rs(tr), C.X.iloc[te].reset_index(drop=True))
            rsi, v2, p_on = P["pred"], B["v2"], P["p_on"]
            np.savez_compressed(f, te=te, rsi=rsi, v2=v2, v1b=B["v1b"], p_on=p_on, log_mean_on=P["log_mean_on"])
        r = unit_metrics(name, kind, tr, te, rsi, v2, p_on)
        rows.append(r)
        print(name, {k: round(r[k], 4) for k in ["rmse_hmass", "rmse_v2", "delta_rmse", "delta_rmse_lo11", "frac_interp"]},
              f"{C.tick() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(C.OUT / "rs_31_hmass_repro_units.csv", index=False)
    R = pd.DataFrame(rows)
    out = {"script": "src/rs_31_hmass_repro.py", "prereg": "docs/30_loop_prereg_hmass_repro_boundary_constraint.md",
           "boundary": BOUNDARY, "units": rows, "summary": summarize(R)}
    # 기록 대조 (중단 조건): 공식 분할, seed 0–9 (rs_29), 내부 15 fold·공간 5 (rs_28)
    chk = {}
    if (R.unit == "official").any():
        o = R[R.unit == "official"].iloc[0]
        hm = C.json.loads((C.OUT / "final_hmass" / "hmass_key_numbers.json").read_text(encoding="utf-8"))
        chk["official"] = {"hmass": [o.rmse_hmass, hm["official_rmse_hmass"]], "v2": [o.rmse_v2, hm["official_rmse_v2"]],
                           "ok": bool(abs(o.rmse_hmass - hm["official_rmse_hmass"]) < 1e-4 and abs(o.rmse_v2 - hm["official_rmse_v2"]) < 1e-4)}
    s29 = {f"seed{r['seed']}": r for r in C.json.loads((C.OUT / "rs_29_hmass_seeds.json").read_text(encoding="utf-8"))["rows"]}
    rec = {}
    for u in ["42", "43", "44", "spatial"]:
        J = C.json.loads((C.OUT / f"rs_28_hybrid_router_{u}.json").read_text(encoding="utf-8"))
        for fo in J["folds"]:
            rec[(f"sp{fo['fold']}" if u == "spatial" else f"r{u}_{fo['fold']}")] = (fo["metrics"]["H_mass"]["rmse"], fo["metrics"]["BB"]["rmse"])
    diffs = []
    for _, r in R.iterrows():
        if r.unit in s29:
            diffs.append((r.unit, r.rmse_hmass - s29[r.unit]["H_mass"], r.rmse_v2 - s29[r.unit]["V2_third"]))
        elif r.unit in rec:
            diffs.append((r.unit, r.rmse_hmass - rec[r.unit][0], r.rmse_v2 - rec[r.unit][1]))
    chk["records"] = {"n_compared": len(diffs), "max_abs_diff": float(max((max(abs(a), abs(b)) for _, a, b in diffs), default=0.0)),
                      "per_unit": {u: [a, b] for u, a, b in diffs}}
    chk["records"]["ok"] = chk["records"]["max_abs_diff"] < 1e-4
    out["record_check"] = chk
    out["elapsed_s"] = C.tick() - t0
    C.save_json(out, C.OUT / "rs_31_hmass_repro.json")
    print(out["summary"], chk.get("official"), chk["records"]["max_abs_diff"])
