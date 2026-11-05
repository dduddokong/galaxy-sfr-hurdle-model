"""RS R33 — §6-4: 완충 구역 공간 5-fold 에서 H_mass vs V2_third (docs/29 §6-4)

Part I 7.0b 의 완충 변형(`BUF_FOLDS`)과 같은 정의: 공간 5-fold(C.SPATIAL)의 각 학습 집합에서 검증 은하로부터
주기 상자 최소상 거리 5 단위 안의 학습 은하를 뺀다(검증 집합은 그대로). 판정은 부호만 읽는다(Part I 규약).
출력: results/rstress/rs_34_hmass_buffer.json, _cache/rs_31_preds/buf<f>.npz (rs_31 과 같은 형식)
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import rs_00_common as C
from rs_09_final_model import predict_rs
from rs_15_seed_splits import fit_rs
from rs_25_cross_stacking import bb
from rs_31_hmass_repro import PRED_DIR, unit_metrics

BUFFER = 5.0


def buffer_units():
    out = []
    for f, (a, b) in enumerate(C.SPATIAL):
        tr, va = np.asarray(a), np.asarray(b)
        P = np.mod(C._P, C.BOX)                     # Part I P_all 과 같은 처리
        d = cKDTree(P[va], boxsize=C.BOX).query(P[tr], k=1)[0]
        out.append((f"buf{f}", "buffer", tr[d > BUFFER], va))
    return out


if __name__ == "__main__":
    t0 = C.tick()
    rows = []
    for name, kind, tr, te in buffer_units():
        f = PRED_DIR / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            rsi, v2, p_on = z["rsi"], z["v2"], z["p_on"]
        else:
            B = bb(tr, te)
            P = predict_rs(fit_rs(tr), C.X.iloc[te].reset_index(drop=True))
            rsi, v2, p_on = P["pred"], B["v2"], P["p_on"]
            np.savez_compressed(f, te=te, rsi=rsi, v2=v2, v1b=B["v1b"], p_on=p_on, log_mean_on=P["log_mean_on"])
        r = unit_metrics(name, kind, tr, te, rsi, v2, p_on)
        rows.append(r)
        print(name, len(tr), {k: round(r[k], 4) for k in ["rmse_hmass", "rmse_v2", "delta_rmse", "delta_mae"]}, f"{C.tick() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    sp = {u["unit"]: u for u in C.json.loads((C.OUT / "rs_31_hmass_repro.json").read_text(encoding="utf-8"))["units"]}
    same_sign = [bool(np.sign(r["delta_rmse"]) == np.sign(sp[f"sp{r['unit'][3:]}"]["delta_rmse"])) for r in rows]
    d = R.delta_rmse.to_numpy()
    out = {"script": "src/rs_34_hmass_buffer.py", "buffer": BUFFER, "units": rows,
           "summary": {"n_units": len(rows), "n_hmass_better": int((d < 0).sum()), "delta_mean": float(d.mean()),
                       "delta_min": float(d.min()), "delta_max": float(d.max()), "n_mae_better": int((R.delta_mae < 0).sum()),
                       "n_same_sign_as_spatial": int(sum(same_sign)), "train_frac_kept_mean": float(np.mean([r["n_train"] for r in rows]) /
                                                                                                  np.mean([sp[f"sp{i}"]["n_train"] for i in range(5)]))},
           "elapsed_s": C.tick() - t0}
    out["verdict"] = "공간 5-fold 와 부호 같음" if all(same_sign) else "공간 5-fold 와 부호가 다른 fold 있음 — 그대로 보고"
    C.save_json(out, C.OUT / "rs_34_hmass_buffer.json")
    print(out["summary"], out["verdict"])
