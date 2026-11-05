"""RS R29 — 루프 2: 경계 선택 절차의 중첩 검증 (사전 등록 docs/30 §2)

바깥 분할 = rs_31 의 단위(seed0–9, sp0–4, r42_0–r44_4; official 제외). 바깥 학습 셋 T 안에서만
KFold(5, shuffle, random_state=0)(rs_28 outer_fold 의 안쪽 정의) OOF 로 RS-I·V2_third 를 예측하고,
경계 후보 {10.5, 11.0, 11.5, 12.0}(결과 전 고정) 중 안쪽 OOF RMSE 최소를 고른다.
바깥 검증 성능은 rs_31 이 저장한 바깥 예측을 재결합해 계산한다(재학습 없음).
사용: rs_31b_boundary_nested.py            안쪽 OOF 계산(단위별 저장, 이어 하기 가능)
      rs_31b_boundary_nested.py --summary  rs_31 예측과 결합해 요약 json 작성
출력: results/rstress/rs_31b_boundary_nested.json, _cache/rs_31b_oof/<unit>.npz
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import rs_00_common as C
from rs_09_final_model import predict_rs
from rs_15_seed_splits import fit_rs
from rs_25_cross_stacking import bb
from rs_31_hmass_repro import BOUNDS, PRED_DIR, units

OOF_DIR = C.CACHE / "rs_31b_oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def inner_oof(T):
    o_rsi, o_v2 = np.empty(len(T)), np.empty(len(T))
    for a, b in KFold(5, shuffle=True, random_state=0).split(T):
        o_v2[b] = bb(T[a], T[b])["v2"]
        o_rsi[b] = predict_rs(fit_rs(T[a]), C.X.iloc[T[b]].reset_index(drop=True))["pred"]
    return o_rsi, o_v2


def run():
    t0 = C.tick()
    for name, kind, tr, te in units():
        if kind == "official":
            continue
        f = OOF_DIR / f"{name}.npz"
        if f.exists():
            continue
        o_rsi, o_v2 = inner_oof(tr)
        np.savez_compressed(f, T=tr, rsi=o_rsi, v2=o_v2)
        yT, lT = C.y[tr], C.LMS[tr]
        g = {b: rmse(np.where(lT < b, o_rsi, o_v2), yT) for b in BOUNDS}
        print(name, {b: round(v, 4) for b, v in g.items()}, "choice", min(g, key=g.get), f"{C.tick() - t0:.0f}s", flush=True)


def summary():
    rows = []
    for name, kind, tr, te in units():
        f, p = OOF_DIR / f"{name}.npz", PRED_DIR / f"{name}.npz"
        if kind == "official" or not f.exists() or not p.exists():
            continue
        z, q = np.load(f), np.load(p)
        assert np.array_equal(z["T"], tr) and np.array_equal(q["te"], te)
        yT, lT, yV, lV = C.y[tr], C.LMS[tr], C.y[te], C.LMS[te]
        inner = {b: rmse(np.where(lT < b, z["rsi"], z["v2"]), yT) for b in BOUNDS}
        outer = {b: rmse(np.where(lV < b, q["rsi"], q["v2"]), yV) for b in BOUNDS}
        choice = min(inner, key=inner.get)
        rows.append({"unit": name, "kind": kind, "choice": choice, **{f"inner_{b}": inner[b] for b in BOUNDS},
                     **{f"outer_{b}": outer[b] for b in BOUNDS}, "outer_nested": outer[choice],
                     "delta_nested_vs_11": outer[choice] - outer[11.0]})
    R = pd.DataFrame(rows)
    out = {"script": "src/rs_31b_boundary_nested.py", "prereg": "docs/30_loop_prereg_hmass_repro_boundary_constraint.md",
           "candidates": BOUNDS, "units": rows, "by_kind": {}}
    for kind, g in R.groupby("kind"):
        k = {"n_units": int(len(g)), "choice_counts": {str(b): int((g.choice == b).sum()) for b in BOUNDS},
             "delta_nested_vs_11": {"mean": float(g.delta_nested_vs_11.mean()), "n_lt0": int((g.delta_nested_vs_11 < 0).sum()),
                                    "n_eq0": int((g.delta_nested_vs_11 == 0).sum())}}
        for b in BOUNDS:
            if b == 11.0:
                continue
            d = (g[f"outer_{b}"] - g["outer_11.0"]).to_numpy()
            k[f"fixed_{b}_vs_11"] = {"mean": float(d.mean()), "n_lt0": int((d < 0).sum()), "min": float(d.min()), "max": float(d.max())}
        k["outer_argmin_counts"] = {str(b): int((g[[f"outer_{x}" for x in BOUNDS]].idxmin(axis=1) == f"outer_{b}").sum()) for b in BOUNDS}
        out["by_kind"][kind] = k
    # 사전 기준(docs/30 §2-4): 대안 경계 b' 가 (i) 안쪽 선택에서 과반, (ii) 고정 b' − 고정 11.0 이 seed ≥ 7/10 · 공간 ≥ 4/5 에서 < 0, (iii) 평균 < 0
    bk = out["by_kind"]
    crit = {}
    for b in BOUNDS:
        if b == 11.0 or "seed" not in bk or "spatial" not in bk:
            continue
        maj = sum(bk[k]["choice_counts"][str(b)] for k in bk) > len(R) / 2
        s_ok = bk["seed"][f"fixed_{b}_vs_11"]["n_lt0"] >= 7 and bk["spatial"][f"fixed_{b}_vs_11"]["n_lt0"] >= 4
        m_ok = all(bk[k][f"fixed_{b}_vs_11"]["mean"] < 0 for k in ("seed", "spatial"))
        crit[str(b)] = {"majority_choice": bool(maj), "wins_seed_spatial": bool(s_ok), "mean_lt0": bool(m_ok), "all": bool(maj and s_ok and m_ok)}
    out["criterion"] = crit
    out["verdict"] = ("변경 기준 충족: " + ",".join(b for b, c in crit.items() if c["all"])) if any(c["all"] for c in crit.values()) else "11.0 유지"
    C.save_json(out, C.OUT / "rs_31b_boundary_nested.json")
    print(R.groupby("kind").choice.value_counts().to_dict(), out["verdict"])
    print({k: {kk: vv for kk, vv in v.items() if kk.startswith("fixed") or kk.startswith("delta")} for k, v in bk.items()})


if __name__ == "__main__":
    summary() if "--summary" in sys.argv else run()
