"""RS R32 — §6-3: H_mass·V2_third 공식 점수의 내부 random_state 잡음 (보고용 — 선택에 쓰지 않음)

hl_11 과 같은 구성 5개 (V1 random_state 묶음, 영역 모델 random_state):
  A (42–46, 42) = 최종 모델, B (52–56, 7), C (62–66, 99), D (72–76, 123), E (82–86, 2024)
RS-I 는 결정적이므로 rs_31 official 예측(log M* < 11 행)을 재사용하고, 구성마다 V2_third 만 다시 학습한다.
판정(docs/29 §6-3): 구성 간 Δ(H_mass − V2_third) 부호가 5/5 같지 않으면 10.2 에 "구성에 따라 역전"으로 적는다.
출력: results/rstress/rs_33_hmass_seed_noise.json
"""
import json

import numpy as np

import hl_00_common as H0
import hl_02_candidates as H2
import rs_00_common as C

CFG = {"A": (range(42, 47), 42), "B": (range(52, 57), 7), "C": (range(62, 67), 99), "D": (range(72, 77), 123),
       "E": (range(82, 87), 2024)}
z = np.load(C.CACHE / "rs_31_preds" / "official.npz")
assert np.array_equal(np.sort(H0.VA), z["te"]) and np.array_equal(H0.TR, C.TR)
order = np.argsort(H0.VA)                       # H0.VA 순서 → 정렬 순서
y = C.y[z["te"]]
lo = C.LMS[z["te"]] < 11.0
dm = H0.DOM[H0.VA]
trd = H0.TR[H0.DOM[H0.TR]]
HL = json.loads((C.ROOT / "results" / "hsloop" / "hl_11_seed_noise.json").read_text(encoding="utf-8"))
rm = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
t0 = C.tick()
out = {"script": "src/rs_33_hmass_seed_noise.py", "configs": {k: [list(v[0]), v[1]] for k, v in CFG.items()}, "units": {}}
for k, (seeds, rs) in CFG.items():
    vs = [np.maximum(np.prod(H0.ver1_parts(H0.TR, H0.VA, s), axis=0), 0.0) for s in seeds]
    base = np.mean(vs, 0)
    v2 = base.copy()
    v2[dm] = (base[dm] + H2.mix(trd, H0.VA[dm], rs=rs, p_cal=True) + H2.mix3(trd, H0.VA[dm], rs=rs)) / 3
    v2 = v2[order]
    hm = np.where(lo, z["rsi"], v2)
    u = {"V2_third": rm(v2), "H_mass": rm(hm), "V2_third_hl11": HL[k]["V2_third"],
         "mae_V2_third": float(np.mean(np.abs(v2 - y))), "mae_H_mass": float(np.mean(np.abs(hm - y)))}
    u["delta"] = u["H_mass"] - u["V2_third"]
    out["units"][k] = u
    print(k, {a: round(b, 4) for a, b in u.items()}, f"{C.tick() - t0:.0f}s", flush=True)
U = out["units"]
assert all(abs(U[k]["V2_third"] - U[k]["V2_third_hl11"]) < 1e-4 for k in U), "V2_third 재계산이 hl_11 과 다르다"
d = np.array([U[k]["delta"] for k in U])
out["summary"] = {"hmass_min": min(U[k]["H_mass"] for k in U), "hmass_max": max(U[k]["H_mass"] for k in U),
                  "hmass_mean": float(np.mean([U[k]["H_mass"] for k in U])),
                  "v2_min": min(U[k]["V2_third"] for k in U), "v2_max": max(U[k]["V2_third"] for k in U),
                  "v2_mean": float(np.mean([U[k]["V2_third"] for k in U])),
                  "delta_mean": float(d.mean()), "delta_min": float(d.min()), "delta_max": float(d.max()),
                  "n_hmass_lower": int((d < 0).sum()), "hmass_rank_A": int(sorted(U[k]["H_mass"] for k in U).index(U["A"]["H_mass"]) + 1)}
out["verdict"] = "5/5 같은 부호" if (d < 0).all() or (d > 0).all() else "구성에 따라 역전"
out["elapsed_s"] = C.tick() - t0
C.save_json(out, C.OUT / "rs_33_hmass_seed_noise.json")
print(out["summary"], out["verdict"])
