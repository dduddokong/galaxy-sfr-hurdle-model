"""최종 후보의 공식 점수가 내부 random_state 만으로 얼마나 흔들리는가 (보고용 — 선택에 쓰지 않음).

구성 5개: (ver1 random_state 묶음, 영역 모델 random_state)
  A (42–46, 42) = hl_10_final 과 같음, B (52–56, 7), C (62–66, 99), D (72–76, 123), E (82–86, 2024)
출력: results/hsloop/hl_11_seed_noise.json
"""
import json

import numpy as np

import hl_00_common as C
import hl_02_candidates as H

CFG = {"A": (range(42, 47), 42), "B": (range(52, 57), 7), "C": (range(62, 67), 99), "D": (range(72, 77), 123),
       "E": (range(82, 87), 2024)}
y = C.y[C.VA]
dm = C.DOM[C.VA]
trd = C.TR[C.DOM[C.TR]]
out = {}
for k, (seeds, rs) in CFG.items():
    vs = [np.maximum(np.prod(C.ver1_parts(C.TR, C.VA, s), axis=0), 0.0) for s in seeds]
    base = np.mean(vs, 0)
    pi, k3 = H.mix(trd, C.VA[dm], rs=rs, p_cal=True), H.mix3(trd, C.VA[dm], rs=rs)
    v2 = base.copy(); v2[dm] = (base[dm] + pi + k3) / 3
    out[k] = {"ver1_single_first_seed": C.metrics(y, vs[0])["rmse_all"], "V1B": C.metrics(y, base)["rmse_all"],
              "V2_third": C.metrics(y, v2)["rmse_all"]}
    print(k, out[k], flush=True)
vals = lambda key: [v[key] for v in out.values()]
out["range"] = {key: [float(min(vals(key))), float(max(vals(key)))] for key in ("ver1_single_first_seed", "V1B", "V2_third")}
out["V2_minus_V1B_each"] = [v["V2_third"] - v["V1B"] for v in list(out.values())[:5]]
(C.OUT / "hl_11_seed_noise.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(out["range"], out["V2_minus_V1B_each"])
