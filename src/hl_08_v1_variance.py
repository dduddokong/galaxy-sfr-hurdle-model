"""루프 L8: ver1 자체의 분산 감소 (전체 모델) — 한 요소씩.

  V1B10 : random_state 42–51 의 10개 평균 (V1B = 42–46 의 5개). 조기종료 내부 분할만 달라진다.
  V1BB5 : 학습 행 부트스트랩 5개(random_state 42, 행 재표본 seed 0–4) 평균 — 행 다양성으로 더 강한 탈상관.
캐시에 추가: results/hsloop/_cache/v1var_inner.npz, v1var_official.npz
그 뒤 영역 결합(1/3: 기준 · P_iso · K_mix3)을 새 기준 위에서 다시 평가.
사용: python hl_08_v1_variance.py
"""
import time

import numpy as np

import hl_00_common as C
from hl_03_blends import blend_dom

t0 = time.time()
inner, off = C.load_cache()


def v1(tr, te, rs):
    p0, m = C.ver1_parts(tr, te, rs)
    return np.maximum(p0 * m, 0.0)


def boot(tr, te, k):
    rng = np.random.default_rng(k)
    return v1(rng.choice(tr, len(tr), replace=True), te, 42)


new_in, new_off = {}, {}
for r in C.REPS:
    b10 = np.full(len(C.TR), np.nan); bb = np.full(len(C.TR), np.nan)
    for a, b in C.INNER[r]:
        tr, te = C.TR[a], C.TR[b]
        extra = [v1(tr, te, rs) for rs in range(47, 52)]
        b10[b] = (5 * inner[f"r{r}_v1b"][b] + sum(extra)) / 10
        bb[b] = np.mean([boot(tr, te, k) for k in range(5)], axis=0)
    new_in[f"r{r}_v1b10"], new_in[f"r{r}_v1bb5"] = b10, bb
    print(f"rep {r}: {time.time() - t0:.0f}s", flush=True)
new_off["v1b10"] = (5 * off["v1b"] + sum(v1(C.TR, C.VA, rs) for rs in range(47, 52))) / 10
new_off["v1bb5"] = np.mean([boot(C.TR, C.VA, k) for k in range(5)], axis=0)
np.savez_compressed(C.CACHE / "v1var_inner.npz", **new_in)
np.savez_compressed(C.CACHE / "v1var_official.npz", **new_off)
print(f"done {time.time() - t0:.0f}s")
