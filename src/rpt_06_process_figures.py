"""연구 과정 보고서(process report) 그림 2개 (2026-11-05).

prc_01_timeline.png   연구 여정 타임라인: 시기별 질문 묶음과 그때의 모델, 주요 전환점(날짜는 기록 파일의 작성일).
prc_02_validation.png (a) 노트북 10.0d 반복 검증의 분할별 Δ RMSE(H_mass − V2_third),
                      (b) 노트북 부록 A 공동 중첩 선택의 분할별 Δ RMSE(N1 − 고정 최종)와 판독 문턱 ±0.01(docs/33 §2-2b).
수치 출처: results/rstress/rs_41_nb_repro.json (제출 노트북 실행본 출력, rs_41_nb_extract.py). 손으로 쓴 값 없음.
타임라인의 단계 이름·날짜는 수치가 아니라 기록(key_idea/README.md, docs/, review/)의 날짜다.
사용: python src/rpt_06_process_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
nb = json.loads((ROOT / "results/rstress/rs_41_nb_repro.json").read_text(encoding="utf-8"))

# Korean-capable font: set KOREAN_FONT=/path/to/font.ttf, otherwise try common system locations.
import os
for f in (os.environ.get("KOREAN_FONT", ""), "C:/Windows/Fonts/malgun.ttf",
          "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if not f:
        continue
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False
NAVY, MID, LIGHT, ACC = "#1F3A5F", "#3F6AA8", "#B8C4D6", "#C0504D"

# ------------------------------------------------------------------ 1. 타임라인
# (시작일, 종료일, 줄, 이름) — 날짜는 10월 1일 = 1 기준의 일 수 (11월 d일 = 31 + d)
d = lambda m, day: day if m == 10 else 31 + day
lanes = ["데이터·지표", "구조와 검정", "고-SFR·해석", "검증 설계"]
# (시작, 끝, 묶음, 이름, 장 번호) — 한 줄에 한 단계, 이름은 막대 오른쪽
bars = [
    (d(10, 10), d(10, 11), 0, "데이터를 처음 보다", "3.1"),
    (d(10, 15), d(10, 16), 0, "기준선 사다리", "3.2"),
    (d(10, 16), d(10, 18), 0, "설명회 — 공식 지표는 무엇에 지배되는가", "3.3"),
    (d(10, 18), d(10, 21), 1, "교차분야 가설·두 단계 구조·중심/위성 재구성", "3.4–3.5"),
    (d(10, 26), d(11, 1), 1, "내부 연구 라운드 → 중간 모델 ENV_OD", "3.6"),
    (d(10, 31), d(11, 1), 1, "중간 직후 사전 등록 검정", "3.7"),
    (d(11, 2), d(11, 4), 2, "고-SFR 표적 V2_third", "3.8"),
    (d(11, 3), d(11, 4), 2, "해석 모델 RS-I → H_mass", "3.9–3.11"),
    (d(11, 3), d(11, 4), 2, "극단 은하는 누구인가", "3.12"),
    (d(11, 4), d(11, 5), 3, "외부 검토 1차 → 재현·경계·제약 검증", "4장"),
    (d(11, 4), d(11, 6), 3, "외부 검토 2·3차 → 공동 중첩 선택·누수 점검", "4장"),
]
marks = [(d(10, 16), "설명회"), (d(10, 20), "1차 종합·조교 검토"), (d(10, 31), "중간 제출"),
         (d(11, 3), "H_mass 채택"), (d(11, 5), "최종 노트북 확정")]
n = len(bars)
fig, ax = plt.subplots(figsize=(11.5, 5.0))
cols = [LIGHT, "#7F98BF", MID, NAVY]
for i, (s, e, lane, name, ch) in enumerate(bars):
    y = n - 1 - i
    ax.barh(y, e - s, left=s, height=0.62, color=cols[lane], edgecolor="white")
    ax.text(e + 0.25, y, f"{name}  ({ch})", ha="left", va="center", fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=3)
for k, (x, name) in enumerate(marks):
    ax.plot([x, x], [-0.6, n - 0.45], color=ACC, lw=0.7, ls=":", alpha=0.55, zorder=0)
    ax.text(x, n - 0.2 + 0.55 * (k % 2), name, ha="center", va="bottom", fontsize=7.8, color=ACC)
ax.set_yticks([])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cols[k], label=lab) for k, lab in enumerate(lanes)], loc="lower left", fontsize=8, frameon=False, ncol=4)
ticks = [d(10, 10), d(10, 15), d(10, 20), d(10, 25), d(10, 31), d(11, 3), d(11, 5)]
ax.set_xticks(ticks); ax.set_xticklabels(["10.10", "10.15", "10.20", "10.25", "10.31", "11.03", "11.05"], fontsize=8.5)
ax.set_xlim(d(10, 9), d(11, 18)); ax.set_ylim(-1.6, n + 0.9)
for sp_ in ("top", "right", "left"):
    ax.spines[sp_].set_visible(False)
ax.set_xlabel("2026년 (막대의 시작·끝 = 그 단계의 기록 파일 작성일; 괄호 = 이 보고서의 장)", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "prc_01_timeline.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ 2. 검증 그림
KINDS = [("seed", "무작위\nseed 0–9"), ("inner", "내부\n5 × 3"), ("spatial", "공간\n블록 5"), ("buffer", "공간 완충\n(부호만)")]
units = nb["units"]
rows = nb["nested_rows"]
FINAL_KEY = "fixed final H_mass"
rng = np.random.default_rng(0)                                        # 점 겹침 방지용 가로 흔들기(값에는 영향 없음)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
ax = axes[0]
for i, (k, lab) in enumerate(KINDS):
    v = [u["delta_rmse"] for u in units if u["kind"] == k]
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=16, color=MID if k != "buffer" else LIGHT, edgecolor=NAVY, lw=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(KINDS))); ax.set_xticklabels([l for _, l in KINDS], fontsize=8.5)
ax.set_ylabel("Δ RMSE = H_mass - V2_third", fontsize=9)
ax.set_title("(a) 고정 조합의 재현 확인 (노트북 10.0d)", fontsize=10)
ax = axes[1]
# 판독 규칙(docs/33 §2-2b)과 같은 묶음: 무작위 = seed + 내부를 합쳐 평균, 공간 = 블록 5. 완충은 부호만 읽으므로 평균선 없음.
dN1 = lambda r: r["seg"]["N1 nested (argmin)"]["All|RMSE"] - r["seg"][FINAL_KEY]["All|RMSE"]
GROUPS = [(("seed", "inner"), "무작위\n(seed 0–9 + 내부 5 × 3)", True), (("spatial",), "공간\n블록 5", True),
          (("buffer",), "공간 완충\n(부호만)", False)]
MARK = {"seed": "o", "inner": "s", "spatial": "o", "buffer": "o"}
for i, (kinds, lab, show_mean) in enumerate(GROUPS):
    allv = []
    for k in kinds:
        v = [dN1(r) for r in rows if r["kind"] == k]
        allv += v
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=16, marker=MARK[k], color=ACC if show_mean else "#E6B0AE",
                   edgecolor="#7A2E2C", lw=0.4, label={"seed": "seed", "inner": "내부"}.get(k) if len(kinds) > 1 else None)
    if show_mean:
        ax.plot([i - 0.28, i + 0.28], [np.mean(allv)] * 2, color="black", lw=1.4)
ax.axhspan(-0.01, 0.01, color="#DDDDDD", zorder=0)
ax.axhline(0, color="black", lw=0.8)
ax.legend(fontsize=8, frameon=False, loc="upper center")
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([g[1] for g in GROUPS], fontsize=8.5)
ax.set_ylabel("Δ RMSE = N1(다시 고른 조합) - 고정 최종", fontsize=9)
ax.set_title("(b) 선택 절차의 성능 (노트북 부록 A; 회색 띠 = ±0.01)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "prc_02_validation.png", dpi=200)
plt.close(fig)
print("→", FIG / "prc_01_timeline.png", FIG / "prc_02_validation.png")
