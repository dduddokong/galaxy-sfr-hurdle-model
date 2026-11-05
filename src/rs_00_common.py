"""RS 루프(질량별 천장 + 차단기·감쇠기) 공용 하네스 — 사전 등록 docs/14_rs_prereg_ceiling_stress.md

분할·평가는 hl 루프(hl_00_common.py)와 같다 (공식 80/20 seed 42, TR 순서 유지, TR 내부 KFold 5).
V1(ver1) 예측은 results/hsloop/_cache 에서 읽기 전용으로 복사해 쓴다(원본 수정 금지).
계산 예산: 병행 hl 루프와 코어를 나누기 위해 스레드 3개로 제한.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize
from sklearn.model_selection import GroupKFold, KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rstress"
CACHE = OUT / "_cache"
FIG = ROOT / "reports" / "figures"
CACHE.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED, BOX = 42, 200.0
REPS = [42, 43, 44]
LN10 = np.log(10.0)

df = pd.read_csv(ROOT / "data" / "raw" / "galaxy_sfr.csv")
y = df.sfr_linear.to_numpy()
N = len(df)
POS = y > 0
LOGY = np.where(POS, np.log10(np.where(POS, y, 1.0)), np.nan)

# 원 특징(로그 질량은 원본 그대로). 하한값(BH 1e5, mass_ratio 1e-6)은 원본 값 그대로 둔다.
X = pd.DataFrame({
    "lms": df.log10_stellar_mass, "lmh": df.log10_halo_mass, "lbh": df.log10_BH_mass,
    "od": df.log10_overdensity_plus_1,
    "dgs": df.log10_stellar_mass - df.log10_main_prog_stellar_mass,    # 지난 기간 별 질량 성장 (dex)
    "dgh": df.log10_halo_mass - df.log10_main_prog_mass,              # 지난 기간 헤일로 성장 (dex)
    "lmr": df.log10_mass_ratio,                                       # -6 = 최근 병합 없음(하한)
})
LMS = X.lms.to_numpy()
# 질량대별 가변 계수 항 (R3b): 대 밖은 기준값(BH = 하한 5, 헤일로 = 대 안 최소)으로 둬서 기여 0 (제약 항) 또는 상수(무제약 항)
BANDS = [(9.3, 10.3), (10.3, 11.0), (11.0, 13.5)]
BAND_FILL = {}
for _i, (_lo, _hi) in enumerate(BANDS, 1):
    _m = (X.lms >= _lo) & (X.lms < _hi)
    BAND_FILL[_i] = (float(X.lbh.min()), float(X.lmh[_m].min()))


def add_band_cols(Xd):
    """lbh_b*, lmh_b* 를 lms·lbh·lmh 로부터 (재)계산 — 시나리오에서 입력을 바꾼 뒤에도 호출."""
    for i, (lo, hi) in enumerate(BANDS, 1):
        m = (Xd.lms >= lo) & (Xd.lms < hi)
        Xd[f"lbh_b{i}"] = np.where(m, Xd.lbh, BAND_FILL[i][0])
        Xd[f"lmh_b{i}"] = np.where(m, Xd.lmh, BAND_FILL[i][1])
    return Xd


add_band_cols(X)

# ver1 입력(MASS12 + OD) — 게이트 비교용
MASS6 = ["log10_halo_mass", "log10_stellar_mass", "log10_BH_mass", "log10_main_prog_mass",
         "log10_main_prog_stellar_mass", "log10_mass_ratio"]
OD = "log10_overdensity_plus_1"
F = df[MASS6 + [OD]].copy()
F["log_mstar_over_mhalo"] = df.log10_stellar_mass - df.log10_halo_mass
F["log_mbh_over_mstar"] = df.log10_BH_mass - df.log10_stellar_mass
F["dlog_mhalo_growth"] = df.log10_halo_mass - df.log10_main_prog_mass
F["dlog_mstar_growth"] = df.log10_stellar_mass - df.log10_main_prog_stellar_mass
F["merger_flag"] = (df.log10_mass_ratio > -6 + 1e-9).astype(int)
F["bh_floor_flag"] = (df.log10_BH_mass <= 5 + 6e-7).astype(int)
MASS12 = MASS6 + ["log_mstar_over_mhalo", "log_mbh_over_mstar", "dlog_mhalo_growth", "dlog_mstar_growth",
                  "merger_flag", "bh_floor_flag"]

IDX = np.arange(N)
TR, VA = train_test_split(IDX, test_size=0.20, random_state=SEED, stratify=POS.astype(int))
VA = np.sort(VA)
INNER = {r: list(KFold(5, shuffle=True, random_state=r).split(TR)) for r in REPS}   # TR 위치 인덱스

_P = df[["pos_x", "pos_y", "pos_z"]].to_numpy(float)
_ijk = np.floor(np.mod(_P, BOX) / 40.0).astype(int)
BLOCK = 25 * _ijk[:, 0] + 5 * _ijk[:, 1] + _ijk[:, 2]
SPATIAL = list(GroupKFold(n_splits=5).split(IDX, groups=BLOCK))                     # 전체 행 인덱스


def v1_cache():
    """hl 루프 V1 캐시를 복사해 읽는다 (원본은 건드리지 않음)."""
    for f in ("base_inner.npz", "base_official.npz"):
        dst = CACHE / ("v1_" + f)
        if not dst.exists():
            shutil.copy2(ROOT / "results" / "hsloop" / "_cache" / f, dst)
    return dict(np.load(CACHE / "v1_base_inner.npz")), dict(np.load(CACHE / "v1_base_official.npz"))


# ---------------------------------------------------------------- 지표
SEG = [("zero", 0, 0), ("0-10", 0, 10), ("10-30", 10, 30), ("30-100", 30, 100), ("100-400", 100, 400),
       ("ge400", 400, np.inf)]


def seg_mask(yt, name, lo, hi):
    if name == "zero":
        return yt == 0
    return (yt > lo) & (yt < hi) if lo == 0 else (yt >= lo) & (yt < hi)


def metrics(yt, p):
    yt, p = np.asarray(yt, float), np.asarray(p, float)
    se = (p - yt) ** 2
    out = {"n": int(len(yt)), "rmse_all": float(np.sqrt(se.mean())), "mae_all": float(np.abs(p - yt).mean()),
           "bias_all": float((p - yt).mean())}
    for name, lo, hi in SEG:
        m = seg_mask(yt, name, lo, hi)
        out[f"n_{name}"] = int(m.sum())
        out[f"rmse_{name}"] = float(np.sqrt(se[m].mean())) if m.any() else None
        out[f"bias_{name}"] = float((p[m] - yt[m]).mean()) if m.any() else None
        out[f"sse_share_{name}"] = float(se[m].sum() / se.sum())
    for t in (100, 400):
        m = yt >= t
        out[f"ratio_med_ge{t}"] = float(np.median(p[m] / yt[m])) if m.any() else None
        out[f"ratio_sum_ge{t}"] = float(p[m].sum() / yt[m].sum()) if m.any() else None
    return out


def log_metrics(ly, mu):
    """SFR>0 행의 log 공간 오차 (dex)."""
    e = mu - ly
    return {"rmse_dex": float(np.sqrt(np.mean(e ** 2))), "mae_dex": float(np.mean(np.abs(e))), "bias_dex": float(e.mean())}


def gate_metrics(pos, p_on, nbins=10):
    """양성 = SFR=0 (I38 ap_zero 와 같은 정의). p_on = P(SFR>0)."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    pz = np.clip(1 - p_on, 1e-6, 1 - 1e-6)
    z = (~pos).astype(int)
    bins = np.minimum((pz * nbins).astype(int), nbins - 1)
    ece = sum(abs(pz[bins == b].mean() - z[bins == b].mean()) * (bins == b).mean() for b in range(nbins) if (bins == b).any())
    return {"ap_zero": float(average_precision_score(z, pz)), "brier": float(brier_score_loss(z, pz)),
            "logloss": float(log_loss(z, pz, labels=[0, 1])), "ece10": float(ece), "frac_zero": float(z.mean()),
            "n": int(len(z))}


# ---------------------------------------------------------------- 천장 C(M*)
def fit_ceiling(lms, ly, q=0.95, width=0.1, min_n=300, lo=9.3):
    """학습 SFR>0 은하의 질량 구간별 log SFR 상위 분위수. 0.1 dex 구간, n<min_n 이면 위로 병합.
    반환: (구간 중앙 질량, 분위수, 구간 n) — 선형 보간·끝값 고정으로 평가."""
    order = np.argsort(lms)
    lms, ly = lms[order], ly[order]
    edges = np.arange(lo, lms.max() + width, width)
    xs, qs, ns = [], [], []
    start = 0
    i = 1
    while start < len(lms):
        # 다음 경계까지 + 최소 개수 확보
        while i < len(edges) and (np.searchsorted(lms, edges[i]) - start < min_n):
            i += 1
        end = np.searchsorted(lms, edges[i]) if i < len(edges) else len(lms)
        if len(lms) - end < min_n:      # 남은 꼬리가 너무 작으면 합친다
            end = len(lms)
        seg = slice(start, end)
        xs.append(float(np.median(lms[seg])))
        qs.append(float(np.quantile(ly[seg], q)))
        ns.append(int(end - start))
        start = end
        i += 1
    return np.array(xs), np.array(qs), np.array(ns)


def eval_ceiling(curve, lms):
    xs, qs = curve[0], curve[1]
    return np.interp(lms, xs, qs)


# ---------------------------------------------------------------- 램프(선형 스플라인) 기저
def ramp_knots(x, n=10, lo=0.01, hi=0.99, n_even=6, min_gap=0.02):
    """매듭 = 학습 최소, 1–99 백분위 n개, 값 공간 등간격 n_even개(1–99.9 백분위 사이), 학습 최대.
    등간격 매듭은 분포가 한쪽에 몰린 변수(BH: 58%가 10^5.01 미만)에서 꼬리 구간이 매듭 1개로 뭉개지는 것을 막는다
    (2026-11-03 R3a 수정). 간격 min_gap 미만 매듭은 합친다. 최소 = 기준점."""
    qlo, qhi = np.quantile(x, [lo, 0.999])
    k = np.concatenate([[x.min()], np.quantile(x, np.linspace(lo, hi, n)), np.linspace(qlo, qhi, n_even), [x.max()]])
    k = np.unique(np.round(k, 6))
    keep = [k[0]]
    for v in k[1:-1]:
        if v - keep[-1] >= min_gap:
            keep.append(v)
    if k[-1] - keep[-1] < min_gap and len(keep) > 1:
        keep[-1] = k[-1]
    else:
        keep.append(k[-1])
    return np.array(keep)


def ramp_basis(x, knots):
    """h_j(x) = clip(x − k_j, 0, k_{j+1} − k_j). 계수 ≥ 0 이면 단조 증가, 첫 매듭(기준점)에서 0.
    학습 범위 밖은 끝값 고정(외삽 없음)."""
    k = knots
    return np.stack([np.clip(x - k[j], 0, k[j + 1] - k[j]) for j in range(len(k) - 1)], axis=1)


def hat_basis(x, knots):
    """1차 B-spline(삼각) 기저. 계수 ≤ 0 이면 f ≤ 0 (비음수 감쇠, 단조 아님). 범위 밖은 끝값 고정."""
    k = knots
    x = np.clip(x, k[0], k[-1])
    cols = []
    for j in range(len(k)):
        v = np.zeros_like(x)
        if j > 0:
            m = (x >= k[j - 1]) & (x <= k[j])
            v[m] = (x[m] - k[j - 1]) / (k[j] - k[j - 1])
        if j < len(k) - 1:
            m = (x >= k[j]) & (x <= k[j + 1])
            v[m] = np.maximum(v[m], (k[j + 1] - x[m]) / (k[j + 1] - k[j]))
        cols.append(v)
    return np.stack(cols, axis=1)


class AdditiveLog:
    """r = a + Σ_k f_k(x_k), f_k = 램프 기저의 선형 결합. 항마다 'free' | 'inc' | 'dec' | 'zero'.
    inc: 계수 ≥ 0 → 기준점(학습 최소) 0, 단조 증가, ≥0. 감쇠항 s_k 로 쓸 때는 r 에서 −s_k 이므로
    '감쇠 증가' = f_k 단조 감소 = 'dec' (계수 ≤ 0). 'neg' = 삼각 기저 × 계수 ≤ 0 → f_k ≤ 0 (비음수 감쇠만, 단조 아님).
    free 항은 학습 평균 0 으로 중심화(절편이 흡수). 제약 항은 기준점 0 (중심화하지 않음)."""

    def __init__(self, terms, n_knots=10, ridge=1e-3):
        self.terms, self.n_knots, self.ridge = terms, n_knots, ridge

    def _design(self, Xd):
        cols, bounds_lo, bounds_hi, slices = [], [], [], {}
        j = 0
        for name, mode in self.terms.items():
            if mode == "zero":
                continue
            B = self._basis(Xd, name)
            cols.append(B)
            m = B.shape[1]
            lo_, hi_ = {"free": (-np.inf, np.inf), "inc": (0, np.inf), "dec": (-np.inf, 0), "neg": (-np.inf, 0)}[mode]
            bounds_lo += [lo_] * m
            bounds_hi += [hi_] * m
            slices[name] = slice(j, j + m)
            j += m
        return np.hstack(cols) if cols else np.zeros((len(Xd), 0)), np.array(bounds_lo), np.array(bounds_hi), slices

    def fit(self, Xd, r, w=None):
        self.knots, self.center = {}, {}
        for name, mode in self.terms.items():
            if mode == "zero":
                continue
            self.knots[name] = ramp_knots(Xd[name].to_numpy(float), self.n_knots)
            if mode == "free":
                self.center[name] = ramp_basis(Xd[name].to_numpy(float), self.knots[name]).mean(0)
        B, lo_, hi_, self.slices = self._design(Xd)
        n, m = B.shape
        A = np.hstack([np.ones((n, 1)), B])
        lo_ = np.concatenate([[-np.inf], lo_])
        hi_ = np.concatenate([[np.inf], hi_])
        lam = np.sqrt(self.ridge * n)
        # 릿지: 기저 열마다 (절편 제외). 열 스케일 차이를 줄이려고 열 표준편차로 나눈 값에 벌점
        sd = A.std(0)
        sd[0] = 1.0
        sd[sd == 0] = 1.0
        R = np.zeros((m, m + 1))
        R[np.arange(m), np.arange(1, m + 1)] = lam * sd[1:]
        AA = np.vstack([A, R])
        bb = np.concatenate([r, np.zeros(m)])
        sol = lsq_linear(AA, bb, bounds=(lo_, hi_), method="trf", lsq_solver="exact", tol=1e-10, max_iter=500)
        self.coef_ = sol.x
        self.a_ = float(sol.x[0])
        return self

    def _basis(self, Xd, name):
        x = Xd[name].to_numpy(float)
        if self.terms[name] == "neg":
            return hat_basis(x, self.knots[name])
        B = ramp_basis(x, self.knots[name])
        if self.terms[name] == "free":
            B = B - self.center[name]
        return B

    def term(self, Xd, name):
        if name not in self.slices:
            return np.zeros(len(Xd))
        return self._basis(Xd, name) @ self.coef_[1:][self.slices[name]]

    def predict(self, Xd):
        return self.a_ + sum(self.term(Xd, n) for n in self.slices)

    def curve(self, name, grid):
        g = pd.DataFrame({name: grid})
        return self.term(g, name)


class AdditiveMean(AdditiveLog):
    """log10 E[y] = off + a + Σ f_k, 포아송 손실 (L-BFGS-B, 경계 제약). 초기값 = log 최소제곱 해 + 평균 맞춤 이동."""

    def __init__(self, terms, n_knots=10, ridge=1e-4):
        super().__init__(terms, n_knots)
        self.ridge_m = ridge

    def fit_mean(self, Xd, off, yv):
        super().fit(Xd, np.log10(yv) - off)                         # 매듭·중심·초기값
        B, lo_, hi_, _ = self._design(Xd)
        A = np.hstack([np.ones((len(B), 1)), B])
        sd = A.std(0); sd[0] = 1.0; sd[sd == 0] = 1.0
        lo_ = np.concatenate([[-np.inf], lo_]); hi_ = np.concatenate([[np.inf], hi_])
        b0 = self.coef_.copy()
        b0[0] += np.log10(yv.mean() / np.mean(10 ** (off + A @ b0)))
        n = len(yv)
        lam = self.ridge_m

        def fg(b):
            eta = LN10 * (off + A @ b)
            m = np.exp(np.clip(eta, -50, 50))
            f = (m - yv * eta).sum() / n + lam * np.sum((sd[1:] * b[1:]) ** 2)
            g = A.T @ (LN10 * (m - yv)) / n
            g[1:] += 2 * lam * sd[1:] ** 2 * b[1:]
            return f, g
        b0 = np.clip(b0, np.where(np.isfinite(lo_), lo_, -1e9), np.where(np.isfinite(hi_), hi_, 1e9))
        res = minimize(fg, b0, jac=True, method="L-BFGS-B", bounds=list(zip(np.where(np.isfinite(lo_), lo_, None),
                                                                           np.where(np.isfinite(hi_), hi_, None))),
                       options={"maxiter": 3000, "maxfun": 6000})
        self.coef_ = res.x
        self.a_ = float(res.x[0])
        self.opt_ = {"success": bool(res.success), "nit": int(res.nit), "fun": float(res.fun)}
        return self


# ---------------------------------------------------------------- 역변환 (log → 선형 조건부 평균)
def smear_factors(mu_tr, ly_tr, lms_tr, width=0.25, min_n=200):
    """학습(교차적합) 잔차로 smearing 계수. const = Duan, bins = 질량 구간별."""
    e = ly_tr - mu_tr
    const = float(np.mean(10 ** e))
    edges = np.arange(9.3, lms_tr.max() + width, width)
    idx = np.clip(np.searchsorted(edges, lms_tr, side="right") - 1, 0, len(edges) - 1)
    fac = {}
    # n 부족 구간은 아래 구간과 병합(위쪽 꼬리)
    groups, cur = [], []
    for b in range(len(edges)):
        cur.append(b)
        if (np.isin(idx, cur)).sum() >= min_n:
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            groups[-1] += cur
        else:
            groups.append(cur)
    table = np.ones(len(edges))
    for g in groups:
        m = np.isin(idx, g)
        table[g] = np.mean(10 ** e[m])
    return {"const": const, "edges": edges, "table": table}


def apply_smear(mu, lms, sm, kind):
    if kind == "naive":
        return 10 ** mu
    if kind == "duan":
        return 10 ** mu * sm["const"]
    if kind == "massbin":
        idx = np.clip(np.searchsorted(sm["edges"], lms, side="right") - 1, 0, len(sm["edges"]) - 1)
        return 10 ** mu * sm["table"][idx]
    raise ValueError(kind)


# ---------------------------------------------------------------- 기록
def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")


def tick():
    return time.time()


# ---------------------------------------------------------------- 크기 팔 정의 (rs_03 과 동일 — docs/14 §3-4)
_HIST = {"dgs": "free", "dgh": "free", "lmr": "free"}
MAG_ARMS = {
    "A": {"lms": "free"},
    "B": {"lms": "free", "od": "free", "lbh": "free", "lmh": "free", **_HIST},
    "C": {"lms": "free", "od": "dec", "lbh": "dec", "lmh": "free", **_HIST},
    "C_h": {"lms": "free", "od": "dec", "lbh": "dec", "lmh": "dec", **_HIST},
    "C_nn": {"lms": "free", "od": "neg", "lbh": "neg", "lmh": "neg", **_HIST},
    "E": {"lms": "free", "od": "dec", "lbh": "inc", "lmh": "inc", **_HIST},
    # R3b: 질량대별 가변 계수 (BH 감쇠·헤일로 효과가 질량대마다 다를 수 있음 — 루프 1·4 관찰)
    "B_band": {"lms": "free", "od": "free", "lbh_b1": "free", "lbh_b2": "free", "lbh_b3": "free",
               "lmh_b1": "free", "lmh_b2": "free", "lmh_b3": "free", **_HIST},
    "C_band": {"lms": "free", "od": "dec", "lbh_b1": "dec", "lbh_b2": "dec", "lbh_b3": "dec",
               "lmh_b1": "free", "lmh_b2": "free", "lmh_b3": "free", **_HIST},
    "C_band_free1": {"lms": "free", "od": "dec", "lbh_b1": "free", "lbh_b2": "dec", "lbh_b3": "dec",
                     "lmh_b1": "free", "lmh_b2": "free", "lmh_b3": "free", **_HIST},
    # R5: 고질량대(log M* ≥ 11) BH 만 무제약 — 평균 척도에서 BH 감쇠 실패가 이 대에 국한되는지
    "C_band_free3": {"lms": "free", "od": "dec", "lbh_b1": "dec", "lbh_b2": "dec", "lbh_b3": "free",
                     "lmh_b1": "free", "lmh_b2": "free", "lmh_b3": "free", **_HIST},
}
