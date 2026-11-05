"""루프 후보 모델 모음 + 실행기.  사용: python hl_02_candidates.py NAME [NAME ...]

각 후보 fn(tr, te, base_te) → te(영역 행) 예측. tr 은 전체 학습 행(영역 밖 포함).
한 실험은 한 요소만 바꾼다 — 기준 구조는 HS_mix(hs_02 와 같음):
  P(별생성 | x) × smear × 10^μ(x) + (1 − P) × E[정지]
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

import hl_00_common as C

y, F, DOM, SFM = C.y, C.F, C.DOM, C.SFM
HGB_REG = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
               early_stopping=False)
HGB_CLS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1.0,
               early_stopping=False)


# ---------------------------------------------------------------- 추가 특성 (타깃 미사용)
def extra_features():
    path = C.CACHE / "extra_feats.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = C.df
    E = pd.DataFrame(index=df.index)
    ms, msp, mr = 10 ** df.log10_stellar_mass, 10 ** df.log10_main_prog_stellar_mass, 10 ** df.log10_mass_ratio
    # 가정: mass_ratio = (작은/큰) 병합 질량비, 큰 쪽 ≈ 주 조상 → 유입 별 질량 ≈ ratio × M*_prog (docs/03, Q11 미확정)
    insitu = ms - msp * (1 + np.where(df.log10_mass_ratio > -6 + 1e-9, mr, 0.0))
    E["log_insitu_dMs"] = np.log10(np.clip(insitu, 1e6, None))
    E["insitu_frac"] = np.clip(insitu / np.clip(ms - msp, 1.0, None), -5, 5)
    E["log_ssfr_past"] = E.log_insitu_dMs - df.log10_main_prog_stellar_mass        # 과거 비성장률 대리
    E["prog_ms_over_mh"] = df.log10_main_prog_stellar_mass - df.log10_main_prog_mass
    E["mbh_over_mh"] = df.log10_BH_mass - df.log10_halo_mass
    # 이웃 (주기 경계 200 가정, docs/05 Q2): 위치·질량만 사용
    P = np.mod(df[["pos_x", "pos_y", "pos_z"]].to_numpy(float), C.BOX)
    tree = cKDTree(P, boxsize=C.BOX)
    lms = df.log10_stellar_mass.to_numpy()
    lmh = df.log10_halo_mass.to_numpy()
    big = np.where(lmh >= 13.0)[0]
    tb = cKDTree(P[big], boxsize=C.BOX)
    d, j = tb.query(P, k=2)
    # 자기 자신 제외한 가장 가까운 큰 헤일로(≥10^13) 은하까지 거리
    self_hit = big[j[:, 0]] == np.arange(len(P))
    E["dist_big_halo"] = np.where(self_hit, d[:, 1], d[:, 0])
    for r in (1.0, 3.0):
        nb = tree.query_ball_point(P, r=r, return_length=True) - 1
        E[f"n_nb_r{int(r)}"] = nb
    # 반경 3 안에서 자기보다 별 질량이 큰 은하 수 (위계 순위)
    lst = tree.query_ball_point(P, r=3.0)
    E["n_more_massive_r3"] = [int((lms[l] > lms[i]).sum()) for i, l in enumerate(lst)]
    # 이웃 평균 속도 대비 상대 속도
    V = df[["vel_x", "vel_y", "vel_z"]].to_numpy(float)
    rel = np.empty(len(P))
    for i, l in enumerate(lst):
        l = [k for k in l if k != i]
        rel[i] = np.linalg.norm(V[i] - V[l].mean(0)) if l else np.nan
    E["rel_speed_r3"] = rel
    E.to_parquet(path)
    return E


EX = None


def feats(names):
    global EX
    G = F
    extra = [n for n in names if n not in F.columns]
    if extra:
        if EX is None:
            EX = extra_features()
        G = pd.concat([F, EX[extra]], axis=1)
    return G[names]


# ---------------------------------------------------------------- 구조 부품
def mix(tr, te, fs=C.HSF, rs=42, mag="log", p_cal=False, cls_kw=None, reg_kw=None):
    """HS_mix 일반형. tr/te 는 영역 행(호출자가 거른다)."""
    X = feats(fs)
    ck, rk = dict(HGB_CLS, **(cls_kw or {})), dict(HGB_REG, **(reg_kw or {}))
    s = SFM[tr]
    clf = HistGradientBoostingClassifier(random_state=rs, **ck)
    if p_cal:   # 학습 행 내부 3-fold OOF 확률로 isotonic 보정
        oof = np.empty(len(tr))
        for a, b in KFold(3, shuffle=True, random_state=rs).split(tr):
            oof[b] = HistGradientBoostingClassifier(random_state=rs, **ck).fit(X.iloc[tr[a]], s[a]).predict_proba(X.iloc[tr[b]])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, s.astype(float))
        p = iso.predict(clf.fit(X.iloc[tr], s).predict_proba(X.iloc[te])[:, 1])
    else:
        p = clf.fit(X.iloc[tr], s).predict_proba(X.iloc[te])[:, 1]
    trs = tr[s]
    if mag == "log":
        res = np.empty(len(trs))
        for a, b in KFold(2, shuffle=True, random_state=rs).split(trs):
            r = HistGradientBoostingRegressor(random_state=rs, **rk).fit(X.iloc[trs[a]], np.log10(y[trs[a]]))
            res[b] = np.log10(y[trs[b]]) - r.predict(X.iloc[trs[b]])
        e_sf = float(np.mean(10 ** res)) * 10 ** HistGradientBoostingRegressor(random_state=rs, **rk).fit(
            X.iloc[trs], np.log10(y[trs])).predict(X.iloc[te])
    elif mag == "log_local":   # 국소 smearing: E[10^res | x] 를 OOF 잔차로 회귀 (이분산 꼬리 평균)
        res = np.empty(len(trs))
        for a, b in KFold(2, shuffle=True, random_state=rs).split(trs):
            r = HistGradientBoostingRegressor(random_state=rs, **rk).fit(X.iloc[trs[a]], np.log10(y[trs[a]]))
            res[b] = np.log10(y[trs[b]]) - r.predict(X.iloc[trs[b]])
        sm = HistGradientBoostingRegressor(random_state=rs, **SMEAR_KW).fit(X.iloc[trs], 10 ** res).predict(X.iloc[te])
        e_sf = np.clip(sm, 0.5, 20.0) * 10 ** HistGradientBoostingRegressor(random_state=rs, **rk).fit(
            X.iloc[trs], np.log10(y[trs])).predict(X.iloc[te])
    elif mag == "linear":
        e_sf = np.maximum(HistGradientBoostingRegressor(random_state=rs, **rk).fit(X.iloc[trs], y[trs]).predict(X.iloc[te]), 0)
    elif mag == "poisson":
        e_sf = HistGradientBoostingRegressor(loss="poisson", random_state=rs, **rk).fit(X.iloc[trs], y[trs]).predict(X.iloc[te])
    q = float(y[tr][~s].mean())
    return p * e_sf + (1 - p) * q


SMEAR_KW = dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=1.0,
                early_stopping=False)


def mix3(tr, te, fs=C.HSF, rs=42, hi=100.0, cuts=None, cls_kw=None, reg_kw=None):
    """다클래스 혼합: 정지 / 별생성을 cuts 경계로 나눈 구간들. 클래스별 log 크기 + 전역 smearing (OOF 2-fold).
    기본 cuts=[hi] → 정지 / SF<hi / SF≥hi."""
    X = feats(fs)
    cuts = [hi] if cuts is None else cuts
    cls = np.where(~SFM[tr], 0, 1 + np.searchsorted(np.asarray(cuts), y[tr], side="right"))
    ck = dict(HGB_CLS, **(cls_kw or {}))
    rk = dict(HGB_REG, **(reg_kw or {}))
    P = HistGradientBoostingClassifier(random_state=rs, **ck).fit(X.iloc[tr], cls).predict_proba(X.iloc[te])
    out = P[:, 0] * float(y[tr][cls == 0].mean())
    for k in range(1, len(cuts) + 2):
        trk = tr[cls == k]
        res = np.empty(len(trk))
        for a, b in KFold(2, shuffle=True, random_state=rs).split(trk):
            r = HistGradientBoostingRegressor(random_state=rs, **rk).fit(X.iloc[trk[a]], np.log10(y[trk[a]]))
            res[b] = np.log10(y[trk[b]]) - r.predict(X.iloc[trk[b]])
        mu = HistGradientBoostingRegressor(random_state=rs, **rk).fit(X.iloc[trk], np.log10(y[trk])).predict(X.iloc[te])
        out = out + P[:, k] * float(np.mean(10 ** res)) * 10 ** mu
    return out


def dom(tr):
    return tr[DOM[tr]]


def bag(fn, tr, te, k=8, seed=42):
    rng = np.random.default_rng(seed)
    return np.mean([fn(rng.choice(tr, len(tr), replace=True), te, 42 + i) for i in range(k)], axis=0)


NEW_ALL = ["log_insitu_dMs", "insitu_frac", "log_ssfr_past", "prog_ms_over_mh", "mbh_over_mh",
           "dist_big_halo", "n_nb_r1", "n_nb_r3", "n_more_massive_r3", "rel_speed_r3"]

CANDS = {
    # 기준 재현
    "B_V1": (lambda tr, te, b: b, "v1", "기준 V1 (항등)"),
    "B_V1B": (lambda tr, te, b: b, "v1b", "기준 V1B (항등)"),
    "B_HSmix": (lambda tr, te, b: mix(dom(tr), te), "v1", "HS_mix 단독"),
    "B_I39": (lambda tr, te, b: 0.5 * mix(dom(tr), te) + 0.5 * b, "v1", "I39: 0.5 HS_mix + 0.5 V1"),
    # 크기 head 목적함수 (한 요소만 변경)
    "M_lin": (lambda tr, te, b: mix(dom(tr), te, mag="linear"), "v1", "HS_mix, 별생성 크기 = 선형 제곱손실"),
    "M_pois": (lambda tr, te, b: mix(dom(tr), te, mag="poisson"), "v1", "HS_mix, 별생성 크기 = Poisson"),
    # 확률 보정
    "P_iso": (lambda tr, te, b: mix(dom(tr), te, p_cal=True), "v1", "HS_mix, P(별생성) isotonic 보정"),
    # 특징 (한 묶음씩)
    "X_insitu": (lambda tr, te, b: mix(dom(tr), te, fs=C.HSF + ["log_insitu_dMs", "insitu_frac", "log_ssfr_past"]), "v1",
                 "HS_mix + 내부 생성 성장 3종"),
    "X_prog": (lambda tr, te, b: mix(dom(tr), te, fs=C.HSF + ["prog_ms_over_mh", "mbh_over_mh"]), "v1",
               "HS_mix + 조상 M*/Mh, BH/Mh"),
    "X_nbr": (lambda tr, te, b: mix(dom(tr), te, fs=C.HSF + ["dist_big_halo", "n_nb_r1", "n_nb_r3", "n_more_massive_r3", "rel_speed_r3"]),
              "v1", "HS_mix + 이웃 5종"),
    "H_local": (lambda tr, te, b: mix(dom(tr), te, mag="log_local"), "v1", "HS_mix, 별생성 크기 국소(이분산) smearing"),
    "K_mix3": (lambda tr, te, b: mix3(dom(tr), te), "v1", "3클래스 혼합 (정지/SF<100/SF≥100)"),
    "K_mix3_400": (lambda tr, te, b: mix3(dom(tr), te, hi=400.0), "v1", "3클래스 혼합 (정지/SF<400/SF≥400)"),
    "K3_insitu": (lambda tr, te, b: mix3(dom(tr), te, fs=C.HSF + ["log_insitu_dMs", "insitu_frac", "log_ssfr_past"]), "v1", "K_mix3 + 내부 생성 성장 3종"),
    "K3_prog": (lambda tr, te, b: mix3(dom(tr), te, fs=C.HSF + ["prog_ms_over_mh", "mbh_over_mh"]), "v1", "K_mix3 + 조상 M*/Mh, BH/Mh"),
    "K3_nbr": (lambda tr, te, b: mix3(dom(tr), te, fs=C.HSF + ["dist_big_halo", "n_nb_r1", "n_nb_r3", "n_more_massive_r3", "rel_speed_r3"]), "v1", "K_mix3 + 이웃 5종"),
    "K3_cap": (lambda tr, te, b: mix3(dom(tr), te, cls_kw=dict(max_leaf_nodes=15, min_samples_leaf=10)), "v1", "K_mix3, 분류기 용량↑ (잎 15, 최소 10)"),
    "K3_regz": (lambda tr, te, b: mix3(dom(tr), te, cls_kw=dict(min_samples_leaf=40, max_iter=100)), "v1", "K_mix3, 분류기 규제↑ (최소 40, 반복 100)"),
    "K4_30_100": (lambda tr, te, b: mix3(dom(tr), te, cuts=[30.0, 100.0]), "v1", "4클래스: 정지 / SF<30 / 30–100 / ≥100"),
    "K3_hi50": (lambda tr, te, b: mix3(dom(tr), te, hi=50.0), "v1", "3클래스 경계 50"),
    "K3_hi200": (lambda tr, te, b: mix3(dom(tr), te, hi=200.0), "v1", "3클래스 경계 200"),
    "K3_regz2": (lambda tr, te, b: mix3(dom(tr), te, cls_kw=dict(min_samples_leaf=80, max_iter=100)), "v1", "K3_regz 에서 분류기 최소 잎 80"),
    "K3_regz_r": (lambda tr, te, b: mix3(dom(tr), te, cls_kw=dict(min_samples_leaf=40, max_iter=100), reg_kw=dict(min_samples_leaf=40)), "v1", "K3_regz 에서 크기 회귀 최소 잎 40"),
    "K3_regz_bag": (lambda tr, te, b: bag(lambda t, e, r: mix3(t, e, rs=r, cls_kw=dict(min_samples_leaf=40, max_iter=100)), dom(tr), te), "v1", "K3_regz 부트스트랩 8개 평균"),
    "K3_regz3": (lambda tr, te, b: mix3(dom(tr), te, cls_kw=dict(min_samples_leaf=150, max_iter=100)), "v1", "분류기 최소 잎 150"),
    "K3_regz2_it50": (lambda tr, te, b: mix3(dom(tr), te, cls_kw=dict(min_samples_leaf=80, max_iter=50)), "v1", "분류기 최소 잎 80, 반복 50"),
    "K3_regz2_l3": (lambda tr, te, b: mix3(dom(tr), te, cls_kw=dict(min_samples_leaf=80, max_iter=100, max_leaf_nodes=3)), "v1", "분류기 최소 잎 80, 잎 3"),
    "X_all": (lambda tr, te, b: mix(dom(tr), te, fs=C.HSF + NEW_ALL), "v1", "HS_mix + 새 특성 10종"),
}


def run(name):
    fn, base, note = CANDS[name]
    t0 = time.time()
    res = C.evaluate(name, fn, base_key=base, note=note)
    C.log_result(res)
    return res


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        run(nm)
