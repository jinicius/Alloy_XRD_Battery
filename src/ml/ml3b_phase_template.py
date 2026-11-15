"""ML3 (개정): 봉우리 모양 함수 비교 + 이동을 허용한 기준 패턴 분해로 2상 공존 vs 연속 전이 판정.

1) 대칭 pseudo-Voigt vs 비대칭(split) pseudo-Voigt: 1봉우리/2봉우리 BIC가 모양 함수에 따라 달라지는지
2) 기준 패턴 분해: 순수 H2(충전 초기)와 순수 H3(충전 끝) 실측 패턴을 기준으로,
   각 스캔을 '한 상(이동+퍼짐)' 또는 '두 상의 합'으로 맞춘다. NMF와 달리 봉우리 이동을 허용한다.
   단일상 모델의 불일치(1-R²)가 크면 두 상이 공존한다는 뜻이다.
3) 구조 지표(단일상 불일치 최대값, 공존 상 간 Δc, c 분포 폭)와 300사이클 유지율 비교

사용법: python ml3_phase_template.py <source_data.xlsx> <project.xlsx> <out_dir>
"""
import sys, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
warnings.filterwarnings("ignore")

SRC, PROJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LP, LC = 0.49594, 1.5406
K = np.log(2)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
X = pd.read_excel(SRC, sheet_name=None, header=None)
COL = {"s-LNO": "#1d4ed8", "d-LNO": "#ea580c", "d-LNO dense": "#f59e0b", "project A": "#1d4ed8", "project B": "#7c3aed", "project C": "#ea580c"}


def frames(sheet, n):
    d = X[sheet]; V = d.iloc[1, 1:n + 1].astype(float).to_numpy(); x = d.iloc[2:, 0].astype(float).to_numpy()
    Y = d.iloc[2:, 1:n + 1].astype(float).to_numpy(); m = (x > 5.55) & (x < 6.85); o = np.argsort(V)
    V, Y = V[o], Y[m][:, o]; ch = V >= 3.95
    return V[ch], x[m], Y[:, ch]


S = frames("Figure 3c-(003) reflection", 13)
D = frames("Figure 3b-(003) reflection", 16)
hd = X["Figure 3a for d-LNO"]; Vh = pd.to_numeric(hd.iloc[0, 1:], errors="coerce").to_numpy()
xh = pd.to_numeric(hd.iloc[1:, 0], errors="coerce").to_numpy(); Yh = hd.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
mh = (xh > 5.55) & (xh < 6.85); ch = Vh >= 3.95
Dd = (Vh[ch], xh[mh], Yh[mh][:, ch])
FITWIN = (5.7, 6.6)   # 맞춤은 이 구간에서만. 기준 패턴은 더 넓게(5.55–6.85) 잘라 이동해도 꼬리가 잘리지 않게 함
xr = pd.read_excel(PROJ, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601); tt = xr.iloc[:, 0].to_numpy(float)
P = {s: (np.array([4.08, 4.30, 4.50, 4.80]), tt, np.c_[[xr[f"{s}_{v}V"].to_numpy(float) for v in ("4.08", "4.30", "4.50", "4.80")]].T) for s in "ABC"}
res = {}

# ---------------------------------------------------------------- 1) 대칭 vs 비대칭 pseudo-Voigt
def spv(x, A, c, wl, wr, e):
    w = np.where(x < c, wl, wr); u = (x - c) / w
    return A * (e / (1 + u * u) + (1 - e) * np.exp(-K * u * u))


def fit_peaks(x, y, npk, sym):
    base = np.median(np.r_[y[:4], y[-4:]]); span = x[-1] - x[0]; c0 = x[np.argmax(y)]; H = y.max() - base
    def f(q, *p):
        m = p[-2] + p[-1] * (q - x.mean())
        for i in range(npk):
            if sym: A, c, w, e = p[4 * i:4 * i + 4]; m = m + spv(q, A, c, w, w, e)
            else: A, c, wl, wr, e = p[5 * i:5 * i + 5]; m = m + spv(q, A, c, wl, wr, e)
        return m
    one = [H, c0, .04, .3] if sym else [H, c0, .04, .04, .3]
    lo1 = [0, x[0], 1e-3, 0] if sym else [0, x[0], 1e-3, 1e-3, 0]
    hi1 = [np.inf, x[-1], span, 1] if sym else [np.inf, x[-1], span, span, 1]
    starts = [None] if npk == 1 else [c0 + d for d in np.linspace(-.25, .25, 11) if abs(d) > 1e-6] + \
        [x[i] for i in find_peaks(y, prominence=0.03 * H)[0]]
    best = None
    for s2 in starts:
        p0 = list(one); lo = list(lo1); hi = list(hi1)
        if npk == 2:
            sec = list(one); sec[0] = H * .3; sec[1] = s2; p0 += sec; lo += lo1; hi += hi1
        p0 += [base, 0]; lo += [-np.inf, -np.inf]; hi += [np.inf, np.inf]
        try:
            p, _ = curve_fit(f, x, y, p0=p0, bounds=(lo, hi), maxfev=20000); r = np.sum((y - f(x, *p)) ** 2)
            if best is None or r < best[0]: best = (r, p, f)
        except Exception:
            pass
    return best


def dbic(x, y, sym):
    n = len(x); k1, k2 = (6, 10) if sym else (7, 12)
    r1 = fit_peaks(x, y, 1, sym); r2 = fit_peaks(x, y, 2, sym)
    return n * np.log(r1[0] / n) + k1 * np.log(n) - (n * np.log(r2[0] / n) + k2 * np.log(n)), r1


shape_rows = []
for lab, (V, x, Y) in (("s-LNO", S), ("d-LNO", D)):
    fm = (x > FITWIN[0]) & (x < FITWIN[1]); x, Y = x[fm], Y[fm]
    for j, v in enumerate(V):
        b_sym, r1s = dbic(x, Y[:, j], True); b_spl, r1a = dbic(x, Y[:, j], False)
        asym = r1a[1][3] / r1a[1][2]
        shape_rows.append(dict(sample=lab, V=round(v, 3), dBIC_sym=round(b_sym, 1), dBIC_split=round(b_spl, 1), asym_wr_wl=round(asym, 2)))
shape = pd.DataFrame(shape_rows); shape.to_csv(f"{OUT}/ml3_shape_bic.csv", index=False)
res["shape_bic_all_above_10"] = {"sym": bool((shape.dBIC_sym > 10).all()), "split": bool((shape.dBIC_split > 10).all())}

# ---------------------------------------------------------------- 2) 기준 패턴(템플릿) 분해
def prep_t(x, y):
    k0 = max(len(y) // 7, 3); b = np.linspace(np.percentile(y[:k0], 20), np.percentile(y[-k0:], 20), len(y)); t = np.clip(y - b, 0, None)   # 양쪽 바닥(하위 20 %)을 이은 직선
    n = len(t); k = max(n // 10, 2); w = np.ones(n); r = 0.5 * (1 - np.cos(np.linspace(0, np.pi, k)))
    w[:k], w[-k:] = r, r[::-1]                     # 양 끝 10 %를 부드럽게 0으로 (이동시켜도 계단이 생기지 않게)
    t = t * w; return t / np.trapezoid(t, x)


def moved(x, t, d, s):
    h = x[1] - x[0]
    if s > 0.25 * h: t = gaussian_filter1d(t, s / h, mode="constant")
    return np.interp(x - d, x, t, left=0, right=0)


def fit_t(x, y, temps, starts, smax, xw=None, fm=None, dmax=0.6):
    if xw is None: xw, fm = x, np.ones_like(x, bool)
    best = None; base = np.median(np.r_[y[:4], y[-4:]]); A0 = max(np.trapezoid(y - base, x), 1e-6)
    for st in starts:
        p0, lo, hi = [], [], []
        for d0 in st: p0 += [A0 / len(st), d0, smax * 0.05]; lo += [0, -dmax, 0]; hi += [np.inf, dmax, smax]
        p0 += [base, 0]; lo += [-np.inf, -np.inf]; hi += [np.inf, np.inf]
        def model(p):
            m = p[-2] + p[-1] * (x - x.mean())
            for i, t in enumerate(temps): m = m + p[3 * i] * moved(xw, t, p[3 * i + 1], p[3 * i + 2])[fm]
            return m
        r = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi), max_nfev=4000, x_scale="jac")
        rss = np.sum(r.fun ** 2)
        if best is None or rss < best[0]: best = (rss, r.x, model(r.x))
    return best


def c_of(tt_, lam): return 3 * lam / (2 * np.sin(np.radians(tt_ / 2)))


def decompose(V, xw, Yw, lam, label, smax=0.15, win=None, smax2=None):
    smax2 = smax2 or smax
    dm = 0.6 if lam == LP else 2.0   # 이동 허용 범위 (논문 싱크로트론 2θ / 과제 Cu 2θ)
    T2, T3 = prep_t(xw, Yw[:, 0]), prep_t(xw, Yw[:, -1])
    m2, m3 = np.trapezoid(xw * T2, xw), np.trapezoid(xw * T3, xw)
    fm = (xw > win[0]) & (xw < win[1]) if win else np.ones_like(xw, bool)
    x, Y = xw[fm], Yw[fm]
    rows, fits = [], {}
    for j, v in enumerate(V):
        y = Y[:, j]; xm = x[np.argmax(y)]; sst = np.sum((y - y.mean()) ** 2)
        a = fit_t(x, y, [T2], [(xm - m2,), (0,)], smax, xw, fm, dm); b = fit_t(x, y, [T3], [(xm - m3,), (0,)], smax, xw, fm, dm)
        one, which = (a, 2) if a[0] <= b[0] else (b, 3)
        cen = (m2 if which == 2 else m3) + one[1][1]; sig = one[1][2]
        th = np.radians(cen / 2); sig_c = c_of(cen, lam) * (np.radians(sig) / 2) / np.tan(th)   # c 분포 폭 (Å, 1σ)
        two = fit_t(x, y, [T2, T3], [(d1, d2) for d1 in (0, xm - m2) for d2 in (0, xm - m3, .5 * (xm - m3))], smax2, xw, fm, dm)
        p = two[1]; ca, cb = c_of(m2 + p[1], lam), c_of(m3 + p[4], lam)
        share = min(p[0], p[3]) / (p[0] + p[3])
        rows.append(dict(sample=label, V=round(float(v), 3), misfit_1phase=one[0] / sst, misfit_2phase=two[0] / sst,
                         c_1phase=round(c_of(cen, lam), 4), sigma_c=round(sig_c, 4), f_H3=round(p[3] / (p[0] + p[3]), 3),
                         minor_share=round(share, 3), dc_phases=round(abs(ca - cb), 3)))
        fits[round(float(v), 3)] = (x, y, one[2], two[2], p, (T2, T3), xw, fm)
    return pd.DataFrame(rows), fits


tabs, FITS = [], {}
# 2상 모델에서는 각 상이 끝점 봉우리 모양을 거의 유지해야 함 (퍼짐 σ ≤ 0.03°, Cu 기준 약 0.09°).
# 이 제약이 없으면 넓게 퍼진 두 성분이 한 봉우리 모양을 흉내 내서 분해가 하나로 정해지지 않음.
for lab, data, lam, smax, smax2 in (("s-LNO", S, LP, .15, .03), ("d-LNO", D, LP, .15, .03), ("d-LNO dense", Dd, LP, .15, .03),
                             ("project A", P["A"], LC, .5, .09), ("project B", P["B"], LC, .5, .09), ("project C", P["C"], LC, .5, .09)):
    V, x, Y = data
    if lab.startswith("project"):
        i = np.argmax(Y[:, 0]); m = (x > x[i] - 0.7) & (x < x[i] + 1.8); x, Y = x[m], Y[m]   # 충전하면 2θ가 커지는 쪽으로 이동
    df, fits = decompose(V, x, Y, lam, lab, smax, None if lab.startswith("project") else FITWIN, smax2); tabs.append(df); FITS[lab] = fits
T = pd.concat(tabs, ignore_index=True)
T["misfit_1phase"] = T["misfit_1phase"].astype(float); T["misfit_2phase"] = T["misfit_2phase"].astype(float)
T.to_csv(f"{OUT}/ml3_template_table.csv", index=False)

# ---------------------------------------------------------------- 3) 구조 지표 ↔ 300사이클 유지율
c2 = X["Figure 2c"]
cs = c2.iloc[1:, 0:2].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(); cd = c2.iloc[1:, 2:4].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
ret = {"s-LNO": cs[299, 1] / cs[0, 1] * 100, "d-LNO": cd[299, 1] / cd[0, 1] * 100, "project A": 55.5, "project B": 82.0, "project C": 95.0}
summ = []
for lab in ("s-LNO", "d-LNO", "project A", "project B", "project C"):
    t = T[T["sample"] == lab]; base = t["misfit_1phase"].iloc[[0, -1]].mean(); k = t["misfit_1phase"].idxmax(); r = T.loc[k]
    summ.append(dict(sample=lab, max_misfit_1phase=float(r["misfit_1phase"]), at_V=float(r["V"]), endpoint_misfit=float(base),
                     ratio_to_endpoints=round(float(r["misfit_1phase"] / base), 1), dc_at_max=float(r["dc_phases"]),
                     minor_share_at_max=float(r["minor_share"]), gain_2phase=float(r["misfit_1phase"] / r["misfit_2phase"]), max_sigma_c=float(t["sigma_c"].max()), retention_300=round(ret[lab], 1)))
Sm = pd.DataFrame(summ)
# '2상 공존' 판정: 두 성분이 모두 15 % 이상, 0.1 Å 이상 떨어져 있고, 2상 모델이 불일치를 절반 이하로 줄여야 함
Sm["two_phase"] = (Sm["minor_share_at_max"] >= 0.15) & (Sm["dc_at_max"] >= 0.1) & (Sm["gain_2phase"] >= 2)
# 논문 프레임은 봉우리가 넓게 겹쳐 2상 분해(분율, Δc)가 제약 조건에 따라 달라짐 → 판정·Δc를 쓰지 않음
Sm["two_phase"] = Sm["two_phase"].astype(object)
pap = Sm["sample"].isin(["s-LNO", "d-LNO"])
Sm.loc[pap, "two_phase"] = "n/a"; Sm.loc[pap, ["dc_at_max", "minor_share_at_max"]] = np.nan
Sm = Sm.drop(columns=["endpoint_misfit", "ratio_to_endpoints"]); Sm.to_csv(f"{OUT}/ml3_structure_vs_retention.csv", index=False)
res["summary"] = Sm.to_dict(orient="records")
dd = T[T["sample"] == "d-LNO dense"]
res["d-LNO dense"] = {"max_misfit_1phase": float(dd["misfit_1phase"].max()), "at_V": float(dd.loc[dd["misfit_1phase"].idxmax(), "V"]),
                      "max_sigma_c": float(dd["sigma_c"].max()), "sigma_c_peak_V": float(dd.loc[dd["sigma_c"].idxmax(), "V"]),
                      "f_H3_50pct_V": float(np.interp(0.5, dd["f_H3"].cummax(), dd["V"]))}

# ---------------------------------------------------------------- figures
tcu = lambda t2: 2 * np.degrees(np.arcsin(LC * np.sin(np.radians(t2 / 2)) / LP))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for lab in ("s-LNO", "d-LNO"):
    s_ = shape[shape["sample"] == lab]
    ax[0].plot(s_.V, s_.dBIC_sym, "o--", color=COL[lab], alpha=.5, label=f"{lab} symmetric pV")
    ax[0].plot(s_.V, s_.dBIC_split, "s-", color=COL[lab], label=f"{lab} split (asymmetric) pV")
ax[0].axhline(10, color="gray", ls=":"); ax[0].set_yscale("log"); ax[0].set_xlabel("Voltage (V)"); ax[0].set_ylabel("ΔBIC (1 − 2 peaks)")
ax[0].set_title("Asymmetric shape does not remove the 2-peak bias"); ax[0].legend(fontsize=8)
fm = (S[1] > FITWIN[0]) & (S[1] < FITWIN[1]); x, Y = S[1][fm], S[2][fm]; y = Y[:, -1]
for sym, col, nm in ((True, "#94a3b8", "symmetric pV"), (False, "#0f172a", "split pV")):
    r = fit_peaks(x, y, 1, sym); ax[1].plot(tcu(x), (y - r[2](x, *r[1])) / y.max() * 100, color=col, lw=2, label=nm)
ax[1].axhline(0, color="gray", lw=.8); ax[1].set_xlabel("2θ (Cu Kα eq.)"); ax[1].set_ylabel("residual (% of peak)")
ax[1].set_title("s-LNO 4.79 V (single H3 phase): residual"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ml3_shape.png", dpi=180)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for a, lab, v in ((ax[0], "s-LNO", 4.303), (ax[1], "d-LNO", 4.4)):
    x, y, f1, f2, p, (T2, T3), xw, fm = FITS[lab][v]; sc = y.max()
    mis = np.sum((y - f1) ** 2) / np.sum((y - y.mean()) ** 2)
    a.plot(tcu(x), y / sc, "k.", ms=5, label="data")
    a.plot(tcu(x), f1 / sc, color="#dc2626", lw=2, label=f"best single phase (shift + broadening)\n1 − R² = {mis:.4f}")
    a.plot(tcu(x), (y - f1) / sc * 5 - 0.25, color="#475569", lw=1.3, label="residual × 5")
    a.axhline(-0.25, color="#cbd5e1", lw=.8)
    a.set_ylim(-0.6, 1.1); a.set_title(f"{lab} {v:.2f} V"); a.set_xlabel("2θ (Cu Kα eq.)"); a.set_ylabel("normalized intensity"); a.legend(fontsize=8, loc="upper left")
a = ax[2]
for lab, st in (("s-LNO", "o-"), ("d-LNO", "s-"), ("d-LNO dense", "-")):
    t = T[T["sample"] == lab]; a.plot(t.V, t.misfit_1phase, st, color=COL[lab], lw=1.5 if lab != "d-LNO dense" else 1, ms=5, label=lab)
a.set_yscale("log"); a.set_ylim(1e-6, 3e-2); a.set_xlabel("Voltage (V)"); a.set_ylabel("single-phase misfit (1 − R²)"); a.set_title("Can one (shifted, broadened) phase explain the scan?")
a.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ml3_phase.png", dpi=180)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
a = ax[0]
for lab, st in (("s-LNO", "o-"), ("d-LNO", "s-"), ("d-LNO dense", "-")):
    t = T[T["sample"] == lab]; a.plot(t.V, t.sigma_c, st, color=COL[lab], ms=5, label=lab)
a.set_xlabel("Voltage (V)"); a.set_ylabel("extra c spread σ_c (Å)"); a.set_title("Peak broadening during charge (vs 4.0 V scan)"); a.legend(fontsize=9)
a = ax[1]
for grp, mk, ls in ((("s-LNO", "d-LNO"), "o", "-"), (("project A", "project B", "project C"), "D", "--")):
    g = Sm[Sm["sample"].isin(grp)].sort_values("max_misfit_1phase")
    a.plot(g["max_misfit_1phase"], g["retention_300"], ls, color="#94a3b8", lw=1, zorder=1)
    for _, r in g.iterrows():
        a.scatter(r["max_misfit_1phase"], r["retention_300"], s=110, marker=mk, color=COL[r["sample"]], edgecolor="k", zorder=3)
        if r["sample"] == "s-LNO": tag = "single phase fails at 4.30 V"
        elif r["sample"] == "d-LNO": tag = "one broadened phase"
        else: tag = f"two phases, Δc = {r['dc_at_max']:.2f} Å" if r["two_phase"] else "single phase"
        off = {"project C": (-10, -34), "d-LNO": (10, 10)}.get(r["sample"], (8, -6))
        a.annotate(f"{r['sample']}\n{tag}", (r["max_misfit_1phase"], r["retention_300"]), textcoords="offset points", xytext=off, fontsize=8.5)
a.set_xscale("log"); a.set_xlim(5e-4, 1.5); a.set_ylim(50, 100); a.set_xlabel("max single-phase misfit during charge (1 − R²)"); a.set_ylabel("300-cycle retention (%)")
a.set_title("Two-phase character vs cycle life (○ paper, ◇ project)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ml3_link.png", dpi=180)

json.dump(res, open(f"{OUT}/ml3_v2.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(shape.to_string(index=False)); print(T.drop(columns=[]).to_string(index=False)); print(Sm.to_string(index=False)); print(json.dumps(res["d-LNO dense"], indent=1))
