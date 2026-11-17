"""ML5 (응용): 소성 중 in-situ 싱크로트론 XRD를 최대한 활용한 준안정 경로 분석.

A. 저자 Rietveld 결과(Fig. 4e, Table S1)로 하는 속도론 분석 4가지
   A1 JMAK(Avrami) 상변태 속도론: 600·650 °C 층상 분율 f(t)
   A2 Arrhenius 겉보기 활성화 에너지 (두 온도의 속도 상수)
   A3 결정 성장 법칙 D^m − D0^m = K·t
   A4 ex-situ(Table S1, 600 °C 3~48 h) ↔ in-situ(Fig. 4e) 비교
B. 원본 XRD 패턴에서 직접 추정해 저자 Rietveld 값과 비교 3가지
   B1 (003) 면적 / (104)+(200) 면적 → 층상 분율의 시간 변화 (JMAK 상수 비교)
   B2 (003) FWHM → Scherrer 결정 크기
   B3 (003) 위치 → c 격자상수 vs 3 × (Li층 + Ni층 두께) (Fig. 4c, 온도 상승 중)

판독은 과제와 같은 기준: 원본 그대로, 배경 = 창 양 끝 중앙값, 2θ = 꼭대기 3점 포물선, FWHM = 절반 높이 선형 보간.

사용법: python ml5_sintering_xrd.py <source_data.xlsx> <out_dir>
"""
import sys, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

SRC, OUT = sys.argv[1], sys.argv[2]
L_ISO = 0.49594      # Fig. 4d의 2θ 축 (0.49594 Å 기준으로 환산돼 있음: 2열 d, 3열 Q로 확인)
L_RAMP = 0.1666      # Fig. 4a 온도 상승 측정 (NSLS-II 28-ID-1)
R = 8.314
rng = np.random.default_rng(0)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COL = {600: "#1d4ed8", 650: "#dc2626"}
X = pd.read_excel(SRC, sheet_name=None, header=None)
res = {}

# ------------------------------------------------------------------ data
e = X["Figure 4e"].iloc[2:]
KIN = {}
for T, c0 in ((650, 0), (600, 4)):
    a = e.iloc[:, c0:c0 + 3].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
    KIN[T] = dict(t=a[:, 0], D=a[:, 1], f=a[:, 2])          # t: 분, D: 결정 크기(nm), f: 층상 분율
S1 = pd.DataFrame({"t_h": [3, 6, 12, 18, 36, 48], "RS_wt": [20.1, 13.0, 7.3, 5.1, 2.3, 2.2], "DS_nm": [20.8, 26.6, 38.5, 43.5, 56.9, 66.5]})
S1["f_L"] = 1 - S1.RS_wt / 100


def heat(sheet):
    d = X[sheet]; tt = pd.to_numeric(d.iloc[2:, 0], errors="coerce").to_numpy()
    hdr = pd.to_numeric(d.iloc[1, 3:], errors="coerce").to_numpy(); Y = d.iloc[2:, 3:].apply(pd.to_numeric, errors="coerce").to_numpy()
    ok = ~np.isnan(hdr); return tt, hdr[ok], Y[:, ok]


ISO = {600: heat("Figure 4d-Part I-600C"), 650: heat("Figure 4d-Part II-650C")}
RAMP = heat("Figure 4a")


# ------------------------------------------------------------------ A1 JMAK
def jmak(t, f0, fi, k, n): return f0 + (fi - f0) * (1 - np.exp(-(k * t) ** n))


def fit_jmak(t, f):
    p, _ = curve_fit(jmak, t, f, p0=[f[0], min(f[-1] + .05, 1.1), 1 / max(t[-1] / 3, 1), 1],
                     bounds=([-1, f[0] - .05, 1e-7, .2], [2, 2, 1, 4]), maxfev=40000)
    fit = jmak(t, *p); return p, fit, 1 - np.sum((f - fit) ** 2) / np.sum((f - f.mean()) ** 2)


def boot(fun, t, y, fit, nb=300):
    r = y - fit; out = []
    for _ in range(nb):
        try: out.append(fun(t, fit + rng.choice(r, len(r)))[0])
        except Exception: pass
    return np.array(out)


A = {}
for T in (600, 650):
    t, f = KIN[T]["t"], KIN[T]["f"]
    p, fit, r2 = fit_jmak(t, f); B = boot(fit_jmak, t, f, fit)
    ci = lambda i: np.percentile(B[:, i], [2.5, 97.5]).round(4).tolist()
    A[T] = dict(f0=p[0], f_inf=p[1], k_per_min=p[2], n=p[3], R2=r2, k_CI=ci(2), n_CI=ci(3), f_inf_CI=ci(1),
                t_half_min=float(np.log(2) ** (1 / p[3]) / p[2]), kboot=B[:, 2])
res["A1_JMAK"] = {T: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in A[T].items() if k != "kboot"} for T in A}

# ------------------------------------------------------------------ A2 Arrhenius (두 온도)
T1, T2 = 873.15, 923.15
Ea = R * np.log(A[650]["k_per_min"] / A[600]["k_per_min"]) / (1 / T1 - 1 / T2) / 1000
m_ = min(len(A[600]["kboot"]), len(A[650]["kboot"]))
Eab = R * np.log(A[650]["kboot"][:m_] / A[600]["kboot"][:m_]) / (1 / T1 - 1 / T2) / 1000
res["A2_Ea_kJ_mol"] = dict(Ea=round(Ea, 1), CI95=np.percentile(Eab, [2.5, 97.5]).round(1).tolist(), note="온도 2점 → 거친 추정")


# ------------------------------------------------------------------ A3 결정 성장
def grow(t, D0, K, m): return (D0 ** m + K * t) ** (1 / m)


def fit_grow(t, D):
    p, _ = curve_fit(grow, t, D, p0=[D[0], 1, 2], bounds=([0, 0, .5], [100, 1e8, 6]), maxfev=40000)
    fit = grow(t, *p); return p, fit, 1 - np.sum((D - fit) ** 2) / np.sum((D - D.mean()) ** 2)


G = {}
for T in (600, 650):
    t, D = KIN[T]["t"], KIN[T]["D"]; p, fit, r2 = fit_grow(t, D); Bg = boot(fit_grow, t, D, fit, 200)
    G[T] = dict(D0=p[0], K=p[1], m=p[2], R2=r2, m_CI=np.percentile(Bg[:, 2], [2.5, 97.5]).round(2).tolist())
res["A3_growth"] = {T: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in G[T].items()} for T in G}

# ------------------------------------------------------------------ A4 ex-situ vs in-situ (600 °C)
p6 = [A[600][k] for k in ("f0", "f_inf", "k_per_min", "n")]
S1["f_L_insitu_JMAK"] = jmak(S1.t_h * 60, *p6)
S1["D_insitu_growth"] = grow(S1.t_h * 60, G[600]["D0"], G[600]["K"], G[600]["m"])
res["A4_exsitu_vs_insitu"] = S1.round(3).to_dict(orient="records")
res["A4_note"] = "in-situ 측정은 12 h까지. 12 h 초과는 외삽"


# ------------------------------------------------------------------ B: 원본 패턴 판독
def read_peak(x, y, lo, hi, edge=0.05):
    m = (x >= lo) & (x <= hi); x, y = x[m], y[m]
    bg = np.median(np.r_[y[x < lo + edge], y[x > hi - edge]]); yb = y - bg
    area = np.trapezoid(yb, x); i = int(np.argmax(yb)); h = yb[i]
    if 0 < i < len(x) - 1:
        a, b, c = np.polyfit(x[i - 1:i + 2], yb[i - 1:i + 2], 2); xc = -b / (2 * a); h = c - b * b / (4 * a)
    else: xc = x[i]
    half = h / 2; L = i
    while L > 0 and yb[L] > half: L -= 1
    Rr = i
    while Rr < len(x) - 1 and yb[Rr] > half: Rr += 1
    xl = x[L] + (half - yb[L]) * (x[L + 1] - x[L]) / (yb[L + 1] - yb[L]) if L < i else x[L]
    xr = x[Rr - 1] + (half - yb[Rr - 1]) * (x[Rr] - x[Rr - 1]) / (yb[Rr] - yb[Rr - 1]) if Rr > i else x[Rr]
    return dict(area=area, pos=xc, fwhm=xr - xl, height=h)


WIN003, WIN104 = (5.70, 6.20), (13.45, 14.15)
Bres = {}
for T in (600, 650):
    tt, tm, Y = ISO[T]; rows = []
    for j in range(Y.shape[1]):
        p3 = read_peak(tt, Y[:, j], *WIN003); p4 = read_peak(tt, Y[:, j], *WIN104)
        th = np.radians(p3["pos"] / 2); beta = np.radians(p3["fwhm"])
        rows.append(dict(t=tm[j], ratio=p3["area"] / p4["area"], fwhm003=p3["fwhm"], pos003=p3["pos"],
                         D_scherrer_nm=0.9 * L_ISO / (beta * np.cos(th)) / 10, c_A=3 * L_ISO / (2 * np.sin(th))))
    Bres[T] = pd.DataFrame(rows)

# B1: 비율 곡선에 JMAK → 속도 상수·지수를 Rietveld 결과와 비교 (비율은 단위가 달라도 k, n은 비교 가능)
B1 = {}
for T in (600, 650):
    b = Bres[T]; f = np.interp(b.t, KIN[T]["t"], KIN[T]["f"])   # 같은 시각의 Rietveld 값
    p, fit, r2 = fit_jmak(b.t.to_numpy(), b.ratio.to_numpy()); Bb = boot(fit_jmak, b.t.to_numpy(), b.ratio.to_numpy(), fit, 200)
    lin = np.polyfit(b.ratio, f, 1); fl = np.polyval(lin, b.ratio)
    B1[T] = dict(k_per_min=p[2], k_CI=np.percentile(Bb[:, 2], [2.5, 97.5]).round(5).tolist(), n=p[3], n_CI=np.percentile(Bb[:, 3], [2.5, 97.5]).round(2).tolist(),
                 R2_jmak=r2, pearson_vs_rietveld=float(np.corrcoef(b.ratio, f)[0, 1]), R2_linear_calib=1 - np.sum((f - fl) ** 2) / np.sum((f - f.mean()) ** 2))
    Bres[T]["f_rietveld"] = f; Bres[T]["f_from_ratio"] = fl
res["B1_003_ratio"] = {T: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in B1[T].items()} for T in B1}
Ea_b = R * np.log(B1[650]["k_per_min"] / B1[600]["k_per_min"]) / (1 / T1 - 1 / T2) / 1000
res["B1_Ea_kJ_mol"] = round(Ea_b, 1)

# B2: Scherrer vs Rietveld 크기
B2 = {}
for T in (600, 650):
    b = Bres[T]; D = np.interp(b.t, KIN[T]["t"], KIN[T]["D"]); Bres[T]["D_rietveld"] = D
    B2[T] = dict(pearson=float(np.corrcoef(b.D_scherrer_nm, D)[0, 1]), ratio_scherrer_to_rietveld=[round(float((b.D_scherrer_nm / D).min()), 2), round(float((b.D_scherrer_nm / D).max()), 2)],
                 scherrer_start_end=[round(float(b.D_scherrer_nm.iloc[0]), 1), round(float(b.D_scherrer_nm.iloc[-1]), 1)],
                 rietveld_start_end=[round(float(D[0]), 1), round(float(D[-1]), 1)])
res["B2_scherrer"] = B2

# B3: 온도 상승 중 c (003) vs 3 × (Li층 + Ni층)
c4 = X["Figure 4c"].iloc[2:].apply(pd.to_numeric, errors="coerce").dropna(subset=[3, 4]).to_numpy()
tt, temps, Y = RAMP; rows = []
for Tm, li, ni in zip(c4[:, 0], c4[:, 3], c4[:, 4]):
    j = int(np.argmin(abs(temps - Tm)))
    if abs(temps[j] - Tm) > 3: continue
    p3 = read_peak(tt, Y[:, j], 1.90, 2.12, edge=0.02); c_x = 3 * L_RAMP / (2 * np.sin(np.radians(p3["pos"] / 2)))
    rows.append(dict(T=Tm, c_from_003=c_x, c_rietveld=3 * (li + ni), fwhm003=p3["fwhm"]))
B3 = pd.DataFrame(rows); d3 = B3.c_from_003 - B3.c_rietveld
res["B3_c_lattice"] = dict(n=len(B3), mean_diff_A=round(float(d3.mean()), 4), max_abs_diff_A=round(float(d3.abs().max()), 4),
                           pearson=round(float(np.corrcoef(B3.c_from_003, B3.c_rietveld)[0, 1]), 4),
                           c_range_from_003=[round(float(B3.c_from_003.min()), 3), round(float(B3.c_from_003.max()), 3)],
                           above_630C=dict(n=int((B3["T"] > 630).sum()), mean_diff_A=round(float(d3[B3["T"] > 630].mean()), 4), max_abs_diff_A=round(float(d3[B3["T"] > 630].abs().max()), 4)),
                           below_630C=dict(n=int((B3["T"] <= 630).sum()), mean_diff_A=round(float(d3[B3["T"] <= 630].mean()), 4), max_abs_diff_A=round(float(d3[B3["T"] <= 630].abs().max()), 4)))

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
a = ax[0]
for T in (600, 650):
    t, f = KIN[T]["t"], KIN[T]["f"]; a.plot(t / 60, f, "o", ms=3, color=COL[T], alpha=.5)
    tg = np.linspace(0, t.max(), 300); a.plot(tg / 60, jmak(tg, *[A[T][k] for k in ("f0", "f_inf", "k_per_min", "n")]), color=COL[T], lw=2,
                                              label=f"{T} °C: n = {A[T]['n']:.2f}, k = {A[T]['k_per_min'] * 60:.2f} /h")
a.set_xlabel("Isothermal time (h)"); a.set_ylabel("Layered fraction (Rietveld)"); a.set_title("A1  JMAK kinetics"); a.legend(fontsize=9)
a = ax[1]
inv = np.array([1 / T1, 1 / T2]) * 1000; lk = np.log([A[600]["k_per_min"], A[650]["k_per_min"]])
a.plot(inv, lk, "ko-", ms=8); a.set_xlabel("1000 / T (K⁻¹)"); a.set_ylabel("ln k (k in min⁻¹)")
a.set_title(f"A2  Arrhenius: Ea ≈ {Ea:.0f} kJ/mol\n(95 % {res['A2_Ea_kJ_mol']['CI95'][0]:.0f}–{res['A2_Ea_kJ_mol']['CI95'][1]:.0f}, 2 temperatures)")
a = ax[2]
for T in (600, 650):
    t, D = KIN[T]["t"], KIN[T]["D"]; a.plot(t / 60, D, "o", ms=3, color=COL[T], alpha=.5)
    a.plot(t / 60, grow(t, G[T]["D0"], G[T]["K"], G[T]["m"]), color=COL[T], lw=2, label=f"{T} °C: m = {G[T]['m']:.2f}")
a.set_xlabel("Isothermal time (h)"); a.set_ylabel("Crystal size (nm, Rietveld)"); a.set_title("A3  Growth law  Dᵐ − D₀ᵐ = K·t"); a.legend(fontsize=9)
a = ax[3]; tg = np.linspace(0, 48 * 60, 400)
a.plot(tg / 60, jmak(tg, *p6), color=COL[600], lw=2, label="in-situ JMAK (600 °C)"); a.axvspan(KIN[600]["t"].max() / 60, 48, color="#e2e8f0", alpha=.6, zorder=0)
a.plot(KIN[600]["t"] / 60, KIN[600]["f"], ".", color=COL[600], alpha=.3)
a.plot(S1.t_h, S1.f_L, "ks", ms=8, label="ex-situ Table S1 (1 − RS)")
a.text(30, 0.62, "extrapolated", color="#64748b", fontsize=9)
a.set_xlabel("Time at 600 °C (h)"); a.set_ylabel("Layered fraction"); a.set_title("A4  ex-situ vs in-situ"); a.set_ylim(.4, 1.02); a.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ml5_kinetics.png", dpi=170)

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
a = ax[0]
for T in (600, 650):
    b = Bres[T]; a.plot(b.t / 60, b.f_rietveld, "-", color=COL[T], lw=2, label=f"{T} °C Rietveld")
    a.plot(b.t / 60, b.f_from_ratio, "o", ms=3, color=COL[T], alpha=.6, label=f"{T} °C from (003)/(104) area")
a.set_xlabel("Isothermal time (h)"); a.set_ylabel("Layered fraction")
a.set_title(f"B1  (003) area ratio vs Rietveld\nk ratio: Rietveld {A[650]['k_per_min'] / A[600]['k_per_min']:.2f}×, XRD {B1[650]['k_per_min'] / B1[600]['k_per_min']:.2f}×"); a.legend(fontsize=8)
a = ax[1]
for T in (600, 650):
    b = Bres[T]; a.plot(b.D_rietveld, b.D_scherrer_nm, "o", ms=3, color=COL[T], label=f"{T} °C (r = {B2[T]['pearson']:.3f})")
lim = [0, max(KIN[600]["D"].max(), KIN[650]["D"].max()) * 1.1]; a.plot(lim, lim, "k--", lw=1, label="y = x")
a.set_xlabel("Crystal size, Rietveld (nm)"); a.set_ylabel("Scherrer size from (003) FWHM (nm)"); a.set_title("B2  Scherrer (003) vs Rietveld"); a.legend(fontsize=9)
a = ax[2]
a.plot(B3["T"], B3.c_rietveld, "s-", color="#334155", ms=5, label="3 × (Li slab + Ni slab), Rietveld")
a.plot(B3["T"], B3.c_from_003, "o", color="#ea580c", ms=5, label="3d(003) from peak position")
a.set_xlabel("Temperature (°C)"); a.set_ylabel("c (Å)"); a.axvspan(B3["T"].min() - 5, 630, color="#e2e8f0", alpha=.6, zorder=0); a.text(B3["T"].min() + 5, B3.c_rietveld.min() - 0.005, "weak, broad (003)", fontsize=9, color="#64748b")
a.set_title(f"B3  c lattice during ramp (> 630 °C: mean diff {res['B3_c_lattice']['above_630C']['mean_diff_A']:+.4f} Å)"); a.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ml5_xrd.png", dpi=170)

for T in (600, 650): Bres[T].to_csv(f"{OUT}/ml5_iso_{T}C_peak_reading.csv", index=False)
B3.to_csv(f"{OUT}/ml5_ramp_c_lattice.csv", index=False); S1.to_csv(f"{OUT}/ml5_exsitu_vs_insitu.csv", index=False)
json.dump(res, open(f"{OUT}/ml5.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float, ensure_ascii=False))
