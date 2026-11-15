"""1) NMF 상 분해 + 끝점 선형 혼합 검정, 2) 1봉우리 vs 2봉우리 BIC 판정."""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.optimize import curve_fit, nnls

SRC, PROJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LP, LC = 0.49594, 1.5406
dof = lambda tt, l: l / (2 * np.sin(np.radians(tt / 2)))
tcu = lambda tt: 2 * np.degrees(np.arcsin(LC / (2 * dof(tt, LP))))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
X = pd.read_excel(SRC, sheet_name=None, header=None)
res = {}


def frames(sheet, n):
    d = X[sheet]; V = d.iloc[1, 1:n + 1].astype(float).to_numpy(); x = d.iloc[2:, 0].astype(float).to_numpy()
    Y = d.iloc[2:, 1:n + 1].astype(float).to_numpy(); m = (x > 5.7) & (x < 6.6)
    o = np.argsort(V); return V[o], x[m], Y[m][:, o]


S = frames("Figure 3c-(003) reflection", 13)   # s-LNO charge
D = frames("Figure 3b-(003) reflection", 16)   # d-LNO charge
hd = X["Figure 3a for d-LNO"]; Vh = pd.to_numeric(hd.iloc[0, 1:], errors="coerce").to_numpy()
xh = pd.to_numeric(hd.iloc[1:, 0], errors="coerce").to_numpy(); Yh = hd.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
mh = (xh > 5.7) & (xh < 6.6); Hd = (Vh, xh[mh], Yh[mh])


def prep(Y):
    Y = np.clip(Y, 0, None)
    return Y / Y.sum(axis=0, keepdims=True)


# ---------- 1a. 끝점 선형 혼합 검정 (two-phase면 중간 패턴 = 두 끝 패턴의 비음수 합)
def mix_test(V, x, Y, v0, v1, vm):
    Yn = prep(Y)
    i0, i1, im = (int(np.argmin(abs(V - v))) for v in (v0, v1, vm))
    A = np.c_[Yn[:, i0], Yn[:, i1]]
    w, r = nnls(A, Yn[:, im])
    fit = A @ w
    r2 = 1 - np.sum((Yn[:, im] - fit) ** 2) / np.sum((Yn[:, im] - Yn[:, im].mean()) ** 2)
    return dict(V0=round(V[i0], 3), V1=round(V[i1], 3), Vm=round(V[im], 3), w=np.round(w, 3).tolist(), R2=round(r2, 4)), (x, Yn[:, im], fit, A)


mt_s, plot_s = mix_test(*S, 4.21, 4.42, 4.30)
mt_d, plot_d = mix_test(*D, 4.21, 4.40, 4.30)
res["mix_test"] = {"s-LNO": mt_s, "d-LNO": mt_d}

# ---------- 1b. NMF on d-LNO dense heatmap (74 scans) and s-LNO frames: components needed
def nmf_curve(Y, kmax=5):
    Yn = prep(Y); out = []
    for k in range(1, kmax + 1):
        m = NMF(n_components=k, init="nndsvda", max_iter=4000, random_state=0).fit(Yn.T)
        rec = m.inverse_transform(m.transform(Yn.T)).T
        out.append(1 - np.sum((Yn - rec) ** 2) / np.sum((Yn - Yn.mean(axis=1, keepdims=True)) ** 2))
    return out


# 전이 구간만 (4.15–4.50 V)
def window(V, x, Y, lo=4.15, hi=4.50):
    m = (V >= lo) & (V <= hi); return V[m], x, Y[:, m]


Vw, xw, Yw = window(*Hd)
res["nmf_explained_dLNO_dense_4.15-4.50"] = [round(v, 4) for v in nmf_curve(Yw)]
Vs, xs, Ys = window(*S, 4.15, 4.50)
res["nmf_frames_in_window"] = {"s-LNO": int(Ys.shape[1]), "d-LNO dense": int(Yw.shape[1])}

# d-LNO dense: k=2 weights vs voltage (does weight switch abruptly or glide?)
Yn = prep(Hd[2]); m2 = NMF(n_components=3, init="nndsvda", max_iter=5000, random_state=0)
Wd = m2.fit_transform(Yn.T); Hc = m2.components_
cent = [float((tcu(Hd[1]) * h).sum() / h.sum()) for h in Hc]
order = np.argsort(cent); Wd = Wd[:, order]; Hc = Hc[order]; cent = np.array(cent)[order]
Wd = Wd / Wd.sum(axis=1, keepdims=True)

# ---------- 2. BIC 1-peak vs 2-peak
def pv(x, A, x0, w, eta):
    return A * (eta / (1 + 4 * ((x - x0) / w) ** 2) + (1 - eta) * np.exp(-4 * np.log(2) * ((x - x0) / w) ** 2))


def fit_bic(x, y):
    n = len(x); x0 = x[np.argmax(y)]; base = np.median(np.r_[y[:5], y[-5:]])
    f1 = lambda q, A, c, w, e, b0, b1: pv(q, A, c, w, e) + b0 + b1 * (q - x.mean())
    p1, c1 = curve_fit(f1, x, y, p0=[y.max(), x0, (x[-1] - x[0]) / 10, .3, base, 0],
                       bounds=([0, x[0], 1e-3, 0, -np.inf, -np.inf], [np.inf, x[-1], x[-1] - x[0], 1, np.inf, np.inf]), maxfev=40000)
    r1 = np.sum((y - f1(x, *p1)) ** 2)
    f2 = lambda q, A, c, w, e, A2, c2, w2, e2, b0, b1: pv(q, A, c, w, e) + pv(q, A2, c2, w2, e2) + b0 + b1 * (q - x.mean())
    best = None
    # 두 번째 봉우리 초기값: 일정 간격 격자 + 실제 국소 최대점 (간격이 넓은 분리 봉우리도 잡도록)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(y, prominence=0.05 * (y.max() - base))
    starts = [x0 + dx for dx in np.linspace(-(x[-1] - x[0]) / 3, (x[-1] - x[0]) / 3, 9) if abs(dx) > 1e-9]
    starts += [x[i] for i in pk if abs(x[i] - x0) > 1e-9]
    for s2 in starts:
        try:
            p2, c2 = curve_fit(f2, x, y, p0=[y.max(), x0, p1[2] * .8, .3, y.max() * .3, s2, p1[2] * .8, .3, base, 0],
                               bounds=([0, x[0], 1e-3, 0, 0, x[0], 1e-3, 0, -np.inf, -np.inf],
                                       [np.inf, x[-1], x[-1] - x[0], 1, np.inf, x[-1], x[-1] - x[0], 1, np.inf, np.inf]), maxfev=40000)
            r2 = np.sum((y - f2(x, *p2)) ** 2)
            if best is None or r2 < best[0]: best = (r2, p2, c2)
        except Exception:
            pass
    r2, p2, c2 = best
    bic1 = n * np.log(r1 / n) + 6 * np.log(n); bic2 = n * np.log(r2 / n) + 10 * np.log(n)
    # minor-component share and separation in FWHM units
    a1, a2 = p2[0] * p2[2], p2[4] * p2[6]
    share = min(a1, a2) / (a1 + a2); sep = abs(p2[5] - p2[1]) / ((p2[2] + p2[6]) / 2)
    cerr = np.sqrt(np.diag(c1))[1]
    return dict(dBIC=round(bic1 - bic2, 1), minor_share=round(share, 3), sep_FWHM=round(sep, 2), center=p1[1], center_err=cerr,
                comps=sorted([p2[1], p2[5]]))


rows = []
for lab, (V, x, Y) in (("s-LNO", S), ("d-LNO", D)):
    for j, v in enumerate(V):
        if v < 3.95: continue
        r = fit_bic(tcu(x), Y[:, j])
        c = 3 * LC / (2 * np.sin(np.radians(r["center"] / 2)))
        dc = 3 * LC * np.cos(np.radians(r["center"] / 2)) / (2 * np.sin(np.radians(r["center"] / 2)) ** 2) * np.radians(r["center_err"] / 2)
        rows.append(dict(sample=lab, V=round(v, 3), dBIC=r["dBIC"], minor_share=r["minor_share"], sep_FWHM=r["sep_FWHM"],
                         c_1peak=round(c, 3), c_err=round(dc, 4)))
# project data
xr = pd.read_excel(PROJ, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601); tt = xr.iloc[:, 0].to_numpy(float)
for s in "ABC":
    for v in ("4.08", "4.30", "4.50", "4.80"):
        y = xr[f"{s}_{v}V"].to_numpy(float); i = np.argmax(y); m = (tt > tt[i] - 0.9) & (tt < tt[i] + 0.9)
        if s in "AB" and v == "4.30": m = (tt > 18.2) & (tt < 19.8)
        r = fit_bic(tt[m], y[m]); rows.append(dict(sample=f"project {s}", V=float(v), dBIC=r["dBIC"], minor_share=r["minor_share"], sep_FWHM=r["sep_FWHM"], c_1peak=None, c_err=None))
bic = pd.DataFrame(rows)
bic.to_csv(f"{OUT}/bic_table.csv", index=False)
res["bic"] = rows

# ---------- figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for a, (x, ym, fit, A), lab, mt in ((ax[0], plot_s, "s-LNO", mt_s), (ax[1], plot_d, "d-LNO", mt_d)):
    a.plot(tcu(x), A[:, 0], color="#94a3b8", lw=1, label=f"{mt['V0']} V")
    a.plot(tcu(x), A[:, 1], color="#cbd5e1", lw=1, ls="--", label=f"{mt['V1']} V")
    a.plot(tcu(x), ym, "k.", ms=4, label=f"{mt['Vm']} V (data)")
    a.plot(tcu(x), fit, color="#dc2626", lw=2, label=f"mix of ends, R² = {mt['R2']:.3f}")
    a.set_title(f"{lab}: is 4.30 V a mix of 4.21 & 4.4 V?"); a.set_xlabel("2θ (Cu Kα eq.)"); a.legend(fontsize=9)
a = ax[2]
for k, col in zip(range(3), ("#1d4ed8", "#16a34a", "#dc2626")):
    a.plot(Vh, Wd[:, k], color=col, lw=2, label=f"component {k+1} (2θ≈{cent[k]:.2f}°)")
a.axvspan(4.15, 4.50, color="#fde68a", alpha=.4, zorder=0)
a.set_xlabel("Voltage (V)"); a.set_ylabel("NMF weight"); a.set_title("d-LNO dense operando: NMF (k = 3)"); a.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_nmf_mix.png", dpi=180)

fig, a = plt.subplots(figsize=(8, 4.4))
for lab, col in (("s-LNO", "#1d4ed8"), ("d-LNO", "#ea580c")):
    b = bic[bic["sample"] == lab]; a.plot(b["V"], b["dBIC"].clip(lower=-50), "o-", color=col, label=lab)
a.axhline(10, color="gray", ls="--", lw=1); a.text(3.97, 14, "ΔBIC = 10 (strong for 2 peaks)", fontsize=9, color="gray")
a.set_yscale("symlog"); a.set_xlabel("Voltage (V)"); a.set_ylabel("ΔBIC (1-peak − 2-peak)"); a.legend(); a.set_title("Peak-count model selection, paper frames")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bic.png", dpi=180)
json.dump(res, open(f"{OUT}/a12.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "bic"}, indent=1, default=float))
pd.set_option("display.width", 200); print(bic.to_string(index=False))
