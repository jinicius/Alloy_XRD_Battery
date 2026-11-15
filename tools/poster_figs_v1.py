"""Poster figures (English labels, large fonts)."""
import math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SRC, PROJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LP, LC = 0.49594, 1.5406
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.4, "xtick.major.width": 1.3, "ytick.major.width": 1.3, "legend.frameon": False})
X = pd.read_excel(SRC, sheet_name=None, header=None)
num = lambda d: d.apply(pd.to_numeric, errors="coerce")
dof = lambda tt, l: l / (2 * np.sin(np.radians(tt / 2)))
tcu = lambda tt: 2 * np.degrees(np.arcsin(LC / (2 * dof(tt, LP))))
VC = {"4.08": "#2563eb", "4.30": "#16a34a", "4.50": "#f59e0b", "4.80": "#dc2626"}
A_C, C_C, B_C, GRAY = "#1d4ed8", "#ea580c", "#7c3aed", "#64748b"
xr = pd.read_excel(PROJ, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601)
tt = xr.iloc[:, 0].to_numpy(float)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=200, transparent=False, facecolor="white")
    plt.close(fig)


# 1) A/B/C raw XRD (stacked panels, shared x)
fig, axs = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True)
for a, s in zip(axs, "ABC"):
    for v, col in VC.items():
        a.plot(tt, xr[f"{s}_{v}V"], color=col, lw=2, label=f"{v} V")
    a.text(17.6, 820, f"Sample {s}", fontsize=22, fontweight="bold")
    a.set_ylim(0, 1000)
    a.set_yticks([0, 500, 1000])
    a.set_ylabel("counts")
axs[0].legend(ncol=4, loc="upper right", fontsize=15, bbox_to_anchor=(1.0, 1.18))
axs[-1].set_xlabel("2θ (deg, Cu Kα)")
axs[-1].set_xlim(17.5, 20.5)
save(fig, "p_raw.png")

# 2) contraction vs retention
fig, a = plt.subplots(figsize=(8, 5.6))
pts = [("A", 6.94, 55.5, A_C), ("B", 5.45, 82.0, B_C), ("C", 4.40, 95.0, C_C)]
for s, c, r, col in pts:
    a.scatter(c, r, s=900, color=col, zorder=3)
    a.text(c, r, s, ha="center", va="center", color="white", fontsize=22, fontweight="bold", zorder=4)
a.plot([p[1] for p in pts], [p[2] for p in pts], color=GRAY, lw=1.5, ls="--", zorder=1)
a.set_xlabel("c-axis contraction 4.08→4.80 V (%)")
a.set_ylabel("300-cycle retention (%)")
a.set_xlim(3.8, 7.6)
a.set_ylim(45, 105)
save(fig, "p_tradeoff_abc.png")

# 3) 4.30 V shape: paper s-LNO fit vs project A
d = X["Figure 3c-(003) reflection"]
V = d.iloc[1, 1:14].astype(float).to_numpy()
xx = d.iloc[2:, 0].astype(float).to_numpy()
j = int(np.argmin(abs(V - 4.30)))
y = d.iloc[2:, 1 + j].astype(float).to_numpy()
m = (xx > 5.55) & (xx < 6.75)
x, y = xx[m], y[m]


def pv(x, A, x0, w, eta):
    return A * (eta / (1 + 4 * ((x - x0) / w) ** 2) + (1 - eta) * np.exp(-4 * np.log(2) * ((x - x0) / w) ** 2))


f = lambda q, *p: pv(q, *p[0:4]) + pv(q, *p[4:8]) + p[8]
p, _ = curve_fit(f, x, y, p0=[y.max() * .5, 5.95, .14, .3, y.max(), 6.12, .14, .3, 0],
                 bounds=([0, 5.85, .05, 0, 0, 6.0, .05, 0, -np.inf], [np.inf, 6.06, .4, 1, np.inf, 6.3, .4, 1, np.inf]), maxfev=50000)
fig, a = plt.subplots(figsize=(9, 5.8))
q = np.linspace(x.min(), x.max(), 600)
sc = f(q, *p).max()
a.fill_between(tcu(q), 0, pv(q, *p[0:4]) / sc, color=A_C, alpha=.30, label="H2-like  c = 14.33 Å")
a.fill_between(tcu(q), 0, pv(q, *p[4:8]) / sc, color="#dc2626", alpha=.25, label="H3-like  c = 13.94 Å")
a.plot(tcu(x), y / sc, "o", ms=5, color="#0f172a", label="Paper s-LNO, 4.30 V")
ya = xr["A_4.30V"].to_numpy(float)
ya = (ya - np.median(ya[:60])) / (ya.max() - np.median(ya[:60]))
a.plot(tt, ya, color="#16a34a", lw=2.2, ls="--", label="Project A, 4.30 V")
a.set_xlim(17.9, 20.1)
a.set_ylim(-0.03, 1.12)
a.set_xlabel("2θ (deg, Cu Kα equivalent)")
a.set_ylabel("normalized intensity")
a.legend(fontsize=14, loc="upper right")
save(fig, "p_fit430.png")

# 4) c–V: paper vs project
d3 = num(X["Figure 3d"].iloc[1:])
p02 = pd.read_excel(PROJ, sheet_name="02_물성·성능", header=None)
r0 = p02.index[p02.iloc[:, 0].astype(str).str.startswith("Voltage")][0]
cv = p02.iloc[r0 + 1:r0 + 30, :4].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
fig, a = plt.subplots(figsize=(9, 6))
for cols, lab, col in (((4, 5), "Paper s-LNO", A_C), ((8, 9), "Paper d-LNO", C_C)):
    arr = d3.iloc[:, list(cols)].dropna().to_numpy()
    a.plot(arr[:, 0], arr[:, 1], "o", ms=6, color=col, label=lab)
a.plot(cv[:, 0], cv[:, 1], color=A_C, lw=1.6, alpha=.6, label="Project curve A")
a.plot(cv[:, 0], cv[:, 3], color=C_C, lw=1.6, alpha=.6, label="Project curve C")
for s, col, P_ in (("A", "#1e3a8a", [(4.08, 14.411), (4.30, 14.304), (4.30, 13.719), (4.50, 13.622), (4.80, 13.411)]),
                   ("C", "#9a3412", [(4.08, 14.216), (4.30, 14.119), (4.50, 13.857), (4.80, 13.591)])):
    P_ = np.array(P_)
    a.plot(P_[:, 0], P_[:, 1], "*", ms=20, color=col, mec="white", label=f"Our reading {s}")
a.axvspan(4.47, 4.53, color="#fde68a", alpha=.6, zorder=0)
a.set_xlim(3.6, 4.85)
a.set_xlabel("Voltage (V)")
a.set_ylabel("c (Å)")
a.legend(fontsize=13, ncol=2, loc="lower left")
save(fig, "p_cv.png")

# 5) FWHM
fig, a = plt.subplots(figsize=(9, 5.6))
lab = ["A\n4.08", "A\n4.30", "A\n4.80", "C\n4.08", "C\n4.30", "C\n4.80"]
paper = [0.46, 0.86, 0.46, 0.68, 0.79, 0.95]
proj = [0.090, 0.075, 0.091, 0.075, 0.075, 0.076]
k = np.arange(6)
a.bar(k - .2, paper, .4, color=GRAY, label="Paper (Cu Kα eq.)")
a.bar(k + .2, proj, .4, color="#0e8c7f", label="Project data")
for i, v in enumerate(paper):
    a.text(i - .2, v + .015, f"{v:.2f}", ha="center", fontsize=13)
a.set_xticks(k, lab)
a.set_ylabel("FWHM (deg)")
a.set_ylim(0, 1.08)
a.legend(fontsize=14, loc="upper left")
save(fig, "p_fwhm.png")

# 6) -dc/dV
fig, a = plt.subplots(figsize=(9, 5.6))
for cols, labl, col in (((4, 5), "s-LNO (A)", A_C), ((8, 9), "d-LNO (C)", C_C)):
    arr = d3.iloc[:, list(cols)].dropna().to_numpy()
    arr = arr[np.argsort(arr[:, 0])]
    arr = arr[arr[:, 0] > 3.9]
    dv = np.diff(arr[:, 0]); ok = dv > 0.005
    sl = -np.diff(arr[:, 1])[ok] / dv[ok]
    mid = ((arr[1:, 0] + arr[:-1, 0]) / 2)[ok]
    a.plot(mid, sl, "o-", ms=5, lw=2.2, color=col, label=f"{labl}  max {sl.max():.1f} Å/V")
a.axhline(0, color=GRAY, lw=1)
a.set_xlim(3.9, 4.85)
a.set_xlabel("Voltage (V)")
a.set_ylabel("−dc/dV (Å/V)")
a.legend(fontsize=15)
save(fig, "p_dcdv.png")

# 7) volume change
e = num(X["Figure 3e"].iloc[1:])
vals = {}
for name, ac, cc in (("s-LNO (A)", (4, 5), (4, 5)), ("d-LNO (C)", (0, 1), (8, 9))):
    aa = e.iloc[:, list(ac)].dropna().to_numpy(); c_ = d3.iloc[:, list(cc)].dropna().to_numpy()
    at = lambda arr, v: arr[np.argmin(abs(arr[:, 0] - v)), 1]
    vol = lambda v: math.sqrt(3) / 2 * at(aa, v) ** 2 * at(c_, v)
    vals[name] = [(at(c_, 4.8) - at(c_, 4.08)) / at(c_, 4.08) * 100, (at(aa, 4.8) - at(aa, 4.08)) / at(aa, 4.08) * 100,
                  (vol(4.8) - vol(4.08)) / vol(4.08) * 100]
fig, a = plt.subplots(figsize=(9, 5.6))
k = np.arange(3)
for i, (name, col) in enumerate((("s-LNO (A)", A_C), ("d-LNO (C)", C_C))):
    v = vals[name]
    a.bar(k + (i - .5) * .36, v, .36, color=col, label=name)
    for bx, bv in zip(k + (i - .5) * .36, v):
        a.text(bx, bv - .3, f"{bv:.1f}%", ha="center", va="top", fontsize=14)
a.axhline(0, color=GRAY, lw=1)
a.set_xticks(k, ["c-axis", "a-axis", "cell volume"])
a.set_ylabel("change 4.08→4.80 V (%)")
a.set_ylim(-9.2, 0.8)
a.legend(fontsize=15, loc="lower center")
save(fig, "p_volume.png")
print("done")
