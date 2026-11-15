"""PPT poster figures, each drawn at its placement size (inches)."""
import math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SRC, PROJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LP, LC = 0.49594, 1.5406
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 10.5,
                     "axes.linewidth": 1.0})
X = pd.read_excel(SRC, sheet_name=None, header=None)
num = lambda d: d.apply(pd.to_numeric, errors="coerce")
dof = lambda tt, l: l / (2 * np.sin(np.radians(tt / 2)))
tcu = lambda tt: 2 * np.degrees(np.arcsin(LC / (2 * dof(tt, LP))))
VC = {"4.08": "#2563eb", "4.30": "#16a34a", "4.50": "#f59e0b", "4.80": "#dc2626"}
A_C, B_C, C_C, GRAY = "#1d4ed8", "#7c3aed", "#ea580c", "#64748b"
SC = {"A": A_C, "B": B_C, "C": C_C}
xr = pd.read_excel(PROJ, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601)
tt = xr.iloc[:, 0].to_numpy(float)
p02 = pd.read_excel(PROJ, sheet_name="02_물성·성능", header=None)
r0 = p02.index[p02.iloc[:, 0].astype(str).str.startswith("Voltage")][0]
cv = p02.iloc[r0 + 1:r0 + 30, :4].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
r1 = p02.index[p02.iloc[:, 0].astype(str).str.startswith("Cycle")][0]
cyc = p02.iloc[r1 + 1:r1 + 40, :4].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
d3 = num(X["Figure 3d"].iloc[1:])


def fig(w, h):
    return plt.subplots(figsize=(w, h))


def save(f, name):
    f.tight_layout(pad=0.4)
    f.savefig(f"{OUT}/{name}.png", dpi=220, facecolor="white")
    plt.close(f)


# raw A/B/C
f, axs = plt.subplots(3, 1, figsize=(6.4, 6.9), sharex=True)
for a, s in zip(axs, "ABC"):
    for v, col in VC.items():
        a.plot(tt, xr[f"{s}_{v}V"], color=col, lw=1.4, label=f"{v} V")
    a.axvline(tt[np.argmax(xr[f"{s}_4.08V"])], color=GRAY, ls="--", lw=0.9)
    a.text(17.58, 830, f"Sample {s}", fontsize=14, fontweight="bold", color=SC[s])
    a.set_ylim(0, 1000); a.set_yticks([0, 500, 1000]); a.set_ylabel("counts")
axs[0].legend(ncol=4, loc="upper right", bbox_to_anchor=(1.0, 1.22))
axs[-1].set_xlabel("2θ (deg, Cu Kα)"); axs[-1].set_xlim(17.5, 20.5)
save(f, "f_raw")

# 4.30 V overlay A/B/C
f, a = fig(6.75, 4.1)
for s in "ABC":
    a.plot(tt, xr[f"{s}_4.30V"], color=SC[s], lw=1.6, label=f"Sample {s}")
for x0, lab in ((18.595, "A 18.595°"), (19.395, "A 19.395°"), (18.695, "B 18.695°"), (19.070, "B 19.070°"), (18.840, "C 18.840°")):
    pass
a.annotate("A: two peaks, 0.80° apart", xy=(19.395, 650), xytext=(19.6, 760), fontsize=10.5, color=A_C, arrowprops=dict(arrowstyle="->", color=A_C))
a.annotate("B: weak high-angle peak", xy=(19.07, 350), xytext=(19.3, 470), fontsize=10.5, color=B_C, arrowprops=dict(arrowstyle="->", color=B_C))
a.annotate("C: single peak", xy=(18.84, 915), xytext=(17.75, 930), fontsize=10.5, color=C_C, arrowprops=dict(arrowstyle="->", color=C_C))
a.set_xlim(17.6, 20.3); a.set_ylim(0, 1050)
a.set_xlabel("2θ (deg, Cu Kα)"); a.set_ylabel("counts"); a.set_title("(003) at 4.30 V, project data")
a.legend(loc="upper left", bbox_to_anchor=(0, 0.88))
save(f, "f_430abc")

# c from XRD points
f, a = fig(6.75, 4.1)
P = {"A": [(4.08, 14.411), (4.30, 14.304), (4.50, 13.622), (4.80, 13.411)], "B": [(4.08, 14.319), (4.30, 14.228), (4.50, 13.722), (4.80, 13.539)],
     "C": [(4.08, 14.216), (4.30, 14.119), (4.50, 13.857), (4.80, 13.591)]}
for s, pts in P.items():
    p = np.array(pts); a.plot(p[:, 0], p[:, 1], "o-", color=SC[s], lw=1.8, ms=7, label=f"{s}: −{(p[0,1]-p[-1,1])/p[0,1]*100:.2f}%")
a.plot([4.30], [13.719], "s", color=A_C, ms=8, mfc="white", mew=2, label="A high-angle (H3-like)")
a.plot([4.30], [13.951], "s", color=B_C, ms=8, mfc="white", mew=2, label="B high-angle")
a.set_xlabel("Voltage (V)"); a.set_ylabel("c = 3d(003) (Å)"); a.set_title("c-axis from (003), 4.08 → 4.80 V")
a.legend(loc="lower left")
save(f, "f_cpoints")

# project c–V curve
f, a = fig(4.45, 3.2)
for k, s in ((1, "A"), (2, "B"), (3, "C")):
    a.plot(cv[:, 0], cv[:, k], color=SC[s], lw=1.8, label=s)
a.set_xlabel("Voltage (V)"); a.set_ylabel("c (Å)"); a.set_title("c–V curve (sheet 02)"); a.legend(loc="lower left")
save(f, "f_cvproj")

# capacity
f, a = fig(4.45, 3.2)
for k, s in ((1, "A"), (2, "B"), (3, "C")):
    a.plot(cyc[:, 0], cyc[:, k], color=SC[s], lw=1.8, label=f"{s}: {cyc[-1,k]/cyc[0,k]*100:.1f}%")
a.set_xlabel("Cycle"); a.set_ylabel("Capacity (mAh/g)"); a.set_title("300-cycle capacity"); a.legend(loc="lower left")
save(f, "f_capacity")

# contraction vs retention
f, a = fig(4.6, 3.2)
for s, c, r in (("A", 6.94, 55.5), ("B", 5.45, 82.0), ("C", 4.40, 95.0)):
    a.scatter(c, r, s=360, color=SC[s], zorder=3); a.text(c, r, s, ha="center", va="center", color="white", fontweight="bold", zorder=4)
a.plot([6.94, 5.45, 4.40], [55.5, 82, 95], ls="--", color=GRAY, lw=1)
a.set_xlabel("c contraction (%)"); a.set_ylabel("Retention (%)"); a.set_title("Structure ↔ life"); a.set_xlim(3.9, 7.5); a.set_ylim(45, 105)
save(f, "f_tradeabc")

# heatmaps (paper Fig. 3a)
for name, sheet, hdr in (("f_heat_s", "Figure 3a for s-LNO", False), ("f_heat_d", "Figure 3a for d-LNO", True)):
    d = X[sheet]
    body = d.iloc[1:] if hdr else d
    body = num(body)
    x = body.iloc[:, 0].to_numpy(); Z = body.iloc[:, 2:].to_numpy()
    m = (x > 5.7) & (x < 6.6)
    f, a = fig(3.5, 4.2)
    a.imshow(Z[m].T, aspect="auto", origin="lower", cmap="magma", extent=[tcu(x[m][0]), tcu(x[m][-1]), 0, Z.shape[1]],
             vmin=np.percentile(Z[m], 5), vmax=np.percentile(Z[m], 99.5))
    a.set_xlabel("2θ (Cu Kα eq.)"); a.set_ylabel("scan # (time →)")
    a.set_title("s-LNO (A) operando" if name.endswith("s") else "d-LNO (C) operando")
    save(f, name)

# waterfalls (paper Fig. 3b/3c charge frames)
for name, sheet, n, lab in (("f_wf_s", "Figure 3c-(003) reflection", 13, "s-LNO (A)"), ("f_wf_d", "Figure 3b-(003) reflection", 16, "d-LNO (C)")):
    d = X[sheet]; V = d.iloc[1, 1:n + 1].astype(float).to_numpy(); x = d.iloc[2:, 0].astype(float).to_numpy(); Y = d.iloc[2:, 1:n + 1].astype(float).to_numpy()
    m = (x > 5.7) & (x < 6.6)
    f, a = fig(6.9, 4.3)
    cmap = plt.get_cmap("coolwarm")
    order = np.argsort(V)
    for i, j in enumerate(order):
        y = Y[m, j] / Y[m].max()
        a.plot(tcu(x[m]), y + i * 0.35, color=cmap(i / max(1, len(order) - 1)), lw=1.3)
        a.text(tcu(x[m])[-1] + 0.03, i * 0.35 + 0.05, f"{V[j]:.2f} V", fontsize=8.5, va="bottom")
    a.set_xlim(tcu(x[m])[0], tcu(x[m])[-1] + 0.35); a.set_yticks([])
    a.set_xlabel("2θ (deg, Cu Kα equivalent)"); a.set_title(f"{lab}: (003) during charge (offset)")
    save(f, name)

# c–V paper vs project
f, a = fig(6.9, 4.5)
for cols, lab, col in (((4, 5), "Paper s-LNO", A_C), ((8, 9), "Paper d-LNO", C_C)):
    arr = d3.iloc[:, list(cols)].dropna().to_numpy(); a.plot(arr[:, 0], arr[:, 1], "o", ms=4, color=col, label=lab)
a.plot(cv[:, 0], cv[:, 1], color=A_C, lw=1.2, alpha=.6, label="Project curve A")
a.plot(cv[:, 0], cv[:, 3], color=C_C, lw=1.2, alpha=.6, label="Project curve C")
for s, col in (("A", "#1e3a8a"), ("C", "#9a3412")):
    p = np.array(P[s] + ([(4.30, 13.719)] if s == "A" else [])); a.plot(p[:, 0], p[:, 1], "*", ms=13, color=col, mec="white", label=f"Our reading {s}")
a.axvspan(4.47, 4.53, color="#fde68a", alpha=.6, zorder=0)
a.set_xlim(3.6, 4.85); a.set_xlabel("Voltage (V)"); a.set_ylabel("c (Å)"); a.set_title("Lattice c: paper Rietveld vs our reading")
a.legend(ncol=2, loc="lower left", fontsize=9.5)
save(f, "f_cv")

# 4.30 V fit
d = X["Figure 3c-(003) reflection"]; V = d.iloc[1, 1:14].astype(float).to_numpy(); xx = d.iloc[2:, 0].astype(float).to_numpy()
y = d.iloc[2:, 1 + int(np.argmin(abs(V - 4.30)))].astype(float).to_numpy(); m = (xx > 5.55) & (xx < 6.75); x, y = xx[m], y[m]
pv = lambda x, A, x0, w, eta: A * (eta / (1 + 4 * ((x - x0) / w) ** 2) + (1 - eta) * np.exp(-4 * np.log(2) * ((x - x0) / w) ** 2))
fn = lambda q, *p: pv(q, *p[0:4]) + pv(q, *p[4:8]) + p[8]
p, _ = curve_fit(fn, x, y, p0=[y.max() * .5, 5.95, .14, .3, y.max(), 6.12, .14, .3, 0],
                 bounds=([0, 5.85, .05, 0, 0, 6.0, .05, 0, -np.inf], [np.inf, 6.06, .4, 1, np.inf, 6.3, .4, 1, np.inf]), maxfev=50000)
f, a = fig(6.9, 4.5)
q = np.linspace(x.min(), x.max(), 600); sc = fn(q, *p).max()
a.fill_between(tcu(q), 0, pv(q, *p[0:4]) / sc, color=A_C, alpha=.3, label="H2-like, c = 14.33 Å")
a.fill_between(tcu(q), 0, pv(q, *p[4:8]) / sc, color="#dc2626", alpha=.25, label="H3-like, c = 13.94 Å")
a.plot(tcu(x), y / sc, "o", ms=3.5, color="#0f172a", label="Paper s-LNO 4.30 V (R² 0.9995)")
ya = xr["A_4.30V"].to_numpy(float); ya = (ya - np.median(ya[:60])) / (ya.max() - np.median(ya[:60]))
a.plot(tt, ya, color="#16a34a", lw=1.6, ls="--", label="Project A 4.30 V")
a.set_xlim(17.9, 20.1); a.set_ylim(-0.03, 1.12); a.set_xlabel("2θ (deg, Cu Kα equivalent)"); a.set_ylabel("normalized intensity")
a.set_title("4.30 V: two-phase, but overlapped in the paper"); a.legend(loc="upper right", fontsize=9.5)
save(f, "f_fit430")

# FWHM
f, a = fig(6.9, 3.0)
lab = ["A 4.08", "A 4.30", "A 4.80", "C 4.08", "C 4.30", "C 4.80"]; paper = [0.46, 0.86, 0.46, 0.68, 0.79, 0.95]; proj = [0.090, 0.075, 0.091, 0.075, 0.075, 0.076]
k = np.arange(6); a.bar(k - .2, paper, .4, color=GRAY, label="Paper (Cu Kα eq.)"); a.bar(k + .2, proj, .4, color="#0e8c7f", label="Project")
for i, v in enumerate(paper): a.text(i - .2, v + .02, f"{v:.2f}", ha="center", fontsize=9)
a.set_xticks(k, lab, fontsize=10); a.set_ylabel("FWHM (deg)"); a.set_ylim(0, 1.1); a.legend(loc="upper left", ncol=2)
save(f, "f_fwhm")

# retention paper vs project
c2 = num(X["Figure 2c"].iloc[1:])
f, a = fig(6.9, 3.0)
for cols, lab, col in (((0, 1), "Paper s-LNO", A_C), ((2, 3), "Paper d-LNO", C_C)):
    arr = c2.iloc[:, list(cols)].dropna().to_numpy(); arr = arr[arr[:, 0] <= 300]
    a.plot(arr[:, 0], arr[:, 1] / arr[0, 1] * 100, ls="--", lw=1.6, color=col, label=lab)
for k_, s in ((1, "A"), (2, "B"), (3, "C")):
    a.plot(cyc[:, 0], cyc[:, k_] / cyc[0, k_] * 100, lw=2, color=SC[s], label=f"Project {s}")
a.set_xlabel("Cycle"); a.set_ylabel("Retention (%)"); a.set_ylim(50, 102); a.legend(ncol=3, loc="lower left", fontsize=9.5)
save(f, "f_retention")

# dQ/dV
q2 = num(X["Figure 2d"].iloc[1:])
f, a = fig(7.3, 3.55)
for cols, lab, col in (((2, 3), "s-LNO (A)", A_C), ((0, 1), "d-LNO 600C-6h (C)", C_C)):
    arr = q2.iloc[:, list(cols)].dropna().to_numpy(); arr = arr[:int(np.argmax(arr[:, 0])) + 1]; arr = arr[np.argsort(arr[:, 0])]
    a.plot(arr[:, 0], arr[:, 1], color=col, lw=1.6, label=lab)
a.axvspan(4.12, 4.26, color="#fde68a", alpha=.5, zorder=0); a.text(4.19, 1250, "H2→H3", ha="center")
a.set_xlim(3.63, 4.75); a.set_ylim(-30, 1400); a.set_xlabel("Voltage (V)"); a.set_ylabel("dQ/dV (mAh/g/V)")
a.set_title("First-charge dQ/dV (paper Fig. 2d)"); a.legend(loc="upper left")
save(f, "f_dqdv")

# bottom row: volume, dcdv, a-axis, sintering
e = num(X["Figure 3e"].iloc[1:]); vals = {}
for nm, ac, cc in (("s-LNO (A)", (4, 5), (4, 5)), ("d-LNO (C)", (0, 1), (8, 9))):
    aa = e.iloc[:, list(ac)].dropna().to_numpy(); c_ = d3.iloc[:, list(cc)].dropna().to_numpy()
    at = lambda arr, v: arr[np.argmin(abs(arr[:, 0] - v)), 1]; vol = lambda v: math.sqrt(3) / 2 * at(aa, v) ** 2 * at(c_, v)
    vals[nm] = [(at(c_, 4.8) - at(c_, 4.08)) / at(c_, 4.08) * 100, (at(aa, 4.8) - at(aa, 4.08)) / at(aa, 4.08) * 100, (vol(4.8) - vol(4.08)) / vol(4.08) * 100]
f, a = fig(6.85, 3.9); k = np.arange(3)
for i, (nm, col) in enumerate((("s-LNO (A)", A_C), ("d-LNO (C)", C_C))):
    a.bar(k + (i - .5) * .36, vals[nm], .36, color=col, label=nm)
    for bx, bv in zip(k + (i - .5) * .36, vals[nm]): a.text(bx, bv - .25, f"{bv:.1f}%", ha="center", va="top", fontsize=9.5)
a.axhline(0, color=GRAY, lw=.8); a.set_xticks(k, ["c-axis", "a-axis", "cell volume"]); a.set_ylabel("Δ 4.08→4.80 V (%)"); a.set_ylim(-9.3, .8)
a.set_title("Lattice & volume change"); a.legend(loc="lower center")
save(f, "f_volume")

f, a = fig(6.85, 3.9)
for cols, lab, col in (((4, 5), "s-LNO (A)", A_C), ((8, 9), "d-LNO (C)", C_C)):
    arr = d3.iloc[:, list(cols)].dropna().to_numpy(); arr = arr[np.argsort(arr[:, 0])]; arr = arr[arr[:, 0] > 3.9]
    dv = np.diff(arr[:, 0]); ok = dv > 0.005; sl = -np.diff(arr[:, 1])[ok] / dv[ok]; mid = ((arr[1:, 0] + arr[:-1, 0]) / 2)[ok]
    a.plot(mid, sl, "o-", ms=3.5, lw=1.6, color=col, label=f"{lab}: max {sl.max():.1f} Å/V")
a.axhline(0, color=GRAY, lw=.8); a.set_xlim(3.9, 4.85); a.set_xlabel("Voltage (V)"); a.set_ylabel("−dc/dV (Å/V)"); a.set_title("Collapse speed of c"); a.legend()
save(f, "f_dcdv")

f, a = fig(6.85, 3.9)
for ac, lab, col in (((4, 5), "s-LNO (A)", A_C), ((0, 1), "d-LNO (C)", C_C)):
    arr = e.iloc[:, list(ac)].dropna().to_numpy(); arr = arr[np.argsort(arr[:, 0])]
    a.plot(arr[:, 0], arr[:, 1], "o-", ms=3, lw=1.4, color=col, label=lab)
a.set_xlabel("Voltage (V)"); a.set_ylabel("a (Å)"); a.set_title("Lattice a during charge (paper Fig. 3e)"); a.legend()
save(f, "f_aaxis")

b = X["Figure 2b"].iloc[1:].astype(float).to_numpy()
f, a = fig(6.85, 3.9); a2 = a.twinx()
a.plot(b[:, 3], b[:, 1], "o-", color="#0e8c7f", lw=1.8, label="Initial capacity"); a2.plot(b[:, 3], b[:, 2], "s--", color="#dc2626", lw=1.8, label="50-cycle retention")
for r in b: a.annotate(f"{int(r[0])} h", (r[3], r[1]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
a.set_xlabel("Rocksalt fraction (wt%) at 600 °C"); a.set_ylabel("Capacity (mAh/g)", color="#0e8c7f"); a2.set_ylabel("Retention (%)", color="#dc2626")
a2.spines["right"].set_visible(True); a.set_ylim(180, 240); a2.set_ylim(70, 104); a.set_title("Sintering time: capacity vs stability")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, loc="center right")
save(f, "f_sinter")
print("ok")

# agreement map replaces s-LNO heatmap (source sheet duplicated d-LNO columns)
f, a = fig(3.5, 4.2)
labs = ["A 4.08", "A 4.50", "A 4.80", "C 4.08", "C 4.30", "C 4.50", "C 4.80"]
dc = [14.411 - 14.412, 13.622 - 13.444, 13.411 - 13.414, 14.216 - 14.221, 14.119 - 14.113, 13.857 - 13.708, 13.591 - 13.589]
cols = ["#dc2626" if abs(v) > 0.05 else "#0e8c7f" for v in dc]
yy = np.arange(len(labs))[::-1]
a.barh(yy, dc, color=cols)
for y_, v in zip(yy, dc):
    a.text(v + (0.008 if v >= 0 else -0.008), y_, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9.5)
a.axvspan(-0.01, 0.01, color="#e3f4f1", zorder=0)
a.set_yticks(yy, labs); a.set_xlim(-0.06, 0.24); a.set_xlabel("c(ours) − c(paper) (Å)")
a.set_title("Agreement with paper")
save(f, "f_agree")
