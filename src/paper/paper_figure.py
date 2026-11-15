"""논문 Source Data vs 과제 데이터 비교 그림 (4패널)."""
import math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SRC, PROJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LP, LC = 0.49594, 1.5406
d_of = lambda tt, lam: lam / (2 * np.sin(np.radians(tt / 2)))
to_cu = lambda tt: 2 * np.degrees(np.arcsin(LC / (2 * d_of(tt, LP))))
for n in ["Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"]:
    if any(n == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = n
        break
plt.rcParams["axes.unicode_minus"] = False

# 과제 데이터
xr = pd.read_excel(PROJ, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601)
X = xr.iloc[:, 0].to_numpy(float)
cv = pd.read_excel(PROJ, sheet_name="02_물성·성능", header=None)
r0 = cv.index[cv.iloc[:, 0].astype(str).str.startswith("Voltage")][0]
cvd = cv.iloc[r0 + 1:r0 + 30, :4].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()


def frames(sheet, n):
    d = pd.read_excel(SRC, sheet_name=sheet, header=None)
    V = d.iloc[1, 1:n + 1].astype(float).to_numpy()
    x = d.iloc[2:, 0].astype(float).to_numpy()
    Y = d.iloc[2:, 1:n + 1].astype(float).to_numpy()
    return V, x, Y


sV, sx, sY = frames("Figure 3c-(003) reflection", 13)
dV, dx, dY = frames("Figure 3b-(003) reflection", 16)
f3d = pd.read_excel(SRC, sheet_name="Figure 3d", header=None).iloc[1:].apply(pd.to_numeric, errors="coerce")
s_c = f3d.iloc[:, 4:6].dropna().to_numpy()
d_c = f3d.iloc[:, 8:10].dropna().to_numpy()

COL = {"4.08": "#2563eb", "4.30": "#16a34a", "4.50": "#f59e0b", "4.80": "#dc2626"}
fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=170)


def peak_panel(a, V, x, Y, proj, title):
    m = (x > 5.6) & (x < 6.7)
    for v in ["4.08", "4.30", "4.80"]:
        j = int(np.argmin(np.abs(V - float(v))))
        y = Y[m, j]
        a.plot(to_cu(x[m]), y / y.max(), color=COL[v], lw=2.2, alpha=0.9, label=f"논문 {V[j]:.2f} V")
        yp = xr[f"{proj}_{v}V"].to_numpy(float)
        yp = (yp - np.median(yp[:60])) / (yp.max() - np.median(yp[:60]))
        a.plot(X, yp, color=COL[v], lw=1, ls="--", label=f"과제 {proj}_{v}V")
    a.set_xlim(17.8, 20.3)
    a.set_ylim(-0.05, 1.12)
    a.set_xlabel("2θ (deg, Cu Kα 환산)")
    a.set_ylabel("정규화 세기 (최대 = 1)")
    a.set_title(title, fontsize=12)
    a.legend(fontsize=8, ncol=2, loc="upper right")
    a.grid(alpha=0.25)


peak_panel(ax[0, 0], sV, sx, sY, "A", "(a) A vs 논문 s-LNO: (003) 모양 (실선 논문, 점선 과제)")
peak_panel(ax[0, 1], dV, dx, dY, "C", "(b) C vs 논문 d-LNO: (003) 모양 (실선 논문, 점선 과제)")

# (c) c–전압
a = ax[1, 0]
a.plot(s_c[:, 0], s_c[:, 1], "o", ms=4, color="#2563eb", label="논문 s-LNO (Rietveld)")
a.plot(d_c[:, 0], d_c[:, 1], "o", ms=4, color="#f97316", label="논문 d-LNO (Rietveld)")
a.plot(cvd[:, 0], cvd[:, 1], "-", color="#2563eb", lw=1, label="과제 02시트 A 곡선")
a.plot(cvd[:, 0], cvd[:, 3], "-", color="#f97316", lw=1, label="과제 02시트 C 곡선")
pts = {"A": [(4.08, 14.411), (4.30, 14.304), (4.30, 13.719), (4.50, 13.622), (4.80, 13.411)],
       "C": [(4.08, 14.216), (4.30, 14.119), (4.50, 13.857), (4.80, 13.591)]}
for k, colr in (("A", "#1e3a8a"), ("C", "#9a3412")):
    p = np.array(pts[k])
    a.plot(p[:, 0], p[:, 1], "*", ms=13, color=colr, mec="white", label=f"우리 판독 {k} (01_XRD)")
a.annotate("4.50 V: 과제 값이\n논문보다 0.15~0.18 Å 큼", xy=(4.5, 13.62), xytext=(4.52, 14.05),
           fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
a.set_xlim(3.6, 4.85)
a.set_xlabel("전압 (V)")
a.set_ylabel("c (Å)")
a.set_title("(c) c–전압: 논문 Rietveld vs 과제 데이터", fontsize=12)
a.legend(fontsize=8, loc="lower left")
a.grid(alpha=0.25)

# (d) FWHM
a = ax[1, 1]
lab = ["A·s-LNO\n4.08 V", "A·s-LNO\n4.30 V", "A·s-LNO\n4.80 V", "C·d-LNO\n4.08 V", "C·d-LNO\n4.30 V", "C·d-LNO\n4.80 V"]
paper = [0.456, 0.860, 0.456, 0.681, 0.791, 0.952]
proj = [0.090, 0.075, 0.091, 0.075, 0.075, 0.076]
xx = np.arange(len(lab))
a.bar(xx - 0.2, paper, 0.4, color="#64748b", label="논문 (Cu Kα 환산, 장비 넓어짐 포함)")
a.bar(xx + 0.2, proj, 0.4, color="#16a34a", label="과제 데이터 (우리 판독)")
for i, (p, q) in enumerate(zip(paper, proj)):
    a.text(i - 0.2, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
    a.text(i + 0.2, q + 0.01, f"{q:.3f}", ha="center", fontsize=8)
a.text(1 - 0.2, 0.90, "두 상 겹침", ha="center", fontsize=8, color="#b91c1c")
a.set_xticks(xx, lab, fontsize=9)
a.set_ylabel("FWHM (deg, Cu Kα 기준)")
a.set_ylim(0, 1.05)
a.set_title("(d) FWHM: 논문은 C가 더 넓고, 과제는 반대", fontsize=12)
a.legend(fontsize=8, loc="upper left")
a.grid(alpha=0.25, axis="y")

fig.text(0.01, 0.005, "논문: Yao et al., Nat. Commun. 16, 10199 (2025) Source Data Fig. 3b·3c·3d (λ = 0.49594 Å, 배경 제거된 값). "
         "2θ는 같은 d를 Cu Kα(1.5406 Å)로 본 값으로 환산. (a)(b)는 모양 비교를 위해 각 곡선 최대값을 1로 정규화.",
         fontsize=8, color="dimgray")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT)
