"""보고서·포스터용 추가 그림: 4.30 V 두 성분 피팅, dQ/dV, -dc/dV, 소성 시간 trade-off, 부피 변화."""
import math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import curve_fit

SRC, OUT = sys.argv[1], sys.argv[2]
LP, LC = 0.49594, 1.5406
for n in ["Noto Sans CJK KR", "Malgun Gothic"]:
    if any(n == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = n
        break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
X = pd.read_excel(SRC, sheet_name=None, header=None)
num = lambda d: d.apply(pd.to_numeric, errors="coerce")
dof = lambda tt, l: l / (2 * np.sin(np.radians(tt / 2)))
tcu = lambda tt: 2 * np.degrees(np.arcsin(LC / (2 * dof(tt, LP))))
BLUE, ORANGE, GRAY, TEAL = "#2563eb", "#f97316", "#64748b", "#0e8c7f"


def pv(x, A, x0, w, eta):
    return A * (eta / (1 + 4 * ((x - x0) / w) ** 2) + (1 - eta) * np.exp(-4 * np.log(2) * ((x - x0) / w) ** 2))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=220)
    plt.close(fig)


# 1) s-LNO 4.30 V 두 성분 피팅
d = X["Figure 3c-(003) reflection"]
V = d.iloc[1, 1:14].astype(float).to_numpy()
x = d.iloc[2:, 0].astype(float).to_numpy()
j = int(np.argmin(abs(V - 4.30)))
y = d.iloc[2:, 1 + j].astype(float).to_numpy()
m = (x > 5.55) & (x < 6.75)
x, y = x[m], y[m]
f = lambda xx, *p: pv(xx, *p[0:4]) + pv(xx, *p[4:8]) + p[8]
p, _ = curve_fit(f, x, y, p0=[y.max() * .5, 5.95, .14, .3, y.max(), 6.12, .14, .3, 0],
                 bounds=([0, 5.85, .05, 0, 0, 6.0, .05, 0, -np.inf], [np.inf, 6.06, .4, 1, np.inf, 6.3, .4, 1, np.inf]), maxfev=50000)
fig, a = plt.subplots(figsize=(7.4, 4.8))
xx = np.linspace(x.min(), x.max(), 600)
a.plot(tcu(x), y / 1000, "o", ms=4, color="#334155", label="논문 s-LNO 4.30 V (원데이터)")
a.plot(tcu(xx), f(xx, *p) / 1000, color="#111827", lw=1.6, label="두 성분 합 (R² 0.9995)")
a.fill_between(tcu(xx), 0, pv(xx, *p[0:4]) / 1000, color=BLUE, alpha=.35, label=f"H2 쪽 c = {3*dof(p[1],LP):.2f} Å")
a.fill_between(tcu(xx), 0, pv(xx, *p[4:8]) / 1000, color="#dc2626", alpha=.30, label=f"H3 쪽 c = {3*dof(p[5],LP):.2f} Å")
for tt, lab in ((18.595, "과제 A\n14.30 Å"), (19.395, "과제 A\n13.72 Å")):
    a.axvline(tt, color=GRAY, ls="--", lw=1)
    a.text(tt, 12.6, lab, ha="center", fontsize=10, color=GRAY)
a.set_xlim(17.9, 20.1)
a.set_ylim(0, 14.5)
a.set_xlabel("2θ (deg, Cu Kα 환산)")
a.set_ylabel("세기 (×1000 counts)")
a.legend(fontsize=10, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
save(fig, "fig_fit430.png")

# 2) dQ/dV
q = num(X["Figure 2d"].iloc[1:])
fig, a = plt.subplots(figsize=(7.4, 4.8))
for cols, lab, col in (((2, 3), "s-LNO (A)", BLUE), ((0, 1), "d-LNO 600C-6h (C)", ORANGE)):
    arr = q.iloc[:, list(cols)].dropna().to_numpy()
    arr = arr[:int(np.argmax(arr[:, 0])) + 1]          # 첫 충전 구간만
    arr = arr[np.argsort(arr[:, 0])]
    a.plot(arr[:, 0], arr[:, 1], color=col, lw=1.8, label=lab)
a.axvspan(4.12, 4.26, color="#fde68a", alpha=.5, zorder=0)
a.text(4.19, 1230, "H2→H3\n1167 vs 329", ha="center", fontsize=11)
a.set_xlim(3.63, 4.75)
a.set_ylim(-50, 1450)
a.set_xlabel("전압 (V)")
a.set_ylabel("dQ/dV (mAh/g/V)")
a.legend(frameon=False, loc="upper left")
save(fig, "fig_dqdv.png")

# 3) -dc/dV
d3 = num(X["Figure 3d"].iloc[1:])
fig, a = plt.subplots(figsize=(7.4, 4.8))
for cols, lab, col in (((4, 5), "s-LNO (A)", BLUE), ((8, 9), "d-LNO (C)", ORANGE)):
    arr = d3.iloc[:, list(cols)].dropna().to_numpy()
    arr = arr[np.argsort(arr[:, 0])]
    arr = arr[arr[:, 0] > 3.9]
    dv = np.diff(arr[:, 0]); ok = dv > 0.005
    sl = -np.diff(arr[:, 1])[ok] / dv[ok]
    mid = ((arr[1:, 0] + arr[:-1, 0]) / 2)[ok]
    a.plot(mid, sl, "o-", ms=4, color=col, lw=1.6, label=f"{lab}: 최대 {sl.max():.1f} Å/V")
a.axhline(0, color=GRAY, lw=.8)
a.set_xlabel("전압 (V)")
a.set_ylabel("−dc/dV (Å/V)")
a.set_xlim(3.9, 4.85)
a.legend(frameon=False)
save(fig, "fig_dcdv.png")

# 4) 소성 시간 trade-off
b = X["Figure 2b"].iloc[1:].astype(float).to_numpy()
fig, a = plt.subplots(figsize=(7.4, 4.8))
a2 = a.twinx()
a.plot(b[:, 3], b[:, 1], "o-", color=TEAL, lw=2, ms=7, label="초기 방전 용량")
a2.plot(b[:, 3], b[:, 2], "s--", color="#dc2626", lw=2, ms=7, label="50사이클 유지율")
for r in b:
    a.annotate(f"{int(r[0])}h", (r[3], r[1]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10, color="#334155")
a.set_xlabel("암염상 비율 (wt%) — 600 °C 소성 시간이 길수록 ←")
a.set_ylabel("용량 (mAh/g)", color=TEAL)
a2.set_ylabel("유지율 (%)", color="#dc2626")
a2.spines["right"].set_visible(True)
a.set_ylim(180, 240)
a2.set_ylim(70, 104)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, frameon=False, loc="center right")
save(fig, "fig_tradeoff.png")

# 5) 부피 변화 (a·c 모두 사용)
e = num(X["Figure 3e"].iloc[1:])
vals = {}
for name, ac, cc in (("s-LNO (A)", (4, 5), (4, 5)), ("d-LNO (C)", (0, 1), (8, 9))):
    aa = e.iloc[:, list(ac)].dropna().to_numpy(); cc_ = d3.iloc[:, list(cc)].dropna().to_numpy()
    at = lambda arr, v: arr[np.argmin(abs(arr[:, 0] - v)), 1]
    vol = lambda v: math.sqrt(3) / 2 * at(aa, v) ** 2 * at(cc_, v)
    vals[name] = ((at(cc_, 4.80) - at(cc_, 4.08)) / at(cc_, 4.08) * 100,
                  (at(aa, 4.80) - at(aa, 4.08)) / at(aa, 4.08) * 100,
                  (vol(4.80) - vol(4.08)) / vol(4.08) * 100)
fig, a = plt.subplots(figsize=(7.4, 4.8))
cats = ["c축", "a축", "단위격자 부피"]
xx = np.arange(3)
for k, (name, col) in enumerate((("s-LNO (A)", BLUE), ("d-LNO (C)", ORANGE))):
    v = vals[name]
    bars = a.bar(xx + (k - .5) * .36, v, .36, color=col, label=name)
    for bx, bv in zip(xx + (k - .5) * .36, v):
        a.text(bx, bv - .35, f"{bv:.2f}%", ha="center", va="top", fontsize=11)
a.axhline(0, color=GRAY, lw=.8)
a.set_xticks(xx, cats)
a.set_ylabel("4.08 → 4.80 V 변화율 (%)")
a.set_ylim(-9, 0.8)
a.legend(frameon=False, loc="lower center")
save(fig, "fig_volume.png")
print({k: [round(t, 2) for t in v] for k, v in vals.items()})
