"""3) 변화점 탐지: c–V를 연속 3구간 직선으로 맞춰 H2→H3 시작·끝 전압을 찾고, 잔차 부트스트랩으로 신뢰구간."""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, PROJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
X = pd.read_excel(SRC, sheet_name=None, header=None)
d3 = X["Figure 3d"].iloc[1:].apply(pd.to_numeric, errors="coerce")
rng = np.random.default_rng(0)


def hinge_fit(v, c, b1, b2):
    A = np.c_[np.ones_like(v), v, np.clip(v - b1, 0, None), np.clip(v - b2, 0, None)]
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    return coef, A @ coef


def best_bp(v, c, grid):
    best = None
    for i, b1 in enumerate(grid):
        for b2 in grid[i + 3:]:
            if ((v > b1) & (v < b2)).sum() < 2: continue
            coef, f = hinge_fit(v, c, b1, b2); sse = np.sum((c - f) ** 2)
            if best is None or sse < best[0]: best = (sse, b1, b2, coef, f)
    return best


def analyse(v, c, lo=4.05, hi=4.82, nboot=300):
    m = (v >= lo) & (v <= hi); v, c = v[m], c[m]
    grid = np.round(np.arange(lo + .02, hi - .02, 0.005), 3)
    sse, b1, b2, coef, f = best_bp(v, c, grid)
    slopes = [coef[1], coef[1] + coef[2], coef[1] + coef[2] + coef[3]]
    res = c - f; bs = []
    for _ in range(nboot):
        cb = f + rng.choice(res, size=len(res), replace=True)
        _, bb1, bb2, _, _ = best_bp(v, cb, grid[::2]); bs.append((bb1, bb2))
    bs = np.array(bs)
    return dict(onset=b1, end=b2, onset_CI=np.percentile(bs[:, 0], [2.5, 97.5]).round(3).tolist(),
                end_CI=np.percentile(bs[:, 1], [2.5, 97.5]).round(3).tolist(), width=round(b2 - b1, 3),
                slope_mid=round(-slopes[1], 2), drop_mid=round(-slopes[1] * (b2 - b1), 3), n=int(len(v))), (v, c, f)


out = {}; curves = {}
for lab, cols in (("s-LNO", (4, 5)), ("d-LNO", (8, 9))):
    a = d3.iloc[:, list(cols)].dropna().to_numpy(); a = a[np.argsort(a[:, 0])]
    out[lab], curves[lab] = analyse(a[:, 0], a[:, 1])
p02 = pd.read_excel(PROJ, sheet_name="02_물성·성능", header=None)
r0 = p02.index[p02.iloc[:, 0].astype(str).str.startswith("Voltage")][0]
cv = p02.iloc[r0 + 1:r0 + 30, :4].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
for k, s in ((1, "project A"), (2, "project B"), (3, "project C")):
    out[s], curves[s] = analyse(cv[:, 0], cv[:, k], nboot=0 or 50)

# dQ/dV H2→H3 peak sharpness
q = X["Figure 2d"].iloc[1:].apply(pd.to_numeric, errors="coerce")
for lab, cols in (("s-LNO", (2, 3)), ("d-LNO", (0, 1))):
    a = q.iloc[:, list(cols)].dropna().to_numpy(); a = a[:int(np.argmax(a[:, 0])) + 1]; a = a[np.argsort(a[:, 0])]
    m = (a[:, 0] > 4.08) & (a[:, 0] < 4.32); v, y = a[m, 0], a[m, 1]; i = np.argmax(y)
    base = np.median(np.r_[y[:4], y[-4:]]); half = base + (y[i] - base) / 2
    above = v[y >= half]
    out[lab]["dQdV_peak_V"] = round(float(v[i]), 3); out[lab]["dQdV_peak"] = round(float(y[i]), 1)
    out[lab]["dQdV_FWHM_V"] = round(float(above.max() - above.min()), 3)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a, labs in ((ax[0], ("s-LNO", "d-LNO")), (ax[1], ("project A", "project B", "project C"))):
    for lab, col in zip(labs, ("#1d4ed8", "#ea580c", "#7c3aed") if len(labs) == 2 else ("#1d4ed8", "#7c3aed", "#ea580c")):
        v, c, f = curves[lab]; o = out[lab]
        a.plot(v, c, "o", ms=4, color=col, alpha=.7); a.plot(v, f, "-", color=col, lw=2, label=f"{lab}: {o['onset']:.3f}–{o['end']:.3f} V, {o['slope_mid']} Å/V")
        a.axvline(o["onset"], color=col, ls=":", lw=1); a.axvline(o["end"], color=col, ls=":", lw=1)
    a.set_xlabel("Voltage (V)"); a.set_ylabel("c (Å)"); a.legend(fontsize=9)
ax[0].set_title("Paper: 3-segment change-point fit"); ax[1].set_title("Project sheet 02 curves")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_changepoint.png", dpi=180)
json.dump(out, open(f"{OUT}/a3.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
