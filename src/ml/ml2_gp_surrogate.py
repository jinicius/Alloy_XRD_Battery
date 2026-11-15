"""4) 소성 시간 → 구조(암염상 비율, 층상 결정 크기) → 성능(용량, 50사이클 유지율) 가우시안 프로세스 대리 모델."""
import sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

SRC, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
b = pd.read_excel(SRC, sheet_name="Figure 2b", header=None).iloc[1:].astype(float).to_numpy()
t, cap, ret, rs_fig = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
# Table S1 (SI): heating time → RS wt%, layered domain size (nm)
S1 = {3: (20.1, 20.8), 6: (13.0, 26.6), 12: (7.3, 38.5), 18: (5.1, 43.5), 36: (2.3, 56.9), 48: (2.2, 66.5)}
rs = np.array([S1[int(k)][0] for k in t]); dsL = np.array([S1[int(k)][1] for k in t])
lt = np.log10(t)


def gp():
    k = C(1.0, (1e-2, 1e2)) * RBF(0.5, (0.1, 5)) + WhiteKernel(1e-2, (1e-5, 1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=20, random_state=0)


def loo(xin, y):
    pred = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i; g = gp().fit(xin[m].reshape(-1, 1), y[m]); pred.append(g.predict(xin[[i]].reshape(-1, 1))[0])
    pred = np.array(pred)
    return round(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), 3), round(float(np.sqrt(np.mean((y - pred) ** 2))), 2)


out = {"data": {"time_h": t.tolist(), "capacity": cap.tolist(), "retention50": ret.tolist(), "RS_wt%": rs.tolist(), "DS_L_nm": dsL.tolist()}}
# correlations (structure ↔ performance)
for nm, xv in (("RS_wt%", rs), ("DS_L_nm", dsL), ("log_time", lt)):
    out[f"spearman_{nm}_vs_capacity"] = round(spearmanr(xv, cap)[0], 3)
    out[f"spearman_{nm}_vs_retention"] = round(spearmanr(xv, ret)[0], 3)
# LOO for GP models
out["LOO_R2_RMSE"] = {
    "log t → RS": loo(lt, rs), "log t → capacity": loo(lt, cap), "log t → retention": loo(lt, ret),
    "RS → capacity": loo(rs, cap), "RS → retention": loo(rs, ret),
}
# fit on all; predict objective over time grid with posterior sampling
grid = np.linspace(np.log10(2), np.log10(60), 300)
gc = gp().fit(lt.reshape(-1, 1), cap); gr = gp().fit(lt.reshape(-1, 1), ret); gs = gp().fit(lt.reshape(-1, 1), rs)
mc, sc = gc.predict(grid.reshape(-1, 1), return_std=True); mr, sr = gr.predict(grid.reshape(-1, 1), return_std=True)
ms, ss = gs.predict(grid.reshape(-1, 1), return_std=True)
Sc = gc.sample_y(grid.reshape(-1, 1), 2000, random_state=1); Sr = gr.sample_y(grid.reshape(-1, 1), 2000, random_state=2)
obj = Sc * np.clip(Sr, 0, 100) / 100        # 50사이클 후 남는 용량
best = 10 ** grid[np.argmax(obj, axis=0)]
out["retained_capacity_opt_time_h"] = {"median": round(float(np.median(best)), 1), "CI95": np.percentile(best, [2.5, 97.5]).round(1).tolist()}
mo = obj.mean(axis=1); out["retained_capacity_max_mean"] = round(float(mo.max()), 1); out["at_time_h"] = round(float(10 ** grid[np.argmax(mo)]), 1)
# constraint: highest capacity with P(retention ≥ 95%) ≥ 0.9
pr = (Sr >= 95).mean(axis=1); ok = pr >= 0.9
out["max_capacity_with_P(ret≥95%)≥0.9"] = {"time_h": round(float(10 ** grid[ok][np.argmax(mc[ok])]), 1), "capacity": round(float(mc[ok].max()), 1)} if ok.any() else None
out["observed_retained"] = dict(zip([f"{int(x)}h" for x in t], np.round(cap * ret / 100, 1).tolist()))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
tg = 10 ** grid
a = ax[0]; a.fill_between(tg, ms - 2 * ss, ms + 2 * ss, color="#94a3b8", alpha=.35); a.plot(tg, ms, color="#334155"); a.plot(t, rs, "o", color="#0f172a")
a.set_xscale("log"); a.set_xlabel("Sintering time at 600 °C (h)"); a.set_ylabel("Rocksalt fraction (wt%)"); a.set_title(f"time → structure (LOO R² {out['LOO_R2_RMSE']['log t → RS'][0]})")
a = ax[1]; a2 = a.twinx()
a.fill_between(tg, mc - 2 * sc, mc + 2 * sc, color="#0e8c7f", alpha=.2); a.plot(tg, mc, color="#0e8c7f"); a.plot(t, cap, "o", color="#0e8c7f")
a2.fill_between(tg, mr - 2 * sr, mr + 2 * sr, color="#dc2626", alpha=.15); a2.plot(tg, mr, color="#dc2626"); a2.plot(t, ret, "s", color="#dc2626")
a.set_xscale("log"); a.set_xlabel("Sintering time (h)"); a.set_ylabel("Capacity (mAh/g)", color="#0e8c7f"); a2.set_ylabel("50-cycle retention (%)", color="#dc2626")
a2.spines["right"].set_visible(True); a.set_title("GP surrogate ±2σ")
a = ax[2]; lo, hi = np.percentile(obj, [2.5, 97.5], axis=1)
a.fill_between(tg, lo, hi, color="#7c3aed", alpha=.2); a.plot(tg, mo, color="#7c3aed", lw=2); a.plot(t, cap * ret / 100, "o", color="#4c1d95")
a.axvline(out["at_time_h"], color="#7c3aed", ls="--"); a.set_xscale("log"); a.set_xlabel("Sintering time (h)"); a.set_ylabel("Capacity after 50 cycles (mAh/g)")
a.set_title(f"Objective: best ≈ {out['at_time_h']} h")
for a_ in ax:
    a_.set_xticks([3, 6, 12, 24, 48]); a_.set_xticklabels(["3", "6", "12", "24", "48"]); a_.minorticks_off()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gp.png", dpi=180)
json.dump(out, open(f"{OUT}/a4.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
