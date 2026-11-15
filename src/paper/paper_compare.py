"""논문 Source Data(Fig. 3b·3c·3d)와 과제 데이터(A·C)를 같은 판독 기준으로 비교한다."""
import math, sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

SRC = sys.argv[1]           # source_data_for_main_figures.xlsx
PROJ = sys.argv[2]          # 과제 원본 엑셀
LP, LC = 0.49594, 1.5406    # 논문 방사광 / Cu Kα 파장 (Å)

d_of = lambda tt, lam: lam / (2 * np.sin(np.radians(tt / 2)))
tt_of = lambda d, lam: 2 * np.degrees(np.arcsin(lam / (2 * d)))
to_cu = lambda tt: tt_of(d_of(tt, LP), LC)          # 논문 2θ → 같은 d를 Cu Kα로 본 2θ


def half_cross(x, y, i, level, step):
    while 0 < i < len(y) - 1:
        j = i + step
        if y[j] < level:
            return x[i] + (x[j] - x[i]) * (y[i] - level) / (y[i] - y[j])
        i = j
    return np.nan


def read(x, y, edge_w):
    """석진 기준: 배경 = 양 끝 edge_w 구간 중앙값, 봉우리 = 잡음 20σ 이상, 2θ = 구간 최대점, FWHM = 반치 직선 보간."""
    edge = (x <= x[0] + edge_w) | (x >= x[-1] - edge_w)
    bg, sd = np.median(y[edge]), np.std(y[edge])
    pk, _ = find_peaks(y, prominence=max(20 * sd, 0.05 * (y.max() - bg)))
    vals = [p + int(np.argmin(y[p:q + 1])) for p, q in zip(pk[:-1], pk[1:])]
    b = [0] + vals + [len(x) - 1]
    out = []
    for k in range(len(pk)):
        ip = b[k] + int(np.argmax(y[b[k]:b[k + 1] + 1]))
        lv = bg + (y[ip] - bg) / 2
        L, R = half_cross(x, y, ip, lv, -1), half_cross(x, y, ip, lv, 1)
        if L < x[b[k]] or R > x[b[k + 1]]:
            L = R = np.nan
        out.append(dict(tt=x[ip], L=L, R=R, h=y[ip] - bg,
                        valley=(y[vals[0]] - bg) / (y[ip] - bg) if vals else np.nan))
    return out, bg, sd


def pv2(x, *p):
    out = p[-1] + 0 * x
    for k in range(0, len(p) - 1, 4):
        A, x0, w, eta = p[k:k + 4]
        out = out + A * (eta / (1 + 4 * ((x - x0) / w) ** 2) + (1 - eta) * np.exp(-4 * np.log(2) * ((x - x0) / w) ** 2))
    return out


def paper_frames(sheet, volts, n_charge):
    d = pd.read_excel(SRC, sheet_name=sheet, header=None)
    V = d.iloc[1, 1:].astype(float).to_numpy()
    X = d.iloc[2:, 0].astype(float).to_numpy()
    Y = d.iloc[2:, 1:].astype(float).to_numpy()
    m = (X > 5.55) & (X < 6.75)
    res = []
    for v in volts:
        j = int(np.argmin(np.abs(V[:n_charge] - v)))   # 첫 충전 구간 프레임만
        x, y = X[m], Y[m, j]
        pk, bg, sd = read(x, y, 0.10)                  # λ0.496에서 0.10° ≈ Cu Kα 0.3°
        res.append((V[j], x, y, pk, bg, sd))
    return res


if __name__ == "__main__":
    rows = []
    for lab, sheet, volts, n in [("s-LNO (=A)", "Figure 3c-(003) reflection", [4.08, 4.21, 4.30, 4.42, 4.50, 4.80], 13),
                                 ("d-LNO (=C)", "Figure 3b-(003) reflection", [4.08, 4.21, 4.30, 4.40, 4.50, 4.80], 16)]:
        for v, x, y, pk, bg, sd in paper_frames(sheet, volts, n):
            for k, p in enumerate(pk):
                c = 3 * d_of(p["tt"], LP)
                fw_cu = to_cu(p["R"]) - to_cu(p["L"])
                rows.append(dict(시료=lab, 프레임전압=round(v, 3), 봉우리=f"{k + 1}/{len(pk)}",
                                 논문2θ=round(p["tt"], 3), 논문FWHM=round(p["R"] - p["L"], 3),
                                 c=round(c, 3), Cu환산2θ=round(to_cu(p["tt"]), 3), Cu환산FWHM=round(fw_cu, 3),
                                 골깊이=round(p["valley"], 2) if not np.isnan(p["valley"]) else "",
                                 잡음비=round(p["h"] / sd)))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))
    tab.to_csv("논문_원데이터_판독.csv", index=False, encoding="utf-8-sig")
