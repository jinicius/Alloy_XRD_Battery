"""
배경·판독 방법을 바꿔도 결과가 같은지 확인하는 스크립트 (민감도 점검)

방법 1  현재 기준   : 배경 = 양 끝 0.3° 중앙값(상수), 2θ = 최대 데이터 점, FWHM = 반치 직선 보간
방법 2  직선 배경   : 배경 = 왼쪽 끝 중앙값과 오른쪽 끝 중앙값을 잇는 직선, 2θ·FWHM 읽는 법은 1과 같음
방법 3  pseudo-Voigt 피팅 : 봉우리 모양 = η·로렌츠 + (1−η)·가우시안, 배경 = 직선.
                     봉우리 ±0.3° 구간(봉우리 2개면 둘을 한꺼번에)에 최소제곱 피팅. 2θ·FWHM이 연속값

사용법: python xrd_method_check.py <원본 엑셀>
"""
import math
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

LAM = 1.5406
VOLTS = ["4.08V", "4.30V", "4.50V", "4.80V"]
c_of = lambda tt: 3 * LAM / (2 * math.sin(math.radians(tt / 2)))


def half_cross(x, y, ip, level, step):
    i = ip
    while 0 < i < len(y) - 1:
        j = i + step
        if y[j] < level[j]:
            return x[i] + (x[j] - x[i]) * (y[i] - level[i]) / ((y[i] - level[i]) - (y[j] - level[j]))
        i = j
    return np.nan


def read_peaks(x, y, base, peaks):
    """base: 각 점의 배경값 배열. 봉우리별 (2θ, FWHM)."""
    valleys = [p + int(np.argmin(y[p:q + 1])) for p, q in zip(peaks[:-1], peaks[1:])]
    bounds = [0] + valleys + [len(x) - 1]
    out = []
    for k in range(len(peaks)):
        lo, hi = bounds[k], bounds[k + 1]
        ip = lo + int(np.argmax(y[lo:hi + 1]))
        level = base + (y[ip] - base[ip]) / 2
        out.append((x[ip], half_cross(x, y, ip, level, +1) - half_cross(x, y, ip, level, -1)))
    return out


def pv(x, A, x0, w, eta):
    g = np.exp(-4 * math.log(2) * ((x - x0) / w) ** 2)
    l = 1 / (1 + 4 * ((x - x0) / w) ** 2)
    return A * (eta * l + (1 - eta) * g)


def fit_pv(x, y, peaks):
    lo = max(x[peaks[0]] - 0.3, x[0]); hi = min(x[peaks[-1]] + 0.3, x[-1])
    m = (x >= lo) & (x <= hi)
    xs, ys = x[m], y[m]
    n = len(peaks)

    def model(xx, *p):
        out = p[-2] + p[-1] * (xx - xs.mean())
        for k in range(n):
            out = out + pv(xx, *p[4 * k:4 * k + 4])
        return out

    p0, lb, ub = [], [], []
    for pk in peaks:
        p0 += [y[pk], x[pk], 0.08, 0.3]
        lb += [0, x[pk] - 0.05, 0.01, 0.0]
        ub += [5 * y[pk], x[pk] + 0.05, 0.5, 1.0]
    p0 += [np.median(ys), 0]; lb += [-np.inf, -np.inf]; ub += [np.inf, np.inf]
    p, _ = curve_fit(model, xs, ys, p0=p0, bounds=(lb, ub), maxfev=20000)
    return [(p[4 * k + 1], p[4 * k + 2], p[4 * k + 3]) for k in range(n)]


def run(path):
    df = pd.read_excel(path, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601)
    x = df.iloc[:, 0].to_numpy(float)
    left, right = x <= 17.8, x >= 20.2
    rows = []
    for s in "ABC":
        for v in VOLTS:
            y = df[f"{s}_{v}"].to_numpy(float)
            edge = left | right
            noise = np.std(y[edge])
            peaks, _ = find_peaks(y, prominence=20 * noise, distance=20)
            const = np.full_like(y, np.median(y[edge]))
            bl, br = np.median(y[left]), np.median(y[right])
            line = bl + (br - bl) * (x - 17.65) / (20.35 - 17.65)
            m1 = read_peaks(x, y, const, peaks)
            m2 = read_peaks(x, y, line, peaks)
            m3 = fit_pv(x, y, peaks)
            for k in range(len(peaks)):
                rows.append({
                    "Sample": f"{s}_{v}", "봉우리": k + 1,
                    "2θ_1 현재": round(m1[k][0], 3), "2θ_2 직선": round(m2[k][0], 3), "2θ_3 피팅": round(m3[k][0], 4),
                    "FWHM_1": round(m1[k][1], 4), "FWHM_2": round(m2[k][1], 4), "FWHM_3": round(m3[k][1], 4),
                    "η(로렌츠 비율)": round(m3[k][2], 2),
                    "c_1": round(c_of(m1[k][0]), 3), "c_2": round(c_of(m2[k][0]), 3), "c_3": round(c_of(m3[k][0]), 3),
                    "배경 왼쪽": round(bl, 1), "배경 오른쪽": round(br, 1),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tab = run(sys.argv[1] if len(sys.argv) > 1 else "배터리_XRD_Project.xlsx")
    tab.to_csv("XRD_방법비교.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))
