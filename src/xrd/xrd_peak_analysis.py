"""
ALLOY 배터리조 XRD 프로젝트 - (003) 피크 판독 스크립트 (분석 절차 01~04단계)

사용법
    python xrd_peak_analysis.py <엑셀 파일 경로> <시료>
    예) python xrd_peak_analysis.py 배터리_XRD_Project.xlsx A

출력 (실행한 폴더에 생성)
    XRD_시료A_그래프.png   : 4전압 겹침 그래프 (4.08 V 기준선, 4.30 V 봉우리 2θ 표기)
    XRD_시료A_피크표.csv   : 전압·봉우리별 2θ, θ, d, c, FWHM, 대칭성, Δc

판독 기준 (팀 공통)
    1. 원본 데이터 그대로 사용. 스무딩·정규화·세로 offset 없음
    2. 배경(바닥) = 양 끝 0.3° 구간(17.50~17.80°, 20.20~20.50°) counts의 중앙값
    3. 봉우리 = (배경 대비) 돌출 높이가 잡음 표준편차의 20배 이상인 극대점
       어깨 후보 = 돌출 높이가 잡음의 10배 이상 20배 미만인 작은 극대점 (봉우리 ±0.4° 안)
    4. 2θ = 봉우리 구간 안의 실제 최대 counts 위치 (엑셀 INDEX/MATCH/MAX와 같은 값, 0.005° 단위)
    5. FWHM = (최고점 - 배경)의 절반 높이에서 왼쪽·오른쪽 교차점 사이 폭.
       교차점은 이웃 두 데이터 점 사이를 직선 보간
    6. 대칭성 = 오른쪽 반폭 / 왼쪽 반폭 (중심은 꼭대기 3점 포물선 보정). 0.8~1.25면 대칭
    7. d = λ / (2 sinθ), θ = 2θ/2, λ = 1.5406 Å (Cu Kα) / c = 3d
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import find_peaks

LAMBDA = 1.5406                       # Å, Cu Kα
VOLTS = ["4.08V", "4.30V", "4.50V", "4.80V"]
COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]   # 저전압 파랑 → 고전압 빨강
BG_WIN = 0.30                         # 배경 계산용 양 끝 구간 폭 (deg)
PEAK_SIGMA = 20                       # 봉우리 기준: 잡음의 20배
SHOULDER_SIGMA = 10                    # 어깨 후보 기준: 잡음의 10배


def set_korean_font():
    for name in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"]:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def d_spacing(two_theta):
    return LAMBDA / (2 * np.sin(np.radians(two_theta / 2)))


def half_crossing(x, y, i_peak, level, direction):
    """봉우리에서 direction(-1 왼쪽 / +1 오른쪽)으로 가며 y가 level 아래로 처음 내려가는 2θ (직선 보간)."""
    i = i_peak
    while 0 < i < len(y) - 1:
        j = i + direction
        if y[j] < level:
            return x[i] + (x[j] - x[i]) * (y[i] - level) / (y[i] - y[j])
        i = j
    return np.nan


def analyze(path, sample):
    df = pd.read_excel(path, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601)
    x = df.iloc[:, 0].to_numpy(float)
    assert len(x) == 601 and np.allclose(np.diff(x), 0.005), "2θ 열이 17.5~20.5°, 0.005° 간격이 아님"

    rows, curves = [], {}
    for v in VOLTS:
        y = df[f"{sample}_{v}"].to_numpy(float)
        edge = (x <= x[0] + BG_WIN) | (x >= x[-1] - BG_WIN)
        bg, noise = np.median(y[edge]), np.std(y[edge])
        curves[v] = y

        peaks, prop = find_peaks(y, prominence=PEAK_SIGMA * noise, distance=20)
        small, _ = find_peaks(y, prominence=SHOULDER_SIGMA * noise, distance=10)
        shoulders = [i for i in small if i not in peaks]

        # 봉우리가 2개 이상이면 봉우리 사이 가장 낮은 점(골)에서 구간을 나눔
        valleys = [peaks[k] + int(np.argmin(y[peaks[k]:peaks[k + 1] + 1]))
                   for k in range(len(peaks) - 1)]
        bounds = [0] + valleys + [len(x) - 1]
        for k in range(len(peaks)):
            lo, hi = bounds[k], bounds[k + 1]
            # 구간 최대값 위치 (엑셀 INDEX/MATCH/MAX와 동일하게 실제 데이터 점)
            ip = lo + int(np.argmax(y[lo:hi + 1]))
            level = bg + (y[ip] - bg) / 2
            left = half_crossing(x, y, ip, level, -1)
            right = half_crossing(x, y, ip, level, +1)
            # 골이 반치보다 높으면 두 봉우리가 겹친 것 → 그 방향 FWHM 무효
            if left < x[lo] or right > x[hi]:
                left = right = np.nan
            fwhm = right - left
            # 대칭성: 0.005° 격자 영향을 줄이려고 꼭대기 3점 포물선으로 중심을 다시 잡고,
            # 반치 지점의 오른쪽 반폭 / 왼쪽 반폭을 비교
            y0, y1, y2 = y[ip - 1], y[ip], y[ip + 1]
            center = x[ip] + 0.005 * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            ratio = (right - center) / (center - left)
            valley_txt = ""
            if len(peaks) > 1:
                v_idx = valleys[min(k, len(valleys) - 1)]
                valley_txt = f"골 {x[v_idx]:.3f}° · {y[v_idx]:.0f} counts"
            tt = round(x[ip], 3)
            d = d_spacing(tt)
            near_sh = [round(x[s], 3) for s in shoulders if abs(x[s] - x[ip]) < 0.4]
            rows.append({
                "Sample": f"{sample}_{v}",
                "봉우리 수": len(peaks),
                "봉우리 번호": k + 1,
                "2θ (deg)": tt,
                "θ (deg)": tt / 2,
                "d (Å)": round(d, 4),
                "c = 3d (Å)": round(3 * d, 3),
                "FWHM (deg)": round(fwhm, 3),
                "대칭비 R/L": round(ratio, 2),
                "대칭": "대칭" if 0.8 <= ratio <= 1.25 else "비대칭",
                "어깨 후보": ", ".join(map(str, near_sh)) if near_sh else "없음",
                "봉우리 사이 골": valley_txt,
                "최고점 counts": round(y[ip], 1),
                "배경 counts": round(bg, 1),
                "잡음 σ": round(noise, 1),
            })

    tab = pd.DataFrame(rows)
    c_ref = tab.loc[tab["Sample"] == f"{sample}_4.08V", "c = 3d (Å)"].iloc[0]
    tt_ref = tab.loc[tab["Sample"] == f"{sample}_4.08V", "2θ (deg)"].iloc[0]
    tab["Δ2θ vs 4.08V"] = (tab["2θ (deg)"] - tt_ref).round(3)
    tab["Δc vs 4.08V (Å)"] = (tab["c = 3d (Å)"] - c_ref).round(3)
    tab["c 변화율 (%)"] = ((tab["c = 3d (Å)"] - c_ref) / c_ref * 100).round(2)
    return x, curves, tab, tt_ref


def plot(x, curves, tab, tt_ref, sample, out):
    set_korean_font()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for v, col in zip(VOLTS, COLORS):
        ax.plot(x, curves[v], color=col, lw=1.3, label=v.replace("V", " V"))
    ax.axvline(tt_ref, color="gray", ls="--", lw=1.2)
    ax.text(tt_ref - 0.03, ax.get_ylim()[1] * 0.97, f"4.08 V 기준선\n{tt_ref:.3f}°",
            ha="right", va="top", fontsize=9, color="dimgray")
    for _, r in tab[tab["Sample"] == f"{sample}_4.30V"].iterrows():
        ax.annotate(f"{r['2θ (deg)']:.3f}°", xy=(r["2θ (deg)"], r["최고점 counts"]),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=10, color=COLORS[1], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=COLORS[1], lw=0.8))
    ax.set_xlim(17.5, 20.5)
    ax.set_xticks(np.arange(17.5, 20.51, 0.5))
    ax.set_ylim(0, max(v.max() for v in curves.values()) * 1.12)
    ax.set_xlabel("2θ (deg)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"Sample {sample} · (003) peak vs charge voltage (Cu Kα)")
    ax.legend(title="Voltage", loc="upper right")
    ax.grid(alpha=0.25)
    fig.text(0.01, 0.005, "처리 조건: 원본 counts 그대로 표시(배경 제거·스무딩·정규화·세로 offset 없음). FWHM 계산 때만 배경(양 끝 0.3° 중앙값)을 기준으로 함.",
             fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "배터리_XRD_Project.xlsx"
    sample = (sys.argv[2] if len(sys.argv) > 2 else "A").upper()
    x, curves, tab, tt_ref = analyze(path, sample)
    tab.to_csv(f"XRD_시료{sample}_피크표.csv", index=False, encoding="utf-8-sig")
    plot(x, curves, tab, tt_ref, sample, f"XRD_시료{sample}_그래프.png")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(tab.to_string(index=False))
