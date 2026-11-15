"""
ALLOY 배터리조 XRD 프로젝트 - 시료 A·B·C 통합 판독 스크립트 (분석 절차 01~04단계)

사용법
    python xrd_all_samples.py <원본 엑셀 경로>
    예) python xrd_all_samples.py 배터리_XRD_Project.xlsx

출력 (실행한 폴더에 생성)
    XRD_시료A_그래프.png / XRD_시료B_그래프.png / XRD_시료C_그래프.png
        : 시료별 4전압 겹침 그래프 (4.08 V 기준선, 4.30 V 봉우리 2θ 표기, 처리 조건 캡션)
    XRD_ABC_피크표.csv
        : 전압·봉우리별 2θ, θ, d, c, FWHM, 대칭비, 어깨, 골, Δ2θ, Δc (판독 근거 전체)
    배터리조_XRD_Project_ABC.xlsx
        : 원본 복사본. 03_분석 피크 계산표 17~30행 노란 칸(A·B·C·F·G·H열)만 채움.
          원본 데이터·수식 칸(D·E열)은 그대로 둠

판독 기준 (팀 공통)
    1. 원본 데이터 그대로 사용. 스무딩·정규화·세로 offset 없음
    2. 배경(바닥) = 양 끝 0.3° 구간(17.50~17.80°, 20.20~20.50°) counts의 중앙값
       잡음 σ = 같은 구간 counts의 표준편차
    3. 봉우리 = 돌출 높이(prominence)가 잡음 σ의 20배 이상인 극대점
       어깨 후보 = 돌출 높이가 σ의 10배 이상 20배 미만인 작은 극대점 (봉우리 ±0.4° 안)
    4. 2θ = 봉우리 구간 안의 실제 최대 counts 위치 (엑셀 INDEX/MATCH/MAX와 같은 값, 0.005° 단위)
       봉우리가 2개 이상이면 둘 사이 가장 낮은 점(골)에서 구간을 나눔
    5. FWHM = (최고점 − 배경)의 절반 높이에서 왼쪽·오른쪽 교차점 사이 폭
       교차점은 이웃 두 데이터 점 사이를 직선 보간. 골이 반치보다 높으면(겹침) 산출하지 않음
    6. 대칭성 = 오른쪽 반폭 / 왼쪽 반폭 (중심은 꼭대기 3점 포물선 보정). 0.8~1.25면 대칭
    7. θ = 2θ/2, d = λ / (2 sinθ), λ = 1.5406 Å (Cu Kα), c = 3d
    8. 이동량은 같은 시료의 4.08 V 대비로만 계산 (시료 간 절대 위치 차이는 '이동'이 아님)

필요 패키지
    pip install pandas openpyxl matplotlib scipy
"""
import math
import re
import sys
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import find_peaks

LAMBDA = 1.5406                        # Å, Cu Kα
SAMPLES = ["A", "B", "C"]
VOLTS = ["4.08V", "4.30V", "4.50V", "4.80V"]
COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]   # 저전압 파랑 → 고전압 빨강
BG_WIN = 0.30                          # 배경 계산용 양 끝 구간 폭 (deg)
PEAK_SIGMA = 20                        # 봉우리 기준: 잡음의 20배
SHOULDER_SIGMA = 10                    # 어깨 후보 기준: 잡음의 10배
TABLE_ROWS = range(17, 31)             # 03_분석 피크 계산표 행 (14줄)


# ---------------------------------------------------------------- 계산
def d_spacing(two_theta):
    return LAMBDA / (2 * math.sin(math.radians(two_theta / 2)))


def half_crossing(x, y, i_peak, level, direction):
    """봉우리에서 direction(-1 왼쪽 / +1 오른쪽)으로 가며 y가 level 아래로 처음 내려가는 2θ (직선 보간)."""
    i = i_peak
    while 0 < i < len(y) - 1:
        j = i + direction
        if y[j] < level:
            return x[i] + (x[j] - x[i]) * (y[i] - level) / (y[i] - y[j])
        i = j
    return np.nan


def load(path):
    df = pd.read_excel(path, sheet_name="01_XRD", header=3, usecols="A:M", nrows=601)
    x = df.iloc[:, 0].to_numpy(float)
    assert len(x) == 601 and np.allclose(np.diff(x), 0.005), "2θ 열이 17.5~20.5°, 0.005° 간격이 아님"
    return df, x


def analyze_sample(df, x, sample):
    rows, curves = [], {}
    for v in VOLTS:
        y = df[f"{sample}_{v}"].to_numpy(float)
        edge = (x <= x[0] + BG_WIN) | (x >= x[-1] - BG_WIN)
        bg, noise = float(np.median(y[edge])), float(np.std(y[edge]))
        curves[v] = y

        peaks, _ = find_peaks(y, prominence=PEAK_SIGMA * noise, distance=20)
        small, _ = find_peaks(y, prominence=SHOULDER_SIGMA * noise, distance=10)
        shoulders = [i for i in small if i not in peaks]

        valleys = [peaks[k] + int(np.argmin(y[peaks[k]:peaks[k + 1] + 1]))
                   for k in range(len(peaks) - 1)]
        bounds = [0] + valleys + [len(x) - 1]
        for k in range(len(peaks)):
            lo, hi = bounds[k], bounds[k + 1]
            ip = lo + int(np.argmax(y[lo:hi + 1]))
            level = bg + (y[ip] - bg) / 2
            left = half_crossing(x, y, ip, level, -1)
            right = half_crossing(x, y, ip, level, +1)
            if left < x[lo] or right > x[hi]:
                left = right = np.nan
            y0, y1, y2 = y[ip - 1], y[ip], y[ip + 1]
            center = x[ip] + 0.005 * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            ratio = (right - center) / (center - left)
            tt = round(float(x[ip]), 3)
            d = d_spacing(tt)
            v_idx = valleys[min(k, len(valleys) - 1)] if valleys else None
            rows.append({
                "Sample": f"{sample}_{v}", "시료": sample, "전압": v,
                "봉우리 수": len(peaks), "봉우리 번호": k + 1,
                "2θ (deg)": tt, "θ (deg)": tt / 2, "d (Å)": round(d, 4), "c = 3d (Å)": round(3 * d, 3),
                "FWHM (deg)": round(right - left, 3) if not np.isnan(right - left) else np.nan,
                "대칭비 R/L": round(ratio, 2) if not np.isnan(ratio) else np.nan,
                "어깨 후보": ", ".join(f"{x[s]:.3f}" for s in shoulders if abs(x[s] - x[ip]) < 0.4) or "없음",
                "골 2θ": round(float(x[v_idx]), 3) if v_idx is not None else np.nan,
                "골 counts": round(float(y[v_idx]), 1) if v_idx is not None else np.nan,
                "최고점 counts": round(float(y[ip]), 1), "배경 counts": round(bg, 1), "잡음 σ": round(noise, 1),
            })
    tab = pd.DataFrame(rows)
    ref = tab[tab["전압"] == "4.08V"].iloc[0]
    tab["Δ2θ vs 4.08V"] = (tab["2θ (deg)"] - ref["2θ (deg)"]).round(3)
    tab["Δc vs 4.08V (Å)"] = (tab["c = 3d (Å)"] - ref["c = 3d (Å)"]).round(3)
    tab["c 변화율 (%)"] = (tab["Δc vs 4.08V (Å)"] / ref["c = 3d (Å)"] * 100).round(2)
    return x, curves, tab


# ---------------------------------------------------------------- 엑셀 G·H열 문장 (관찰 / 해석)
def sym_text(r):
    if np.isnan(r["대칭비 R/L"]):
        return "대칭성 판단 불가(겹침)"
    return f"{'대칭' if 0.8 <= r['대칭비 R/L'] <= 1.25 else '비대칭'}({r['대칭비 R/L']:.2f})"


def write_sentences(tab):
    """설명서 3.3절 허용 표현 범위에서 관찰(G)·해석(H) 문장을 만든다. 상전이 판정(05단계)은 쓰지 않음."""
    out = []
    singles = tab[tab["봉우리 수"] == 1].set_index("전압")["c = 3d (Å)"]
    for _, r in tab.iterrows():
        sh = "어깨 없음" if r["어깨 후보"] == "없음" else f"어깨 후보 {r['어깨 후보']}°"
        move = "" if r["전압"] == "4.08V" else (
            f"4.08 V 대비 {r['Δ2θ vs 4.08V']:+.2f}° {'고각' if r['Δ2θ vs 4.08V'] > 0 else '저각'} 이동. ")
        c_txt = f"c = 3 × {r['d (Å)']:.4f} = {r['c = 3d (Å)']:.2f} Å."
        if r["봉우리 수"] == 1:
            g = f"단일 봉우리 · {sym_text(r)} · {sh}. {move}".strip()
            if r["전압"] == "4.30V":
                g += " 측정한 4.30 V에서는 분리된 봉우리가 관찰되지 않음"
            if r["전압"] == "4.08V":
                g += " 이 위치를 기준선으로 사용"
                h = f"{c_txt} 시료 {r['시료']}의 출발 c값(이후 이동량의 기준)"
            else:
                h = f"{c_txt} 4.08 V 대비 {r['Δc vs 4.08V (Å)']:+.2f} Å({r['c 변화율 (%)']:+.1f}%)."
                if r["전압"] == "4.80V":
                    h += f" 4.08→4.80 V c 수축률 {-r['c 변화율 (%)']:.2f}%(기준 4.08 V). a축·부피는 (003)만으로 산출 불가"
        else:
            n, k = r["봉우리 수"], r["봉우리 번호"]
            side = "저각" if k == 1 else ("고각" if k == n else f"{k}번째")
            g = f"{r['전압'][:-1]} V에서 봉우리 {n}개 중 {side} 쪽. {move}{sym_text(r)} · {sh}."
            if k == 1:
                bg, half = r["배경 counts"], r["배경 counts"] + (r["최고점 counts"] - r["배경 counts"]) / 2
                if r["골 counts"] < bg + 5 * r["잡음 σ"]:
                    sep = "골이 바닥까지 내려감 = 완전히 분리됨"
                elif r["골 counts"] < half:
                    sep = "골이 반치 아래까지 내려감 = 분리됨(일부 겹침)"
                else:
                    sep = "골이 반치보다 높음 = 겹침"
                g += f" 두 봉우리 사이 {sep}({r['골 2θ']:.3f}°, {r['골 counts']:.0f} counts)"
            else:
                first = tab[(tab["Sample"] == r["Sample"]) & (tab["봉우리 번호"] == 1)].iloc[0]
                comp = "낮음" if r["최고점 counts"] < first["최고점 counts"] else "높음"
                g += f" 높이는 저각 성분보다 {comp}({r['최고점 counts']:.0f} vs {first['최고점 counts']:.0f} counts)"
            others = singles.drop(r["전압"], errors="ignore")
            near_v = (others - r["c = 3d (Å)"]).abs().idxmin()
            h = (f"{c_txt} 4.08 V 대비 {r['Δc vs 4.08V (Å)']:+.2f} Å({r['c 변화율 (%)']:+.1f}%). "
                 f"{near_v[:-1]} V의 c({others[near_v]:.2f} Å)에 가까운 성분")
            if k == n:
                h += ". 같은 전압에서 c가 다른 두 성분이 공존(상전이 판정은 05단계). 높이 비로 양적 비율은 산출 불가"
        hkl = "(003)" if r["봉우리 수"] == 1 else f"(003) {side} 성분"
        out.append((r["Sample"], hkl, r["2θ (deg)"], r["FWHM (deg)"], g, h))
    return out


# ---------------------------------------------------------------- 엑셀 기록 (시트 XML 직접 수정)
def fill_excel(src, dst, entries):
    """openpyxl로 저장하면 원본 조건부 서식 일부가 사라지므로 03_분석 시트 XML을 직접 고친다."""
    if len(entries) > len(TABLE_ROWS):
        raise ValueError(f"계산표 칸 부족: {len(entries)}줄 필요, {len(TABLE_ROWS)}줄 있음")
    with zipfile.ZipFile(src) as z:
        files = {n: z.read(n) for n in z.namelist()}
    wb = files["xl/workbook.xml"].decode("utf-8")
    sheet_rid = re.search(r'<x:sheet name="03_분석"[^>]*r:id="([^"]+)"', wb).group(1)
    rels = files["xl/_rels/workbook.xml.rels"].decode("utf-8")
    target = re.search(rf'Id="{sheet_rid}"[^>]*Target="([^"]+)"|Target="([^"]+)"[^>]*Id="{sheet_rid}"', rels)
    sheet_path = "xl/" + (target.group(1) or target.group(2)).lstrip("/").replace("xl/", "")
    s = files[sheet_path].decode("utf-8")

    def put(ref, kind, val):
        nonlocal s
        m = re.search(rf'<x:c r="{ref}" s="(\d+)"\s*/>', s)
        if not m:
            raise ValueError(f"{ref} 칸이 비어 있지 않거나 구조가 다름")
        st = m.group(1)
        new = (f'<x:c r="{ref}" s="{st}" t="inlineStr"><x:is><x:t>{escape(str(val))}</x:t></x:is></x:c>'
               if kind == "t" else f'<x:c r="{ref}" s="{st}"><x:v>{val}</x:v></x:c>')
        s = s[:m.start()] + new + s[m.end():]

    for row, (sample, hkl, tt, fwhm, g, h) in zip(TABLE_ROWS, entries):
        put(f"A{row}", "t", sample)
        put(f"B{row}", "t", hkl)
        put(f"C{row}", "n", tt)
        if not (isinstance(fwhm, float) and math.isnan(fwhm)):
            put(f"F{row}", "n", fwhm)
        put(f"G{row}", "t", g)
        put(f"H{row}", "t", h)
        th = tt / 2
        for col, val in (("D", th), ("E", LAMBDA / (2 * math.sin(math.radians(th))))):
            s = re.sub(rf'(<x:c r="{col}{row}" s="\d+") t="str">(<x:f>.*?</x:f>)</x:c>',
                       rf'\1>\2<x:v>{val:.10g}</x:v></x:c>', s, count=1)
        s = s.replace(f'<x:row r="{row}">', f'<x:row r="{row}" ht="78" customHeight="1">', 1)
    files[sheet_path] = s.encode("utf-8")
    if "calcPr" not in wb:
        wb = wb.replace("</x:workbook>", '<x:calcPr fullCalcOnLoad="1" /></x:workbook>')
    files["xl/workbook.xml"] = wb.encode("utf-8")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)


# ---------------------------------------------------------------- 그래프
def set_korean_font():
    for name in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"]:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_sample(x, curves, tab, sample, out):
    set_korean_font()
    tt_ref = tab[tab["전압"] == "4.08V"]["2θ (deg)"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for v, col in zip(VOLTS, COLORS):
        ax.plot(x, curves[v], color=col, lw=1.3, label=v.replace("V", " V"))
    ax.set_ylim(0, max(c.max() for c in curves.values()) * 1.12)
    ax.axvline(tt_ref, color="gray", ls="--", lw=1.2)
    ax.text(tt_ref - 0.03, ax.get_ylim()[1] * 0.97, f"4.08 V 기준선\n{tt_ref:.3f}°",
            ha="right", va="top", fontsize=9, color="dimgray")
    for _, r in tab[tab["전압"] == "4.30V"].iterrows():
        ax.annotate(f"{r['2θ (deg)']:.3f}°", xy=(r["2θ (deg)"], r["최고점 counts"]),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=10, color=COLORS[1], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=COLORS[1], lw=0.8))
    ax.set_xlim(17.5, 20.5)
    ax.set_xticks(np.arange(17.5, 20.51, 0.5))
    ax.set_xlabel("2θ (deg)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"Sample {sample} · (003) peak vs charge voltage (Cu Kα)")
    ax.legend(title="Voltage", loc="upper right")
    ax.grid(alpha=0.25)
    fig.text(0.01, 0.005, "처리 조건: 원본 counts 그대로 표시(배경 제거·스무딩·정규화·세로 offset 없음). "
             "FWHM 계산 때만 배경(양 끝 0.3° 중앙값)을 기준으로 함.",
             fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- 실행
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "배터리_XRD_Project.xlsx"
    df, x = load(src)
    tables, entries = [], []
    for smp in SAMPLES:
        x, curves, tab = analyze_sample(df, x, smp)
        plot_sample(x, curves, tab, smp, f"XRD_시료{smp}_그래프.png")
        tables.append(tab)
        entries += write_sentences(tab)
    full = pd.concat(tables, ignore_index=True)
    full.to_csv("XRD_ABC_피크표.csv", index=False, encoding="utf-8-sig")
    fill_excel(src, "배터리조_XRD_Project_ABC.xlsx", entries)

    pd.set_option("display.width", 200)
    cols = ["Sample", "봉우리 번호", "2θ (deg)", "d (Å)", "c = 3d (Å)", "FWHM (deg)", "대칭비 R/L",
            "Δ2θ vs 4.08V", "Δc vs 4.08V (Å)", "c 변화율 (%)"]
    print(full[cols].to_string(index=False))
    print(f"\n계산표 {len(entries)}줄 기록 → 배터리조_XRD_Project_ABC.xlsx (17~{16 + len(entries)}행)")
