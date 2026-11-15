# Co-free LiNiO₂ XRD 판독 · 논문 검증 · ML 분석

ALLOY 배터리조 XRD 프로젝트 코드입니다.
LiNiO₂ 양극(시료 A·B·C)의 (003) XRD 봉우리로 충전 중 H2→H3 c축 붕괴를 판독하고,
원 논문 데이터로 검증한 뒤, 머신러닝·데이터 분석 4가지를 추가로 돌렸습니다.

> Reading of (003) XRD peaks for Co-free LiNiO₂ cathodes (H2→H3 collapse), validation against
> Yao et al., *Nat. Commun.* 16, 10199 (2025), and four ML / data-science analyses
> (change-point detection, GP surrogate, NMF, BIC model selection).

Team: Kwon Ye Rim, Kim Min Su, Baek Si Yeon, Jeong Jun Su, Han Seok Jin, Hwang Min Jun

## 폴더 구조

```
src/
  xrd/    xrd_peak_analysis.py    시료 1개 판독 (봉우리 2θ, d, c, FWHM, 대칭비)
          xrd_all_samples.py      시료 A·B·C 통합 판독 → 그래프, CSV, 엑셀 03_분석 표 채우기
          xrd_method_check.py     배경 처리 방식에 따른 결과 민감도 비교
  paper/  paper_compare.py        논문 Source Data ↔ 과제 데이터 수치 비교
          paper_figure.py         비교 그림 (4패널)
          paper_figure_3panel.py  비교 그림 (3패널)
          dev_figures.py          4.30 V 2봉우리 피팅, dQ/dV, dc/dV, 성능 trade-off, 부피 변화
  ml/     ml1_changepoint.py      [ML1] c–V 3구간 변화점 탐지 + 부트스트랩 신뢰구간
          ml2_gp_surrogate.py     [ML2] 소성 시간 → 구조 → 성능 가우시안 프로세스 대리 모델
          ml3_nmf_bic.py          [ML3 첫 시도] NMF 상 분해·끝점 혼합 검정, [ML4] 1 vs 2봉우리 BIC
          ml3b_phase_template.py  [ML3 개정] 대칭/비대칭 pseudo-Voigt BIC, 이동 허용 기준 패턴 분해, 구조–유지율 비교
tools/    fix_team_excel.py       팀 엑셀 검토 후 수정
          poster_figs_v1/v2.py    포스터용 영문 그림
results/  figures/, ml/           실행 결과 그림·표 (재현 확인용)
data/     원본 데이터 넣는 곳 (저장소에는 없음, data/README.md 참고)
```

## 실행

```bash
pip install -r requirements.txt
# data/ 에 project.xlsx, source_data.xlsx 넣기
bash run_all.sh data/project.xlsx data/source_data.xlsx    # 결과 → out/
```

개별 실행 예:

```bash
python src/xrd/xrd_all_samples.py data/project.xlsx
python src/ml/ml1_changepoint.py data/source_data.xlsx data/project.xlsx out/
python src/ml/ml2_gp_surrogate.py data/source_data.xlsx out/
python -W ignore src/ml/ml3_nmf_bic.py data/source_data.xlsx data/project.xlsx out/   # 5~6분
python src/ml/ml3b_phase_template.py data/source_data.xlsx data/project.xlsx out/     # 약 10분
```

## 판독 기준 (팀 공통)

1. 원본 데이터 그대로 사용. 스무딩·정규화·세로 offset 없음
2. 배경 = 양 끝 0.3° 구간 counts의 중앙값, 잡음 σ = 같은 구간 표준편차
3. 봉우리 = 돌출 높이 ≥ 20σ, 어깨 = 10~20σ
4. 2θ = 구간 내 최대 counts 위치 (엑셀 INDEX/MATCH/MAX와 같음). 봉우리 2개면 골에서 구간 분리
5. FWHM = (최고점 − 배경)의 절반 높이 폭, 선형 보간
6. 대칭비 = 오른쪽 반폭 / 왼쪽 반폭 (꼭대기 3점 포물선 보정), 0.8~1.25면 대칭
7. d = λ / 2sinθ (Cu Kα, λ = 1.5406 Å), c = 3d
8. 이동량은 같은 시료 4.08 V 대비로만 계산

논문 싱크로트론 데이터(λ = 0.49594 Å)는 d를 구한 뒤 Cu Kα 2θ로 바꿔 비교합니다.

## 주요 결과

| 분석 | 결과 |
|---|---|
| 논문 검증 | 끝점 c값 0.006 Å 이내 일치, 수축률 A 6.94 vs 6.92 %, C 4.40 vs 4.44 % |
| ML1 변화점 | s-LNO 붕괴 시작 4.22 V, d-LNO 4.29 V (95 % CI ±0.01 V). dQ/dV 봉우리는 둘 다 ~4.19 V → d-LNO는 전기화학 반응과 격자 붕괴가 약 90 mV 어긋남 |
| ML2 GP 대리 모델 | 암염상 비율 ↔ 용량 ρ = −1, ↔ 유지율 ρ = +1 (n = 6). 50사이클 후 남는 용량은 6~36 h에서 평평(192~196 mAh/g) |
| ML3 상 분해 | NMF는 봉우리 이동 때문에 실패 → 이동 허용 기준 패턴 분해로 개정. s-LNO 4.30 V만 단일상으로 설명 안 됨(불일치가 d-LNO 최대의 6~8배), d-LNO는 한 상의 c 분포가 0.16 Å까지 넓어지며 연속 수축. 2상 성격이 강할수록 300사이클 유지율 낮음 (논문 s<d, 과제 A<B<C) |
| ML4 BIC | 과제 데이터: 4.30 V A·B만 2봉우리(ΔBIC 2160, 1707). 논문 실측은 비대칭 pseudo-Voigt로 바꿔도 전 구간 2봉우리 선호 → 상대 비교만 유효 |

## 출처

- Yao et al., "Topotaxially grown composite cathodes for cobalt-free high-energy long-life Li-ion batteries," *Nature Communications* 16, 10199 (2025). doi:10.1038/s41467-025-63258-9
- 데이터는 논문 Source Data와 Supplementary Information(Table S1)을 사용했습니다. 원 데이터 권리는 저자에게 있습니다.
