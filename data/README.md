# data/

원본 데이터는 저장소에 올리지 않습니다. 아래 두 파일을 이 폴더에 넣고 실행하세요.

| 파일명 (권장) | 내용 | 출처 |
|---|---|---|
| `project.xlsx` | 배터리조 XRD Project 원본 엑셀 (01_XRD, 02_물성·성능, 03_분석 시트) | ALLOY 배터리조 내부 자료 |
| `source_data.xlsx` | 논문 본문 그림 원자료 (Source Data, Fig. 2–3) | Yao et al., *Nat. Commun.* 16, 10199 (2025), doi:10.1038/s41467-025-63258-9 → Source data |

참고
- 논문 Source Data의 `Figure 3a for s-LNO` 시트는 앞 75열이 d-LNO와 같은 값으로 들어 있어 분석에서 뺐습니다.
- Table S1(소성 시간별 암염상 비율, 결정 크기)은 보충자료(SI) PDF에서 옮겨 `src/ml/ml2_gp_surrogate.py` 안에 상수로 넣었습니다.
