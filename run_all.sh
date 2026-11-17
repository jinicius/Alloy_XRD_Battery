#!/usr/bin/env bash
# 전체 분석 재현: bash run_all.sh data/project.xlsx data/source_data.xlsx
set -e
PROJ=${1:-data/project.xlsx}
SRC=${2:-data/source_data.xlsx}
OUT=out; mkdir -p $OUT/xrd $OUT/paper $OUT/ml
(cd $OUT/xrd && python ../../src/xrd/xrd_all_samples.py ../../$PROJ && python ../../src/xrd/xrd_method_check.py ../../$PROJ)
python src/paper/paper_compare.py $SRC $PROJ > $OUT/paper/compare.txt
python src/paper/paper_figure_3panel.py $SRC $PROJ $OUT/paper/paper_vs_project_3panel.png
python src/paper/dev_figures.py $SRC $OUT/paper
python src/ml/ml1_changepoint.py $SRC $PROJ $OUT/ml
python src/ml/ml2_gp_surrogate.py $SRC $OUT/ml
python -W ignore src/ml/ml3_nmf_bic.py $SRC $PROJ $OUT/ml   # 약 5~6분
python src/ml/ml3b_phase_template.py $SRC $PROJ $OUT/ml   # 약 10분
python src/ml/ml5_sintering_xrd.py $SRC $OUT/ml
echo "done -> $OUT/"
