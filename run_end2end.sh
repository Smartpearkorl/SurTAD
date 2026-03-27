#!/bin/bash

echo "===== Running: 选择测试集 ====="
python /home/qh/TDD/SurTAD/QT_label.py
wait

echo "===== Running: 推理中 ====="

/home/qh/TDD/SurTAD/run_end2end.sh
wait