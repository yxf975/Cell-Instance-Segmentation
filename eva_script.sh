#!/usr/bin/env bash

CONFIG=$1
MODEL=$2
echo "config path：$CONFIG";
echo "model path：$MODEL";

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m ./evaluation/hp_tuning.py --config=$CONFIG model=$MODEL
