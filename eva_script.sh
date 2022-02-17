#!/usr/bin/env bash

CONFIG=$1
MODEL=$2
echo "第一个参数为：$CONFIG";
echo "第二个参数为：$MODEL";

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m ./evaluation/hp_tuning.py $CONFIG $MODEL
