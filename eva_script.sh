#!/usr/bin/env bash

CONFIG=$1
MODEL=$2
echo "config path：$CONFIG";
echo "model path：$MODEL";

python ./evaluation/hp_tuning.py $CONFIG $MODEL &&
python ./evaluation/evaluation.py $CONFIG $MODEL &&
