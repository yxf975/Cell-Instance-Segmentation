#!/usr/bin/env bash

CONFIG=$1
MODEL=$2

python -m ./evaluation/hp_tuning.py $CONFIG $MODEL
python -m ./evaluation/evalution.py $CONFIG $MODEL
