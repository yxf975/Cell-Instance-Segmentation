#!/bin/sh
cd /bhome/jgeng/Projects/CellSegmentation/Cell-Instance-Segmentation;

CONFIG=$1
MODEL=$2

python3 -m ./evaluation/hp_tuning.py $CONFIG $MODEL &&
python3 -m ./evaluation/evalution.py $CONFIG $MODEL
