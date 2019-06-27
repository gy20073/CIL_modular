#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_SqnoiseShoulder_exptownv9v23_notown0102 \
    -m 0.05

