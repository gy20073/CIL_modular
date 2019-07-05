#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_SqnoiseShoulder_full_nopark_div8 \
    -m 0.05

