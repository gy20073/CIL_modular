#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_SqnoiseShoulder_exptownv2_notown0102_mergefollowstraight \
    -m 0.05

