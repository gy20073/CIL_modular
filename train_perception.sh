#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_SqnoiseShoulder_noshoulder \
    -m 0.05

