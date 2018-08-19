#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_matthias_nowp_shallow \
    -m 0.1
