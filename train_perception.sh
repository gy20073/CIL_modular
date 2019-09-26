#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_SqnoiseShoulder_rfsv6_withTL_fixTL \
    -m 0.05


