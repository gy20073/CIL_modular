#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_SqnoiseShoulder_rfsv6_withTL \
    -m 0.05
