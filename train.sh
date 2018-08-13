#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_from_alldata_noseg_splitv0_v4 \
    -m 0.3
