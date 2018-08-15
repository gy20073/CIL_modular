#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_from_alldata_noseg_splitv0_v6_fastdecay \
    -m 0.1
