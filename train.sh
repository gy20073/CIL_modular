#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_from_alldata_noseg \
    -m 0.4
