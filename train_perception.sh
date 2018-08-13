#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_from_alldata_perception_v4 \
    -m 0.15
