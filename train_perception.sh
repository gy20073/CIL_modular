#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_base_3cam_conaug_share_1cam_middlesplit \
    -m 0.1

