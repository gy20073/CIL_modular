#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_base_newseg_noiser_TL_lane_structure02_goodsteer_3cam \
    -m 0.1

