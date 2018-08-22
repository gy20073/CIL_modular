#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_perception_straight3constantaug_lessdrop_yangv2net_segonly_wd2e_5 \
    -m 0.1
