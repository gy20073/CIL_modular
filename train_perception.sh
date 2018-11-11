#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_wp2town3cam_3town \
    -m 0.05
