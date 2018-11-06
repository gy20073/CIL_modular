#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_wp2town3cam_stacked_control \
    -m 0.05
