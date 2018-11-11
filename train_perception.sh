#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_wp2town3cam_correct_town2 \
    -m 0.05
