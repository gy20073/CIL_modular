#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_wp2town3cam_2p3town_ft \
    -m 0.05
