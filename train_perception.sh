#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_wp2town3cam_parallel_control_2p2town_real \
    -m 0.05
