#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_PcSensordropLessmap_rfsv5_lanecolor_accurate_map \
    -m 0.05
