#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_base_3cam_conaug_noshare \
    -m 0.1

