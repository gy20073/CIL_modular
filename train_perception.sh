#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_base_newseg \
    -m 0.1

