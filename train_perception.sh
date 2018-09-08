#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v4_base_newseg_noiser_TL_structure_noise \
    -m 0.1

