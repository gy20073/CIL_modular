#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v5_perception_allpercep_wd5e_6 \
    -m 0.1
