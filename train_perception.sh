#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
ulimit -Sn 60000
python chauffeur.py train \
    -e mm45_v5_perception_allpercep_nowd \
    -m 0.1

