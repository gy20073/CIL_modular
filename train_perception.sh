#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_SqnoiseShoulder_v19v25v26_expmaplessshoulder_onshoulder \
    -m 0.05


