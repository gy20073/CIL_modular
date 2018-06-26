#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
python chauffeur.py train \
    -e mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_yang_alldata \
    -m 0.4
