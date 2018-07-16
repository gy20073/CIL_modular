#!/usr/bin/env bash

python \
tools/compute_statistics_carla.py \
results/mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster \
-w 1 |& tee -a results/mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster.summary
