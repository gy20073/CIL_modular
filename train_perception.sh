#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
ulimit -Sn 600000
python chauffeur.py train \
    -e mm45_v4_base_newseg_noiser_TL_lane_structure02_goodsteer_waypoint_zoom_stdnorm_v5_3cam_abn \
    -m 0.05
