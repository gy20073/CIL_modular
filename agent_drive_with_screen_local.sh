#!/usr/bin/env bash

# you could switch the carla config through the command line, thus it's easy to loop over all carla config files.
# some pending flags
# --carla-config
# --show_screen
# --noise

# ./CarlaUE4.sh Town03 -windowed -ResX=90 -ResY=90
# CARLA_VERSION="0.9.X" bash agent_drive_with_screen_local.sh town03_positions/intersections.csv

positions_file=$1
python2 chauffeur.py \
    drive \
    --log \
    --debug \
    --driver-config "9cam_agent_carla_acquire_rc_yang_human_demo" --positions_file $positions_file

