#!/usr/bin/env bash

# you could switch the carla config through the command line, thus it's easy to loop over all carla config files.
# some pending flags
# --carla-config
# --show_screen
# --noise


positions_file=$1
python2 chauffeur.py \
    drive \
    --log \
    --debug \
    --driver-config "9cam_agent_carla_acquire_rc_yang_human_demo" --positions_file $positions_file
