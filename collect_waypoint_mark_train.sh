#!/usr/bin/env bash

inputds="exptown_v19_weather"
expid="mm45_v4_SqnoiseShoulder_exptownv19_notown0102"
policy_gpu=2

townid="11"
# config ends

outputds=$inputds"_way"

# TODO modify drive_multi_config, 9cam_agent_carla_acquire_rc_batch_095 and then
python drive_multi_config.py

#pkill -f -9 CarlaUE4

python utils/compute_waypoints.py -input=$inputds -output=$outputds
python utils/mark_h5_townid.py --dataset=$outputds --townid=$townid

# TODO: get the exp config ready
export CUDA_VISIBLE_DEVICES=$policy_gpu
ulimit -Sn 600000
python chauffeur.py train \
    -e $expid \
    -m 0.05
