#!/usr/bin/env bash


TownName="Exp_Town01_01CrossWalk"
townid="12"

port=2400
declare -a stages=("normal" "park_withcar" "park_nocar" "shoulder")

par_level=4
# config ends

TownName="Exp_Town01_02Shoulder"
townid="13"

port=2500
declare -a stages=("normal")

par_level=2
# config ends



for stage in "${stages[@]}"
do
    inputds=$TownName"_"$stage
    outputds=$inputds"_way"

    python drive_multi_config.py --townname $TownName --port $port --parallel $par_level --mode $stage
    python utils/compute_waypoints.py -input=$inputds -output=$outputds
    python utils/mark_h5_townid.py --dataset=$outputds --townid=$townid
done
