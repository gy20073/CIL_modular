#!/usr/bin/env bash

# resource related
gpu_carla="0"
gpu_agent="1"
port="2009"
# test related
city_name="Town02"
exp_id="mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_yang_alldata"


# launch carla
export CUDA_VISIBLE_DEVICES=$gpu_carla
/scratch/yang/aws_data/carla_0.8.4/CarlaUE4.sh /Game/Maps/$city_name -carla-server -carla-world-port=$port &

sleep 20

# launch the client
export CUDA_VISIBLE_DEVICES=$gpu_agent
ulimit -Sn 60000

python \
drive_interfaces/carla/comercial_cars/run_test_cvpr.py \
-e $exp_id \
-s $exp_id \
-l 127.0.0.1 \
-p $port \
-cy $city_name \
-m 0.25 || true #TestTownTrainWeather

