#!/usr/bin/env bash

expid="mm45_v4_SqnoiseShoulder_rfsv6_withTL_lessmap"
OFFSET=90
declare -a GPU=(7 7 7)

# setting output related
output_prefix="/home/yang/data/aws_data/CIL_modular_data/benchmark_all/"
mkdir $output_prefix$expid
output_prefix=$output_prefix$expid"/"

# the town01
output_folder="/home/yang/data/aws_data/CIL_modular_data/_benchmarks_results/"$expid"_1,2,3,4,5,6,7,8,9,10,11,12,13,14_YangExp3cam_Town01/_images/"
ln -s $output_folder $output_prefix"Town01"
gpu_str="["${GPU[0]}","${GPU[1]}","${GPU[2]}"]"
# this has the form of "[2,3,4]"
python eval_par.py -gpu $gpu_str -expid $expid &


# RFS parked car
this_output=$output_prefix$"RFS_parked_car/"
mkdir $this_output
this_output=$this_output$"video"
/home/yang/data/aws_data/carla_rfs/CarlaUE4.sh -benchmark -fps=5 -carla-world-port=$((2600 + OFFSET)) &
export CARLA_VERSION="0.9.auto2"
sleep 25
python utils/eval_inenv_carla_090.py \
    --gpu ${GPU[0]} \
    --condition 2.0 \
    --expid $expid \
    --output_video_path $this_output \
    --starting_points "town03_intersections/positions_file_RFS_MAP.parked_car_attract.txt" \
    --parked_car "town03_intersections/positions_file_RFS_MAP.parking_v2.txt" \
    --townid "10" \
    --port $((2600 + OFFSET)) &

#RFS shoulder
this_output=$output_prefix$"RFS_shoulder/"
mkdir $this_output
this_output=$this_output$"video"
/home/yang/data/aws_data/carla_rfs/CarlaUE4.sh -benchmark -fps=5 -carla-world-port=$((2700 + OFFSET)) &
export CARLA_VERSION="0.9.auto2"
sleep 25
python utils/eval_inenv_carla_090.py \
    --gpu ${GPU[1]} \
    --condition 2.0 \
    --expid $expid \
    --output_video_path $this_output \
    --starting_points "town03_intersections/positions_file_RFS_MAP.extra_explore_v3.txt" \
    --parked_car "" \
    --townid "10" \
    --port $((2700 + OFFSET)) &


#ExpTown random
this_output=$output_prefix$"ExpTown_parked_car/"
mkdir $this_output
this_output=$this_output$"video"
/home/yang/data/aws_data/carla_095/CarlaUE4.sh Exp_Town -benchmark -fps=5 -carla-world-port=$((2800 + OFFSET)) &
export CARLA_VERSION="0.9.5"
sleep 25
python utils/eval_inenv_carla_090.py \
    --gpu ${GPU[2]} \
    --condition 2.0 \
    --expid $expid \
    --output_video_path $this_output \
    --starting_points "town03_intersections/positions_file_Exp_Town.parking_attract.txt" \
    --parked_car "town03_intersections/positions_file_Exp_Town.parking.txt" \
    --townid "11" \
    --port $((2800 + OFFSET)) &
