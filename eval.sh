#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
ulimit -Sn 60000

rm ./temp/*png
rm ./temp/directions.txt
rm ./temp/carla_0_debug.png

python \
drive_interfaces/carla/comercial_cars/run_test_cvpr.py \
-e mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster \
-s mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster \
-l 127.0.0.1 \
-p 2003 \
-t 2 \
-cy Town02 \
-w 1 \
-m 0.25 || true #TestTownTrainWeather

ffmpeg -i ./temp/%05d.png -c:v libx264  ./temp/out.mp4
rm ./temp/0*png
