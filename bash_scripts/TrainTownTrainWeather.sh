#!/bin/bash
echo "Activating environment"
source activate chauffeur
echo "Taking a break..."
sleep 5
#./CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000 -benchmark -fps=10  ##Train Town

: <<'COMMENT'
#####CVPR25
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_cvpr25_rgb_cluster -s results/mm01_cvpr25_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_cvpr25_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_cvpr25_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm02_cvpr25_seg_gt_cluster -s results/mm02_cvpr25_seg_gt_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm02_cvpr25_seg_gt_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm02_cvpr25_seg_gt_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_cvpr25_seg_gt_oh_cluster -s results/mm03_cvpr25_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_cvpr25_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_cvpr25_seg_gt_oh_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_cvpr25_seg_erfnet_cluster -s results/mm04_cvpr25_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_cvpr25_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_cvpr25_seg_erfnet_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm11_cvpr25_mm01_aug_cluster -s results/mm11_cvpr25_mm01_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm11_cvpr25_mm01_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm11_cvpr25_mm01_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm41_cvpr25_mm04_aug_cluster -s results/mm41_cvpr25_mm04_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm41_cvpr25_mm04_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm41_cvpr25_mm04_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm12_cvpr25_mm01_no_schedule_new_weights_cluster -s results/mm12_cvpr25_mm01_no_schedule_new_weights_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm12_cvpr25_mm01_no_schedule_new_weights_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm12_cvpr25_mm01_no_schedule_new_weights_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm22_cvpr25_mm02_no_schedule_new_weights_cluster -s results/mm22_cvpr25_mm02_no_schedule_new_weights_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm22_cvpr25_mm02_no_schedule_new_weights_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm22_cvpr25_mm02_no_schedule_new_weights_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm42_cvpr25_mm04_no_schedule_new_weights_cluster -s results/mm42_cvpr25_mm04_no_schedule_new_weights_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm42_cvpr25_mm04_no_schedule_new_weights_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm42_cvpr25_mm04_no_schedule_new_weights_cluster_TrainT_TrainW1.summary
COMMENT

: <<'COMMENT'
#####CVPR5
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_cvpr5_rgb_cluster -s results/mm01_cvpr5_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_cvpr5_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_cvpr5_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm02_cvpr5_seg_gt_cluster -s results/mm02_cvpr5_seg_gt_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm02_cvpr5_seg_gt_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm02_cvpr5_seg_gt_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_cvpr5_seg_gt_oh_cluster -s results/mm03_cvpr5_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_cvpr5_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_cvpr5_seg_gt_oh_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_cvpr5_seg_erfnet_cluster -s results/mm04_cvpr5_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_cvpr5_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_cvpr5_seg_erfnet_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm11_cvpr5_mm01_aug_cluster -s results/mm11_cvpr5_mm01_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm11_cvpr5_mm01_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm11_cvpr5_mm01_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm41_cvpr5_mm04_aug_cluster -s results/mm41_cvpr5_mm04_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm41_cvpr5_mm04_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm41_cvpr5_mm04_aug_cluster_TrainT_TrainW1.summary
COMMENT

: <<'COMMENT'
#####CVPR1
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_cvpr1_rgb_cluster -s results/mm01_cvpr1_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_cvpr1_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_cvpr1_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm02_cvpr1_seg_gt_cluster -s results/mm02_cvpr1_seg_gt_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm02_cvpr1_seg_gt_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm02_cvpr1_seg_gt_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_cvpr1_seg_gt_oh_cluster -s results/mm03_cvpr1_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_cvpr1_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_cvpr1_seg_gt_oh_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_cvpr1_seg_erfnet_cluster -s results/mm04_cvpr1_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_cvpr1_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_cvpr1_seg_erfnet_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm11_cvpr1_mm01_aug_cluster -s results/mm11_cvpr1_mm01_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm11_cvpr1_mm01_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm11_cvpr1_mm01_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm41_cvpr1_mm04_aug_cluster -s results/mm41_cvpr1_mm04_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm41_cvpr1_mm04_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm41_cvpr1_mm04_aug_cluster_TrainT_TrainW1.summary


: <<'COMMENT'
#####RC1
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc1_rgb_cluster -s results/mm01_rc1_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc1_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc1_rgb_cluster_TrainT_TrainW1.summary

#####RC5
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc5_rgb_cluster -s results/mm01_rc5_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc5_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc5_rgb_cluster_TrainT_TrainW1.summary

#####RC25
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc25_rgb_cluster -s results/mm01_rc25_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc25_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc25_rgb_cluster_TrainT_TrainW1.summary

#####RC1
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_rc1_seg_gt_oh_cluster -s results/mm03_rc1_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_rc1_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_rc1_seg_gt_oh_cluster_TrainT_TrainW1.summary

#####RC5
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_rc5_seg_gt_oh_cluster -s results/mm03_rc5_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_rc5_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_rc5_seg_gt_oh_cluster_TrainT_TrainW1.summary

#####RC25
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_rc25_seg_gt_oh_cluster -s results/mm03_rc25_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_rc25_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_rc25_seg_gt_oh_cluster_TrainT_TrainW1.summary


#####RC1
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc1_seg_erfnet_cluster -s results/mm04_rc1_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc1_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc1_seg_erfnet_cluster_TrainT_TrainW1.summary

#####RC5
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc5_seg_erfnet_cluster -s results/mm04_rc5_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc5_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc5_seg_erfnet_cluster_TrainT_TrainW1.summary

#####RC25
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc25_seg_erfnet_cluster -s results/mm04_rc25_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc25_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc25_seg_erfnet_cluster_TrainT_TrainW1.summary
COMMENT

: <<'COMMENT'

##########################1h
#####RC11wpz
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc11_wpz_1h_rgb_cluster -s results/mm01_rc11_wpz_1h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc11_wpz_1h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc11_wpz_1h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc11_wpz_1h_seg_erfnet_cluster -s results/mm04_rc11_wpz_1h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc11_wpz_1h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc11_wpz_1h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC7w
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc7_w_1h_rgb_cluster -s results/mm01_rc7_w_1h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc7_w_1h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc7_w_1h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc7_w_1h_seg_erfnet_cluster -s results/mm04_rc7_w_1h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc7_w_1h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc7_w_1h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC6pz
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc6_pz_1h_rgb_cluster -s results/mm01_rc6_pz_1h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc6_pz_1h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc6_pz_1h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc6_pz_1h_seg_erfnet_cluster -s results/mm04_rc6_pz_1h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc6_pz_1h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc6_pz_1h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC3p
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc3_p_1h_rgb_cluster -s results/mm01_rc3_p_1h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc3_p_1h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc3_p_1h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc3_p_1h_seg_erfnet_cluster -s results/mm04_rc3_p_1h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc3_p_1h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc3_p_1h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC3z
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc3_z_1h_rgb_cluster -s results/mm01_rc3_z_1h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc3_z_1h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc3_z_1h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc3_z_1h_seg_erfnet_cluster -s results/mm04_rc3_z_1h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc3_z_1h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc3_z_1h_seg_erfnet_cluster_TrainT_TrainW1.summary

##########################2h
#####RC11wpz
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc11_wpz_2h_rgb_cluster -s results/mm01_rc11_wpz_2h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc11_wpz_2h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc11_wpz_2h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc11_wpz_2h_seg_erfnet_cluster -s results/mm04_rc11_wpz_2h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc11_wpz_2h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc11_wpz_2h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC7w
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc7_w_2h_rgb_cluster -s results/mm01_rc7_w_2h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc7_w_2h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc7_w_2h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc7_w_2h_seg_erfnet_cluster -s results/mm04_rc7_w_2h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc7_w_2h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc7_w_2h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC6pz
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc6_pz_2h_rgb_cluster -s results/mm01_rc6_pz_2h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc6_pz_2h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc6_pz_2h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc6_pz_2h_seg_erfnet_cluster -s results/mm04_rc6_pz_2h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc6_pz_2h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc6_pz_2h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC3p
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc3_p_2h_rgb_cluster -s results/mm01_rc3_p_2h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc3_p_2h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc3_p_2h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc3_p_2h_seg_erfnet_cluster -s results/mm04_rc3_p_2h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc3_p_2h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc3_p_2h_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC3z
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc3_z_2h_rgb_cluster -s results/mm01_rc3_z_2h_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc3_z_2h_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc3_z_2h_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc3_z_2h_seg_erfnet_cluster -s results/mm04_rc3_z_2h_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc3_z_2h_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc3_z_2h_seg_erfnet_cluster_TrainT_TrainW1.summary
COMMENT

: <<'COMMENT'
#####RC11wpz
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm41_rc11_wpz_1h_mm04_aug_cluster -s results/mm41_rc11_wpz_1h_mm04_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm41_rc11_wpz_1h_mm04_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm41_rc11_wpz_1h_mm04_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm42_rc11_wpz_1h_mm04_rssAll_cluster -s results/mm42_rc11_wpz_1h_mm04_rssAll_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm42_rc11_wpz_1h_mm04_rssAll_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm42_rc11_wpz_1h_mm04_rssAll_cluster_TrainT_TrainW1.summary



#####RC3pN
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc3_p_1h_N_rgb_cluster -s results/mm01_rc3_p_1h_N_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc3_p_1h_N_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc3_p_1h_N_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc3_p_1h_N_seg_erfnet_cluster -s results/mm04_rc3_p_1h_N_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc3_p_1h_N_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc3_p_1h_N_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC3zN
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc3_z_1h_N_rgb_cluster -s results/mm01_rc3_z_1h_N_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc3_z_1h_N_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc3_z_1h_N_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc3_z_1h_N_seg_erfnet_cluster -s results/mm04_rc3_z_1h_N_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc3_z_1h_N_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc3_z_1h_N_seg_erfnet_cluster_TrainT_TrainW1.summary


#####RC6pzN
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc6_pz_1h_N_rgb_cluster -s results/mm01_rc6_pz_1h_N_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc6_pz_1h_N_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc6_pz_1h_N_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc6_pz_1h_N_seg_erfnet_cluster -s results/mm04_rc6_pz_1h_N_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc6_pz_1h_N_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc6_pz_1h_N_seg_erfnet_cluster_TrainT_TrainW1.summary

COMMENT





: <<'COMMENT'
#####RC28wpzM
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc28_wpz_M_rgb_cluster -s results/mm01_rc28_wpz_M_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc28_wpz_M_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc28_wpz_M_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_rc28_wpz_M_seg_gt_oh_cluster -s results/mm03_rc28_wpz_M_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_rc28_wpz_M_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_rc28_wpz_M_seg_gt_oh_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc28_wpz_M_seg_erfnet_cluster -s results/mm04_rc28_wpz_M_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc28_wpz_M_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc28_wpz_M_seg_erfnet_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm11_rc28_wpz_M_mm01_aug_cluster -s results/mm11_rc28_wpz_M_mm01_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm11_rc28_wpz_M_mm01_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm11_rc28_wpz_M_mm01_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster -s results/mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm45_rc28_wpz_M_mm41_cityscapes_aug_cluster_TrainT_TrainW1.summary






CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm41_rc28_wpz_M_mm04_aug_cluster -s results/mm41_rc28_wpz_M_mm04_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm41_rc28_wpz_M_mm04_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm41_rc28_wpz_M_mm04_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm42_rc28_wpz_M_mm04_rssAll_cluster -s results/mm42_rc28_wpz_M_mm04_rssAll_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm42_rc28_wpz_M_mm04_rssAll_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm42_rc28_wpz_M_mm04_rssAll_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm43_rc28_wpz_M_mm04_rssAll_aug_cluster -s results/mm43_rc28_wpz_M_mm04_rssAll_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm43_rc28_wpz_M_mm04_rssAll_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm43_rc28_wpz_M_mm04_rssAll_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm44_rc28_wpz_M_mm04_cityscapes_aug_cluster -s results/mm44_rc28_wpz_M_mm04_cityscapes_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm44_rc28_wpz_M_mm04_cityscapes_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm44_rc28_wpz_M_mm04_cityscapes_aug_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm46_rc28_wpz_M_mm41_rssAll_cluster -s results/mm46_rc28_wpz_M_mm41_rssAll_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm46_rc28_wpz_M_mm41_rssAll_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm46_rc28_wpz_M_mm41_rssAll_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm47_rc28_wpz_M_mm41_rssAll_aug_cluster -s results/mm47_rc28_wpz_M_mm41_rssAll_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm47_rc28_wpz_M_mm41_rssAll_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm47_rc28_wpz_M_mm41_rssAll_aug_cluster_TrainT_TrainW1.summary



#####RC20wpzM
CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm01_rc20_wpz_M_rgb_cluster -s results/mm01_rc20_wpz_M_rgb_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm01_rc20_wpz_M_rgb_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm01_rc20_wpz_M_rgb_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm03_rc20_wpz_M_seg_gt_oh_cluster -s results/mm03_rc20_wpz_M_seg_gt_oh_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm03_rc20_wpz_M_seg_gt_oh_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm03_rc20_wpz_M_seg_gt_oh_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm04_rc20_wpz_M_seg_erfnet_cluster -s results/mm04_rc20_wpz_M_seg_erfnet_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm04_rc20_wpz_M_seg_erfnet_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm04_rc20_wpz_M_seg_erfnet_cluster_TrainT_TrainW1.summary



CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm11_rc20_wpz_M_mm01_aug_cluster -s results/mm11_rc20_wpz_M_mm01_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm11_rc20_wpz_M_mm01_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm11_rc20_wpz_M_mm01_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm41_rc20_wpz_M_mm04_aug_cluster -s results/mm41_rc20_wpz_M_mm04_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm41_rc20_wpz_M_mm04_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm41_rc20_wpz_M_mm04_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm42_rc20_wpz_M_mm04_rssAll_cluster -s results/mm42_rc20_wpz_M_mm04_rssAll_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm42_rc20_wpz_M_mm04_rssAll_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm42_rc20_wpz_M_mm04_rssAll_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm43_rc20_wpz_M_mm04_rssAll_aug_cluster -s results/mm43_rc20_wpz_M_mm04_rssAll_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm43_rc20_wpz_M_mm04_rssAll_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm43_rc20_wpz_M_mm04_rssAll_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm44_rc20_wpz_M_mm04_cityscapes_aug_cluster -s results/mm44_rc20_wpz_M_mm04_cityscapes_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm44_rc20_wpz_M_mm04_cityscapes_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm44_rc20_wpz_M_mm04_cityscapes_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm45_rc20_wpz_M_mm41_cityscapes_aug_cluster -s results/mm45_rc20_wpz_M_mm41_cityscapes_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm45_rc20_wpz_M_mm41_cityscapes_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm45_rc20_wpz_M_mm41_cityscapes_aug_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm46_rc20_wpz_M_mm41_rssAll_cluster -s results/mm46_rc20_wpz_M_mm41_rssAll_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm46_rc20_wpz_M_mm41_rssAll_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm46_rc20_wpz_M_mm41_rssAll_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm47_rc20_wpz_M_mm41_rssAll_aug_cluster -s results/mm47_rc20_wpz_M_mm41_rssAll_aug_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm47_rc20_wpz_M_mm41_rssAll_aug_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm47_rc20_wpz_M_mm41_rssAll_aug_cluster_TrainT_TrainW1.summary



CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm400_rc28_wpz_M_mm04_steering_cluster -s results/mm400_rc28_wpz_M_mm04_steering_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm400_rc28_wpz_M_mm04_steering_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm400_rc28_wpz_M_mm04_steering_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm401_rc28_wpz_M_mm45_steering_cluster -s results/mm401_rc28_wpz_M_mm45_steering_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm401_rc28_wpz_M_mm45_steering_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm401_rc28_wpz_M_mm45_steering_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm100_rc28_wpz_M_mm01_steering_cluster -s results/mm100_rc28_wpz_M_mm01_steering_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm100_rc28_wpz_M_mm01_steering_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm100_rc28_wpz_M_mm01_steering_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm101_rc28_wpz_M_mm11_steering_cluster -s results/mm101_rc28_wpz_M_mm11_steering_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm101_rc28_wpz_M_mm11_steering_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm101_rc28_wpz_M_mm11_steering_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm300_rc28_wpz_M_mm03_steering_cluster -s results/mm300_rc28_wpz_M_mm03_steering_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm300_rc28_wpz_M_mm03_steering_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm300_rc28_wpz_M_mm03_steering_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm404_rc28_wpz_M_mm402_small_lr_cluster -s results/mm404_rc28_wpz_M_mm402_small_lr_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm404_rc28_wpz_M_mm402_small_lr_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm404_rc28_wpz_M_mm402_small_lr_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm405_rc28_wpz_M_mm403_small_lr_cluster -s results/mm405_rc28_wpz_M_mm403_small_lr_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm405_rc28_wpz_M_mm403_small_lr_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm405_rc28_wpz_M_mm403_small_lr_cluster_TrainT_TrainW1.summary


CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm402_rc28_wpz_M_mm04_finetune_cluster -s results/mm402_rc28_wpz_M_mm04_finetune_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm402_rc28_wpz_M_mm04_finetune_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm402_rc28_wpz_M_mm04_finetune_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm403_rc28_wpz_M_mm45_finetune_cluster -s results/mm403_rc28_wpz_M_mm45_finetune_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm403_rc28_wpz_M_mm45_finetune_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm403_rc28_wpz_M_mm45_finetune_cluster_TrainT_TrainW1.summary

CUDA_VISIBLE_DEVICES='0' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm407_rc28_wpz_M_mm403_medium_lr_cluster -s results/mm407_rc28_wpz_M_mm403_medium_lr_cluster_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.25 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm407_rc28_wpz_M_mm403_medium_lr_cluster_TrainT_TrainW1 -w 1 |& tee -a results/mm407_rc28_wpz_M_mm403_medium_lr_cluster_TrainT_TrainW1.summary

COMMENT

CUDA_VISIBLE_DEVICES='1' python2 drive_interfaces/carla/comercial_cars/run_test_cvpr.py -e mm108_rc28_wpz_M_mm11_dr -s results/mm108_rc28_wpz_M_mm11_dr_TrainT_TrainW1  -l 127.0.0.1 -p 2000 -t 2 -cy carla_1 -w 1  -m 0.1 || true #TrainTownTrainWeather
echo "Taking a break..."
sleep 10
echo "Computings stats..."
python tools/compute_statistics_carla.py results/mm108_rc28_wpz_M_mm11_dr_TrainT_TrainW1 -w 1 |& tee -a results/mm108_rc28_wpz_M_mm11_dr_TrainT_TrainW1.summary

echo "ALL DONE"
